import copy
from functools import partial
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
)
from ogpo.networks import Actor, Value, ActorVectorField, EditActor, LogParam
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    compute_flow_bc_loss_online,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    compute_td_target,
    compute_td_loss,
    get_q_values as _get_q_values,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
)

class DSRLEXPOAgent(flax.struct.PyTreeNode):
    """Soft Actor-Critic (SAC) noise agent + base flow matching actor agent with Best-of-N sampling.
    Uses separate TrainState objects for actor, critic, and BC flow networks.

    DSRL+EXPO Extension:
    - action_critic_network: Evaluates Q(obs, refined_action) for training edit actor
    - edit_actor_network: Refines flow-refined actions with learned edits
    """

    rng: Any
    actor_network: TrainState  # SAC actor + temperature (noise policy)
    critic_network: TrainState  # Critic + target critic (trained on real actions)
    bc_flow_network: TrainState  # Flow matching actor (frozen in online RL)
    noise_critic_network: TrainState  # Distilled critic operating in noise space

    # DSRL+EXPO components
    action_critic_network: TrainState  # Q_action: Critic + target critic (evaluates final actions)
    edit_actor_network: TrainState  # Edit policy + temperature

    config: Any = nonpytree_field()

    def _denoise_noise_batch(self, observations, noises):
        """Denoise a batch of noise vectors through the frozen BC flow policy."""
        def actor_fn(obs, actions, t, is_encoded=False):
            return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn,
            observations=observations,
            rng=None,
            flow_steps=self.config['flow_steps'],
            act_dim=self.config['noise_dim'],
            act_min=-1.0,
            act_max=1.0,
            clip_intermediate=False,
            noises=noises,
            is_encoded=False,
        )
        return jnp.clip(actions, -1, 1)

    def _scale_noise_for_diffusion(self, noise):
        """Scale actor noise [-1, 1] to diffusion input range [-am, am]."""
        return noise * self.config['action_magnitude']

    def _normalize_noise_for_critic(self, noise):
        """Normalize noise to actor's output range [-1, 1] for noise critic."""
        return jnp.tanh(noise)

    def noise_to_actions(self, observations, actor_noise):
        """Full pipeline: actor noise [-1,1] -> scale -> denoise -> actions [-1,1]."""
        scaled_noise = self._scale_noise_for_diffusion(actor_noise)
        return self._denoise_noise_batch(observations, scaled_noise)

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss (standard single-sample TD target).

        Matches original DSRL: sample ONE noise from actor, denoise → action,
        evaluate target_critic. No Best-of-N for TD targets (BoN causes
        maximization bias → Q-value explosion).
        """
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.actor_network.select('actor')(batch['next_observations'])

        # Sample single noise from actor, scale + denoise → action
        next_noise = next_dist.sample(seed=sample_rng)
        next_actions = self.noise_to_actions(batch['next_observations'], next_noise)

        # Evaluate Q-values with target critic
        next_qs = self.critic_network.select('target_critic')(
            batch['next_observations'], actions=next_actions
        )

        # Aggregate Q-values across ensemble
        rng, agg_rng = jax.random.split(rng)
        next_q = aggregate_q_values(
            next_qs, method=self.config['q_agg'],
            rng=agg_rng, num_qs=self.config['num_qs'],
        )

        # Get temperature for entropy bonus
        temp = self.actor_network.select('temp')()

        # Compute TD target with optional entropy backup
        if self.config['entropy_backup']:
            # Entropy is computed in noise space (the policy's action space)
            next_log_probs = jnp.clip(next_dist.log_prob(next_noise), -100, 100)
            next_q_with_entropy = next_q - temp * next_log_probs
            # horizon_length=1: DSRL stores chunk-level transitions (match reference)
            target_q = compute_td_target(
                rewards=batch['rewards'],
                masks=batch['masks'],
                next_q=next_q_with_entropy,
                discount=self.config['discount'],
                horizon_length=1,
                clip_min=-500.0,
                clip_max=500.0,
            )
        else:
            target_q = compute_td_target(
                rewards=batch['rewards'],
                masks=batch['masks'],
                next_q=next_q,
                discount=self.config['discount'],
                horizon_length=1,
                clip_min=-500.0,
                clip_max=500.0,
            )

        # Current Q-values (critic operates on denoised actions, NOT noise)
        # batch["true_actions"] = flattened executed actions; batch["actions"] = noise
        q = self.critic_network.select('critic')(batch['observations'], actions=batch["true_actions"], params=grad_params)

        # MSE loss
        critic_loss, _ = compute_td_loss(q_pred=q, target_q=target_q, loss_type="mse")

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'temp': temp,
            'next_actions_mean': next_actions.mean(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_actions_std': next_actions.std(),
            'rewards_mean': batch['rewards'].mean(),
            'rewards_min': batch['rewards'].min(),
            'rewards_max': batch['rewards'].max(),
        }
        if self.config['entropy_backup']:
            info['next_log_probs_mean'] = next_log_probs.mean()

        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss with integrated temperature loss.

        Uses the noise critic (distilled from main critic) to evaluate noise actions,
        avoiding the semantic mismatch of feeding noise to an action-trained critic.
        """
        # Sample noise with current policy
        rng, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)
        # Safety: clip log_probs to prevent overflow when policy becomes deterministic
        log_probs = jnp.clip(log_probs, -100, 100)

        # Get Q-values from noise critic (operates in noise space)
        qs = self.noise_critic_network.select('noise_critic')(batch['observations'], actions=actions)
        if self.config['q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)
        
        # Get temperature
        temp = self.actor_network.select('temp')()
        
        # SAC actor loss: maximize Q - temperature * entropy
        actor_loss = jnp.mean((temp * log_probs - q))

        temp_param = self.actor_network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()
        
        total_loss = actor_loss + temp_loss

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(),
            'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(),
            'temp': temp,
            'actor_mode': dist.mode().mean(),
        }
    
    def actor_total_loss(self, batch, grad_params, rng):
        """Actor loss for separate optimization."""
        a_loss, a_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info
    
    def actor_bc_loss(self, batch, grad_params, rng):
        """Compute BC loss on actor network using successful noise samples.

        The success buffer contains transitions where batch["actions"] is the noise
        that led to successful outcomes. We train the actor to imitate this successful noise.

        Uses MSE loss between actor output and successful noise to avoid issues with
        computing log_prob on tanh-transformed distributions.

        Args:
            batch: Batch with successful transitions where batch["actions"] contains noise
            grad_params: Parameters for gradient computation
            rng: Random number generator

        Returns:
            Weighted BC loss and info dict
        """
        # batch["actions"] contains the noise that led to successful outcomes (tanh-squashed)
        successful_noise = batch["actions"]

        # Get distribution from current actor
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)

        # Get the mode/mean of the distribution (already tanh-squashed)
        if hasattr(dist, 'mode'):
            actor_noise = dist.mode()
        else:
            actor_noise = dist.mean()

        # Compute MSE loss between actor output and successful noise
        bc_loss_unweighted = jnp.mean(jnp.square(actor_noise - successful_noise))

        # Apply bc_coeff weighting
        bc_loss = self.config["bc_coeff"] * bc_loss_unweighted

        return bc_loss, {
            'actor_bc_loss': bc_loss,
            'actor_bc_loss_unweighted': bc_loss_unweighted,
            'actor_noise_mean': actor_noise.mean(),
            'actor_noise_std': actor_noise.std(),
            'successful_noise_mean': successful_noise.mean(),
            'successful_noise_std': successful_noise.std(),
            'bc_mse_error': bc_loss_unweighted,
        }

    def noise_critic_distill_loss(self, batch, grad_params, rng):
        """Distill main critic's knowledge into noise critic.

        Trains noise_critic(obs, normalized_noise) ≈ critic(obs, denoise(scaled_noise)).
        Matches reference DSRL: noise is normalized to [-1,1] for noise_critic,
        scaled by action_magnitude for diffusion.
        """
        batch_size = batch['observations'].shape[0]
        noise_dim = self.config['noise_dim']

        # 1. Sample random noise (broad coverage, not from actor)
        rng, noise_rng = jax.random.split(rng)
        raw_noise = jax.random.normal(noise_rng, (batch_size, noise_dim))

        # 2. Normalize to [-1, 1] (match actor's tanh output distribution)
        normalized_noise = self._normalize_noise_for_critic(raw_noise)

        # 3. Scale for diffusion input (match reference's unscale_action)
        scaled_noise = self._scale_noise_for_diffusion(normalized_noise)

        # 4. Denoise via frozen BC flow → real actions
        denoised_actions = self._denoise_noise_batch(batch['observations'], scaled_noise)

        # 5. Q targets from main critic (stop gradient)
        target_qs = jax.lax.stop_gradient(
            self.critic_network.select('critic')(batch['observations'], actions=denoised_actions)
        )

        # 6. Noise critic prediction on NORMALIZED noise ([-1, 1] range)
        pred_qs = self.noise_critic_network.select('noise_critic')(
            batch['observations'], actions=normalized_noise, params=grad_params
        )

        # 7. MSE distillation loss
        distill_loss = jnp.mean((pred_qs - target_qs) ** 2)

        return distill_loss, {
            'noise_critic_distill_loss': distill_loss,
            'noise_critic_pred_q_mean': pred_qs.mean(),
            'noise_critic_target_q_mean': target_qs.mean(),
        }

    def noise_critic_total_loss(self, batch, grad_params, rng):
        """Noise critic distillation loss for separate optimization."""
        nc_loss, nc_info = self.noise_critic_distill_loss(batch, grad_params, rng)
        info = {f'noise_critic/{k}': v for k, v in nc_info.items()}
        return nc_loss, info

    def critic_total_loss(self, batch, grad_params, rng):
        """Critic loss for separate optimization."""
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    def bc_loss(self, batch, grad_params, rng):
        """Compute the Flow Matching actor loss for offline pretraining.

        Uses 'valid' mask from offline dataset.
        Uses bc_helper for standard flow matching BC loss computation.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        rng, targets_rng = jax.random.split(rng)
        x_t, vel, t = get_flow_targets(
            batch['observations'],
            batch_actions,
            targets_rng,
        )
        pred = self.bc_flow_network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        return bc_flow_loss, {
            'bc_flow_loss': bc_flow_loss,
        }

    def bc_loss_online(self, batch, grad_params, rng):
        """Compute the Flow Matching actor loss for online RL success buffer.

        Uses bc_coeff weighting and treats all timesteps as valid.
        Note: batch["actions"] contains noise, batch["true_actions"] contains actual executed actions.
        Uses bc_helper for online BC loss computation.
        """
        def model_fn(obs, x_t, t):
            return self.bc_flow_network.select('actor_bc_flow')(obs, x_t, t, params=grad_params)

        return compute_flow_bc_loss_online(
            batch, rng, model_fn,
            self.config["action_chunking"],
            self.config["horizon_length"],
            self.config["action_dim"],
            bc_coeff=self.config["bc_coeff"],
            action_key="true_actions"
        )

    # ========== DSRL+EXPO: Action Critic and Edit Actor Methods ==========

    def action_critic_loss(self, batch, grad_params, rng):
        """Compute the action critic loss for DSRL+EXPO.

        This critic evaluates Q(obs, refined_action) where refined_action is the
        final action after flow refinement and optional edit.

        Args:
            batch: Must contain 'refined_actions' field with final executed actions
            grad_params: Parameters for gradient computation
            rng: Random number generator

        Returns:
            Tuple of (loss, info_dict)
        """
        # Check if edit actor is enabled
        if not self.config['use_edit_actor']:
            return 0.0, {}

        # Sample next actions: noise → flow → (optionally edit)
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.actor_network.select('actor')(batch['next_observations'])

        # Sample num_action_samples noise samples
        batch_size = batch['next_observations'].shape[0]
        sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
        next_noises = jnp.stack([next_dist.sample(seed=key) for key in sample_keys], axis=0)

        # Apply flow refinement to get base actions
        if self.config["noise_chunk_length"] > 0:
            next_noises_expanded = jnp.tile(
                next_noises, self.config["horizon_length"]
            )
        else:
            next_noises_expanded = next_noises

        # Scale noise by action_magnitude before diffusion (match reference DSRL)
        next_noises_scaled = self._scale_noise_for_diffusion(next_noises_expanded)

        # Compute flow-refined actions for all samples
        # Shape: (num_samples, batch_size, action_dim)
        batch_next_obs = jnp.repeat(
            batch['next_observations'][None, ...],
            self.config["num_action_samples"],
            axis=0
        )
        next_base_actions = jax.vmap(
            lambda obs, noise: self.compute_flow_actions(obs, noise),
            in_axes=(0, 0)
        )(batch_next_obs, next_noises_scaled)

        # Apply edit actor if we're past edit_start_step
        n_edit_samples = self.config['n_edit_samples']
        if n_edit_samples > 0:
            # Sample edits for a subset of base actions
            rng, edit_rng = jax.random.split(rng)
            # Take first n_edit_samples for editing
            edit_base_actions = next_base_actions[:n_edit_samples]
            edit_obs = batch_next_obs[:n_edit_samples]

            # Sample edits from edit actor using vmap for both distribution creation and sampling
            edit_keys = jax.random.split(edit_rng, n_edit_samples)
            
            # Vmap both distribution creation AND sampling together
            def sample_edit(obs, base_act, key):
                edit_dist = self.edit_actor_network.select('edit_actor')(obs, base_act)
                return edit_dist.sample(seed=key)
            
            edits = jax.vmap(sample_edit)(edit_obs, edit_base_actions, edit_keys)
            edited_actions = edit_base_actions + edits

            # Combine: original base actions + edited actions
            next_refined_actions = jnp.concatenate([next_base_actions, edited_actions], axis=0)
            batch_next_obs_all = jnp.concatenate([
                batch_next_obs,
                edit_obs
            ], axis=0)
        else:
            next_refined_actions = next_base_actions
            batch_next_obs_all = batch_next_obs

        # Evaluate all refined actions with target action critic
        next_qs_all = self.action_critic_network.select('target_action_critic')(
            batch_next_obs_all, actions=next_refined_actions
        )

        # Aggregate Q-values across ensemble
        rng, agg_rng = jax.random.split(rng)
        next_qs_all = aggregate_q_values(
            next_qs_all, method=self.config['q_agg'],
            rng=agg_rng, num_qs=self.config.get('action_num_qs', self.config['num_qs']),
        )

        # Best-of-N: select action with highest Q-value
        best_idx = jnp.argmax(next_qs_all, axis=0)
        next_q = jnp.take_along_axis(next_qs_all, best_idx[None, :], axis=0).squeeze(0)

        # Compute TD target (horizon_length=1: chunk-level transitions, match reference)
        target_q = compute_td_target(
            rewards=batch['rewards'],
            masks=batch['masks'],
            next_q=next_q,
            discount=self.config['discount'],
            horizon_length=1,
            clip_min=-500.0,
            clip_max=500.0,
        )

        # Current Q-values on refined actions
        q = self.action_critic_network.select('action_critic')(
            batch['observations'],
            actions=batch["refined_actions"],
            params=grad_params
        )

        # MSE loss
        action_critic_loss, _ = compute_td_loss(q_pred=q, target_q=target_q, loss_type="mse")

        info = {
            'action_critic_loss': action_critic_loss,
            'action_q_mean': q.mean(),
            'action_q_max': q.max(),
            'action_q_min': q.min(),
            'action_target_q_mean': target_q.mean(),
        }

        return action_critic_loss, info

    def edit_actor_loss(self, batch, grad_params, rng):
        """Compute the edit actor loss for DSRL+EXPO.

        The edit actor learns to refine flow-refined actions by maximizing
        Q_action(obs, base_action + edit) - temperature * entropy.

        Uses base_actions from replay buffer (off-policy) to match EXPO's approach.

        Args:
            batch: Training batch (must contain 'base_actions' field)
            grad_params: Parameters for gradient computation
            rng: Random number generator

        Returns:
            Tuple of (loss, info_dict)
        """
        # Check if edit actor is enabled
        if not self.config['use_edit_actor']:
            return 0.0, {}

        # Use base actions from replay buffer (off-policy, like EXPO)
        base_actions = batch['base_actions']

        # Sample edits from edit actor
        rng, edit_rng = jax.random.split(rng)
        edit_dist = self.edit_actor_network.select('edit_actor')(
            batch['observations'],
            base_actions,
            params=grad_params
        )
        edits, log_probs = edit_dist.sample_and_log_prob(seed=edit_rng)

        # Apply edits to base actions
        refined_actions = base_actions + edits

        # Evaluate with action critic
        qs = self.action_critic_network.select('action_critic')(
            batch['observations'],
            actions=refined_actions
        )

        # Aggregate Q-values across ensemble
        rng, agg_rng = jax.random.split(rng)
        q = aggregate_q_values(
            qs, method=self.config['q_agg'],
            rng=agg_rng, num_qs=self.config.get('action_num_qs', self.config['num_qs']),
        )

        # Get temperature
        edit_temp = self.edit_actor_network.select('edit_temp')()

        # Edit actor loss: maximize Q - temperature * entropy
        edit_actor_loss = jnp.mean((edit_temp * log_probs - q))

        temp_param = self.edit_actor_network.select('edit_temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        edit_temp_loss = (temp_param * (entropy - self.config['edit_target_entropy'])).mean()

        total_loss = edit_actor_loss + edit_temp_loss

        return total_loss, {
            'edit_actor_loss': edit_actor_loss,
            'edit_temp_loss': edit_temp_loss,
            'edit_total_loss': total_loss,
            'edit_log_probs_mean': log_probs.mean(),
            'edit_entropy': entropy,
            'edit_q_mean': q.mean(),
            'edit_temp': edit_temp,
            'edit_magnitude': jnp.abs(edits).mean(),
        }

    def action_critic_total_loss(self, batch, grad_params, rng):
        """Action critic loss for separate optimization."""
        if not self.config['use_edit_actor']:
            return 0.0, {}
        c_loss, c_info = self.action_critic_loss(batch, grad_params, rng)
        info = {f'action_critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    def edit_actor_total_loss(self, batch, grad_params, rng):
        """Edit actor loss for separate optimization."""
        if not self.config['use_edit_actor']:
            return 0.0, {}
        a_loss, a_info = self.edit_actor_loss(batch, grad_params, rng)
        info = {f'edit_actor/{k}': v for k, v in a_info.items()}
        return a_loss, info
    
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(
        agent,
        batch: dict
    ) -> Tuple['DSRLEXPOAgent', dict]:
        """Apply gradient update to actor, critic, and noise critic networks (online RL)."""
        new_rng, rng1, rng2, rng3 = jax.random.split(agent.rng, 4)

        # Update critic network (trained on real actions from replay)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        # Multi-step noise critic distillation (use updated critic for targets)
        noise_critic_grad_steps = int(agent.config.get('noise_critic_grad_steps', 1))
        agent_for_nc = agent.replace(critic_network=new_critic_state)
        current_nc_state = agent.noise_critic_network
        for _ in range(noise_critic_grad_steps):
            rng3, nc_rng = jax.random.split(rng3)
            def nc_loss_fn(p, _rng=nc_rng, _agent=agent_for_nc):
                return _agent.noise_critic_total_loss(batch, p, _rng)
            current_nc_state, nc_info = current_nc_state.apply_loss_fn(nc_loss_fn)
        new_noise_critic_state = current_nc_state

        # Update actor network (uses updated noise critic for Q-values)
        def actor_loss_fn(p):
            return agent.replace(
                noise_critic_network=new_noise_critic_state
            ).actor_total_loss(batch, p, rng1)
        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # Combine info
        info = {**actor_info, **critic_info, **nc_info}

        return agent.replace(
            actor_network=new_actor_state,
            critic_network=new_critic_state,
            noise_critic_network=new_noise_critic_state,
            rng=new_rng
        ), info

    @staticmethod
    def _update_offline(
        agent,
        batch: dict
    ) -> Tuple['DSRLBestOfNAgent', dict]:
        """Apply gradient update to BC flow network only (offline training)."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        
        def bc_loss_fn(p):
            return agent.bc_loss(batch, p, rng1)  # Offline BC loss (with 'valid' mask)
        
        new_bc_flow_state, info = agent.bc_flow_network.apply_loss_fn(bc_loss_fn)
        
        return agent.replace(bc_flow_network=new_bc_flow_state, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @staticmethod
    def _update_with_bc(
        agent,
        batch_tuple: Tuple[dict, dict, bool]
    ) -> Tuple['DSRLEXPOAgent', dict]:
        """Apply gradient update to actor (with BC on success buffer), critic, and noise critic.

        The BC flow network is NOT updated during online RL - it remains frozen after
        offline pretraining. Only the actor (noise policy) is adapted using successful
        noise samples from the success buffer.

        DSRL+EXPO Extension:
        If use_edit_actor is enabled, also updates action_critic and edit_actor.
        """
        batch, batch_success, success_flag = batch_tuple
        new_rng, rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(agent.rng, 7)

        # Update critic network (trained on real actions from replay)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        # Multi-step noise critic distillation (use updated critic for targets)
        noise_critic_grad_steps = int(agent.config.get('noise_critic_grad_steps', 1))
        agent_for_nc = agent.replace(critic_network=new_critic_state)
        current_nc_state = agent.noise_critic_network
        for _ in range(noise_critic_grad_steps):
            rng6, nc_rng = jax.random.split(rng6)
            def nc_loss_fn(p, _rng=nc_rng, _agent=agent_for_nc):
                return _agent.noise_critic_total_loss(batch, p, _rng)
            current_nc_state, nc_info = current_nc_state.apply_loss_fn(nc_loss_fn)
        new_noise_critic_state = current_nc_state

        # Update actor network with RL loss + optional BC loss on successful noise
        agent_for_actor = agent.replace(noise_critic_network=new_noise_critic_state)
        def actor_loss_fn(p):
            # RL loss on regular batch (uses updated noise critic)
            rl_loss, rl_info = agent_for_actor.actor_total_loss(batch, p, rng1)

            # Conditionally add BC loss on success batch
            def add_bc_loss():
                bc_loss, bc_info = agent.actor_bc_loss(batch_success, p, rng3)
                combined_info = {**rl_info, **bc_info}
                return rl_loss + bc_loss, combined_info

            def no_bc_loss():
                # Add dummy BC metrics for consistent logging
                dummy_bc_info = {
                    'actor_bc_loss': 0.0,
                    'actor_bc_loss_unweighted': 0.0,
                    'actor_noise_mean': 0.0,
                    'actor_noise_std': 0.0,
                    'successful_noise_mean': 0.0,
                    'successful_noise_std': 0.0,
                    'bc_mse_error': 0.0,
                }
                return rl_loss, {**rl_info, **dummy_bc_info}

            return jax.lax.cond(success_flag, add_bc_loss, no_bc_loss)

        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        # DSRL+EXPO: Update action critic and edit actor if enabled
        if agent.config['use_edit_actor'] and agent.action_critic_network is not None:
            # Update action critic network (Q_action)
            def action_critic_loss_fn(p):
                return agent.action_critic_total_loss(batch, p, rng4)
            new_action_critic_state, action_critic_info = agent.action_critic_network.apply_loss_fn(
                action_critic_loss_fn
            )
            agent.target_update(new_action_critic_state, 'action_critic')

            # Update edit actor network
            def edit_actor_loss_fn(p):
                return agent.edit_actor_total_loss(batch, p, rng5)
            new_edit_actor_state, edit_actor_info = agent.edit_actor_network.apply_loss_fn(
                edit_actor_loss_fn
            )

            # Combine all info
            info = {**actor_info, **critic_info, **nc_info, **action_critic_info, **edit_actor_info}

            return agent.replace(
                actor_network=new_actor_state,
                critic_network=new_critic_state,
                noise_critic_network=new_noise_critic_state,
                action_critic_network=new_action_critic_state,
                edit_actor_network=new_edit_actor_state,
                rng=new_rng
            ), info
        else:
            # Original DSRL behavior (no edit actor)
            info = {**actor_info, **critic_info, **nc_info}

            return agent.replace(
                actor_network=new_actor_state,
                critic_network=new_critic_state,
                noise_critic_network=new_noise_critic_state,
                rng=new_rng
            ), info

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        """Update the agent and return a new agent with information dictionary."""
        if batch_success is None:
            # Original behavior: only update critic
            agent, infos = jax.lax.scan(self._update, self, batch)
        else:
            # New behavior: update critic + optionally BC flow on success buffer
            # For each UTD step, create tuple of (batch, batch_success, success_flag)
            def scan_fn(agent, inputs):
                batch_i, batch_success_i = inputs
                return self._update_with_bc(agent, (batch_i, batch_success_i, success_flag))
            
            agent, infos = jax.lax.scan(scan_fn, self, (batch, batch_success))
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @partial(jax.jit, static_argnames=('deterministic',))
    def sample_noise(
        self,
        observations,
        rng=None,
        deterministic=False,
    ):
        """Sample best of N noise from the actor policy with BON Q-value selection."""
        if deterministic:
            # For deterministic mode, just use mode/mean (no Best-of-N)
            dist = self.actor_network.select('actor')(observations)
            if hasattr(dist, 'mode'):
                noise = dist.mode()
            else:
                noise = dist.mean()
        else:
            # Sample N noises from actor and select best based on noise critic Q-value
            dist = self.actor_network.select('actor')(observations)
            rng, sample_rng = jax.random.split(rng)
            sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
            noises = jnp.stack([dist.sample(seed=key) for key in sample_keys], axis=0)  # (num_samples, batch_size, action_dim)

            # Evaluate Q-values using noise critic (operates in noise space)
            batch_obs = jnp.repeat(observations[None, ...], self.config["num_action_samples"], axis=0)
            qs = self.noise_critic_network.select('noise_critic')(batch_obs, actions=noises)  # (num_qs, num_samples, batch_size)

            # BON Q-aggregation: always min or subsample-min (never mean)
            if self.config['subsample_bon']:
                rng, subsample_key = jax.random.split(rng)
                subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                qs = qs[subsample_idxs].min(axis=0)  # (num_samples, batch_size)
            else:
                qs = qs.min(axis=0)  # (num_samples, batch_size)

            # NaN guard: replace NaN Q-values with -inf so they're never selected
            qs = jnp.where(jnp.isnan(qs), -jnp.inf, qs)

            # Best-of-N: select noise with highest Q-value
            best_idx = jnp.argmax(qs, axis=0, keepdims=True)  # (1, batch_size)
            noise = jnp.take_along_axis(noises, best_idx[..., None], axis=0).squeeze(0)  # (batch_size, action_dim)
        return noise
    
    
    @partial(jax.jit, static_argnames=('deterministic', 'return_info'))
    def sample_actions(
        self,
        observations,
        rng=None,
        deterministic=False,
        return_info=False,
    ):
        """Sample actions from the actor policy with Best-of-N sampling.

        DSRL+EXPO Extension:
        If use_edit_actor is enabled, also samples edit refinements and evaluates
        them with the action critic for Best-of-N selection.

        Args:
            observations: Current observations
            rng: Random number generator
            deterministic: If True, use mode/mean instead of sampling
            return_info: If True, return dict with {actions, base_actions, noise}
                        If False, return only actions (default, for backward compatibility)

        Returns:
            If return_info=False: actions (final actions to execute)
            If return_info=True: dict with 'actions', 'base_actions', 'noise'
        """
        dist = self.actor_network.select('actor')(observations)

        if deterministic:
            # For deterministic mode, just use mode/mean (no Best-of-N)
            if hasattr(dist, 'mode'):
                noises = dist.mode()
            else:
                noises = dist.mean()
        else:
            # Sample N actions from actor and select best based on Q-value
            rng, sample_rng = jax.random.split(rng)
            sample_keys = jax.random.split(sample_rng, self.config["num_action_samples"])
            noises = jnp.stack([dist.sample(seed=key) for key in sample_keys], axis=0)  # (num_samples, batch_size, action_dim)

            # Evaluate Q-values using noise critic (operates in noise space)
            batch_obs = jnp.repeat(observations[None, ...], self.config["num_action_samples"], axis=0)
            qs = self.noise_critic_network.select('noise_critic')(batch_obs, actions=noises)  # (num_qs, num_samples, batch_size)

            # BON Q-aggregation: always min or subsample-min (never mean)
            if self.config['subsample_bon']:
                rng, subsample_key = jax.random.split(rng)
                subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                qs = qs[subsample_idxs].min(axis=0)  # (num_samples, batch_size)
            else:
                qs = qs.min(axis=0)  # (num_samples, batch_size)

            # NaN guard: replace NaN Q-values with -inf so they're never selected
            qs = jnp.where(jnp.isnan(qs), -jnp.inf, qs)

            # Best-of-N: select action with highest Q-value
            best_idx = jnp.argmax(qs, axis=0, keepdims=True)  # (1, batch_size)
            noises = jnp.take_along_axis(noises, best_idx[..., None], axis=0).squeeze(0)  # (batch_size, action_dim)

        # Expand noise for action chunking
        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            noises_expanded = jnp.tile(noises, self.config["horizon_length"])
        else:
            noises_expanded = noises

        # Scale noise by action_magnitude before diffusion (match reference DSRL)
        scaled_noise = self._scale_noise_for_diffusion(noises_expanded)
        # Apply flow refinement to get base actions
        base_actions = self.compute_flow_actions(observations, scaled_noise)

        # DSRL+EXPO: Apply edit actor if enabled
        if self.config['use_edit_actor'] and self.action_critic_network is not None and not deterministic:
            n_edit_samples = self.config['n_edit_samples']
            if n_edit_samples > 0:
                # Generate multiple edit refinements
                rng, edit_rng = jax.random.split(rng)
                edit_keys = jax.random.split(edit_rng, n_edit_samples)

                # Sample edits for each observation
                edit_dist = self.edit_actor_network.select('edit_actor')(observations, base_actions)
                edits = jnp.stack([edit_dist.sample(seed=key) for key in edit_keys], axis=0)  # (n_edit_samples, batch_size, action_dim)

                # Apply edits to base actions
                # Handle both batched and unbatched base_actions
                base_actions_expanded = base_actions[None, ...] if base_actions.ndim == 1 else base_actions[None, :, :]
                edited_actions = base_actions_expanded + edits  # (n_edit_samples, batch_size, action_dim)

                # Combine base actions with edited actions
                all_actions = jnp.concatenate([base_actions_expanded, edited_actions], axis=0)  # (1 + n_edit_samples, batch_size, action_dim)

                # Evaluate all actions with Q_action
                batch_obs_all = jnp.repeat(observations[None, ...], 1 + n_edit_samples, axis=0)
                action_qs = self.action_critic_network.select('target_action_critic')(batch_obs_all, actions=all_actions)

                # BON Q-aggregation: always min or subsample-min (never mean)
                if self.config['subsample_bon']:
                    rng, subsample_key = jax.random.split(rng)
                    subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                    action_qs = action_qs[subsample_idxs].min(axis=0)
                else:
                    action_qs = action_qs.min(axis=0)

                # NaN guard: replace NaN Q-values with -inf so they're never selected
                action_qs = jnp.where(jnp.isnan(action_qs), -jnp.inf, action_qs)

                # Select best action
                best_idx = jnp.argmax(action_qs, axis=0, keepdims=True)
                # Handle both batched and unbatched cases
                if action_qs.ndim == 1:
                    # Unbatched: action_qs is (1 + n_edit_samples,), best_idx is (1,)
                    final_actions = jnp.take_along_axis(all_actions, best_idx[:, None], axis=0).squeeze(0)
                else:
                    # Batched: action_qs is (1 + n_edit_samples, batch_size), best_idx is (1, batch_size)
                    final_actions = jnp.take_along_axis(all_actions, best_idx[:, :, None], axis=0).squeeze(0)

                if return_info:
                    return {
                        'actions': final_actions,
                        'base_actions': base_actions,
                        'noise': noises,
                    }
                return final_actions

        if return_info:
            return {
                'actions': base_actions,
                'base_actions': base_actions,
                'noise': noises,
            }
        return base_actions
    
    @jax.jit
    def noise_to_actions_public(self, observations, actor_noise):
        """Public API: actor noise [-1,1] -> scale -> denoise -> actions [-1,1].

        Handles unbatched observations (via compute_flow_actions internals).
        Use this from the runner instead of calling compute_flow_actions directly.
        """
        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actor_noise = jnp.tile(actor_noise, self.config["horizon_length"])
        scaled_noise = self._scale_noise_for_diffusion(actor_noise)
        return self.compute_flow_actions(observations, scaled_noise)

    @jax.jit
    def sample_actions_normal(
        self,
        observations,
        rng=None,
    ):
        """Sample actions from the actor policy."""
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions = self.compute_flow_actions(observations, noises)
        return actions

    def get_q_values(self, observations, actions, is_encoded=False):
        """Computes the Q-value for a given state-action pair for logging."""
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=is_encoded,
        )

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises=None,
        params=None,
        rng=None,  # Accept rng for compatibility with evaluation code
    ):
        """Compute actions from the BC flow model using pg_helper ODE function."""
        # Handle unbatched observations (e.g., from runner's data collection)
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['noise_dim']

        # Handle encoder
        if self.config['encoder'] is not None:
            if params is not None:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations, params=params)
            else:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations)

        # Create wrapper function for actor network
        def actor_fn(obs, actions, t, is_encoded=True):
            if params is not None:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded, params=params)
            else:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn,
            observations=observations,
            rng=rng,
            flow_steps=self.config['flow_steps'],
            act_dim=act_dim,
            act_min=-1.0,
            act_max=1.0,
            clip_intermediate=False,  # DSRL+EXPO doesn't clip intermediate actions
            noises=noises,
            is_encoded=True,
        )

        # Remove batch dimension if it was added
        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        """Restore actor (SAC noise policy) parameters from checkpoint.

        DSRL+EXPO's actor_network has {actor, temp} — no target_actor.
        """
        import pickle
        with open(actor_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent_dict = load_dict['agent']

        if 'actor_network' in agent_dict:
            src = agent_dict['actor_network']['params']
            if 'modules_actor' in src:
                self.actor_network.params['modules_actor'] = src['modules_actor']
                if 'modules_temp' in src:
                    self.actor_network.params['modules_temp'] = src['modules_temp']
                print(f"Restored DSRL+EXPO actor from {actor_path}")
            else:
                raise ValueError(f"No modules_actor in actor_network checkpoint: {actor_path}")
        elif 'network' in agent_dict:
            src = agent_dict['network']['params']
            if 'modules_actor' in src:
                self.actor_network.params['modules_actor'] = src['modules_actor']
                print(f"Restored DSRL+EXPO actor from unified checkpoint: {actor_path}")
            else:
                raise ValueError(f"No modules_actor in network checkpoint: {actor_path}")
        else:
            raise ValueError(f"Unknown checkpoint format in {actor_path}")

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]
        
        if config["noise_chunk_length"] == -1:
            noise_dim = full_action_dim
            ex_noise = full_actions
        elif config["noise_chunk_length"] == 1:
            noise_dim = action_dim * config["noise_chunk_length"]
            ex_noise = jnp.concatenate([ex_actions] * config["noise_chunk_length"], axis=-1)
        else:
            raise ValueError(f"Invalid noise chunk length: {config['noise_chunk_length']}")
        
        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            
        # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
        # Matches OGPO so DSRL+EXPO sees proprio through a dedicated tower
        # instead of buried inside 2048D image features.
        _two_tier_kwargs = dict(
            obs_two_tier=config.get('obs_two_tier', False),
            two_tier_img_dim=config.get('_two_tier_img_dim', 0),
            two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
            two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
        )

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
            **_two_tier_kwargs,
        )

        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=noise_dim,  # This handles both chunked and non-chunked actions
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,  # Enable tanh squashing for bounded actions
            state_dependent_std=True,  # Enable state-dependent std for SAC
            # No low/high: actor outputs in [-1, 1] (standard tanh bounds).
            # Original DSRL-NA actor operates in the env's [-1, 1] action space.
            # Using action_magnitude here causes a log_det scaling in log_prob that
            # makes entropy negative, driving temperature up and Q-values to -inf.
            **_two_tier_kwargs,
        )

        temp_def = LogParam(init_value=config['init_temperature'])

        # Time embedding
        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        # Define flow matching actor network.
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )
        
        # Separate network definitions
        actor_nets = {
            'actor': actor_def,
            'temp': temp_def,
        }
        actor_args = {
            'actor': (ex_observations,),
            'temp': (),
        }
        
        critic_nets = {
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def),
        }
        # Main critic is trained on real (denoised) actions
        critic_args = {
            'critic': (ex_observations, full_actions),
            'target_critic': (ex_observations, full_actions),
        }

        # Noise critic: same architecture, takes noise as input
        noise_critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
            **_two_tier_kwargs,
        )
        noise_critic_nets = {'noise_critic': noise_critic_def}
        noise_critic_args = {'noise_critic': (ex_observations, ex_noise)}

        bc_flow_nets = {
            'actor_bc_flow': actor_bc_flow_def,
        }
        bc_flow_args = {
            'actor_bc_flow': (ex_observations, full_actions, ex_times),
        }

        # Create separate network definitions
        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        noise_critic_net_def = ModuleDict(noise_critic_nets)
        bc_flow_net_def = ModuleDict(bc_flow_nets)

        # Create separate optimizers - ensure learning rates are Python floats
        actor_lr = float(config['lr_actor'])
        critic_lr = float(config['lr_critic'])
        noise_critic_lr = float(config.get('lr_noise_critic', config['lr_critic']))
        bc_flow_lr = float(config['lr_actor_bc_flow'])

        actor_opt_type = config['optimizer_actor']
        critic_opt_type = config['optimizer_critic']
        bc_flow_opt_type = config['optimizer_actor_bc_flow']

        clip_norm = float(config.get('clip_grad_norm', 1000.0))

        def get_optimizer(lr, opt_type):
            # lr is already a Python float
            if opt_type == 'adam':
                base = optax.adam(learning_rate=lr)
            elif opt_type == 'adamw':
                base = optax.adamw(learning_rate=lr)
            elif opt_type == 'sgd':
                base = optax.sgd(learning_rate=lr)
            else:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
            return optax.chain(optax.clip_by_global_norm(clip_norm), base)

        actor_tx = get_optimizer(actor_lr, actor_opt_type)
        critic_tx = get_optimizer(critic_lr, critic_opt_type)
        noise_critic_tx = get_optimizer(noise_critic_lr, critic_opt_type)
        bc_flow_tx = get_optimizer(bc_flow_lr, bc_flow_opt_type)

        # Initialize parameters
        rng, actor_rng, critic_rng, nc_rng, bc_flow_rng = jax.random.split(rng, 5)
        actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
        critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
        noise_critic_params = noise_critic_net_def.init(nc_rng, **noise_critic_args)['params']
        bc_flow_params = bc_flow_net_def.init(bc_flow_rng, **bc_flow_args)['params']

        # Create separate train states
        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        noise_critic_state = TrainState.create(noise_critic_net_def, noise_critic_params, noise_critic_tx)
        bc_flow_state = TrainState.create(bc_flow_net_def, bc_flow_params, bc_flow_tx)
        
        # Initialize target networks
        critic_state.params[f'modules_target_critic'] = critic_state.params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config["noise_dim"] = noise_dim

        if config['target_entropy'] is None:
            config['target_entropy'] = -float(noise_dim) / 2

        # DSRL+EXPO: Initialize action critic and edit actor if enabled
        action_critic_state = None
        edit_actor_state = None

        if config['use_edit_actor']:
            # Add encoder for action critic and edit actor
            if config['encoder'] is not None:
                encoders['action_critic'] = encoder_module()
                encoders['edit_actor'] = encoder_module()

            # Define action critic (evaluates Q(obs, refined_action))
            action_critic_def = Value(
                hidden_dims=config['action_critic_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['action_num_qs'],
                encoder=encoders.get('action_critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                **_two_tier_kwargs,
            )

            # Define edit actor
            edit_actor_def = EditActor(
                hidden_dims=config['edit_actor_hidden_dims'],
                action_dim=full_action_dim,
                layer_norm=config['actor_layer_norm'],
                encoder=encoders.get('edit_actor'),
                edit_action_scale=config['edit_action_scale'],
                state_dependent_std=True,
                **_two_tier_kwargs,
            )

            edit_temp_def = LogParam(init_value=config['edit_init_temperature'])

            # Network definitions
            action_critic_nets = {
                'action_critic': action_critic_def,
                'target_action_critic': copy.deepcopy(action_critic_def),
            }
            action_critic_args = {
                'action_critic': (ex_observations, full_actions),
                'target_action_critic': (ex_observations, full_actions),
            }

            edit_actor_nets = {
                'edit_actor': edit_actor_def,
                'edit_temp': edit_temp_def,
            }
            edit_actor_args = {
                'edit_actor': (ex_observations, full_actions),
                'edit_temp': (),
            }

            # Create network definitions
            action_critic_net_def = ModuleDict(action_critic_nets)
            edit_actor_net_def = ModuleDict(edit_actor_nets)

            # Get optimizers
            action_critic_lr = config['lr_action_critic']
            edit_actor_lr = config['lr_edit_actor']

            action_critic_opt_type = config['optimizer_action_critic']
            edit_actor_opt_type = config['optimizer_edit_actor']

            action_critic_tx = get_optimizer(action_critic_lr, action_critic_opt_type)
            edit_actor_tx = get_optimizer(edit_actor_lr, edit_actor_opt_type)

            # Initialize parameters
            rng, action_critic_rng, edit_actor_rng = jax.random.split(rng, 3)
            action_critic_params = action_critic_net_def.init(action_critic_rng, **action_critic_args)['params']
            edit_actor_params = edit_actor_net_def.init(edit_actor_rng, **edit_actor_args)['params']

            # Create train states
            action_critic_state = TrainState.create(action_critic_net_def, action_critic_params, action_critic_tx)
            edit_actor_state = TrainState.create(edit_actor_net_def, edit_actor_params, edit_actor_tx)

            # Initialize target action critic
            action_critic_state.params[f'modules_target_action_critic'] = action_critic_state.params[f'modules_action_critic']

            # Set edit target entropy
            if config['edit_target_entropy'] is None:
                config['edit_target_entropy'] = -float(full_action_dim) / 2

        return cls(
            rng,
            actor_state,
            critic_state,
            bc_flow_state,
            noise_critic_state,
            action_critic_state,
            edit_actor_state,
            config=flax.core.FrozenDict(**config)
        )