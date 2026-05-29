import copy
from typing import Any, Tuple, List, NamedTuple, Dict
from functools import partial

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import distrax
import pickle
import numpy as np

from ogpo.networks.encoders import encoder_modules, vit_encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import ActorVectorField, Value, EditPolicy
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    compute_flow_bc_loss_online,
)
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
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


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


@partial(jax.jit, static_argnames=('critic_fn', 'agg_method', 'num_qs', 'subsample_size'))
def compute_q(critic_fn, critic_params, observations, actions, is_encoded=False,
              agg_method="min", rng=None, num_qs=None, subsample_size=2):
    q_values = critic_fn(observations, actions, params=critic_params, is_encoded=is_encoded)
    return aggregate_q_values(q_values, method=agg_method, rng=rng, num_qs=num_qs, subsample_size=subsample_size)


class EXPOAgent(flax.struct.PyTreeNode):
    """
    EXpressive Policy Optimization (EXPO) Agent
    Uses a Flow Matching base policy and a Gaussian Edit policy.
    """
    rng: Any
    actor_network: TrainState  # Base Policy (Flow)
    critic_network: TrainState # Q-Function Ensemble
    edit_network: TrainState   # Edit Policy (Residual)
    temp: TrainState           # Temperature for Edit Policy SAC objective
    config: Any = nonpytree_field()

    def actor_total_loss(self, batch: dict, batch_success: dict,  grad_params: flax.core.FrozenDict, edit_grad_params: flax.core.FrozenDict, success_flag: bool,  rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """ Combined Actor loss: SAC Edit Loss + optional BC Regularization. """
        rng, edit_key, combined_key = jax.random.split(rng, 3)
        
        # 1. SAC Edit Loss
        sac_loss, sac_info = self.edit_policy_loss(batch, edit_grad_params, edit_key)
        
        # 2. Combined BC Regularization (Optional)
        bc_loss = 0.0
        bc_info = {'combined_bc_loss': 0.0}
        if self.config.use_bc_regularization:
            # Use success buffer if available, otherwise fallback to main batch
            # We must call the loss function inside the branches to avoid pytree mismatch of the batches themselves
            def true_fn(_):
                return self.bc_loss_combined(batch_success, grad_params, edit_grad_params, combined_key)
            
            def false_fn(_):
                return self.bc_loss_combined(batch, grad_params, edit_grad_params, combined_key)
            
            bc_loss, bc_info = jax.lax.cond(success_flag, true_fn, false_fn, None)
            
        total_actor_loss = sac_loss + self.config.bc_coeff * bc_loss
        
        info = {
            **sac_info,
            'actor_bc_regularization_loss': bc_loss,
            'total_actor_loss': total_actor_loss,
            **bc_info
        }
        return total_actor_loss, info

    @partial(jax.jit, static_argnames=('is_encoded', 'use_target_critic'))
    def _get_otf_actions(
        self,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
        actor_params: flax.core.FrozenDict,
        edit_params: flax.core.FrozenDict,
        critic_params: flax.core.FrozenDict,
        is_encoded: bool = False,
        use_target_critic: bool = True,
    ) -> jnp.ndarray:
        """ Implements the On-the-Fly (OTF) policy action selection. """
        B = observations.shape[0]
        N = self.config.n_action_samples
        M = self.config.n_edit_samples
        
        # 1. Sample N actions from Base Policy
        rng, key = jax.random.split(rng)
        
        # Sample N base actions for each observation
        # We need to vectorized this. 
        # observations shape: (B, obs_dim)
        # repeated_obs shape: (B * N, obs_dim)
        repeated_obs = jnp.repeat(observations, N, axis=0)
        
        base_actions, _, _, _ = self.sample_base_actions_with_noise(
            repeated_obs,
            grad_params=actor_params,
            rng=key,
            use_target=True,
            is_encoded=is_encoded
        )
        # base_actions shape: (B * N, full_act_dim)
        
        # 2. Pick M actions to edit
        # For simplicity, if M > 0, we'll edit the first M samples per batch item or just all N if M=N
        if M > 0:
            rng, key = jax.random.split(rng)
            # Sample edit for first M base actions
            # We reshape to (B, N, act_dim) to select the first M
            base_actions_reshaped = base_actions.reshape(B, N, -1)
            actions_to_edit = base_actions_reshaped[:, :M, :].reshape(B * M, -1)
            obs_for_edit = jnp.repeat(observations, M, axis=0)
            
            edit_dist = self.edit_network.apply_fn(
                {"params": edit_params},
                obs_for_edit,
                actions_to_edit,
                is_encoded=is_encoded
            )
            edits = edit_dist.sample(seed=key)
            scale = self.config.edit_policy_bound
            edited_actions = actions_to_edit + edits * scale
            
            # Combine all candidate actions: (B, N + M, act_dim)
            all_actions = jnp.concatenate([base_actions_reshaped, edited_actions.reshape(B, M, -1)], axis=1)
            num_candidates = N + M
        else:
            all_actions = base_actions.reshape(B, N, -1)
            num_candidates = N

        # 3. Evaluate all with Critic
        obs_repeated_all = jnp.repeat(observations, num_candidates, axis=0)
        actions_flat_all = all_actions.reshape(B * num_candidates, -1)
        
        if use_target_critic:
            critic_fn = self.critic_network.select('target_critic')
        else:
            critic_fn = self.critic_network.select('critic')

        rng, subsample_rng = jax.random.split(rng)
        qs = compute_q(
            critic_fn, critic_params, obs_repeated_all, actions_flat_all,
            is_encoded=is_encoded, agg_method=self.config.q_agg,
            rng=subsample_rng, num_qs=self.config.num_qs,
            subsample_size=self.config.get('num_min_qs', 2),
        )
        qs = qs.reshape(B, num_candidates)

        # 4. Select Best
        best_indices = jnp.argmax(qs, axis=1)
        batch_indices = jnp.arange(B)
        best_actions = all_actions[batch_indices, best_indices]
        
        return best_actions

    def critic_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Standard TD loss using OTF next actions with q_helper functions."""
        rng, next_action_key = jax.random.split(rng)

        obs = batch['observations']
        next_obs = batch['next_observations']

        if self.config["action_chunking"]:
            actions_chunk = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            actions_chunk = batch["actions"][..., 0, :]

        # Use target networks for next action selection
        next_actions = self._get_otf_actions(
            next_obs,
            next_action_key,
            self.actor_network.params,
            self.edit_network.params,
            self.critic_network.params,
            use_target_critic=True
        )

        rng, subsample_rng = jax.random.split(rng)
        next_qs = compute_q(
            self.critic_network.select('target_critic'),
            self.critic_network.params,
            next_obs,
            next_actions,
            agg_method=self.config.q_agg,
            rng=subsample_rng,
            num_qs=self.config.num_qs,
            subsample_size=self.config.get('num_min_qs', 2),
        )

        # Compute TD target (EXPO uses horizon_length=1)
        target_q = compute_td_target(
            rewards=batch['rewards'],
            masks=batch['masks'],
            next_q=next_qs,
            discount=self.config.discount,
            horizon_length=1,
        )

        # Get current Q-values
        current_qs_ensemble = self.critic_network.select('critic')(obs, actions_chunk, params=grad_params)

        # Compute TD loss
        td_loss, _ = compute_td_loss(q_pred=current_qs_ensemble, target_q=target_q, loss_type="mse")

        info = {
            'critic_loss': td_loss,
            'q_mean': current_qs_ensemble.mean(),
            'target_q_mean': target_q.mean(),
        }
        return td_loss, info

    def edit_policy_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """ SAC-style RL loss for the edit policy. """
        rng, base_key, edit_key = jax.random.split(rng, 3)

        obs = batch['observations']

        # Sample base actions from target flow policy (not dataset actions)
        # so the edit policy trains on the same distribution it sees at inference
        base_actions, _, _, _ = self.sample_base_actions_with_noise(
            obs,
            grad_params=self.actor_network.params,
            rng=base_key,
            use_target=True
        )

        edit_dist = self.edit_network.apply_fn(
            {"params": grad_params},
            obs,
            base_actions,
        )
        edit_actions, log_prob = edit_dist.sample_and_log_prob(seed=edit_key)

        scale = self.config.edit_policy_bound
        scaled_edits = edit_actions * scale
        log_prob = log_prob - edit_actions.shape[-1] * jnp.log(scale)

        final_actions = base_actions + scaled_edits

        rng, q_rng = jax.random.split(rng)
        qs = compute_q(
            self.critic_network.select('critic'),
            self.critic_network.params,
            obs,
            final_actions,
            agg_method=self.config.q_agg,
            rng=q_rng,
            num_qs=self.config.num_qs,
            subsample_size=self.config.get('num_min_qs', 2),
        )

        alpha = self.temp.apply_fn({"params": self.temp.params})
        loss = (alpha * log_prob - qs).mean()

        info = {
            'edit_policy_loss': loss,
            'edit_log_prob': log_prob.mean(),
            'edit_entropy': -log_prob.mean(),
            'edit_q_mean': qs.mean(),
        }
        return loss, info

    def bc_loss_combined(
        self,
        batch: dict,
        actor_grad_params: flax.core.FrozenDict,
        edit_grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """BC loss on combined actions (Flow + Edit Mode) to match dataset actions."""
        if self.config["action_chunking"]:
            dataset_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            dataset_actions = batch["actions"][..., 0, :]

        obs = batch['observations']
        rng, base_key = jax.random.split(rng)

        base_actions, _, _, _ = self.sample_base_actions_with_noise(
            obs,
            grad_params=actor_grad_params,
            rng=base_key,
            use_target=False
        )

        edit_dist = self.edit_network.apply_fn(
            {"params": edit_grad_params},
            obs,
            base_actions
        )
        edit_mode = edit_dist.mode()
        scale = self.config.edit_policy_bound
        scaled_edit = edit_mode * scale

        combined_actions = base_actions + scaled_edit

        loss = jnp.mean(jnp.square(combined_actions - dataset_actions))

        return loss, {'combined_bc_loss': loss}

    def bc_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Standard Flow Matching BC loss for the Base Policy.

        Uses bc_helper for action preprocessing and target generation.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]
        obs = batch['observations']
        rng, targets_rng = jax.random.split(rng)

        x_t, vel, t = get_flow_targets(obs, batch_actions, targets_rng)

        pred = self.actor_network.select('actor')(
            obs,
            x_t,
            t,
            params=grad_params
        )

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        info = {'base_policy_bc_loss': bc_flow_loss}
        return bc_flow_loss, info

    def update_temperature(self, entropy: float) -> Tuple[TrainState, Dict[str, float]]:
        def temperature_loss_fn(params):
            log_temp = params['log_temp']
            temp = jnp.exp(log_temp)
            loss = temp * (entropy - self.config.target_entropy)
            return loss, {"temp": temp}

        new_temp_state, info = self.temp.apply_loss_fn(temperature_loss_fn)
        return new_temp_state, info

    @staticmethod
    def _update(agent, batch_tuple, success_flag) -> Tuple['EXPOAgent', dict]:
        """ Unified update step for fine-tuning. """
        batch, batch_success = batch_tuple
        new_rng, critic_key, actor_key, base_key = jax.random.split(agent.rng, 4)

        # 1. Update Critic
        def critic_loss_fn(p):
            return agent.critic_loss(batch, p, critic_key)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)

        # 2. Update Edit Policy & Combined Regularization
        def actor_loss_fn(p_edit):
            return agent.actor_total_loss(batch, batch_success, agent.actor_network.params, p_edit, success_flag, actor_key)
        new_edit_state, actor_info = agent.edit_network.apply_loss_fn(actor_loss_fn)

        # 3. Update Temperature
        new_temp_state, temp_info = agent.update_temperature(actor_info["edit_entropy"])

        # 4. Update Base Policy (Use success buffer if available, else RL batch)
        def base_policy_loss_fn(p):
            def true_fn(_):
                return agent.bc_loss(batch_success, p, base_key)
            def false_fn(_):
                return agent.bc_loss(batch, p, base_key)
            return jax.lax.cond(success_flag, true_fn, false_fn, None)
        new_actor_state, base_info = agent.actor_network.apply_loss_fn(base_policy_loss_fn)

        # Target updates
        agent.target_update(new_actor_state, 'actor')
        agent.target_update(new_critic_state, 'critic')

        info = {**critic_info, **actor_info, **base_info, **temp_info}
        
        return agent.replace(
            rng=new_rng,
            actor_network=new_actor_state,
            critic_network=new_critic_state,
            edit_network=new_edit_state,
            temp=new_temp_state,
        ), info

    @jax.jit
    def get_q_values(self, observations, actions, is_encoded=False):
        """Computes the Q-value for a given state-action pair for logging."""
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg="min",  # EXPO always uses min aggregation
            is_encoded=is_encoded,
        )

    def target_update(self, network, module_name):
        """Update the target network."""
        if module_name in ['actor', 'noise_net']:
            tau = self.config['actor_tau']
        else:
            tau = self.config['tau']
            
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
        """ Sample actions for environment interaction using the OTF policy. """
        add_batch_dim = False
        if observations.ndim == 1 or (self.config.encoder and observations.ndim == 3):
            add_batch_dim = True
            observations = observations[None]

        actions = self._get_otf_actions(
            observations,
            rng,
            self.actor_network.params,
            self.edit_network.params,
            self.critic_network.params,
            use_target_critic=True
        )

        if add_batch_dim:
            actions = actions[0]
        return actions

    @jax.jit
    def sample_actions_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
        """Vectorized Best-of-N for EXPO (N base + M edits, pick best).

        Same as sample_actions but always returns batched output (B, act_dim)
        for consistency with the runner's BON dispatch.
        """
        expected_single_ndim = 3 if self.config.encoder else 1
        if observations.ndim == expected_single_ndim:
            observations = observations[None]

        return self._get_otf_actions(
            observations,
            rng,
            self.actor_network.params,
            self.edit_network.params,
            self.critic_network.params,
            use_target_critic=True
        )

    @jax.jit
    def sample_base_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
        """Sample base actions without edits (uses current policy, not target)."""
        add_batch_dim = False
        if observations.ndim == 1 or (self.config.encoder and observations.ndim == 3):
            add_batch_dim = True
            observations = observations[None]

        actions, _, _, _ = self.sample_base_actions_with_noise(
            observations,
            grad_params=self.actor_network.params,
            rng=rng,
            use_target=True
        )

        if add_batch_dim:
            actions = actions[0]
        return actions

    @jax.jit
    def compute_flow_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
        """Alias for sample_base_actions for BC evaluation compatibility."""
        return self.sample_base_actions(observations, rng)

    def sample_base_actions_with_noise(
        self,
        observations: jnp.ndarray,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray = None,
        use_target: bool = True,
        is_encoded: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Generate actions from the Flow Matching base policy. """
        params = grad_params
        actor_module_name = 'target_actor' if use_target else 'actor'

        if not is_encoded and self.config.encoder:
            observations = self.actor_network(
                    params=params,
                    method='call_submodule',
                    submodule=actor_module_name,
                    submethod='encode',
                    observations=observations,
                )
        
        B = observations.shape[0]
        act_dim = self.config.full_act_dim
        rng, key = jax.random.split(rng)
        
        # Denoising chain starting from noise
        curr_actions = jax.random.normal(key, (B, act_dim))
        chains = [curr_actions]
        
        flow_steps = self.config.flow_steps
        dt = 1.0 / flow_steps
        
        for i in range(flow_steps):
            rng, key = jax.random.split(rng)
            t = jnp.full((B, 1), i / flow_steps)
            
            vels = self.actor_network.select(actor_module_name)(
                observations, 
                curr_actions, 
                t, 
                params=params, 
                is_encoded=True
            )
            
            curr_actions = curr_actions + vels * dt
            
            if self.config['clip_intermediate_actions']:
                curr_actions = jnp.clip(curr_actions, -1.0, 1.0)
            
            # Simplified: No intermediate noise for base flow in OGPO/ReinFlow style usually
            # But EXPO implementation suggests some noise. We'll keep it deterministic for now
            # unless constant_noise_std is set.
            if self.config.constant_noise_std > 0:
                rng, noise_key = jax.random.split(rng)
                curr_actions += jax.random.normal(noise_key, curr_actions.shape) * self.config.constant_noise_std
            
            chains.append(curr_actions)
            
        final_actions = jnp.clip(curr_actions, -1.0, 1.0)
        chains = jnp.stack(chains, axis=1)
            
        return final_actions, chains, None, None
    
    @partial(jax.jit)
    def bc_update(self, batch: dict) -> Tuple['EXPOAgent', dict]:
        """ Imitation learning update for the base policy. """
        new_rng, base_key = jax.random.split(self.rng)
        
        def base_policy_loss_fn(p):
            return self.bc_loss(batch, p, base_key)
        
        new_actor_state, base_info = self.actor_network.apply_loss_fn(base_policy_loss_fn)
        self.target_update(new_actor_state, 'actor')
        
        base_info = {f'offline_agent/{k}': v for k,v in base_info.items()}
        
        return self.replace(rng=new_rng, actor_network=new_actor_state), base_info

    @jax.jit
    def batch_update(self, batch: dict, batch_success: dict, success_flag: bool) -> Tuple['EXPOAgent', dict]:
        """ Batch of online updates. """
        batch_tuple = (batch, batch_success)
        agent, infos = jax.lax.scan(partial(self._update, success_flag=success_flag), self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @staticmethod
    def _critic_only_update(agent, batch) -> Tuple['EXPOAgent', dict]:
        new_rng, critic_key = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_loss(batch, p, critic_key)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @jax.jit
    def batch_critic_only_update(self, batch):
        agent, infos = jax.lax.scan(self._critic_only_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=bool(self.config.encoder))

    @classmethod
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config: Any) -> 'EXPOAgent':
        rng = jax.random.PRNGKey(seed)
        
        action_dim = ex_actions.shape[-1]
        full_act_dim = action_dim * (config.horizon_length if config.action_chunking else 1)
        config['full_act_dim'] = full_act_dim
        
        # Initial examples
        ex_obs = ex_observations[None, ...]
        ex_full_actions = jnp.tile(ex_actions[None, ...], (1, config.horizon_length if config.action_chunking else 1))
        ex_time = jnp.array([[0.0]])
        
        encoders = dict()
        if config.encoder:
            encoder_module = vit_encoder_modules[config.encoder] if 'vit' in config.encoder else encoder_modules[config.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['edit'] = encoder_module()
        
        # Time embedding
        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
        # Mirrors OGPO so the proprio signal isn't drowned in 2048D image features.
        _two_tier_kwargs = dict(
            obs_two_tier=config.get('obs_two_tier', False),
            two_tier_img_dim=config.get('_two_tier_img_dim', 0),
            two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
            two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
        )

        # 1. Base Policy
        actor_def = ActorVectorField(
            hidden_dims=config.actor_hidden_dims,
            action_dim=full_act_dim,
            layer_norm=config.actor_layer_norm,
            encoder=encoders.get('actor'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )
        actor_nets = {'actor': actor_def, 'target_actor': copy.deepcopy(actor_def)}
        actor_net_def = ModuleDict(actor_nets)

        # 2. Critic
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            critic_loss_type='mse',
            **_two_tier_kwargs,
        )
        critic_nets = {'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)}
        critic_net_def = ModuleDict(critic_nets)

        # 3. Edit Policy
        edit_def = EditPolicy(
            hidden_dims=config.edit_hidden_dims,
            action_dim=full_act_dim,
            encoder=encoders.get('edit'),
            edit_bound=config.edit_policy_bound,
            layer_norm=config['edit_layer_norm'],
            **_two_tier_kwargs,
        )
        
        # 4. Temperature
        temp_def = Temperature(config['init_temperature'])
        
        # Optimizers - ensure all values are Python floats
        clip_grad_norm = float(config.clip_grad_norm)
        def create_tx(lr, wd=0.0):
            return optax.chain(optax.clip_by_global_norm(clip_grad_norm), optax.adamw(float(lr), weight_decay=float(wd)))

        actor_tx = create_tx(config.actor_lr, config.actor_weight_decay)
        critic_tx = create_tx(config.critic_lr, config.critic_weight_decay)
        edit_tx = create_tx(config.edit_lr, config['edit_weight_decay'])
        temp_tx = optax.adam(float(config.alpha_lr))
        
        # Initialization
        rng, actor_key, critic_key, edit_key, temp_key = jax.random.split(rng, 5)
        
        actor_params = actor_net_def.init(actor_key, **{'actor': (ex_obs, ex_full_actions, ex_time), 'target_actor': (ex_obs, ex_full_actions, ex_time)})['params']
        actor_params['modules_target_actor'] = actor_params['modules_actor']
        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        
        critic_params = critic_net_def.init(critic_key, **{'critic': (ex_obs, ex_full_actions), 'target_critic': (ex_obs, ex_full_actions)})['params']
        critic_params['modules_target_critic'] = critic_params['modules_critic']
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        
        edit_params = edit_def.init(edit_key, ex_obs, ex_full_actions)['params']
        edit_state = TrainState.create(edit_def, edit_params, edit_tx)
        
        temp_params = temp_def.init(temp_key)['params']
        temp_state = TrainState.create(temp_def, temp_params, temp_tx)
        
        # Store dimensions in config
        config.ob_dims = ex_obs.shape
        config.action_dim = action_dim
        config.full_act_dim = full_act_dim

        if config.target_entropy is None:
            config.target_entropy = -full_act_dim / 2

        print("--- EXPO Agent Created (Flow-based) ---")
        return cls(rng, actor_state, critic_state, edit_state, temp_state, config)