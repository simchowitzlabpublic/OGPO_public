import copy
import pickle
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import ActorVectorField, Value
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    get_flow_targets,
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

class ACFQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking.
    The actor is shared between all Q function of different chunk lengths.
    The actor always predicts action chunks, and is shared between all Q functions.
    There are chunked critics and non-chunked critics.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss using q_helper functions."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        # Sample next actions
        rng, sample_rng = jax.random.split(rng)
        if self.config.best_of_n > 1:
            next_actions = self.sample_actions_BON(batch['next_observations'], rng=sample_rng)
        else:
            next_actions = self.sample_actions(batch['next_observations'], rng=sample_rng)

        # Get next Q-values and aggregate
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        rng, agg_rng = jax.random.split(rng)
        next_q = aggregate_q_values(
            next_qs,
            method=self.config['q_agg'],
            rng=agg_rng,
            num_qs=self.config['num_qs'],
        )

        # Compute TD target
        target_q = compute_td_target(
            rewards=batch['rewards'],
            masks=batch['masks'],
            next_q=next_q,
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
        )

        # Compute TD loss (MSE or HLGauss)
        if self.config["critic_loss_type"] == "hlgauss":
            q, q_logits = self.network.select('critic')(
                batch['observations'], actions=batch_actions, params=grad_params, return_logits=True
            )
            valid = batch.get('valid')
            if valid is not None and valid.ndim > 1:
                valid = valid[..., -1]
            critic_loss, stats = compute_td_loss(
                q_pred=q,
                target_q=target_q,
                valid_mask=valid,
                loss_type="hlgauss",
                q_min=self.config['q_min'],
                q_max=self.config['q_max'],
                num_bins=self.config['num_bins'],
                q_logits=q_logits,
            )
        else:
            q = self.network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)
            valid = batch.get('valid')
            valid_mask = None
            if valid is not None and valid.ndim > 1:
                valid_mask = valid[..., -1]
            critic_loss, stats = compute_td_loss(
                q_pred=q,
                target_q=target_q,
                valid_mask=valid_mask,
                loss_type="mse",
            )

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            **stats
        }

    def _compute_bc_loss(self, rng, grad_params, batch, batch_success, success_flag):
        rng, key = jax.random.split(rng)

        def true_branch(args):
            batch_success, batch, grad_params, key = args
            bc_loss, _ = self.actor_loss(batch_success, grad_params, key, is_bc_loss=True)
            return bc_loss

        def false_branch(args):
            batch_success, batch, grad_params, key = args
            bc_loss, _ = self.actor_loss(batch, grad_params, key, is_bc_loss=True)
            return bc_loss

        bc_loss = jax.lax.cond(success_flag, true_branch, false_branch,
            (batch_success, batch, grad_params, key),
        )
        return bc_loss

    def actor_loss(self, batch, grad_params, rng, is_bc_loss=False):
        """Compute the FQL actor loss.

        Combines BC flow loss with optional distillation and Q losses.
        Uses bc_helper for standard flow matching BC loss computation.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size, action_dim = batch_actions.shape

        # BC flow loss using bc_helper
        rng, targets_rng = jax.random.split(rng)
        x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))
        bc_flow_loss = apply_chunking_mask(
            jnp.square(pred - vel), valid_mask, batch_size,
            self.config["horizon_length"], self.config["action_dim"],
            self.config["action_chunking"]
        )

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            # Q loss with optional normalization (FQL).
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
            if self.config.get('normalize_q_loss', False):
                lam = jax.lax.stop_gradient(1.0 / jnp.maximum(jnp.abs(q).mean(), 1e-8))
                q_loss = lam * q_loss
        elif self.config["actor_type"] == "best-of-n":
            # In best-of-n mode, actor trains with pure BC only.
            # Q-function guides action selection at inference time, not actor training.
            # (BoN-SFT, if desired, can be toggled via bon_sft config flag.)
            if self.config.get('bon_sft', False):
                rng, bon_rng = jax.random.split(rng)
                best_actions = jax.lax.stop_gradient(
                    self.sample_actions_BON(batch['observations'], rng=bon_rng)
                )
                rng, bon_targets_rng = jax.random.split(rng)
                x_t_bon, vel_bon, t_bon = get_flow_targets(
                    batch['observations'], best_actions, bon_targets_rng
                )
                pred_bon = self.network.select('actor_bc_flow')(
                    batch['observations'], x_t_bon, t_bon, params=grad_params
                )
                bon_sft_loss = jnp.mean((pred_bon - vel_bon) ** 2)
                distill_loss = bon_sft_loss
            else:
                distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        if is_bc_loss:
            actor_loss = bc_flow_loss
        else:
            actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }

    @jax.jit
    def total_loss(self, batch, batch_success, success_flag, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss_batch, actor_info_batch = self.actor_loss(batch, grad_params, actor_rng)
        rng, bc_rng = jax.random.split(rng)
        bc_loss = self._compute_bc_loss(bc_rng, grad_params, batch, batch_success, success_flag)
        
        final_actor_loss = bc_loss + self.config['alpha'] * actor_info_batch['distill_loss'] + actor_info_batch.get('q_loss', 0.0)
        
        actor_info_batch['bc_flow_loss'] = bc_loss
        actor_info_batch['actor_loss'] = final_actor_loss
        
        for k, v in actor_info_batch.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + final_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch_tuple, success_flag):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        batch, batch_success = batch_tuple
        def loss_fn(grad_params):
            return agent.total_loss(batch, batch_success, success_flag, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        """BC update: trains flow policy (BC loss) + critic (TD loss).

        Matches the reference QC which uses total_loss for both phases.
        The critic must be trained during BC so Q-values are meaningful
        when online RL starts (BoN selection depends on critic quality).
        """
        new_rng, rng = jax.random.split(self.rng)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        def loss_fn(grad_params):
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, is_bc_loss=True)
            info = {f'critic/{k}': v for k, v in critic_info.items()}
            info.update({f'actor/{k}': v for k, v in actor_info.items()})
            return critic_loss + actor_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def batch_update(self, batch, batch_success, success_flag):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        batch_tuple = (batch, batch_success)
        
        def scan_fn(agent, batch_tuple):
            return self._update(agent, batch_tuple, success_flag)

        agent, infos = jax.lax.scan(scan_fn, self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False):
        """Computes the Q-value for a given state-action pair for logging."""
        return _get_q_values(
            critic_fn=self.network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=is_encoded,
        )
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1),
                ),
            )

            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)
        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * \
                        (self.config['horizon_length'] if self.config["action_chunking"] else 1)
            N = self.config["actor_num_samples"]
            bshape = observations.shape[: -len(self.config['ob_dims'])]
            B = 1
            for s in bshape:
                B *= s

            noises = jax.random.normal(rng, (B * N, action_dim))
            obs_repeated = jnp.repeat(observations.reshape(B, -1), N, axis=0)  # (B*N, obs_dim)

            actions = self.compute_flow_actions(obs_repeated, noises)  # (B*N, action_dim)
            actions = jnp.clip(actions, -1, 1)

            q_s = self.network.select("critic")(obs_repeated, actions)  # (num_qs, B*N)
            if self.config['subsample_bon']:
                rng, subsample_key = jax.random.split(rng)
                subsample_idxs = jax.random.randint(subsample_key, (2,), 0, self.config['num_qs'])
                q = q_s[subsample_idxs].min(axis=0)
            elif self.config['q_agg'] == 'mean':
                q = q_s.mean(axis=0)
            else:
                q = q_s.min(axis=0)

            q = q.reshape(B, N)
            actions = actions.reshape(B, N, action_dim)
            indices = jnp.argmax(q, axis=-1)  # (B,)
            actions = actions[jnp.arange(B), indices]  # (B, action_dim)
            actions = actions.reshape(bshape + (action_dim,))
        return actions

    @jax.jit
    def sample_actions_BON(self, observations, rng=None):
        """Fully vectorized Best-of-N action selection for QC."""
        expected_single_ndim = 3 if self.config['encoder'] else 1
        is_single_obs = observations.ndim == expected_single_ndim
        if is_single_obs:
            observations = observations[None]

        batch_size = observations.shape[0]
        num_samples = self.config['actor_num_samples']
        action_dim = self.config['action_dim'] * \
                     (self.config['horizon_length'] if self.config['action_chunking'] else 1)

        # 1. Sample all B*N noises and expand observations
        rng, noise_key = jax.random.split(rng)
        noises = jax.random.normal(noise_key, (batch_size * num_samples, action_dim))
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]

        # 2. Run flow on all candidates in one pass
        actions = self.compute_flow_actions(obs_expanded, noises)
        actions = jnp.clip(actions, -1, 1)

        # 3. Evaluate Q-ensemble: (num_qs, B*N) → (num_qs, B, N)
        q_s = self.network.select("critic")(obs_expanded, actions)
        q_values = q_s.reshape(self.config['num_qs'], batch_size, num_samples)

        # 4. Aggregate Q-ensemble per candidate (respects q_agg setting)
        if self.config['subsample_bon']:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, self.config['num_qs'])
            qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)
            min_qs = jnp.min(qs_sub, axis=0)
        elif self.config['q_agg'] == 'mean':
            min_qs = jnp.mean(q_values, axis=0)
        else:
            min_qs = jnp.min(q_values, axis=0)

        # 5. Select best candidate per observation
        best_idx = jnp.argmax(min_qs, axis=1)
        actions_reshaped = actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)

        return best_actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises=None,
        rng=None,  # Accept rng for compatibility with evaluation code, but use noises instead
    ):
        """Compute actions from the BC flow model using pg_helper ODE function."""
        # Handle unbatched observations (e.g., from evaluation code)
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['action_dim'] * (self.config['horizon_length'] if self.config['action_chunking'] else 1)

        # Handle encoder
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)

        # Create wrapper function for actor network
        def actor_fn(obs, actions, t, is_encoded=True):
            return self.network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn,
            observations=observations,
            rng=rng,
            flow_steps=self.config['flow_steps'],
            act_dim=act_dim,
            act_min=-1.0,
            act_max=1.0,
            clip_intermediate=False,  # QC doesn't clip intermediate actions
            noises=noises,
            is_encoded=True,
        )

        # Remove batch dimension if it was added
        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        """Restore critic parameters from checkpoint."""
        with open(critic_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent_dict = load_dict['agent']
        if 'network' in agent_dict:
            src = agent_dict['network']['params']
            self.network.params['modules_critic'] = src['modules_critic']
            self.network.params['modules_target_critic'] = src['modules_target_critic']
        elif 'critic_network' in agent_dict:
            src = agent_dict['critic_network']['params']
            self.network.params['modules_critic'] = src['modules_critic']
            self.network.params['modules_target_critic'] = src['modules_target_critic']

    def restore_actor_params(self, actor_path):
        """Restore actor (BC flow) parameters from checkpoint."""
        with open(actor_path, 'rb') as f:
            load_dict = pickle.load(f)
        agent_dict = load_dict['agent']
        if 'network' in agent_dict:
            src = agent_dict['network']['params']
            if 'modules_actor_bc_flow' in src:
                self.network.params['modules_actor_bc_flow'] = src['modules_actor_bc_flow']
                if 'modules_actor_bc_flow_encoder' in src:
                    self.network.params['modules_actor_bc_flow_encoder'] = src['modules_actor_bc_flow_encoder']
        elif 'actor_network' in agent_dict:
            src = agent_dict['actor_network']['params']
            if 'modules_actor' in src:
                self.network.params['modules_actor_bc_flow'] = src['modules_actor']

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

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # if config["critic_loss_type"] == 'hlgauss':
        #     config['q_min'] = -3.0 / (1 - config['discount'])
        #     config['q_max'] = 0.0 / (1 - config['discount'])

        # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
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
            encoder=encoders.get('critic'),  # Keep .get() - encoder may be None
            critic_loss_type=config['critic_loss_type'],
            num_bins=config['num_bins'],
            q_min=config['q_min'],
            q_max=config['q_max'],
            **_two_tier_kwargs,
        )

        # Time embedding
        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
            time_embedding=time_emb_module,
            **_two_tier_kwargs,
        )


        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        clip_grad_norm = float(config.get('clip_grad_norm', 1000.0))
        network_tx = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.adam(learning_rate=float(config['lr'])),
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=config)
