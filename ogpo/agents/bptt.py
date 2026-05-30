import copy
from typing import Any, Tuple, List, NamedTuple
from functools import partial
import numpy as np

import chex
import flax
import jax
import jax.numpy as jnp
import optax
import distrax
import pickle

from ogpo.networks.encoders import encoder_modules, vit_encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.networks import (
    ActorVectorField, Value, ActorVectorFieldTF, ValueTF,
    ActorVectorFieldSimBa, ValueSimBa, NoiseInjectionNetwork,
)
from ogpo.agents.modules.igp_targets_helper import get_flow_targets, get_shortcut_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    get_mip_targets,
    apply_chunking_mask,
)
from ogpo.agents.modules.maintenance import (
    create_lr_schedule,
    create_optimizer,
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
    sample_mip_actions_ode,
)

def _split_keys(rng: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Split a PRNGKey into one key and a list of n subkeys."""
    keys = jax.random.split(rng, n + 1)
    return keys[0], list(keys[1:])


class OGPOBPTTAgent(flax.struct.PyTreeNode):
    """OGPO with Backprop-Through-Time: Directly backprop gradients from Q-values into policy."""
    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    config: Any = nonpytree_field()

    def _compute_deterministic_flow_actions(
        self,
        observations: jnp.ndarray,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        is_encoded: bool = False,
    ) -> jnp.ndarray:
        """Differentiable flow action sampling for BPTT."""
        B = observations.shape[0]
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        if self.config.encoder and not is_encoded:
            observations = self.actor_network(
                method='call_submodule',
                submodule='actor',
                submethod='encode',
                observations=observations,
                params=grad_params,
            )
            is_encoded = True

        # Stop gradient on the noise so gradients only flow through the policy
        actions = jax.lax.stop_gradient(jax.random.normal(rng, (B, act_dim)))

        if self.config.policy_type == 'mip':
            t_0 = jnp.zeros((B, 1))
            a_0_hat = self.actor_network.select('actor')(observations, actions, t_0, params=grad_params, is_encoded=True)
            t_star = jnp.full((B, 1), self.config.mip_t_star)
            actions = self.actor_network.select('actor')(observations, a_0_hat, t_star, params=grad_params, is_encoded=True)
        else:
            # Deterministic flow integration (no noise injection for gradients)
            dt = 1.0 / self.config.flow_steps
            for i in range(self.config.flow_steps):
                t = jnp.full((B, 1), i / self.config.flow_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, params=grad_params, is_encoded=True)
                actions = actions + vels * dt

        actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        return actions

    def actor_loss_bptt(
        self,
        batch: dict,
        batch_success: dict,
        grad_params: flax.core.FrozenDict,
        success_flag: bool,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """BPTT actor loss: maximize Q(s, pi(s)) by backpropping through Q into pi."""
        obs = batch['observations']
        if self.config.ppo_batch_size < self.config.batch_size:
            rng, key = jax.random.split(rng)
            obs = jax.random.choice(key, obs, (self.config.ppo_batch_size,), replace=False)

        obs_encoded = None
        if self.config.encoder:
            obs_encoded = self.actor_network(
                method='call_submodule',
                submodule='actor',
                submethod='encode',
                observations=obs,
                params=grad_params,
            )

        rng, action_rng = jax.random.split(rng)
        num_samples = self.config['bptt_num_samples']

        def sample_and_evaluate(rng_i):
            actions = self._compute_deterministic_flow_actions(
                obs_encoded if self.config.encoder else obs,
                grad_params,
                rng_i,
                is_encoded=self.config.encoder
            )

            if self.config.encoder:
                critic_obs_enc = self.critic_network(
                    method='call_submodule',
                    submodule='target_critic',
                    submethod='encode',
                    observations=obs,
                )
                q_vals = self.critic_network.select('target_critic')(critic_obs_enc, actions, is_encoded=True)
            else:
                q_vals = self.critic_network.select('target_critic')(obs, actions)

            if self.config['q_agg'] == 'min':
                q = q_vals.min(axis=0)
            else:
                q = q_vals.mean(axis=0)

            return q, actions

        sample_rngs = jax.random.split(action_rng, num_samples)
        q_values_per_sample, actions_per_sample = jax.vmap(sample_and_evaluate)(sample_rngs)

        # Maximize Q-values, i.e. minimize -Q
        q_values = q_values_per_sample.mean(axis=0)  # [batch_size]
        pg_loss = -jnp.mean(q_values)

        bc_loss = 0.0
        if self.config.use_bc_regularization:
            rng, key = jax.random.split(rng)
            batch_obs_encoded = None
            batch_success_encoded = None
            if self.config.encoder:
                batch_obs_encoded = self.actor_network(
                    method='call_submodule',
                    submodule='actor',
                    submethod='encode',
                    observations=batch['observations'],
                    params=grad_params
                )
                batch_success_encoded = self.actor_network(
                    method='call_submodule',
                    submodule='actor',
                    submethod='encode',
                    observations=batch_success['observations'],
                    params=grad_params
                )

            bc_loss = self._compute_bc_loss(
                rng, grad_params, batch, batch_success, success_flag,
                obs_encoded=batch_obs_encoded,
                batch_success_encoded=batch_success_encoded
            )

        total_loss = pg_loss + self.config.bc_coeff * bc_loss

        info = {
            'pg_loss': pg_loss,
            'bc_loss': bc_loss,
            'q_values': q_values.mean(),
            'q_min': q_values.min(),
            'q_max': q_values.max(),
            'q_std': q_values.std(),
            'actions_mean': actions_per_sample.mean(),
            'actions_std': actions_per_sample.std(),
            'bptt_num_samples': num_samples,
        }

        return total_loss, info

    def _compute_bc_loss(self, rng, grad_params, batch, batch_success, success_flag, obs_encoded=None, batch_success_encoded=None):
        rng, key = jax.random.split(rng)

        def true_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded = args
            bc_loss, _ = self.bc_loss(batch_success, grad_params, key, obs_encoded=batch_success_encoded)
            return bc_loss

        def false_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded = args
            bc_loss, _ = self.bc_loss(batch, grad_params, key, obs_encoded=obs_encoded)
            return bc_loss

        bc_loss = jax.lax.cond(
            success_flag, true_branch, false_branch,
            (batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded),
        )
        return bc_loss

    def get_td_loss(self, batch, batch_actions, next_actions, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """Compute TD loss using q_helper functions."""
        if self.config.encoder:
            if next_obs_encoded is None:
                next_obs_encoded = self.critic_network(
                    method='call_submodule',
                    submodule='target_critic',
                    submethod='encode',
                    observations=batch['next_observations']
                )
            next_qs = self.critic_network.select('target_critic')(next_obs_encoded, actions=next_actions, is_encoded=True)
        else:
            next_qs = self.critic_network.select('target_critic')(batch['next_observations'], actions=next_actions)

        rng, agg_rng = jax.random.split(rng)
        next_q = aggregate_q_values(
            next_qs,
            method=self.config['q_agg'],
            rng=agg_rng,
            num_qs=self.config.num_qs,
        )

        target_q = compute_td_target(
            rewards=batch['rewards'],
            masks=batch['masks'],
            next_q=next_q,
            discount=self.config['discount'],
            horizon_length=self.config['horizon_length'],
        )

        if self.config["critic_loss_type"] == "hlgauss":
            if self.config.encoder:
                if obs_encoded is None:
                    obs_encoded = self.critic_network(
                        method='call_submodule',
                        submodule='critic',
                        submethod='encode',
                        observations=batch['observations'],
                        params=grad_params
                    )
                q, q_logits = self.critic_network.select('critic')(
                    obs_encoded, actions=batch_actions, params=grad_params, return_logits=True, is_encoded=True
                )
            else:
                q, q_logits = self.critic_network.select('critic')(
                    batch['observations'], actions=batch_actions, params=grad_params, return_logits=True
                )

            valid = batch.get('valid')
            if valid is not None and valid.ndim > 1:
                valid = valid[..., -1]

            td_loss, stats = compute_td_loss(
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
            if self.config.encoder:
                if obs_encoded is None:
                    obs_encoded = self.critic_network(
                        method='call_submodule',
                        submodule='critic',
                        submethod='encode',
                        observations=batch['observations'],
                        params=grad_params
                    )
                q = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=True)
            else:
                q = self.critic_network.select('critic')(batch['observations'], actions=batch_actions, params=grad_params)

            valid = batch.get('valid')
            if valid is not None and valid.ndim > 1:
                valid = valid[..., -1]

            td_loss, stats = compute_td_loss(
                q_pred=q,
                target_q=target_q,
                valid_mask=valid,
                loss_type="mse",
            )

        return td_loss, q, next_q, stats

    def get_calql_diff(self, batch_bc, batch_actions, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        B = batch_bc['observations'].shape[0]
        if self.config.encoder:
            if obs_encoded is None:
                obs_encoded = self.critic_network(
                    method='call_submodule',
                    submodule='critic',
                    submethod='encode',
                    observations=batch_bc['observations'],
                    params=grad_params
                )
            q_pred = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=True)
        else:
            q_pred = self.critic_network.select('critic')(batch_bc['observations'], actions=batch_actions, params=grad_params)

        # CQL Part
        rng, action_rng = jax.random.split(rng)
        cql_random_actions = jax.random.uniform(
            action_rng,
            shape=(B, self.config["cql_n_actions"], self.config["full_act_dim"]),
            minval=-1.0, maxval=1.0,
        )
        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        sample_rngs = jax.random.split(current_a_rng, self.config.cql_n_actions)
        vmapped_sample = jax.vmap(
            lambda obs, rng_i: self.sample_actions(obs, rng=rng_i),
            in_axes=(None, 0)
        )

        if self.config.encoder:
            if next_obs_encoded is None:
                next_obs_encoded = self.critic_network(
                    method='call_submodule',
                    submodule='target_critic',
                    submethod='encode',
                    observations=batch_bc['next_observations']
                )

        vmapped_q_fn = jax.vmap(
            lambda a: self.critic_network.select('critic')(
                obs_encoded if self.config.encoder else batch_bc['observations'],
                actions=a,
                params=grad_params,
                is_encoded=self.config.encoder
            ),
            in_axes=1, out_axes=-1
        )

        cql_current_actions = vmapped_sample(batch_bc['observations'], sample_rngs)
        cql_current_actions = jnp.transpose(cql_current_actions, (1, 0, 2))
        cql_next_actions = vmapped_sample(batch_bc['next_observations'], sample_rngs)
        cql_next_actions = jnp.transpose(cql_next_actions, (1, 0, 2))
        all_actions = jnp.concatenate([cql_random_actions, cql_current_actions, cql_next_actions], axis=1)

        cql_qs = vmapped_q_fn(all_actions)
        chex.assert_shape(cql_qs, (self.config.num_qs, B, self.config["cql_n_actions"]*3))

        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(subsample_key, (self.config.calql_q_subsample,), 0, self.config.num_qs,)
        cql_qs = cql_qs[subsample_idcs]
        q_pred = q_pred[subsample_idcs]

        # CalQL Part
        n_actions_for_calql = self.config.cql_n_actions * 3
        mc_lower_bound = jnp.repeat(batch_bc["mc_returns"].reshape(-1, 1), n_actions_for_calql, axis=1,)
        cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
        cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)], axis=-1)
        cql_qs -= jnp.log(cql_qs.shape[-1]) * self.config.cql_temp

        cql_ood_values = jax.scipy.special.logsumexp(cql_qs / self.config.cql_temp, axis=-1) * self.config.cql_temp
        calql_regularizer = (cql_ood_values - q_pred).mean()
        return calql_regularizer

    def bc_critic_loss(self, batch_bc, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """Standard TD loss with support for hlgauss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch_bc["actions"], (batch_bc["actions"].shape[0], -1))
            next_actions = jnp.reshape(batch_bc["next_actions"], (batch_bc["next_actions"].shape[0], -1))
        else:
            batch_actions = batch_bc["actions"][..., 0, :]
            next_actions = batch_bc["next_actions"][..., 0, :]
        rng, sample_rng = jax.random.split(rng)

        if self.config.encoder:
            if obs_encoded is None:
                obs_encoded = self.critic_network(
                    method='call_submodule',
                    submodule='critic',
                    submethod='encode',
                    observations=batch_bc['observations'],
                    params=grad_params
                )
            if next_obs_encoded is None:
                next_obs_encoded = self.critic_network(
                    method='call_submodule',
                    submodule='target_critic',
                    submethod='encode',
                    observations=batch_bc['next_observations']
                )

        td_loss, q, next_q, stats = self.get_td_loss(
            batch_bc, batch_actions, next_actions, grad_params, rng,
            obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded
        )
        calql_regularizer = self.get_calql_diff(
            batch_bc, batch_actions, grad_params, rng,
            obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded
        )
        critic_loss = td_loss + self.config.cql_alpha * calql_regularizer

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_q_mean': next_q.mean(),
            'calql_regularizer': calql_regularizer.mean(),
            **stats
        }

    def critic_loss(self, batch: dict, batch_bc: dict, grad_params: flax.core.FrozenDict,
                    rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Standard TD loss with support for hlgauss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)

        obs_encoded = None
        next_obs_encoded = None
        if self.config.encoder:
            obs_encoded = self.critic_network(
                method='call_submodule',
                submodule='critic',
                submethod='encode',
                observations=batch['observations'],
                params=grad_params
            )
            next_obs_encoded = self.critic_network(
                method='call_submodule',
                submodule='target_critic',
                submethod='encode',
                observations=batch['next_observations']
            )

        if self.config.encoder:
            next_actor_obs_encoded = self.actor_network(
                method='call_submodule',
                submodule='target_actor',
                submethod='encode',
                observations=batch['next_observations'],
            )
            next_actions = self.sample_actions(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)
        else:
            next_actions = self.sample_actions(batch['next_observations'], rng=sample_rng)

        td_loss, q, next_q, stats = self.get_td_loss(
            batch, batch_actions, next_actions, grad_params, rng,
            obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded
        )

        if batch_bc is not None:
            obs_encoded_bc = None
            if self.config.encoder:
                obs_encoded_bc = self.critic_network(
                    method='call_submodule',
                    submodule='critic',
                    submethod='encode',
                    observations=batch_bc['observations'],
                    params=grad_params
                )

            if self.config["action_chunking"]:
                batch_bc_actions = jnp.reshape(batch_bc["actions"], (batch_bc["actions"].shape[0], -1))
            else:
                batch_bc_actions = batch_bc["actions"][..., 0, :]

            calql_regularizer = self.get_calql_diff(
                batch_bc, batch_bc_actions, grad_params, rng, obs_encoded=obs_encoded_bc
            )

            critic_loss = td_loss + self.config.cql_alpha * calql_regularizer
            stats['calql_regularizer'] = calql_regularizer
        else:
            critic_loss = td_loss
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_actions': next_actions.mean(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_actions_std': next_actions.std(),
            'next_q_mean': next_q.mean(),
            **stats
        }

    def actor_total_loss(
        self,
        batch: dict,
        batch_success: dict,
        grad_params: flax.core.FrozenDict,
        success_flag,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Actor loss for separate optimization."""
        a_loss, a_info = self.actor_loss_bptt(batch, batch_success, grad_params, success_flag, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info

    def critic_total_loss(
        self,
        batch: dict,
        batch_bc: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray
    ) -> Tuple[jnp.ndarray, dict]:
        """Critic loss for separate optimization."""
        c_loss, c_info = self.critic_loss(batch, batch_bc, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['OGPOBPTTAgent', dict]:
        """Offline BC update of the actor network."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.bc_loss(batch, p, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        agent.target_update(new_actor_state, 'actor')

        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        bc_info = {**actor_info}
        new_rng, _ = jax.random.split(agent.rng)
        return agent.replace(actor_network=new_actor_state, rng=new_rng), bc_info

    @staticmethod
    def _update_offline_calql(agent, batch: dict) -> Tuple['OGPOBPTTAgent', dict]:
        """Offline CalQL update of the critic network."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.critic_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.bc_critic_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.bc_critic_loss(batch, p, rng1)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        new_rng, _ = jax.random.split(new_rng)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @jax.jit
    def calql_update(self, batch):
        return self._update_offline_calql(self, batch)

    @staticmethod
    def _update(agent, batch_tuple, success_flag) -> Tuple['OGPOBPTTAgent', dict]:
        """Apply gradient update to actor and critic networks."""
        batch, batch_bc, batch_success = batch_tuple
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=agent.actor_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                return agent.actor_total_loss(batch, batch_success, p, success_flag, rng1)
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        if agent.config.useSimBa:
            def critic_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch,
                    rngs={'dropout': rng1},
                    method=agent.critic_total_loss,
                    mutable=['batch_stats']
                )
                return (loss, info), new_model_state
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        else:
            def critic_loss_fn(p):
                return agent.critic_total_loss(batch, batch_bc, p, rng2)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        info = {**actor_info, **critic_info}
        return agent.replace(actor_network=new_actor_state, critic_network=new_critic_state, rng=new_rng), info

    @jax.jit
    def batch_update(self, batch, batch_bc, batch_success, success_flag):
        batch_tuple = (batch, batch_bc, batch_success)

        def scan_update(agent, batch_tuple):
            return self._update(agent, batch_tuple, success_flag)

        agent, infos = jax.lax.scan(scan_update, self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @staticmethod
    def _update_critic_only(agent, batch_tuple) -> Tuple['OGPOBPTTAgent', dict]:
        """Apply gradient update to critic network only."""
        batch, batch_bc = batch_tuple
        new_rng, sample_rng = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, batch_bc, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @jax.jit
    def batch_q_warmup_update(self, batch, batch_bc):
        """Update the agent and return a new agent with information dictionary."""
        batch_tuple = (batch, batch_bc)
        agent, infos = jax.lax.scan(self._update_critic_only, self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @staticmethod
    def _bc_refine_update(agent, batch_success) -> Tuple['OGPOBPTTAgent', dict]:
        """Apply BC loss update to actor network only using success buffer."""
        new_rng, rng1 = jax.random.split(agent.rng)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables,
                    batch_success,
                    rngs={'dropout': rng1},
                    method=lambda batch, params: agent.bc_loss(batch, params, rng1),
                    mutable=['batch_stats']
                )
                bc_info = {f'actor/bc_{k}': v for k, v in info.items()}
                return (loss, bc_info), new_model_state
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)
        else:
            def actor_loss_fn(p):
                bc_loss_val, bc_info = agent.bc_loss(batch_success, p, rng1)
                info = {f'actor/bc_{k}': v for k, v in bc_info.items()}
                return bc_loss_val, info
            new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        agent.target_update(new_actor_state, 'actor')
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        return agent.replace(actor_network=new_actor_state, rng=new_rng), actor_info

    @jax.jit
    def batch_bc_refine_update(self, batch_success):
        """BC refinement update using only the success buffer."""
        agent, infos = jax.lax.scan(self._bc_refine_update, self, batch_success)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

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
    def sample_actions_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None) -> jnp.ndarray:
        """Fully vectorized Best-of-N action selection for BPTT."""
        expected_single_ndim = 3 if self.config.encoder else 1
        is_single_obs = observations.ndim == expected_single_ndim
        if is_single_obs:
            observations = observations[None]

        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n

        # Generate all B*N candidate actions in one forward pass
        rng, action_key = jax.random.split(rng)
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
        sampled_actions = self.sample_actions(obs_expanded, rng=action_key, is_encoded=False)

        # Evaluate Q-ensemble: (num_qs, B*N) -> (num_qs, B, N)
        q_ensemble = self.critic_network.select('target_critic')(
            obs_expanded, actions=sampled_actions, is_encoded=False
        )
        q_values = q_ensemble.reshape(self.config.num_qs, batch_size, num_samples)

        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, self.config.num_qs)
            qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)
            min_qs = jnp.min(qs_sub, axis=0)
        else:
            min_qs = jnp.min(q_values, axis=0)

        best_idx = jnp.argmax(min_qs, axis=1)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)

        return best_actions

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None, is_encoded: bool = False) -> jnp.ndarray:
        """Sample final actions via the flow policy (deterministic for evaluation)."""
        add_batch_dim = False
        if not is_encoded:
            if (not self.config.encoder and observations.ndim == 1) or (self.config.encoder and observations.ndim == 3):
                add_batch_dim = True
                observations = observations[None]

        actions = self.compute_flow_actions(observations, use_target=True, rng=rng, is_encoded=is_encoded)

        if add_batch_dim:
            actions = actions[0]
        return actions

    @partial(jax.jit, static_argnames=('is_encoded', 'use_target'))
    def compute_flow_actions(self, observations, rng=None, use_target=True, is_encoded=False):
        """Compute actions from the flow model using pg_helper ODE functions."""
        add_batch_dim = False
        if not is_encoded:
            if (self.config.encoder and observations.ndim == 3) or (not self.config.encoder and observations.ndim == 1):
                observations = observations[None, ...]
                add_batch_dim = True

        actor_module_name = 'target_actor' if use_target else 'actor'

        if self.config.encoder and not is_encoded:
            observations = self.actor_network(
                method='call_submodule',
                submodule=actor_module_name,
                submethod='encode',
                observations=observations,
            )
            is_encoded = True

        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, is_encoded=is_encoded)

        if self.config.policy_type == 'mip':
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                is_encoded=True,
            )
        else:
            actions = sample_flow_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,
                clip_value=self.config['denoised_clip_value'],
                is_encoded=True,
            )

        if add_batch_dim:
            actions = actions[0]
        return actions

    @jax.jit
    def bc_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        obs_encoded: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, dict]:
        """Flow matching BC loss.

        MIP mode uses a two-term loss (regression at t=0 + denoising at t=t*);
        standard flow matching predicts velocity only.
        """
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        if self.config.encoder and obs_encoded is None:
            obs_encoded = self.actor_network(
                method='call_submodule', submodule='actor', submethod='encode',
                observations=batch['observations'], params=grad_params
            )
        actor_obs = obs_encoded if self.config.encoder else batch['observations']

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))

        if self.config.policy_type == 'mip':
            x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(
                batch_actions, rng, self.config.mip_t_star
            )

            # Regression at t=0
            pred_0 = self.actor_network.select('actor')(
                actor_obs, x_0, t_0_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_regression_raw = jnp.square(pred_0 - batch_actions)

            # Denoising at t=t*
            pred_t_star = self.actor_network.select('actor')(
                actor_obs, x_t_star, t_star_arr, params=grad_params, is_encoded=self.config.encoder
            )
            loss_denoising_raw = jnp.square(pred_t_star - batch_actions)

            combined_loss_raw = loss_regression_raw + loss_denoising_raw

            bc_loss = apply_chunking_mask(
                combined_loss_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_regression = apply_chunking_mask(
                loss_regression_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )
            loss_denoising = apply_chunking_mask(
                loss_denoising_raw, valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            info = {
                'bc_loss': bc_loss,
                'bc_mip_loss_regression': loss_regression,
                'bc_mip_loss_denoising': loss_denoising,
            }
        else:
            rng, targets_rng = jax.random.split(rng)
            x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

            pred = self.actor_network.select('actor')(
                actor_obs, x_t, t,
                params=grad_params,
                is_encoded=self.config.encoder
            )

            loss_vel = apply_chunking_mask(
                jnp.square(pred - vel), valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            bc_loss = loss_vel
            info = {
                'bc_loss': bc_loss,
                'bc_flow_loss_vel': loss_vel,
            }

        return bc_loss, info

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False):
        """Computes the Q-value for a given state-action pair for logging."""
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=is_encoded,
        )

    def reset_optimizers_with_lr(self,) -> 'OGPOBPTTAgent':
        """Reset optimizers with new LR."""
        actor_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.actor_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.actor_warmup_steps,
            decay_steps=self.config.actor_decay_steps,
            end_value=self.config.actor_end_value
        )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=self.config.critic_scheduler,
            base_lr=self.config.ppo_lr,
            warmup_steps=self.config.critic_warmup_steps,
            decay_steps=self.config.critic_decay_steps,
            end_value=self.config.critic_end_value
        )

        new_actor_tx = create_optimizer(
            lr_schedule=actor_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.actor_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )
        new_critic_tx = create_optimizer(
            lr_schedule=critic_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.critic_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )

        opt_type = "Muon" if self.config.use_muon else "AdamW"
        print(f"Done: reset actor optimizer with {opt_type} (LR: {self.config.ppo_lr}, scheduler: {self.config.actor_scheduler})")
        print(f"Done: reset critic optimizer with {opt_type} (LR: {self.config.critic_lr}, scheduler: {self.config.critic_scheduler})")

        new_actor_opt_state = new_actor_tx.init(self.actor_network.params)
        new_actor_network = self.actor_network.replace(tx=new_actor_tx, opt_state=new_actor_opt_state, step=1)

        new_critic_opt_state = new_critic_tx.init(self.critic_network.params)
        new_critic_network = self.critic_network.replace(tx=new_critic_tx, opt_state=new_critic_opt_state, step=1)

        return self.replace(actor_network=new_actor_network, critic_network=new_critic_network)

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=bool(self.config.encoder))

    @classmethod
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray,
               config: Any, adv_clip_min=None) -> 'OGPOBPTTAgent':
        """Initialize an OGPOBPTTAgent with network states."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]

        if config.action_chunking:
            full_actions = jnp.concatenate([ex_actions] * config.horizon_length, axis=-1)
        else:
            full_actions = ex_actions
        full_act_dim = full_actions.shape[-1]
        config['full_act_dim'] = full_act_dim
        config.adv_clip_min = adv_clip_min

        if config['policy_type'] == 'mip':
            assert 0.0 < config.mip_t_star < 1.0, "mip_t_star must be in (0, 1)"
            print(f"[MIP Mode] Using Minimal Iterative Policy with t*={config.mip_t_star}")
        else:
            print(f"[Flow Mode] Using Flow Matching with {config.flow_steps} steps")

        # Encoders
        encoders = dict()
        if config.encoder:
            if 'vit' in config.encoder:
                encoder_module = vit_encoder_modules[config.encoder]
            else:
                encoder_module = encoder_modules[config.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Resolve backbone per-network (useTF is a convenience shortcut)
        a_backbone = config['actor_backbone']
        c_backbone = config['critic_backbone']
        if config.useTF:
            a_backbone = 'tf'
            c_backbone = 'tf'
        config['actor_backbone'] = a_backbone
        config['critic_backbone'] = c_backbone

        time_emb_type = config.get('time_embedding', 'scalar')
        time_emb_module = None
        if time_emb_type == 'sinusoidal':
            time_emb_module = SinusoidalTimeEmbedding(embed_dim=config.get('time_embedding_dim', 32))

        # Actor definition
        if a_backbone == 'tf':
            actor_def = ActorVectorFieldTF(
                hidden_dim=config.tf_pi_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=False,
                num_layers=config.tf_pi_layers,
                num_heads=config.tf_pi_heads,
                dropout_rate=config.tf_pi_dropout,
                encoder=encoders.get('actor'),
                conditioning_type=config.tf_pi_conditioning,
                time_embedding=time_emb_module,
            )
        elif config.useSimBa:
            actor_def = ActorVectorFieldSimBa(
                hidden_dims=config['simba_actor_hidden_dims'],
                action_dim=full_act_dim,
                encoder=encoders.get('actor'),
                rs_norm_momentum=0.99,
                rs_norm_epsilon=1e-8,
                time_embedding=time_emb_module,
            )
        else:
            actor_def = ActorVectorField(
                hidden_dims=config.actor_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.actor_layer_norm,
                encoder=encoders.get('actor'),
                use_film=config.useFiLM,
                use_denoiser=False,  # BPTT doesn't use denoiser
                time_embedding=time_emb_module,
            )

        # Critic definition
        if c_backbone == 'tf':
            critic_def = ValueTF(
                hidden_dim=config.tf_q_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=True,
                num_ensembles=config.num_qs,
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                num_layers=config.tf_q_layers,
                num_heads=config.tf_q_heads,
                dropout_rate=config.tf_q_dropout,
                conditioning_type=config.tf_q_conditioning,
            )
        elif config.useSimBa:
            critic_def = ValueSimBa(
                hidden_dims=config['simba_critic_hidden_dims'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                rs_norm_momentum=0.99,
                rs_norm_epsilon=1e-8,
            )
        else:
            critic_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=config['num_qs'],
                encoder=encoders.get('critic'),
                critic_loss_type=config['critic_loss_type'],
                num_bins=config['num_bins'],
                q_min=config['q_min'],
                q_max=config['q_max'],
                action_repeat=config['action_repeat'],
                use_film=config.useFiLM,
            )

        ex_observations = ex_observations[None, ...]
        full_actions = full_actions[None, ...]
        ex_times = ex_times[None, ...]

        print(ex_observations.shape, full_actions.shape, ex_times.shape)

        actor_nets = {'actor': actor_def, 'target_actor': copy.deepcopy(actor_def),}
        init_args_tuple = (ex_observations, full_actions, ex_times)
        actor_args = {'actor': init_args_tuple, 'target_actor': init_args_tuple}

        critic_nets = {'critic': critic_def, 'target_critic': copy.deepcopy(critic_def),}
        critic_args = {'critic': (ex_observations, full_actions), 'target_critic': (ex_observations, full_actions)}

        if config.use_constant_noise:
            noise_def = None
        else:
            noise_def = NoiseInjectionNetwork(
                hidden_dims=config.noise_net_hidden_dims,
                action_dim=full_act_dim,
                layer_norm=config.noise_net_layer_norm,
                min_noise_std=config.min_noise_std,
                max_noise_std=config.max_noise_std,
            )
            actor_nets['noise_net'] = noise_def
            actor_nets['target_noise_net'] = copy.deepcopy(noise_def)
            if config.encoder:
                if 'vit' in config.encoder:
                    ex_noise_obs_shape = (encoders['actor'].num_patches * encoders['actor'].embed_dim,)
                else:
                    ex_noise_obs_shape = (encoders['actor'].mlp_hidden_dims[-1],)
                ex_noise_obs = jnp.zeros(ex_noise_obs_shape)
            else:
                ex_noise_obs = ex_observations
            actor_args['noise_net'] = (ex_noise_obs, ex_times)
            actor_args['target_noise_net'] = (ex_noise_obs, ex_times)

        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)

        # Ensure all numeric values are Python scalars
        actor_lr = float(config['actor_lr'])
        critic_lr = float(config['critic_lr'])
        clip_grad_norm = float(config.clip_grad_norm)
        actor_weight_decay = float(config.actor_weight_decay)
        critic_weight_decay = float(config.critic_weight_decay)

        if config.use_constant_scheduler_for_bc:
            actor_lr_schedule = actor_lr
        else:
            actor_lr_schedule = create_lr_schedule(
                scheduler_type=config.actor_scheduler,
                base_lr=actor_lr,
                warmup_steps=int(config.actor_warmup_steps),
                decay_steps=int(config.actor_decay_steps),
                end_value=float(config.actor_end_value)
            )
        critic_lr_schedule = create_lr_schedule(
            scheduler_type=config.critic_scheduler,
            base_lr=critic_lr,
            warmup_steps=int(config.critic_warmup_steps),
            decay_steps=int(config.critic_decay_steps),
            end_value=float(config.critic_end_value)
        )

        if config.use_muon:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=actor_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=actor_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                optax.contrib.muon(
                    learning_rate=critic_lr_schedule,
                    beta=float(config.muon_beta),
                    ns_steps=int(config.muon_ns_steps),
                    weight_decay=critic_weight_decay,
                    nesterov=config.muon_nesterov,
                )
            )
        else:
            @optax.inject_hyperparams
            def adam_optimizer(learning_rate, weight_decay):
                return optax.adamw(learning_rate, weight_decay=weight_decay)
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=actor_lr_schedule, weight_decay=actor_weight_decay)
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=critic_lr_schedule, weight_decay=critic_weight_decay)
            )

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        if config.useSimBa:
            actor_variables = actor_net_def.init(actor_rng, **actor_args)
            critic_variables = critic_net_def.init(critic_rng, **critic_args)

            actor_params = actor_variables.pop('params')
            actor_batch_stats = actor_variables.pop('batch_stats', None)
            critic_params = critic_variables.pop('params')
            critic_batch_stats = critic_variables.pop('batch_stats', None)

            actor_state = TrainState.create(
                apply_fn=actor_net_def.apply,
                params=actor_params,
                tx=actor_tx,
                batch_stats=actor_batch_stats
            )
            critic_state = TrainState.create(
                apply_fn=critic_net_def.apply,
                params=critic_params,
                tx=critic_tx,
                batch_stats=critic_batch_stats
            )
        else:
            actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
            critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
            actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
            critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)

        actor_params = actor_state.params
        actor_params['modules_target_actor'] = actor_params['modules_actor']
        if not config.use_constant_noise:
            actor_params['modules_target_noise_net'] = actor_params['modules_noise_net']

        params = critic_state.params
        params['modules_target_critic'] = params['modules_critic']

        config.ob_dims = ob_dims
        config.action_dim = action_dim

        print(config)

        return cls(rng, actor_state, critic_state, config)
