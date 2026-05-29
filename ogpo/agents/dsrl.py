import copy
from typing import Any, Tuple

import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import Actor, Value, ActorVectorField, LogParam
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets
from ogpo.agents.modules.maintenance import (
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
)
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    apply_chunking_mask,
    compute_flow_bc_loss_online,
)
from ogpo.agents.modules.q_helper import (
    get_q_values as _get_q_values,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
)

class DSRLBestOfNAgent(flax.struct.PyTreeNode):
    """SAC noise agent + frozen BC flow matching.
    Critic operates in noise space. Actor outputs noise in [-am, am].
    Matching the reference DSRL implementation (no noise_critic).
    """

    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    bc_flow_network: TrainState
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_dist = self.actor_network.select('actor')(batch['next_observations'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=sample_rng)

        next_qs = self.critic_network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        temp = self.actor_network.select('temp')()

        if self.config['entropy_backup']:
            next_log_probs = jnp.clip(next_log_probs, -100, 100)
            target_q = batch['rewards'] + \
                (self.config['discount'] ** self.config['horizon_length']) * batch['masks'] * \
                (next_q - temp * next_log_probs)
        else:
            target_q = batch['rewards'] + \
                (self.config['discount'] ** self.config['horizon_length']) * batch['masks'] * next_q

        target_q = jnp.clip(target_q, -500.0, 500.0)

        q = self.critic_network.select('critic')(batch['observations'], actions=batch["actions"], params=grad_params)
        critic_loss = jnp.mean((q - target_q) ** 2)

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
            'target_q_mean': target_q.mean(), 'temp': temp,
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
        rng, sample_rng = jax.random.split(rng)
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        actions, log_probs = dist.sample_and_log_prob(seed=sample_rng)

        qs = self.critic_network.select('critic')(batch['observations'], actions=actions)
        if self.config['q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)

        temp = self.actor_network.select('temp')()
        actor_loss = jnp.mean(temp * log_probs - q)

        temp_param = self.actor_network.select('temp')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        temp_loss = (temp_param * (entropy - self.config['target_entropy'])).mean()

        total_loss = actor_loss + temp_loss
        return total_loss, {
            'total_loss': total_loss, 'actor_loss': actor_loss, 'temp_loss': temp_loss,
            'log_probs_mean': log_probs.mean(), 'entropy': -log_probs.mean(),
            'q_actor_mean': q.mean(), 'temp': temp, 'actor_mode': dist.mode().mean(),
        }

    def actor_total_loss(self, batch, grad_params, rng):
        a_loss, a_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in a_info.items()}
        return a_loss, info

    def actor_bc_loss(self, batch, grad_params, rng):
        successful_noise = batch["actions"]
        dist = self.actor_network.select('actor')(batch['observations'], params=grad_params)
        actor_noise = dist.mode() if hasattr(dist, 'mode') else dist.mean()
        bc_loss_unweighted = jnp.mean(jnp.square(actor_noise - successful_noise))
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

    def critic_total_loss(self, batch, grad_params, rng):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k, v in c_info.items()}
        return c_loss, info

    def bc_loss(self, batch, grad_params, rng):
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

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch: dict) -> Tuple['DSRLBestOfNAgent', dict]:
        new_rng, rng1, rng2 = jax.random.split(agent.rng, 3)

        def actor_loss_fn(p):
            return agent.actor_total_loss(batch, p, rng1)
        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        return agent.replace(
            actor_network=new_actor_state, critic_network=new_critic_state, rng=new_rng,
        ), {**actor_info, **critic_info}

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['DSRLBestOfNAgent', dict]:
        new_rng, rng1 = jax.random.split(agent.rng, 2)
        def bc_loss_fn(p):
            return agent.bc_loss(batch, p, rng1)
        new_bc_flow_state, info = agent.bc_flow_network.apply_loss_fn(bc_loss_fn)
        return agent.replace(bc_flow_network=new_bc_flow_state, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)

    @staticmethod
    def _update_with_bc(agent, batch_tuple: Tuple[dict, dict, bool]) -> Tuple['DSRLBestOfNAgent', dict]:
        batch, batch_success, success_flag = batch_tuple
        new_rng, rng1, rng2, rng3 = jax.random.split(agent.rng, 4)

        def actor_loss_fn(p):
            rl_loss, rl_info = agent.actor_total_loss(batch, p, rng1)

            def add_bc_loss():
                bc_loss, bc_info = agent.actor_bc_loss(batch_success, p, rng3)
                return rl_loss + bc_loss, {**rl_info, **bc_info}

            def no_bc_loss():
                dummy_bc_info = {
                    'actor_bc_loss': 0.0, 'actor_bc_loss_unweighted': 0.0,
                    'actor_noise_mean': 0.0, 'actor_noise_std': 0.0,
                    'successful_noise_mean': 0.0, 'successful_noise_std': 0.0,
                    'bc_mse_error': 0.0,
                }
                return rl_loss, {**rl_info, **dummy_bc_info}

            return jax.lax.cond(success_flag, add_bc_loss, no_bc_loss)

        new_actor_state, actor_info = agent.actor_network.apply_loss_fn(actor_loss_fn)

        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, rng2)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent.target_update(new_critic_state, 'critic')

        return agent.replace(
            actor_network=new_actor_state, critic_network=new_critic_state, rng=new_rng,
        ), {**actor_info, **critic_info}

    @jax.jit
    def batch_update(self, batch, batch_success=None, success_flag=False):
        if batch_success is None:
            agent, infos = jax.lax.scan(self._update, self, batch)
        else:
            def scan_fn(agent, inputs):
                batch_i, batch_success_i = inputs
                return self._update_with_bc(agent, (batch_i, batch_success_i, success_flag))
            agent, infos = jax.lax.scan(scan_fn, self, (batch, batch_success))
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_noise(self, observations, rng=None, deterministic=False):
        dist = self.actor_network.select('actor')(observations)
        if deterministic:
            return dist.mode() if hasattr(dist, 'mode') else dist.mean()
        return dist.sample(seed=rng)

    @jax.jit
    def sample_actions(self, observations, rng=None, deterministic=False):
        dist = self.actor_network.select('actor')(observations)
        if deterministic:
            actions = dist.mode() if hasattr(dist, 'mode') else dist.mean()
        else:
            actions = dist.sample(seed=rng)

        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actions = jnp.concatenate([actions] * self.config["horizon_length"], axis=-1)

        actions = self.compute_flow_actions(observations, actions)
        return actions

    @jax.jit
    def noise_to_actions_public(self, observations, actor_noise):
        if self.config["noise_chunk_length"] > 0:
            assert self.config["noise_chunk_length"] == 1
            actor_noise = jnp.concatenate([actor_noise] * self.config["horizon_length"], axis=-1)
        return self.compute_flow_actions(observations, actor_noise)

    @jax.jit
    def sample_actions_normal(self, observations, rng=None):
        noises = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'] * \
                    (self.config['horizon_length'] if self.config["action_chunking"] else 1),
            ),
        )
        actions = self.compute_flow_actions(observations, noises)
        return actions

    def get_q_values(self, observations, actions, is_encoded=False):
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations, actions=actions,
            q_agg=self.config['q_agg'], is_encoded=is_encoded,
        )

    @jax.jit
    def compute_flow_actions(self, observations, noises=None, params=None, rng=None):
        add_batch_dim = False
        if (self.config['encoder'] is not None and observations.ndim == 3) or \
           (self.config['encoder'] is None and observations.ndim == 1):
            observations = observations[None, ...]
            if noises is not None:
                noises = noises[None, ...]
            add_batch_dim = True

        act_dim = self.config['noise_dim']

        if self.config['encoder'] is not None:
            if params is not None:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations, params=params)
            else:
                observations = self.bc_flow_network.select('actor_bc_flow_encoder')(observations)

        def actor_fn(obs, actions, t, is_encoded=True):
            if params is not None:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded, params=params)
            else:
                return self.bc_flow_network.select('actor_bc_flow')(obs, actions, t, is_encoded=is_encoded)

        actions = sample_flow_actions_ode(
            actor_fn=actor_fn, observations=observations, rng=rng,
            flow_steps=self.config['flow_steps'], act_dim=act_dim,
            act_min=-1.0, act_max=1.0, clip_intermediate=False,
            noises=noises, is_encoded=True,
        )

        if add_batch_dim:
            actions = actions[0]

        return actions

    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
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
                print(f"Restored DSRL actor from {actor_path}")
            else:
                raise ValueError(f"No modules_actor in actor_network checkpoint: {actor_path}")
        elif 'network' in agent_dict:
            src = agent_dict['network']['params']
            if 'modules_actor' in src:
                self.actor_network.params['modules_actor'] = src['modules_actor']
                print(f"Restored DSRL actor from unified checkpoint: {actor_path}")
            else:
                raise ValueError(f"No modules_actor in network checkpoint: {actor_path}")
        else:
            raise ValueError(f"Unknown checkpoint format in {actor_path}")

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
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

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()

        # Two-tier obs encoder (frozen-encoder image runs): splits the encoded
        # [image_features, proprio] obs back into image/proprio towers so
        # proprio isn't buried in a 2048+9D flat MLP. Matches the OGPO setup.
        _two_tier_kwargs = dict(
            obs_two_tier=config.get('obs_two_tier', False),
            two_tier_img_dim=config.get('_two_tier_img_dim', 0),
            two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
            two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
        )

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
            action_dim=noise_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            tanh_squash=True,
            state_dependent_std=True,
            low=-config["action_magnitude"],
            high=config["action_magnitude"],
            **_two_tier_kwargs,
        )

        temp_def = LogParam(init_value=config['init_temperature'])

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
        critic_args = {
            'critic': (ex_observations, ex_noise),
            'target_critic': (ex_observations, ex_noise),
        }

        bc_flow_nets = {
            'actor_bc_flow': actor_bc_flow_def,
        }
        bc_flow_args = {
            'actor_bc_flow': (ex_observations, full_actions, ex_times),
        }

        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)
        bc_flow_net_def = ModuleDict(bc_flow_nets)

        actor_lr = float(config['lr_actor'])
        critic_lr = float(config['lr_critic'])
        bc_flow_lr = float(config['lr_actor_bc_flow'])

        actor_opt_type = config['optimizer_actor']
        critic_opt_type = config['optimizer_critic']
        bc_flow_opt_type = config['optimizer_actor_bc_flow']

        clip_norm = float(config.get('clip_grad_norm', 1000.0))

        def get_optimizer(lr, opt_type):
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
        bc_flow_tx = get_optimizer(bc_flow_lr, bc_flow_opt_type)

        rng, actor_rng, critic_rng, bc_flow_rng = jax.random.split(rng, 4)
        actor_params = actor_net_def.init(actor_rng, **actor_args)['params']
        critic_params = critic_net_def.init(critic_rng, **critic_args)['params']
        bc_flow_params = bc_flow_net_def.init(bc_flow_rng, **bc_flow_args)['params']

        actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
        critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)
        bc_flow_state = TrainState.create(bc_flow_net_def, bc_flow_params, bc_flow_tx)

        critic_state.params[f'modules_target_critic'] = critic_state.params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config["noise_dim"] = noise_dim

        if config['target_entropy'] is None:
            config['target_entropy'] = -float(noise_dim) / 2

        return cls(rng, actor_state, critic_state, bc_flow_state, config=flax.core.FrozenDict(**config))
