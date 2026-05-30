import copy
from typing import Any, Tuple, List, Optional
from functools import partial
import numpy as np

import chex
import flax
import jax
import jax.numpy as jnp
import optax

from ogpo.networks.encoders import encoder_modules, vit_encoder_modules
from ogpo.agents.modules.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogpo.networks import (
    ActorVectorField, Value, ActorVectorFieldTF, ValueTF,
    ActorVectorFieldSimBa, ValueSimBa, NoiseInjectionNetwork,
    OneStepPolicy,
)
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding
from ogpo.agents.modules.igp_targets_helper import get_flow_targets, get_shortcut_targets
from ogpo.agents.modules.bc_helper import (
    preprocess_actions,
    get_mip_targets,
    apply_chunking_mask,
)
from ogpo.agents.modules.maintenance import (
    create_lr_schedule,
    compute_param_norm,
    compute_param_diff_norm,
    restore_critic_params as _restore_critic_params,
    restore_actor_params as _restore_actor_params,
    create_optimizer,
)
from ogpo.agents.modules.q_helper import (
    aggregate_q_values,
    reduce_q_over_samples,
    compute_td_target,
    compute_td_loss,
    compute_mc_regression_loss,
    get_q_values as _get_q_values,
    sample_mip_q_values,
    compute_mip_q_predictions,
    sample_mip_q_ensemble_values,
    compute_mip_q_ensemble_predictions,
)
from ogpo.agents.modules.pg_helper import (
    sample_flow_actions_ode,
    sample_mip_actions_ode,
    sample_flow_actions_ode_with_correction,
    sample_flow_actions_sde,
    sample_mip_actions_sde,
    compute_flow_log_prob,
    compute_mip_log_prob,
    compute_ppo_loss,
    compute_ogpo_advantages,
    compute_fpo_cfm_ratios,
    compute_fpo_ppo_loss,
    compute_awr_cfm_loss,
    compute_cfm_chi2_ratio,
    compute_chi_po_beta,
    compute_kl_reg_beta,
    compute_chi_po_advantages,
    compute_chi_po_ppo_loss,
)

def _split_keys(rng: jnp.ndarray, n: int) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    """Split a PRNGKey into one key and a list of n subkeys."""
    keys = jax.random.split(rng, n + 1)
    return keys[0], list(keys[1:])


class OGPOAgent(flax.struct.PyTreeNode):
    """Online policy-gradient agent for fine-tuning flow-matching policies with RL."""
    rng: Any
    actor_network: TrainState
    critic_network: TrainState
    config: Any = nonpytree_field()
    pi_slow_params: Any = None
    chi_po_drift: Any = None  # scalar: latest mean chi2_ratio from actor, used by critic
    one_step_network: Optional[TrainState] = None

    def _encode_for_actor(self, observations, images=None, params=None, use_target=False):
        """Encode observations for actor. Returns encoded obs (flat features)."""
        if self.config.get('actor_obs', 'state') != 'image' or self.config.get('_encoder_frozen', False):
            return observations
        module = 'target_actor' if use_target else 'actor'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        encoded = self.actor_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)
        if self.config.get('_freeze_encoder_for_bc', False):
            encoded = jax.lax.stop_gradient(encoded)
        return encoded

    def _encode_for_critic(self, observations, images=None, params=None, use_target=False):
        """Encode observations for critic. Returns encoded obs (flat features)."""
        if self.config.get('critic_obs', 'state') != 'image' or self.config.get('_encoder_frozen', False):
            return observations
        module = 'target_critic' if use_target else 'critic'
        kwargs = dict(observations=observations)
        if images is not None:
            kwargs['images'] = images
        if params is not None:
            kwargs['params'] = params
        encoded = self.critic_network(
            method='call_submodule', submodule=module, submethod='encode', **kwargs)
        if self.config.get('_freeze_encoder_for_bc', False):
            encoded = jax.lax.stop_gradient(encoded)
        return encoded

    def _critic_is_encoded(self):
        """Whether critic receives pre-encoded observations (true when critic_obs='image')."""
        return self.config.get('critic_obs', 'state') == 'image'

    def _actor_is_encoded(self):
        """Whether actor receives pre-encoded observations (true when actor_obs='image')."""
        return self.config.get('actor_obs', 'state') == 'image'

    def _get_critic_obs(self, batch):
        """Critic observations: full_states when critic_obs=state (sees object pose), else observations."""
        original = self.config.get('_original_critic_obs', self.config.get('critic_obs', 'state'))
        if original == 'state' and 'full_states' in batch:
            return batch['full_states']
        return batch['observations']

    def _get_critic_next_obs(self, batch):
        """Get the right next_observations for the critic."""
        original = self.config.get('_original_critic_obs', self.config.get('critic_obs', 'state'))
        if original == 'state' and 'next_full_states' in batch:
            return batch['next_full_states']
        return batch['next_observations']

    def _get_critic_images(self, batch):
        """Get images for critic (None when critic_obs=state)."""
        if self.config.get('critic_obs', 'state') != 'image':
            return None
        return batch.get('images')

    def _get_critic_next_images(self, batch):
        """Get next_images for critic (None when critic_obs=state)."""
        if self.config.get('critic_obs', 'state') != 'image':
            return None
        return batch.get('next_images')

    def compute_log_probs_and_entropy(
        self,
        observations: jnp.ndarray,
        chains: jnp.ndarray,
        params: flax.core.FrozenDict,
        get_entropy: bool = False,
        use_target: bool = False,
        is_encoded: bool = False,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute log-probabilities and optionally entropy over the denoising chain."""
        actor_module_name = 'target_actor' if use_target else 'actor'
        noise_module_name = 'target_noise_net' if use_target else 'noise_net'

        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, params=params, use_target=use_target)

        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            logprob, entropy_rate_est, info = compute_mip_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                mip_t_star=self.config.mip_t_star,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
            )
        else:
            logprob, entropy_rate_est, info = compute_flow_log_prob(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                chains=chains,
                flow_steps=self.config.flow_steps,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                normalize_horizon=self.config['normalize_denoising_horizon'],
                normalize_dim=self.config['normalize_act_space_dimension'],
                is_encoded=True,
                params=params,
                get_entropy=get_entropy,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
                ft_flow_steps=self.config.get('ft_flow_steps', 0),
            )

        if get_entropy:
            return logprob, entropy_rate_est, info
        else:
            return logprob, None

    def compute_log_probs(self, observations: jnp.ndarray, chains: jnp.ndarray, params: flax.core.FrozenDict, use_target: bool = False,
                          is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """ Compute sum of log-probabilities over the denoising chain. """
        logprob, _ = self.compute_log_probs_and_entropy(observations, chains, params, get_entropy=False, use_target=use_target, is_encoded=is_encoded, images=images)
        return logprob

    def _sample_actions(self, obs: jnp.ndarray, rng: jnp.ndarray, use_target: bool, num_samples: int, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """ Sample actions from TARGET policy """
        sample_rngs = jax.random.split(rng, num_samples)
        vmapped_sample = jax.vmap(
            lambda obs_g, rng_g: self.sample_actions_with_noise(obs_g, rng=rng_g, use_target=use_target, is_encoded=True),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, use_target=use_target)
        actions, chains, old_lp, sigmas = vmapped_sample(obs_encoded, sample_rngs)

        return actions, chains, old_lp, sigmas

    def _sample_next_actions_for_q(self, obs_encoded: jnp.ndarray, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample multiple next actions (target policy, SDE) for Q-target variance reduction. Returns (G, batch, act_dim)."""
        sample_rngs = jax.random.split(rng, num_samples)

        def sample_single(rng_g):
            actions, _, _, _ = self.sample_actions_with_noise(
                obs_encoded, rng=rng_g, use_target=True, is_encoded=True
            )
            return actions

        return jax.vmap(sample_single)(sample_rngs)

    def _sample_actions_ode(self, obs_encoded: jnp.ndarray, rng: jnp.ndarray, num_samples: int) -> jnp.ndarray:
        """Sample actions from target policy via ODE (no logprob, no chains)."""
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('target_actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        def sample_ode_single(rng_g):
            if self.config.error_correct_ode_to_sde and self.config.use_denoiser:
                return sample_flow_actions_ode_with_correction(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    constant_noise_std=self.config['constant_noise_std'],
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )
            else:
                return sample_flow_actions_ode(
                    actor_fn=actor_fn, observations=obs_encoded, rng=rng_g,
                    flow_steps=self.config.flow_steps, act_dim=act_dim,
                    act_min=self.config.act_min, act_max=self.config.act_max,
                    clip_intermediate=False, clip_value=self.config['denoised_clip_value'],
                    is_encoded=True,
                )

        sample_rngs = jax.random.split(rng, num_samples)
        return jax.vmap(sample_ode_single)(sample_rngs)  # [G, batch, act_dim]

    def _compute_current_log_probs(self, obs: jnp.ndarray, chains: jnp.ndarray, grad_params: flax.core.FrozenDict, obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """ Compute log probabilities under CURRENT policy for actions sampled from target policy """
        vmapped_logprob = jax.vmap(
            lambda obs_g, chains_g: self.compute_log_probs_and_entropy(
                obs_g, chains_g, grad_params, get_entropy=True, is_encoded=True
            ),
            in_axes=(None, 0)
        )

        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)
        lp, entropy_rate_est, info_vmapped = vmapped_logprob(obs_encoded, chains)

        return lp, entropy_rate_est, info_vmapped

    def _compute_ref_log_probs(self, obs_encoded: jnp.ndarray, chains: jnp.ndarray) -> jnp.ndarray:
        """Compute chain-level log probs under slow reference policy. Returns [G, batch]."""
        vmapped_logprob = jax.vmap(
            lambda obs_g, chains_g: self.compute_log_probs_and_entropy(
                obs_g, chains_g, params=self.pi_slow_params, get_entropy=False, is_encoded=True
            ),
            in_axes=(None, 0)
        )
        lp, _ = vmapped_logprob(obs_encoded, chains)
        return lp

    def _sample_from_ref(self, obs_encoded: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
        """Sample actions from the slow reference policy (pi_slow_params) via SDE."""
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)
        ref_params = self.pi_slow_params

        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select('actor')(obs, actions, t, params=ref_params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select('noise_net')(obs, t, params=ref_params)

        if self.config.policy_type == 'mip':
            actions, _, _, _ = sample_mip_actions_sde(
                actor_fn=actor_fn, noise_fn=noise_fn, observations=obs_encoded, rng=rng,
                mip_t_star=self.config.mip_t_star, act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min, act_max=self.config.act_max,
                is_encoded=True, params=ref_params,
            )
        else:
            ft_flow_steps = self.config.get('ft_flow_steps', 0)
            actions, _, _, _ = sample_flow_actions_sde(
                actor_fn=actor_fn, noise_fn=noise_fn, observations=obs_encoded, rng=rng,
                flow_steps=self.config.flow_steps, act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min, act_max=self.config.act_max,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
                is_encoded=True, params=ref_params, ft_flow_steps=ft_flow_steps,
            )
        return actions

    def _compute_fwd_kl_bc_loss(self, rng: jnp.ndarray, grad_params: flax.core.FrozenDict,
                                batch: dict, obs_encoded: jnp.ndarray,
                                images: jnp.ndarray = None) -> Tuple[jnp.ndarray, dict]:
        """Forward KL surrogate: BC flow matching of the current actor onto slow-ref samples (KL(pi_ref || pi))."""
        if self.pi_slow_params is None:
            return jnp.float32(0.0), {}

        rng, sample_rng, bc_rng = jax.random.split(rng, 3)
        actions_ref = self._sample_from_ref(obs_encoded, sample_rng)
        actions_ref = jax.lax.stop_gradient(actions_ref)

        B = actions_ref.shape[0]
        H = self.config["horizon_length"]
        A = self.config["action_dim"]
        if self.config["action_chunking"]:
            actions_ref_3d = actions_ref.reshape(B, H, A)
        else:
            # bc_loss takes batch_actions[..., 0, :]; broadcast across H so any layout works.
            actions_ref_3d = jnp.broadcast_to(actions_ref.reshape(B, 1, A), (B, H, A))  # [B, H, A]

        fake_batch = {**batch, 'actions': actions_ref_3d}
        fake_batch['valid'] = jnp.ones((B, H))
        loss, info = self.bc_loss(fake_batch, grad_params, bc_rng,
                                  obs_encoded=obs_encoded, images=images)
        prefixed = {f'fwd_kl_{k}': v for k, v in info.items()}
        prefixed['fwd_kl_loss'] = loss
        return loss, prefixed

    def _compute_q_values(self, obs: jnp.ndarray, actions: jnp.ndarray, rng: jnp.ndarray,
                          images: jnp.ndarray = None, critic_obs: jnp.ndarray = None,
                          critic_images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute Q-ensemble values. Supports both standard ensemble-Q and MIP-Q.
        Returns (q_agg, q_full) with shapes [G, batch], [G, M, batch]."""
        c_obs = critic_obs if critic_obs is not None else obs
        c_images = critic_images if critic_images is not None else images
        critic_obs_enc = self._encode_for_critic(c_obs, images=c_images, use_target=True)
        is_enc = self._critic_is_encoded()

        if self.config.get('use_mip_q', False):
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            batch_size = critic_obs_enc.shape[0]

            if self.config.get('mip_q_ensemble', False):
                # Single shared noise across actions for fair comparison.
                rng, noise_rng = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_rng,
                    (batch_size, 1),
                    minval=-noise_scale,
                    maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_ensemble_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_sample=noise_sample,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc
                    )  # [num_ensemble_members, batch]

                    if self.config['q_agg'] == 'min':
                        q_agg = q_vals.min(axis=0)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                        q_agg = q_vals[subsample_idxs].min(axis=0)
                    else:
                        q_agg = q_vals.mean(axis=0)

                    return q_agg, q_vals
            else:
                rng, noise_rng = jax.random.split(rng)
                noise_samples = jax.random.uniform(
                    noise_rng,
                    (num_ensemble_members, batch_size, 1),
                    minval=-noise_scale,
                    maxval=noise_scale
                )

                def compute_q_for_group(actions_g, rng_g):
                    target_critic_fn = self.critic_network.select('target_critic')
                    q_vals = sample_mip_q_values(
                        critic_fn=target_critic_fn,
                        observations=critic_obs_enc,
                        actions=actions_g,
                        noise_samples=noise_samples,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc
                    )  # [num_ensemble_members, batch]

                    if self.config['q_agg'] == 'min':
                        q_agg = q_vals.min(axis=0)
                    elif self.config['q_agg'] == 'subsample':
                        subsample_idxs = jax.random.randint(rng_g, (2,), 0, num_ensemble_members)
                        q_agg = q_vals[subsample_idxs].min(axis=0)
                    else:
                        q_agg = q_vals.mean(axis=0)

                    return q_agg, q_vals
        else:
            def compute_q_for_group(actions_g, rng_g):
                q_vals = self.critic_network.select('target_critic')(critic_obs_enc, actions_g, is_encoded=is_enc)  # [M, batch]

                if self.config['q_agg'] == 'min':
                    q_agg = q_vals.min(axis=0)
                elif self.config['q_agg'] == 'subsample':
                    subsample_idxs = jax.random.randint(rng_g, (2,), 0, self.config.num_qs)
                    q_agg = q_vals[subsample_idxs].min(axis=0)
                else:
                    q_agg = q_vals.mean(axis=0)

                return q_agg, q_vals

        subsample_rngs = jax.random.split(rng, actions.shape[0])
        vmapped_critic = jax.vmap(compute_q_for_group, in_axes=(0, 0))
        return vmapped_critic(actions, subsample_rngs)  # [G, batch], [G, M, batch]

    def _compute_advantages(self, obs: jnp.ndarray, actions: jnp.ndarray, rng: jnp.ndarray,
                            obs_encoded: jnp.ndarray = None, images: jnp.ndarray = None,
                            critic_obs: jnp.ndarray = None, critic_images: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute GRPO-style advantages from Q-values using pg_helper."""
        q_agg, q_full = self._compute_q_values(obs, actions, rng, images=images,
                                                critic_obs=critic_obs, critic_images=critic_images)
        if self.config.get('use_value_fn', False) and self.config.get('adv_baseline', 'group_mean') == 'v_fn':
            c_obs = critic_obs if critic_obs is not None else obs
            c_images = critic_images if critic_images is not None else images
            critic_obs_enc = self._encode_for_critic(c_obs, images=c_images, use_target=True)
            is_enc = self._critic_is_encoded()
            v_ens = self.critic_network.select('target_value')(critic_obs_enc, is_encoded=is_enc)  # (num_v, B)
            v_agg = aggregate_q_values(
                v_ens,
                method=self.config.get('v_agg', 'mean'),
                rng=rng,
                num_qs=self.config.get('v_ensemble_size', 10),
            )  # (B,)
            advantages = q_agg - v_agg[None, :]  # (G, B)
            if self.config.get('normalize_group', False):
                advantages = advantages / (advantages.std(axis=0, keepdims=True) + 1e-8)
            return jax.lax.stop_gradient(advantages), q_agg
        return compute_ogpo_advantages(q_agg, q_full, self.config)

    def _compute_bc_loss(self, rng, grad_params, batch, batch_success, success_flag, obs_encoded=None, batch_success_encoded=None, images=None, images_success=None):
        rng, key = jax.random.split(rng)

        def true_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch_success, grad_params, key, obs_encoded=batch_success_encoded, images=images_success)
            return bc_loss

        def false_branch(args):
            batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success = args
            bc_loss, _ = self.bc_loss(batch, grad_params, key, obs_encoded=obs_encoded, images=images)
            return bc_loss

        bc_loss = jax.lax.cond(success_flag, true_branch, false_branch,
            (batch_success, batch, grad_params, key, obs_encoded, batch_success_encoded, images, images_success),
        )
        return bc_loss

    def actor_loss(self, batch: dict, batch_success: dict, grad_params: flax.core.FrozenDict, success_flag: bool, rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Compute PPO surrogate loss + entropy + BC regularization with OGPO advantage calculation."""
        skip_pg_loss = not self.config.get('fql_train_main_policy', True)

        pg_loss = jnp.float32(0.0)
        ent_loss = jnp.float32(0.0)
        ent = jnp.float32(0.0)
        lp = jnp.float32(0.0)
        old_lp = jnp.float32(0.0)
        adv = jnp.zeros(1)
        adv_pre = jnp.zeros(1)
        sigmas = jnp.float32(0.0)
        actions = jnp.zeros(1)
        pg_stats = {}
        info_extra = {}
        G = self.config.group_num_samples

        if not skip_pg_loss:
            obs = batch['observations']
            images = batch.get('images')
            c_obs = self._get_critic_obs(batch)
            c_images = self._get_critic_images(batch)
            if self.config.ppo_batch_size < self.config.batch_size:
                rng, key = jax.random.split(rng)
                idxs = jax.random.choice(key, obs.shape[0], (self.config.ppo_batch_size,), replace=False)
                obs = obs[idxs]
                c_obs = c_obs[idxs]
                if images is not None:
                    images = images[idxs]
                if c_images is not None:
                    c_images = c_images[idxs]

            obs_encoded = self._encode_for_actor(obs, images=images, params=grad_params)

            rng, sample_rng = jax.random.split(rng)

            if self.config.get('use_awr', False):
                actions = self._sample_actions_ode(obs_encoded, sample_rng, G)  # [G, batch, act_dim]

                def new_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('actor')(obs_a, x_tau, t, params=grad_params, is_encoded=True)

                # Advantages: chi2-penalized or standard
                rng, adv_rng = jax.random.split(rng)
                info_extra = {}
                if self.config.get('chi_po', False) and self.pi_slow_params is not None:
                    def ref_actor_fn(obs_a, x_tau, t):
                        return self.actor_network.select('actor')(obs_a, x_tau, t, params=self.pi_slow_params, is_encoded=True)
                    rng, chi_rng = jax.random.split(rng)
                    chi_rngs = jax.random.split(chi_rng, G)
                    chi2_ratio = jax.vmap(lambda a_g, r_g: compute_cfm_chi2_ratio(
                        new_actor_fn, ref_actor_fn, obs_encoded, a_g,
                        n_mc=self.config.get('awr_n_mc', 16), rng=r_g,
                    ))(actions, chi_rngs)  # [G, batch]
                    q_agg, q_full = self._compute_q_values(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)
                    beta = compute_chi_po_beta(q_full, self.config)
                    adv, adv_pre = compute_chi_po_advantages(q_full, chi2_ratio, beta, self.config)
                    info_extra = {'chi_po_beta': beta, 'chi_po_chi2_ratio_mean': chi2_ratio.mean()}
                else:
                    adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)

                rng, awr_rng = jax.random.split(rng)
                awr_n_mc = self.config.get('awr_n_mc', 16)
                awr_weight_max = self.config.get('awr_weight_max', 20.0)
                awr_mode = self.config.get('awr_mode', 'symmetric')

                def compute_awr_single(actions_g, adv_g, rng_g):
                    return compute_awr_cfm_loss(
                        actor_fn=new_actor_fn,
                        observations=obs_encoded,
                        actions=actions_g,
                        advantages=adv_g,
                        n_mc=awr_n_mc,
                        rng=rng_g,
                        awr_beta=self.config.awr_beta,
                        awr_weight_max=awr_weight_max,
                        awr_mode=awr_mode,
                    )

                awr_rngs = jax.random.split(awr_rng, G)
                awr_losses, awr_stats = jax.vmap(compute_awr_single)(actions, adv, awr_rngs)
                pg_loss = awr_losses.mean()

                ent = jnp.float32(0.0)
                ent_loss = jnp.float32(0.0)
                sigmas = jnp.float32(0.0)
                lp = jnp.float32(0.0)
                old_lp = jnp.float32(0.0)

                pg_stats = {f'awr_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in awr_stats.items()}
                pg_stats['ratio'] = jnp.float32(1.0)
                pg_stats['approx_kl'] = jnp.float32(0.0)

            elif self.config.get('use_fpo', False):
                actions = self._sample_actions_ode(obs_encoded, sample_rng, G)  # [G, batch, act_dim]

                def old_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('target_actor')(obs_a, x_tau, t, is_encoded=True)
                def new_actor_fn(obs_a, x_tau, t):
                    return self.actor_network.select('actor')(obs_a, x_tau, t, params=grad_params, is_encoded=True)

                # Advantages: chi2-penalized or standard
                rng, adv_rng = jax.random.split(rng)
                info_extra = {}
                if self.config.get('chi_po', False) and self.pi_slow_params is not None:
                    def ref_actor_fn(obs_a, x_tau, t):
                        return self.actor_network.select('actor')(obs_a, x_tau, t, params=self.pi_slow_params, is_encoded=True)
                    rng, chi_rng = jax.random.split(rng)
                    chi_rngs = jax.random.split(chi_rng, G)
                    chi2_ratio = jax.vmap(lambda a_g, r_g: compute_cfm_chi2_ratio(
                        new_actor_fn, ref_actor_fn, obs_encoded, a_g,
                        n_mc=self.config.fpo_n_mc, rng=r_g,
                    ))(actions, chi_rngs)  # [G, batch]
                    q_agg, q_full = self._compute_q_values(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)
                    beta = compute_chi_po_beta(q_full, self.config)
                    adv, adv_pre = compute_chi_po_advantages(q_full, chi2_ratio, beta, self.config)
                    info_extra = {'chi_po_beta': beta, 'chi_po_chi2_ratio_mean': chi2_ratio.mean()}
                else:
                    adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)

                rng, ratio_rng = jax.random.split(rng)
                def compute_ratio_single(actions_g, rng_g):
                    return compute_fpo_cfm_ratios(
                        old_actor_fn=old_actor_fn, new_actor_fn=new_actor_fn,
                        observations=obs_encoded, actions=actions_g,
                        n_mc=self.config.fpo_n_mc, rng=rng_g,
                        per_sample=self.config.fpo_per_sample,
                        cfm_loss_clamp=self.config.fpo_cfm_loss_clamp,
                        cfm_diff_clamp=self.config.fpo_cfm_diff_clamp,
                        use_huber=self.config.get('fpo_use_huber', False),
                        huber_delta=self.config.get('fpo_huber_delta', 1.0),
                    )
                ratio_rngs = jax.random.split(ratio_rng, G)
                ratios, fpo_stats = jax.vmap(compute_ratio_single)(actions, ratio_rngs)
                # ratios: [G, n_mc, batch] if per_sample, else [G, batch]

                adv_flat = adv.reshape(-1)  # [G * batch]
                if self.config.fpo_per_sample:
                    # [G, n_mc, batch] -> [n_mc, G*batch]
                    n_mc = self.config.fpo_n_mc
                    ratios_reshaped = ratios.transpose(1, 0, 2).reshape(n_mc, -1)
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_reshaped, adv_flat, self.config.clip_epsilon,
                        per_sample=True, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )
                else:
                    ratios_flat = ratios.reshape(-1)  # [G*batch]
                    pg_loss, pg_stats = compute_fpo_ppo_loss(
                        ratios_flat, adv_flat, self.config.clip_epsilon,
                        per_sample=False, use_aspo=self.config.fpo_use_aspo,
                        clip_min_epsilon_multiplier=self.config.clip_min_epsilon_multiplier,
                        adv_clip_min=self.config.adv_clip_min,
                    )

                # FPO has no tractable entropy
                ent = jnp.float32(0.0)
                ent_loss = jnp.float32(0.0)
                sigmas = jnp.float32(0.0)
                lp = jnp.float32(0.0)
                old_lp = jnp.float32(0.0)

                pg_stats.update({f'fpo_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in fpo_stats.items()})

            elif self.config.get('chi_po', False) or self.config.get('kl_reg', False):
                chi_po_on = self.config.get('chi_po', False)
                kl_reg_on = self.config.get('kl_reg', False)
                actions, chains, old_lp, sigmas = self._sample_actions(obs, sample_rng, use_target=True, num_samples=G, obs_encoded=obs_encoded, images=images)
                lp, entropy_rate_est, info_vmapped = self._compute_current_log_probs(obs, chains, grad_params, obs_encoded=obs_encoded)
                ref_lp = self._compute_ref_log_probs(obs_encoded, chains)  # [G, batch]
                log_ratio_raw = jax.lax.stop_gradient(lp - ref_lp)                              # [G, batch]
                chi2_ratio = jax.lax.stop_gradient(jnp.exp(log_ratio_raw)) if chi_po_on else None
                log_ratio = log_ratio_raw if kl_reg_on else None

                # Q-values and regularized advantages (chi_po and/or kl_reg)
                rng, adv_rng = jax.random.split(rng)
                q_agg, q_full = self._compute_q_values(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)
                beta = compute_chi_po_beta(q_full, self.config) if chi_po_on else None
                beta_kl = compute_kl_reg_beta(q_full, self.config) if kl_reg_on else None
                adv, adv_pre = compute_chi_po_advantages(
                    q_full, chi2_ratio, beta, self.config,
                    log_ratio=log_ratio, beta_kl=beta_kl,
                )

                # PPO loss (PPO ratio = exp(lp - old_lp), has gradients)
                lp_flat = lp.reshape(-1)
                old_lp_flat = old_lp.reshape(-1)
                adv_flat = adv.reshape(-1)
                pg_loss, pg_stats = compute_chi_po_ppo_loss(
                    lp_flat, old_lp_flat, adv_flat, beta, self.config, beta_kl=beta_kl,
                )

                ent = jnp.mean(entropy_rate_est)
                ent_loss = -ent
                info_extra = {f'info_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in info_vmapped.items()}
                if chi_po_on:
                    info_extra['chi_po_beta'] = beta
                    info_extra['chi_po_chi2_ratio_mean'] = chi2_ratio.mean()
                    info_extra['chi_po_chi2_ratio_std'] = chi2_ratio.std()
                    info_extra['chi_po_chi2_ratio_max'] = chi2_ratio.max()
                if kl_reg_on:
                    info_extra['kl_reg_beta'] = beta_kl
                    info_extra['kl_reg_log_ratio_mean'] = log_ratio.mean()
                    info_extra['kl_reg_log_ratio_std'] = log_ratio.std()
                    info_extra['kl_reg_log_ratio_max'] = log_ratio.max()

                lp = lp_flat
                old_lp = old_lp_flat
                adv = adv_flat

            else:
                actions, chains, old_lp, sigmas = self._sample_actions(obs, sample_rng, use_target=True, num_samples=G, obs_encoded=obs_encoded, images=images)

                rng, adv_rng = jax.random.split(rng)
                adv, adv_pre = self._compute_advantages(obs, actions, adv_rng, images=images, critic_obs=c_obs, critic_images=c_images)
                lp, entropy_rate_est, info_vmapped = self._compute_current_log_probs(obs, chains, grad_params, obs_encoded=obs_encoded)

                lp = lp.reshape(-1)  # [G * batch]
                old_lp = old_lp.reshape(-1)  # [G * batch]
                adv = adv.reshape(-1)  # [G * batch]
                pg_loss, pg_stats = compute_ppo_loss(lp, old_lp, adv, self.config)

                ent = jnp.mean(entropy_rate_est)
                ent_loss = -ent
                info_extra = {f'info_{k}': v.mean() if hasattr(v, 'mean') else v for k, v in info_vmapped.items()}

        # BC regularization (shared between FPO and OGPO)
        bc_loss = 0.0
        batch_images = batch.get('images')
        batch_obs_encoded = None
        if self.config.use_bc_regularization:
            rng, key = jax.random.split(rng)

            batch_success_images = batch_success.get('images')
            batch_obs_encoded = self._encode_for_actor(batch['observations'], images=batch_images, params=grad_params)
            batch_success_encoded = self._encode_for_actor(batch_success['observations'], images=batch_success_images, params=grad_params)

            bc_loss = self._compute_bc_loss(rng, grad_params, batch, batch_success, success_flag, obs_encoded=batch_obs_encoded, batch_success_encoded=batch_success_encoded, images=batch_images, images_success=batch_success_images)

        # Forward-KL regularization: BC flow-matching against slow-ref samples.
        fwd_kl_loss = jnp.float32(0.0)
        fwd_kl_info = {}
        if self.config.get('fwd_kl_reg', False) and self.pi_slow_params is not None:
            rng, fkl_rng = jax.random.split(rng)
            if batch_obs_encoded is None:
                batch_obs_encoded = self._encode_for_actor(batch['observations'], images=batch_images, params=grad_params)
            fwd_kl_loss, fwd_kl_info = self._compute_fwd_kl_bc_loss(
                fkl_rng, grad_params, batch, batch_obs_encoded, images=batch_images,
            )

        total_loss = (pg_loss
                      + self.config.entropy_coeff * ent_loss
                      + self.config.bc_coeff * bc_loss
                      + self.config.get('fwd_kl_bc_coeff', 0.0) * fwd_kl_loss)
        info = {
            'pg_loss': pg_loss,
            'ent_loss': ent_loss,
            'bc_loss': bc_loss,
            'lp': lp.mean() if hasattr(lp, 'mean') else lp,
            'lp_min': lp.min() if hasattr(lp, 'min') else lp,
            'lp_max': lp.max() if hasattr(lp, 'max') else lp,
            'old_lp': old_lp.mean() if hasattr(old_lp, 'mean') else old_lp,
            'entropy': ent,
            'adv': adv.mean(),
            'adv_min': adv.min(),
            'adv_max': adv.max(),
            'adv_std': adv.std(),
            'adv_pre_norm': adv_pre.mean(),
            'adv_pre_min': adv_pre.min(),
            'adv_pre_max': adv_pre.max(),
            'q_values': adv_pre.mean(),
            'sigmas': sigmas.mean() if hasattr(sigmas, 'mean') else sigmas,
            'actions': actions.mean(),
            'actions_min': actions.min(),
            'actions_max': actions.max(),
            'actions_std': actions.std(),
            'group_num_samples': G,
            'grpo_adv_group_std': jnp.std(adv_pre, axis=0).mean(),
            'grpo_adv_between_group_std': jnp.std(adv_pre.mean(axis=0)),
            **pg_stats,
            **info_extra,
            **fwd_kl_info,
        }

        # Drift statistic passed to critic for pessimism blending (chi_po takes priority if both active).
        if self.config.get('chi_po', False):
            info['_chi2_ratio'] = chi2_ratio  # [G, batch], not logged, used by critic
        elif self.config.get('kl_reg', False):
            info['_log_ratio'] = log_ratio  # [G, batch], not logged, used by critic

        return total_loss, info

    def get_td_loss(self, batch, batch_actions, next_actions, grad_params, rng, obs_encoded=None, next_obs_encoded=None, images=None, next_images=None, next_q_override=None):
        """Compute TD loss using q_helper functions - supports both ensemble-Q and MIP-Q."""
        is_enc = self._critic_is_encoded()
        if next_q_override is not None:
            next_q = next_q_override
        else:
            if next_obs_encoded is None:
                next_obs_encoded = self._encode_for_critic(self._get_critic_next_obs(batch), images=next_images, use_target=True)

            if self.config.get('use_mip_q', False):
                num_ensemble_members = self.config.get('num_ensemble_members', 10)
                noise_scale = self.config.get('mip_q_noise_scale', 1.0)
                mip_t_star = self.config.get('mip_q_t_star', 0.9)
                batch_size = next_obs_encoded.shape[0]

                target_critic_fn = self.critic_network.select('target_critic')

                if self.config.get('mip_q_ensemble', False):
                    rng, noise_rng = jax.random.split(rng)
                    noise_sample = jax.random.uniform(
                        noise_rng, (batch_size, 1),
                        minval=-noise_scale, maxval=noise_scale
                    )
                    next_qs = sample_mip_q_ensemble_values(
                        critic_fn=target_critic_fn,
                        observations=next_obs_encoded,
                        actions=next_actions,
                        noise_sample=noise_sample,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc
                    )
                else:
                    rng, noise_rng = jax.random.split(rng)
                    noise_samples = jax.random.uniform(
                        noise_rng, (num_ensemble_members, batch_size, 1),
                        minval=-noise_scale, maxval=noise_scale
                    )
                    next_qs = sample_mip_q_values(
                        critic_fn=target_critic_fn,
                        observations=next_obs_encoded,
                        actions=next_actions,
                        noise_samples=noise_samples,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc
                    )

                rng, agg_rng = jax.random.split(rng)
                next_q = aggregate_q_values(
                    next_qs,
                    method=self.config['q_agg'],
                    rng=agg_rng,
                    num_qs=num_ensemble_members,
                )
            else:
                next_qs = self.critic_network.select('target_critic')(next_obs_encoded, actions=next_actions, is_encoded=is_enc)

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

        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=images, params=grad_params)

        if self.config.get('use_mip_q', False):
            critic_base_fn = self.critic_network.select('critic')
            def critic_fn_with_params(obs, actions, scalar_input, time, is_encoded, return_logits):
                return critic_base_fn(
                    obs, actions=actions, scalar_input=scalar_input, time=time,
                    params=grad_params, is_encoded=is_encoded, return_logits=return_logits
                )

            if self.config.get('mip_q_ensemble', False):
                if self.config["critic_loss_type"] == "hlgauss":
                    q_0, q, q_0_logits, q_logits = compute_mip_q_ensemble_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc,
                        return_logits=True
                    )
                else:
                    q_0, q = compute_mip_q_ensemble_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        is_encoded=is_enc,
                        return_logits=False
                    )
                    q_0_logits = None
                    q_logits = None
            else:
                if self.config["critic_loss_type"] == "hlgauss":
                    q_0, q, q_0_logits, q_logits = compute_mip_q_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        num_ensemble_members=num_ensemble_members,
                        is_encoded=is_enc,
                        return_logits=True
                    )
                else:
                    q_0, q = compute_mip_q_predictions(
                        critic_fn=critic_fn_with_params,
                        observations=obs_encoded,
                        actions=batch_actions,
                        target_q=target_q,
                        rng=rng,
                        noise_scale=noise_scale,
                        mip_t_star=mip_t_star,
                        num_ensemble_members=num_ensemble_members,
                        is_encoded=is_enc,
                        return_logits=False
                    )
                    q_0_logits = None
                    q_logits = None
        else:
            if self.config["critic_loss_type"] == "hlgauss":
                q, q_logits = self.critic_network.select('critic')(
                    obs_encoded, actions=batch_actions,
                    params=grad_params, return_logits=True, is_encoded=is_enc
                )
            else:
                q = self.critic_network.select('critic')(
                    obs_encoded, actions=batch_actions,
                    params=grad_params, is_encoded=is_enc
                )
                q_logits = None

        # q: [num_qs/num_ensemble_members, batch_size]; target_q: [batch_size]
        valid = batch.get('valid')
        if valid is not None and valid.ndim > 1:
            valid = valid[..., -1]

        if self.config.get('use_mip_q', False):
            # MIP-Q two-term loss: regression at t=0 + denoising at t=t*
            if self.config["critic_loss_type"] == "hlgauss":
                td_loss_0, stats_0 = compute_td_loss(
                    q_pred=q_0,
                    target_q=target_q,
                    valid_mask=valid,
                    loss_type="hlgauss",
                    q_min=self.config['q_min'],
                    q_max=self.config['q_max'],
                    num_bins=self.config['num_bins'],
                    q_logits=q_0_logits,
                )
                td_loss_t_star, stats_t_star = compute_td_loss(
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
                td_loss_0, stats_0 = compute_td_loss(
                    q_pred=q_0,
                    target_q=target_q,
                    valid_mask=valid,
                    loss_type="mse",
                )
                td_loss_t_star, stats_t_star = compute_td_loss(
                    q_pred=q,
                    target_q=target_q,
                    valid_mask=valid,
                    loss_type="mse",
                )

            td_loss = td_loss_0 + td_loss_t_star

            stats = {
                'td_loss': td_loss,
                'td_loss_regression_t0': td_loss_0,
                'td_loss_denoising_tstar': td_loss_t_star,
                **{f'q_0_{k}': v for k, v in stats_0.items()},
                **{f'q_tstar_{k}': v for k, v in stats_t_star.items()},
            }
        else:
            if self.config["critic_loss_type"] == "hlgauss":
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
                td_loss, stats = compute_td_loss(
                    q_pred=q,
                    target_q=target_q,
                    valid_mask=valid,
                    loss_type="mse",
                )

        return td_loss, q, next_q, stats

    def get_calql_diff(self, batch, batch_actions, grad_params, rng, obs_encoded=None, images=None):
        B = self._get_critic_obs(batch).shape[0]
        is_enc = self._critic_is_encoded()
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=images, params=grad_params)
        q_pred = self.critic_network.select('critic')(obs_encoded, actions=batch_actions, params=grad_params, is_encoded=is_enc)

        # Sample actions from current policy (pre-encode obs so sample_actions skips raw images)
        actor_images = batch.get('images')
        actor_obs_encoded = self._encode_for_actor(batch['observations'], images=actor_images, use_target=True)
        next_actor_images = batch.get('next_images')
        next_actor_obs_encoded = self._encode_for_actor(batch['next_observations'], images=next_actor_images, use_target=True)

        rng, action_rng = jax.random.split(rng)
        cql_random_actions = jax.random.uniform(action_rng, shape=(B, self.config["cql_n_actions"], self.config["full_act_dim"]),
                                                minval=-1.0, maxval=1.0,)
        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        sample_rngs = jax.random.split(current_a_rng, self.config.cql_n_actions)
        vmapped_sample = jax.vmap(
            lambda obs, rng_i: self.sample_actions(obs, rng=rng_i, is_encoded=True),
            in_axes=(None, 0)
        )

        vmapped_q_fn = jax.vmap(
            lambda a: self.critic_network.select('critic')(obs_encoded, actions=a, params=grad_params, is_encoded=is_enc),
            in_axes=1, out_axes=-1
        )

        cql_current_actions = vmapped_sample(actor_obs_encoded, sample_rngs)
        cql_current_actions = jnp.transpose(cql_current_actions, (1, 0, 2))
        cql_next_actions = vmapped_sample(next_actor_obs_encoded, sample_rngs)
        cql_next_actions = jnp.transpose(cql_next_actions, (1, 0, 2))
        all_actions = jnp.concatenate([cql_random_actions, cql_current_actions, cql_next_actions], axis=1)

        cql_qs = vmapped_q_fn(all_actions)
        chex.assert_shape(cql_qs,(self.config.num_qs, B, self.config["cql_n_actions"]*3))
        
        rng, subsample_key = jax.random.split(rng)
        subsample_idcs = jax.random.randint(subsample_key, (self.config.calql_q_subsample,), 0, self.config.num_qs,)
        cql_qs = cql_qs[subsample_idcs]
        q_pred = q_pred[subsample_idcs]

        # CalQL lower bound: clamp OOD Q-values to MC returns.
        n_actions_for_calql = self.config.cql_n_actions * 3
        mc_lower_bound = jnp.repeat(batch["mc_returns"].reshape(-1, 1), n_actions_for_calql, axis=1,)
        cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
        cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)],axis=-1)
        cql_qs -= jnp.log(cql_qs.shape[-1]) * self.config.cql_temp
        
        cql_ood_values = jax.scipy.special.logsumexp(cql_qs / self.config.cql_temp, axis=-1)* self.config.cql_temp
        calql_regularizer = (cql_ood_values - q_pred).mean()
        return calql_regularizer
        
    def bc_critic_loss(self, batch, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """Standard TD loss with support for hlgauss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            next_actions = jnp.reshape(batch["next_actions"], (batch["next_actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
            next_actions = batch["next_actions"][..., 0, :]
        rng, sample_rng = jax.random.split(rng)

        images = self._get_critic_images(batch)
        next_images = self._get_critic_next_images(batch)
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=images, params=grad_params)
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(self._get_critic_next_obs(batch), images=next_images, use_target=True)

        td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=images, next_images=next_images)
        calql_regularizer = self.get_calql_diff(batch, batch_actions, grad_params, rng, obs_encoded=obs_encoded, images=images)
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

    def bc_q_critic_loss(self, batch, grad_params, rng, obs_encoded=None, next_obs_encoded=None):
        """TD loss + MC regression loss for bc_q_steps phase (critic-only)."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
            next_actions = jnp.reshape(batch["next_actions"], (batch["next_actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
            next_actions = batch["next_actions"][..., 0, :]

        images = self._get_critic_images(batch)
        next_images = self._get_critic_next_images(batch)
        if obs_encoded is None:
            obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=images, params=grad_params)
        if next_obs_encoded is None:
            next_obs_encoded = self._encode_for_critic(self._get_critic_next_obs(batch), images=next_images, use_target=True)

        td_loss, q, next_q, stats = self.get_td_loss(
            batch, batch_actions, next_actions, grad_params, rng,
            obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded,
            images=images, next_images=next_images)

        mc_loss = compute_mc_regression_loss(q, batch['mc_returns'])
        total_loss = td_loss + self.config.mc_regression_coeff * mc_loss

        extra_info = {}
        if self.config.get('use_value_fn', False):
            is_enc = self._critic_is_encoded()
            next_v_ens = self.critic_network.select('target_value')(next_obs_encoded, is_encoded=is_enc)
            v_pred = self.critic_network.select('value')(obs_encoded, params=grad_params, is_encoded=is_enc)
            if self.config.get('v_pair_with_q', False):
                next_v_per_head = jax.lax.stop_gradient(next_v_ens)  # (num_v, B)
                v_target = compute_td_target(
                    rewards=batch['rewards'],
                    masks=batch['masks'],
                    next_q=next_v_per_head,
                    discount=self.config['discount'],
                    horizon_length=self.config['horizon_length'],
                )  # (num_v, B)
                v_loss = jnp.mean(jnp.square(v_pred - v_target))
            else:
                next_v_agg = aggregate_q_values(
                    next_v_ens,
                    method=self.config.get('v_agg', 'mean'),
                    rng=rng,
                    num_qs=self.config.get('v_ensemble_size', 10),
                )
                v_target = compute_td_target(
                    rewards=batch['rewards'],
                    masks=batch['masks'],
                    next_q=jax.lax.stop_gradient(next_v_agg),
                    discount=self.config['discount'],
                    horizon_length=self.config['horizon_length'],
                )
                v_loss = jnp.mean(jnp.square(v_pred - v_target[None, :]))
            total_loss = total_loss + self.config.get('v_loss_coeff', 1.0) * v_loss
            extra_info = {'v_loss': v_loss, 'v_mean': v_pred.mean()}

        return total_loss, {
            'critic_loss': total_loss,
            'td_loss': td_loss,
            'mc_regression_loss': mc_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'next_q_mean': next_q.mean(),
            'mc_returns_mean': batch['mc_returns'].mean(),
            **extra_info,
            **stats
        }

    def critic_loss(self, batch: dict, grad_params: flax.core.FrozenDict,
                    rng: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """Standard TD loss with support for hlgauss and Q-target variance reduction."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)

        c_images = self._get_critic_images(batch)
        c_next_images = self._get_critic_next_images(batch)

        obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=c_images, params=grad_params)
        next_obs_encoded = self._encode_for_critic(self._get_critic_next_obs(batch), images=c_next_images, use_target=True)

        actor_next_images = batch.get('next_images')
        next_actor_obs_encoded = self._encode_for_actor(batch['next_observations'], images=actor_next_images, use_target=True)

        if self.config.get('use_one_step_for_targets', False) and self.one_step_network is not None:
            next_actions = self.sample_actions_one_step(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)
            td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=c_images, next_images=c_next_images)
            using_one_step = 1.0
            return td_loss, {
                'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
                'next_actions': next_actions.mean(), 'next_actions_min': next_actions.min(),
                'next_actions_max': next_actions.max(), 'next_actions_std': next_actions.std(),
                'next_q_mean': next_q.mean(), 'using_one_step_for_targets': using_one_step, **stats
            }

        if self.config.get('use_value_fn', False):
            # V(s')-based Q-target: no next-action denoising for the critic bootstrap.
            is_enc = self._critic_is_encoded()
            next_v_ens = self.critic_network.select('target_value')(
                next_obs_encoded, is_encoded=is_enc
            )  # (num_v, B)
            rng, v_agg_rng = jax.random.split(rng)
            next_q = aggregate_q_values(
                next_v_ens,
                method=self.config.get('v_agg', 'mean'),
                rng=v_agg_rng,
                num_qs=self.config.get('v_ensemble_size', 10),
            )  # (B,)

            td_loss, q, _, stats = self.get_td_loss(
                batch, batch_actions, None, grad_params, rng,
                obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded,
                images=c_images, next_images=c_next_images,
                next_q_override=next_q,
            )

            v_training_mode = self.config.get('v_training_mode', 'td_offpolicy')
            v_pair = self.config.get('v_pair_with_q', False)
            if v_training_mode == 'td_offpolicy':
                v_pred = self.critic_network.select('value')(
                    obs_encoded, params=grad_params, is_encoded=is_enc
                )  # (num_v, B)
                if v_pair:
                    # Each V_i bootstraps from its own target_V_i(s').
                    next_v_per_head = jax.lax.stop_gradient(next_v_ens)  # (num_v, B)
                    v_target = compute_td_target(
                        rewards=batch['rewards'],
                        masks=batch['masks'],
                        next_q=next_v_per_head,
                        discount=self.config['discount'],
                        horizon_length=self.config['horizon_length'],
                    )  # (num_v, B)
                    v_loss = jnp.mean(jnp.square(v_pred - v_target))
                else:
                    v_target = compute_td_target(
                        rewards=batch['rewards'],
                        masks=batch['masks'],
                        next_q=jax.lax.stop_gradient(next_q),
                        discount=self.config['discount'],
                        horizon_length=self.config['horizon_length'],
                    )  # (B,)
                    v_loss = jnp.mean(jnp.square(v_pred - v_target[None, :]))
                v_sup_info = {}
            elif v_training_mode == 'bc_q_mean':
                # Regress V(s') onto mean_{a'~pi} Q(s', a') over a G-sample ensemble.
                G = self.config.get('q_vr_num_samples', 8)
                rng, sample_rng_v = jax.random.split(rng)
                next_actions_group = self._sample_next_actions_for_q(
                    next_actor_obs_encoded, rng=sample_rng_v, num_samples=G
                )
                next_qs_group = jax.vmap(
                    lambda actions_g: self.critic_network.select('target_critic')(
                        next_obs_encoded, actions=actions_g, is_encoded=is_enc
                    ),
                )(next_actions_group)  # (G, num_qs, B)
                v_pred_next = self.critic_network.select('value')(
                    next_obs_encoded, params=grad_params, is_encoded=is_enc
                )  # (num_v, B)
                if v_pair:
                    # V_i supervised by mean_{a'} Q_i(s', a').
                    q_mean_per_head = reduce_q_over_samples(next_qs_group, 'mean')  # (num_qs, B)
                    v_loss = jnp.mean(jnp.square(v_pred_next - jax.lax.stop_gradient(q_mean_per_head)))
                    v_sup_info = {'v_bc_q_mean_target': q_mean_per_head.mean()}
                else:
                    q_mean_over_a = reduce_q_over_samples(next_qs_group, 'mean').mean(axis=0)  # (B,)
                    v_loss = jnp.mean(jnp.square(v_pred_next - jax.lax.stop_gradient(q_mean_over_a)[None, :]))
                    v_sup_info = {'v_bc_q_mean_target': q_mean_over_a.mean()}
            else:
                raise ValueError(f"Unknown v_training_mode: {v_training_mode}")

            total_loss = td_loss + self.config.get('v_loss_coeff', 1.0) * v_loss

            return total_loss, {
                'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
                'next_q_mean': next_q.mean(),
                'v_loss': v_loss, 'v_mean': next_v_ens.mean(),
                'v_std_over_ensemble': next_v_ens.std(axis=0).mean(),
                **v_sup_info, **stats,
            }

        if self.config.get('q_variance_reduction', False):
            G = self.config['q_vr_num_samples']

            next_actions_group = self._sample_next_actions_for_q(
                next_actor_obs_encoded, rng=sample_rng, num_samples=G
            )  # (G, B, act_dim)

            is_enc = self._critic_is_encoded()
            next_qs_group = jax.vmap(
                lambda actions_g: self.critic_network.select('target_critic')(
                    next_obs_encoded, actions=actions_g, is_encoded=is_enc
                ),
            )(next_actions_group)  # (G, num_qs, B)

            next_qs_reduced = reduce_q_over_samples(
                next_qs_group, method=self.config['q_vr_reduction']
            )  # (num_qs, B)

            rng, agg_rng = jax.random.split(rng)
            next_q = aggregate_q_values(
                next_qs_reduced,
                method=self.config['q_agg'],
                rng=agg_rng,
                num_qs=self.config.num_qs,
            )  # (B,)

            td_loss, q, _, stats = self.get_td_loss(
                batch, batch_actions, None, grad_params, rng,
                obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded,
                images=c_images, next_images=c_next_images,
                next_q_override=next_q,
            )

            q_std_across_samples = next_qs_group.std(axis=0)  # (num_qs, B)

            return td_loss, {
                'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
                'next_actions': next_actions_group.mean(), 'next_actions_min': next_actions_group.min(),
                'next_actions_max': next_actions_group.max(), 'next_actions_std': next_actions_group.std(axis=0).mean(),
                'next_q_mean': next_q.mean(),
                'q_vr_sample_std': q_std_across_samples.mean(), 'q_vr_sample_std_max': q_std_across_samples.max(),
                **stats,
            }
        else:
            next_actions = self.sample_actions(next_actor_obs_encoded, rng=sample_rng, is_encoded=True)

            chi_po_on = self.config.get('chi_po', False)
            kl_reg_on = self.config.get('kl_reg', False)
            if (chi_po_on or kl_reg_on) and self.chi_po_drift is not None:
                is_enc = self._critic_is_encoded()
                if self.config.get('use_mip_q', False):
                    num_ensemble_members = self.config.get('num_ensemble_members', 10)
                    noise_scale = self.config.get('mip_q_noise_scale', 1.0)
                    mip_t_star = self.config.get('mip_q_t_star', 0.9)
                    batch_size = next_obs_encoded.shape[0]
                    target_critic_fn = self.critic_network.select('target_critic')
                    if self.config.get('mip_q_ensemble', False):
                        rng, noise_rng = jax.random.split(rng)
                        noise_sample = jax.random.uniform(
                            noise_rng, (batch_size, 1),
                            minval=-noise_scale, maxval=noise_scale)
                        next_qs = sample_mip_q_ensemble_values(
                            critic_fn=target_critic_fn,
                            observations=next_obs_encoded,
                            actions=next_actions,
                            noise_sample=noise_sample,
                            mip_t_star=mip_t_star,
                            is_encoded=is_enc)
                    else:
                        rng, noise_rng = jax.random.split(rng)
                        noise_samples = jax.random.uniform(
                            noise_rng, (num_ensemble_members, batch_size, 1),
                            minval=-noise_scale, maxval=noise_scale)
                        next_qs = sample_mip_q_values(
                            critic_fn=target_critic_fn,
                            observations=next_obs_encoded,
                            actions=next_actions,
                            noise_samples=noise_samples,
                            mip_t_star=mip_t_star,
                            is_encoded=is_enc)
                else:
                    next_qs = self.critic_network.select('target_critic')(
                        next_obs_encoded, actions=next_actions, is_encoded=is_enc)
                q_mean = next_qs.mean(axis=0)
                q_min = next_qs.min(axis=0)
                if chi_po_on:
                    pessimism_coeff = jax.nn.sigmoid(self.config.chi_po_ensemble_alpha * (self.chi_po_drift - 1.0))
                else:
                    alpha_kl = self.config.get('kl_reg_ensemble_alpha', self.config.chi_po_ensemble_alpha)
                    pessimism_coeff = jax.nn.sigmoid(alpha_kl * self.chi_po_drift)
                next_q = (1.0 - pessimism_coeff) * q_mean + pessimism_coeff * q_min

                td_loss, q, _, stats = self.get_td_loss(
                    batch, batch_actions, None, grad_params, rng,
                    obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded,
                    images=c_images, next_images=c_next_images,
                    next_q_override=next_q)
            else:
                td_loss, q, next_q, stats = self.get_td_loss(batch, batch_actions, next_actions, grad_params, rng, obs_encoded=obs_encoded, next_obs_encoded=next_obs_encoded, images=c_images, next_images=c_next_images)

            return td_loss, {
                'td_loss': td_loss, 'q_mean': q.mean(), 'q_max': q.max(), 'q_min': q.min(),
                'next_actions': next_actions.mean(), 'next_actions_min': next_actions.min(),
                'next_actions_max': next_actions.max(), 'next_actions_std': next_actions.std(),
                'next_q_mean': next_q.mean(), **stats
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
        a_loss, a_info = self.actor_loss(batch, batch_success, grad_params, success_flag, rng)
        info = {f'actor/{k}': v for k,v in a_info.items()}
        return a_loss, info

    def critic_total_loss(self, batch: dict, grad_params: flax.core.FrozenDict, rng: jnp.ndarray):
        c_loss, c_info = self.critic_loss(batch, grad_params, rng)
        info = {f'critic/{k}': v for k,v in c_info.items()}

        # Optional MC regression: ||Q(s,a) - MC_return||^2; needs batch['mc_returns'].
        if self.config.get('mc_regression', False) and 'mc_returns' in batch:
            is_enc = self._critic_is_encoded()
            obs_encoded = self._encode_for_critic(
                self._get_critic_obs(batch),
                images=self._get_critic_images(batch),
                params=grad_params,
            )
            if self.config['action_chunking']:
                batch_actions = jnp.reshape(batch['actions'], (batch['actions'].shape[0], -1))
            else:
                batch_actions = batch['actions'][..., 0, :]
            q_pred = self.critic_network.select('critic')(
                obs_encoded, actions=batch_actions, params=grad_params, is_encoded=is_enc)
            mc_loss = compute_mc_regression_loss(q_pred, batch['mc_returns'])
            c_loss = c_loss + self.config.mc_regression_coeff * mc_loss
            info['critic/mc_regression_loss'] = mc_loss
            info['critic/mc_returns_mean'] = batch['mc_returns'].mean()

        return c_loss, info

    @staticmethod
    def _update_offline(agent, batch: dict) -> Tuple['OGPOAgent', dict]:
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
        agent.pi_slow_update(new_actor_state)

        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        # One-step policy: distillation only during BC (critic untrained, no Q-maximization).
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_offline', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return agent.one_step_distillation_loss(batch, p, rng2, use_q_loss=0.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        bc_info = {**actor_info, **one_step_info}
        new_rng, _ = jax.random.split(agent.rng)
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), bc_info
    
    @staticmethod
    def _update_offline_calql(agent, batch: dict) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to both networks."""
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
        agent._critic_target_update(new_critic_state)

        new_rng, _ = jax.random.split(new_rng)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info
        
    @jax.jit
    def bc_update(self, batch):
        return self._update_offline(self, batch)
    
    @jax.jit
    def calql_update(self, batch):
        return self._update_offline_calql(self, batch)

    @staticmethod
    def _update_bc_q_critic(agent, batch: dict) -> Tuple['OGPOAgent', dict]:
        """Critic-only update with TD + MC regression (bc_q_steps phase)."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        def critic_loss_fn(p):
            return agent.bc_q_critic_loss(batch, p, rng1)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent._critic_target_update(new_critic_state)

        new_rng, _ = jax.random.split(new_rng)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @jax.jit
    def bc_q_update(self, batch):
        return self._update_bc_q_critic(self, batch)

    @staticmethod
    def _update(agent, batch_tuple, success_flag) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to both networks."""
        batch, batch_success = batch_tuple
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
        agent.pi_slow_update(new_actor_state)
        # chi_po_drift = mean chi2_ratio (chi_po) or log_ratio (kl_reg only)
        if agent.config.get('chi_po', False):
            agent = agent.replace(chi_po_drift=actor_info['actor/_chi2_ratio'].mean())
        elif agent.config.get('kl_reg', False):
            agent = agent.replace(chi_po_drift=actor_info['actor/_log_ratio'].mean())
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        use_sb_q = agent.config.get('use_success_buffer_q', False)

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
            agent._critic_target_update(new_critic_state)
        elif use_sb_q:
            # Two TD updates: one on RB batch, one on SB batch (guarded on success_flag).
            agent, critic_info = OGPOAgent.critic_update_sb(
                agent, (batch, batch_success), success_flag, rng2)
            new_critic_state = agent.critic_network
        else:
            def critic_loss_fn(p):
                return agent.critic_total_loss(batch, p, rng2)
            new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
            agent._critic_target_update(new_critic_state)

        # One-step policy: Q-maximization + distillation (critic is trained online).
        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng3, _ = jax.random.split(rng2)
            def one_step_loss_fn(p):
                return agent.one_step_distillation_loss(batch, p, rng3, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        info = {**actor_info, **critic_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, critic_network=new_critic_state, one_step_network=new_one_step_state, rng=new_rng), info
        
    @jax.jit
    def batch_update(self, batch, batch_success, success_flag):
        """Interleaved update: each scan step updates both actor and critic.

        batch: dict of [utd, B, ...] arrays
        batch_success: dict of [utd, B, ...] arrays (success buffer)
        """
        batch_tuple = (batch, batch_success)

        def scan_update(agent, batch_tuple):
            return self._update(agent, batch_tuple, success_flag)

        agent, infos = jax.lax.scan(scan_update, self, batch_tuple)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def v_only_loss(self, batch, grad_params, rng):
        """V-network TD loss only. Used by the extra V updates when utd_v > utd_q."""
        c_images = self._get_critic_images(batch)
        c_next_images = self._get_critic_next_images(batch)
        obs_encoded = self._encode_for_critic(self._get_critic_obs(batch), images=c_images, params=grad_params)
        next_obs_encoded = self._encode_for_critic(self._get_critic_next_obs(batch), images=c_next_images, use_target=True)
        is_enc = self._critic_is_encoded()

        next_v_ens = self.critic_network.select('target_value')(next_obs_encoded, is_encoded=is_enc)  # (num_v, B)
        v_pred = self.critic_network.select('value')(obs_encoded, params=grad_params, is_encoded=is_enc)  # (num_v, B)

        v_training_mode = self.config.get('v_training_mode', 'td_offpolicy')
        v_pair = self.config.get('v_pair_with_q', False)

        if v_training_mode == 'td_offpolicy':
            if v_pair:
                v_target = compute_td_target(
                    rewards=batch['rewards'], masks=batch['masks'],
                    next_q=jax.lax.stop_gradient(next_v_ens),
                    discount=self.config['discount'],
                    horizon_length=self.config['horizon_length'],
                )  # (num_v, B)
                v_loss = jnp.mean(jnp.square(v_pred - v_target))
            else:
                rng, agg_rng = jax.random.split(rng)
                next_v_agg = aggregate_q_values(
                    next_v_ens, method=self.config.get('v_agg', 'mean'),
                    rng=agg_rng, num_qs=self.config.get('v_ensemble_size', 10),
                )
                v_target = compute_td_target(
                    rewards=batch['rewards'], masks=batch['masks'],
                    next_q=jax.lax.stop_gradient(next_v_agg),
                    discount=self.config['discount'],
                    horizon_length=self.config['horizon_length'],
                )
                v_loss = jnp.mean(jnp.square(v_pred - v_target[None, :]))
        elif v_training_mode == 'bc_q_mean':
            G = self.config.get('q_vr_num_samples', 8)
            rng, sample_rng = jax.random.split(rng)
            next_actor_obs_encoded = self._encode_for_actor(
                batch['next_observations'], images=batch.get('next_images'), use_target=True)
            next_actions_group = self._sample_next_actions_for_q(
                next_actor_obs_encoded, rng=sample_rng, num_samples=G)
            next_qs_group = jax.vmap(
                lambda actions_g: self.critic_network.select('target_critic')(
                    next_obs_encoded, actions=actions_g, is_encoded=is_enc),
            )(next_actions_group)  # (G, num_qs, B)
            if v_pair:
                q_mean_per_head = reduce_q_over_samples(next_qs_group, 'mean')  # (num_qs, B)
                v_loss = jnp.mean(jnp.square(v_pred - jax.lax.stop_gradient(q_mean_per_head)))
            else:
                q_mean_over_a = reduce_q_over_samples(next_qs_group, 'mean').mean(axis=0)  # (B,)
                v_loss = jnp.mean(jnp.square(v_pred - jax.lax.stop_gradient(q_mean_over_a)[None, :]))
        else:
            raise ValueError(f"Unknown v_training_mode: {v_training_mode}")

        v_loss_scaled = self.config.get('v_loss_coeff', 1.0) * v_loss
        info = {'critic/v_only_loss': v_loss, 'critic/v_only_mean': v_pred.mean()}
        return v_loss_scaled, info

    @staticmethod
    def _v_update(agent, batch) -> Tuple['OGPOAgent', dict]:
        """Single V-only gradient step (target_value Polyak update included)."""
        new_rng, rng1 = jax.random.split(agent.rng)
        def loss_fn(p):
            return agent.v_only_loss(batch, p, rng1)
        new_critic_state, info = agent.critic_network.apply_loss_fn(loss_fn)
        agent.target_update(new_critic_state, 'value')
        return agent.replace(critic_network=new_critic_state, rng=new_rng), info

    @jax.jit
    def batch_v_only_update(self, batch):
        """Run V-only updates across the UTD dimension. batch: dict of [utd_v_extra, B, ...]."""
        agent, infos = jax.lax.scan(self._v_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @staticmethod
    def critic_update(agent, batch_tuple) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to critic network only."""
        batch = batch_tuple
        new_rng, sample_rng = jax.random.split(agent.rng)
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent._critic_target_update(new_critic_state)
        return agent.replace(critic_network=new_critic_state, rng=new_rng), critic_info

    @staticmethod
    def critic_update_sb(agent, inputs, success_flag, rng) -> Tuple['OGPOAgent', dict]:
        """Critic update: RB batch + conditional success buffer batch.

        rng is passed in (not drawn from agent.rng) so callers that already
        split agent.rng for other sub-updates don't double-consume keys.
        """
        batch, success_batch = inputs
        sample_rng, sb_rng = jax.random.split(rng, 2)

        # TD update on the replay-buffer batch.
        def critic_loss_fn(p):
            return agent.critic_total_loss(batch, p, sample_rng)
        new_critic_state, critic_info = agent.critic_network.apply_loss_fn(critic_loss_fn)
        agent._critic_target_update(new_critic_state)
        agent = agent.replace(critic_network=new_critic_state)

        # Conditional TD update on the success-buffer batch.
        def do_sb(agent):
            def sb_loss_fn(p):
                return agent.critic_total_loss(success_batch, p, sb_rng)
            new_cs, _ = agent.critic_network.apply_loss_fn(sb_loss_fn)
            agent._critic_target_update(new_cs)
            return agent.replace(critic_network=new_cs)

        def skip_sb(agent):
            return agent

        agent = jax.lax.cond(success_flag, do_sb, skip_sb, agent)
        return agent, critic_info

    @staticmethod
    def actor_update(agent, batch_tuple, success_flag) -> Tuple['OGPOAgent', dict]:
        """Apply gradient update to actor network only (used with policy delay)."""
        batch, batch_success = batch_tuple
        new_rng, rng1 = jax.random.split(agent.rng, 2)

        if agent.config.useSimBa:
            def actor_loss_fn(params, batch_stats):
                variables = {'params': params, 'batch_stats': batch_stats}
                (loss, info), new_model_state = agent.actor_network.apply_fn(
                    variables, batch, batch_success,
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
        agent.pi_slow_update(new_actor_state)
        # chi_po_drift = mean chi2_ratio (chi_po) or log_ratio (kl_reg only)
        if agent.config.get('chi_po', False):
            agent = agent.replace(chi_po_drift=actor_info['actor/_chi2_ratio'].mean())
        elif agent.config.get('kl_reg', False):
            agent = agent.replace(chi_po_drift=actor_info['actor/_log_ratio'].mean())
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return agent.one_step_distillation_loss(batch, p, rng2, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        info = {**actor_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), info

    @jax.jit
    def batch_q_warmup_update(self, batch):
        """Critic-only update across the UTD axis. Always TD loss; MC regression
        is added inside critic_total_loss when mc_regression=true."""
        agent, infos = jax.lax.scan(self.critic_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    batch_critic_only_update = batch_q_warmup_update

    @staticmethod
    def _bc_refine_update(agent, batch_success) -> Tuple['OGPOAgent', dict]:
        """Apply BC loss update to actor network only using success buffer."""
        new_rng, rng1 = jax.random.split(agent.rng, 2)

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
        agent.pi_slow_update(new_actor_state)
        if not agent.config.use_constant_noise:
            agent.target_update(new_actor_state, 'noise_net')

        one_step_info = {}
        new_one_step_state = agent.one_step_network
        if agent.one_step_network is not None and agent.config.get('train_one_step_in_online', True):
            rng2, _ = jax.random.split(rng1)
            def one_step_loss_fn(p):
                return agent.one_step_distillation_loss(batch_success, p, rng2, use_q_loss=1.0)
            new_one_step_state, one_step_info = agent.one_step_network.apply_loss_fn(one_step_loss_fn)
            agent.target_update(new_one_step_state, 'one_step')

        info = {**actor_info, **one_step_info}
        return agent.replace(actor_network=new_actor_state, one_step_network=new_one_step_state, rng=new_rng), info

    @jax.jit
    def batch_bc_refine_update(self, batch_success):
        """BC refinement update using only the success buffer."""
        agent, infos = jax.lax.scan(self._bc_refine_update, self, batch_success)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def target_update(self, network, module_name):
        """Polyak update of a target network (actor and critic use different tau)."""
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

    def _critic_target_update(self, network):
        """Polyak update for the critic module, plus V(s) if enabled."""
        self.target_update(network, 'critic')
        if self.config.get('use_value_fn', False):
            self.target_update(network, 'value')

    def pi_slow_update(self, network):
        """Polyak EMA update for pi_slow, the slow reference policy shared by chi_po, kl_reg, and fwd_kl_reg."""
        if self.pi_slow_params is None:
            return
        tau = self.config.get('tau_slow', self.config.get('chi_po_ref_tau', 0.0005))
        new_ref = jax.tree_util.tree_map(
            lambda p, rp: p * tau + rp * (1 - tau),
            network.params['modules_actor'],
            self.pi_slow_params['modules_actor'],
        )
        self.pi_slow_params['modules_actor'] = new_ref
        if 'modules_noise_net' in self.pi_slow_params:
            new_noise_ref = jax.tree_util.tree_map(
                lambda p, rp: p * tau + rp * (1 - tau),
                network.params['modules_noise_net'],
                self.pi_slow_params['modules_noise_net'],
            )
            self.pi_slow_params['modules_noise_net'] = new_noise_ref

    @jax.jit
    def sample_actions_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None, images: jnp.ndarray = None, full_state: jnp.ndarray = None) -> jnp.ndarray:
        """Vectorized Best-of-N action selection: best action -> argmax_a min_ensemble Q(s, a)."""
        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[None]
            if images is not None:
                images = images[None]
            if full_state is not None:
                full_state = full_state[None]

        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n

        rng, action_key = jax.random.split(rng)
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
        images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None
        sampled_actions, _, _, _ = self.sample_actions_with_noise(
            obs_expanded, rng=action_key, use_target=True, is_encoded=False, images=images_expanded
        )

        # Critic uses full_state when originally state-based (state-based critic in image env).
        original_critic_obs = self.config.get('_original_critic_obs', self.config.get('critic_obs', 'state'))
        if full_state is not None and original_critic_obs == 'state':
            c_obs_expanded = jnp.repeat(full_state, num_samples, axis=0)
            c_images_expanded = None
        else:
            c_obs_expanded = obs_expanded
            c_images_expanded = images_expanded
        critic_obs = self._encode_for_critic(c_obs_expanded, images=c_images_expanded, use_target=True)
        is_enc = self._critic_is_encoded()

        if self.config.get('use_mip_q', False):
            num_ensemble_members = self.config.get('num_ensemble_members', 10)
            noise_scale = self.config.get('mip_q_noise_scale', 1.0)
            mip_t_star = self.config.get('mip_q_t_star', 0.9)
            total_candidates = batch_size * num_samples

            target_critic_fn = self.critic_network.select('target_critic')

            if self.config.get('mip_q_ensemble', False):
                rng, noise_key = jax.random.split(rng)
                noise_sample = jax.random.uniform(
                    noise_key, (total_candidates, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                q_values_flat = sample_mip_q_ensemble_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs, actions=sampled_actions,
                    noise_sample=noise_sample, mip_t_star=mip_t_star, is_encoded=is_enc
                )
            else:
                rng, noise_key = jax.random.split(rng)
                noise_samples_base = jax.random.uniform(
                    noise_key, (num_ensemble_members, 1),
                    minval=-noise_scale, maxval=noise_scale
                )
                noise_samples = jnp.broadcast_to(
                    noise_samples_base[:, None, :],
                    (num_ensemble_members, total_candidates, 1)
                )
                q_values_flat = sample_mip_q_values(
                    critic_fn=target_critic_fn,
                    observations=critic_obs, actions=sampled_actions,
                    noise_samples=noise_samples, mip_t_star=mip_t_star, is_encoded=is_enc
                )

            q_values = q_values_flat.reshape(num_ensemble_members, batch_size, num_samples)
            num_ensemble_or_noise = num_ensemble_members
        else:
            q_ensemble = self.critic_network.select('target_critic')(
                critic_obs, actions=sampled_actions, is_encoded=is_enc
            )
            q_values = q_ensemble.reshape(self.config.num_qs, batch_size, num_samples)
            num_ensemble_or_noise = self.config.num_qs

        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, num_ensemble_or_noise)
            qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)  # (2, B, N)
            min_qs = jnp.min(qs_sub, axis=0)                       # (B, N)
        else:
            min_qs = jnp.min(q_values, axis=0)                     # (B, N)

        best_idx = jnp.argmax(min_qs, axis=1)                      # (B,)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)  # (B, act_dim)

        return best_actions

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions(self, observations: jnp.ndarray, rng: jnp.ndarray = None, is_encoded: bool = False, images: jnp.ndarray = None, full_state: jnp.ndarray = None) -> jnp.ndarray:
        """Sample final actions via the flow policy with noise."""
        add_batch_dim = False
        if not is_encoded:
            if observations.ndim == 1:
                add_batch_dim = True
                observations = observations[None]
                if images is not None:
                    images = images[None]

        actions, chains, _, _ = self.sample_actions_with_noise(observations, use_target=True, rng=rng, is_encoded=is_encoded, images=images)

        if add_batch_dim:
            actions = actions[0]
        return actions

    @partial(jax.jit, static_argnames=('is_encoded',))
    def sample_actions_one_step(self, observations: jnp.ndarray, rng: jnp.ndarray = None, is_encoded: bool = False, images: jnp.ndarray = None) -> jnp.ndarray:
        """Sample actions using the one-step policy (direct noise-to-action map; faster than ODE sampling)."""
        if self.one_step_network is None:
            return self.sample_actions(observations, rng, is_encoded, images)

        add_batch_dim = False
        if not is_encoded:
            if observations.ndim == 1:
                add_batch_dim = True
                observations = observations[None]
                if images is not None:
                    images = images[None]

        obs_encoded = self._encode_for_actor(observations, images=images, use_target=True) if not is_encoded else observations

        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)
        noise = jax.random.normal(rng, (obs_encoded.shape[0], act_dim))

        actions = self.one_step_network.select('target_one_step')(
            obs_encoded, noise, is_encoded=True
        )

        actions = jnp.clip(actions, self.config.act_min, self.config.act_max)

        if add_batch_dim:
            actions = actions[0]
        return actions

    @jax.jit
    def sample_actions_one_step_BON(self, observations: jnp.ndarray, rng: jnp.ndarray = None, images: jnp.ndarray = None) -> jnp.ndarray:
        """Best-of-N action selection with the one-step policy (expects raw observations)."""
        if self.one_step_network is None:
            return self.sample_actions_BON(observations, rng, images)

        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[None]
            if images is not None:
                images = images[None]

        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        obs_expanded = jnp.repeat(observations, num_samples, axis=0)  # [B*N, obs_dim]
        images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None

        obs_encoded_actor = self._encode_for_actor(obs_expanded, images=images_expanded, use_target=True)

        rng, noise_rng = jax.random.split(rng)
        noise_samples = jax.random.normal(noise_rng, (batch_size * num_samples, act_dim))

        sampled_actions = self.one_step_network.select('target_one_step')(
            obs_encoded_actor, noise_samples, is_encoded=True
        )
        sampled_actions = jnp.clip(sampled_actions, self.config.act_min, self.config.act_max)

        critic_obs = self._encode_for_critic(obs_expanded, images=images_expanded, use_target=True)

        q_ensemble = self.critic_network.select('target_critic')(
            critic_obs, actions=sampled_actions, is_encoded=bool(self.config.encoder)
        )
        q_values = q_ensemble.reshape(self.config.num_qs, batch_size, num_samples)  # (num_qs, B, N)

        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2, batch_size, num_samples), 0, self.config.num_qs)
            qs_sub = jnp.take_along_axis(q_values, idxs, axis=0)  # (2, B, N)
            min_qs = jnp.min(qs_sub, axis=0)                       # (B, N)
        else:
            min_qs = jnp.min(q_values, axis=0)                     # (B, N)

        best_idx = jnp.argmax(min_qs, axis=1)                      # (B,)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        best_actions = jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)  # (B, act_dim)

        if is_single_obs:
            best_actions = best_actions[0]

        return best_actions

    def _check_add_batch_dim(self, observations, is_encoded):
        """Whether a batch dim must be added (Python bool from static ndim; not jitted)."""
        if is_encoded:
            return False
        return observations.ndim == 1

    def _encode_observations_for_flow(self, observations, is_encoded, images=None):
        """Encode observations for flow sampling. Called from within jitted functions."""
        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images)
        return observations

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions() instead."""
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init.
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            actions = sample_flow_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=False,  # no intermediate clipping in ODE inference
                clip_value=self.config['denoised_clip_value'],
                noises=ode_noises,
                is_encoded=True,
            )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions(self, observations, rng=None, noises=None, is_encoded=False, images=None, full_state=None):
        """Compute actions via pure ODE flow (no SDE-to-ODE correction); standard BC-phase eval."""
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    @partial(jax.jit, static_argnames=('is_encoded', 'noises', '_add_batch_dim'))
    def _compute_flow_actions_corrected_impl(self, observations, rng=None, noises=None, is_encoded=False, _add_batch_dim=False, images=None):
        """Internal jitted implementation. Use compute_flow_actions_corrected() instead."""
        if _add_batch_dim:
            observations = observations[None, ...]
            if images is not None:
                images = images[None, ...]

        batch_size = observations.shape[0]
        observations = self._encode_observations_for_flow(observations, is_encoded, images=images)
        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, is_encoded=True, return_denoiser=False):
            return self.actor_network.select('actor')(obs, actions, t, is_encoded=is_encoded, return_denoiser=return_denoiser)

        # FPO++ zero-sampling: use zeros instead of random noise for ODE init.
        if self.config.get('fpo_zero_sampling', False) and noises is None:
            ode_noises = jnp.zeros((batch_size, act_dim))
        else:
            ode_noises = noises

        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            # MIP does not use correction.
            actions = sample_mip_actions_ode(
                actor_fn=actor_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                noises=ode_noises,
                is_encoded=True,
            )
        elif self.config["use_shortcut"]:
            # Shortcut does not use correction.
            if ode_noises is None:
                actions = jax.random.normal(rng, (batch_size, act_dim))
            else:
                actions = ode_noises
            step_size = 1.0 / self.config.shortcut_inference_steps
            dt_base_val = np.log2(1.0 / step_size)
            dt_base = jnp.full((*observations.shape[:-1], 1), dt_base_val)
            for i in range(self.config.shortcut_inference_steps):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config.shortcut_inference_steps)
                vels = self.actor_network.select('actor')(observations, actions, t, dt_base, is_encoded=True)
                actions = actions + vels * step_size
            actions = jnp.clip(actions, self.config.act_min, self.config.act_max)
        else:
            # Legacy denoiser-trained ODE-with-correction requires both flags;
            # otherwise fall back to plain Euler ODE (correction belongs in SDE sampling, not ODE eval).
            use_legacy_correction = (
                self.config.get('error_correct_ode_to_sde', False)
                and self.config.get('use_denoiser', False)
            )
            if use_legacy_correction:
                actions = sample_flow_actions_ode_with_correction(
                    actor_fn=actor_fn,
                    observations=observations,
                    rng=rng,
                    flow_steps=self.config.flow_steps,
                    act_dim=act_dim,
                    constant_noise_std=self.config['constant_noise_std'],
                    act_min=self.config.act_min,
                    act_max=self.config.act_max,
                    clip_intermediate=False,
                    clip_value=self.config['denoised_clip_value'],
                    noises=ode_noises,
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
                    noises=ode_noises,
                    is_encoded=True,
                )

        if _add_batch_dim:
            actions = actions[0]
        return actions

    def compute_flow_actions_corrected(self, observations, rng=None, noises=None, is_encoded=False, images=None, full_state=None):
        """Compute actions via ODE flow, using the denoiser-trained correction sampler under the legacy flags."""
        add_batch_dim = self._check_add_batch_dim(observations, is_encoded)
        return self._compute_flow_actions_corrected_impl(observations, rng, noises, is_encoded, _add_batch_dim=add_batch_dim, images=images)

    def sample_actions_with_noise(
            self,
            observations: jnp.ndarray,
            grad_params: flax.core.FrozenDict = None,
            rng: jnp.ndarray = None,
            use_target: bool = True,
            is_encoded: bool = False,
            images: jnp.ndarray = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Generate actions, full chains, and log_probs using pg_helper SDE sampling."""
        params = grad_params or self.actor_network.params
        actor_module_name = 'target_actor'
        noise_module_name = 'target_noise_net'

        if not is_encoded:
            observations = self._encode_for_actor(observations, images=images, use_target=use_target)

        act_dim = self.config.action_dim * (self.config.horizon_length if self.config.action_chunking else 1)

        def actor_fn(obs, actions, t, params=None, is_encoded=True):
            return self.actor_network.select(actor_module_name)(obs, actions, t, params=params, is_encoded=is_encoded)

        def noise_fn(obs, t, params=None):
            return self.actor_network.select(noise_module_name)(obs, t, params=params)

        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            actions, chains, logprob, sigmas = sample_mip_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                mip_t_star=self.config.mip_t_star,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                is_encoded=True,
                params=params,
            )
        else:
            ft_flow_steps = self.config.get('ft_flow_steps', 0)
            actions, chains, logprob, sigmas = sample_flow_actions_sde(
                actor_fn=actor_fn,
                noise_fn=noise_fn,
                observations=observations,
                rng=rng,
                flow_steps=self.config.flow_steps,
                act_dim=act_dim,
                use_constant_noise=self.config.use_constant_noise,
                use_tapered_noise=self.config.get('use_tapered_noise', False),
                error_correct_sde_to_ode=self.config.get('error_correct_sde_to_ode', False),
                constant_noise_std=self.config.constant_noise_std,
                min_noise_std=self.config.min_noise_std,
                randn_clip_value=self.config.randn_clip_value,
                act_min=self.config.act_min,
                act_max=self.config.act_max,
                clip_intermediate=self.config['clip_intermediate_actions'],
                clip_value=self.config['denoised_clip_value'],
                is_encoded=True,
                params=params,
                ft_flow_steps=ft_flow_steps,
            )

        # Normalization (matches compute_log_probs_and_entropy); ft_flow_steps uses the effective step count.
        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            logprob_steps = 3
        else:
            ft = self.config.get('ft_flow_steps', 0)
            effective_ft = ft if (0 < ft < self.config.flow_steps) else self.config.flow_steps
            logprob_steps = effective_ft + 1 if effective_ft == self.config.flow_steps else effective_ft
        if self.config['normalize_denoising_horizon']:
            logprob = logprob / logprob_steps

        if self.config['normalize_act_space_dimension']:
            logprob = logprob / act_dim

        return actions, chains, logprob, sigmas

    @jax.jit
    def bc_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        obs_encoded: jnp.ndarray = None,
        images: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray, dict]:
        """Flow-matching BC loss (MIP two-term, flow+denoiser, or standard velocity prediction)."""
        batch_actions = preprocess_actions(batch, self.config["action_chunking"])
        batch_size = batch_actions.shape[0]

        if images is None:
            images = batch.get('images')
        if obs_encoded is None:
            obs_encoded = self._encode_for_actor(batch['observations'], images=images, params=grad_params)
        actor_obs = obs_encoded if self._actor_is_encoded() else batch['observations']

        valid_mask = batch.get("valid", jnp.ones((batch_size, self.config["horizon_length"])))

        if self.config.policy_type == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif self.config.policy_type == 'mip':
            # MIP two-term loss: regression at t=0 + denoising at t=t*.
            x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(
                batch_actions, rng, self.config.mip_t_star
            )

            pred_0 = self.actor_network.select('actor')(
                actor_obs, x_0, t_0_arr, params=grad_params, is_encoded=self._actor_is_encoded()
            )
            loss_regression_raw = jnp.square(pred_0 - batch_actions)

            pred_t_star = self.actor_network.select('actor')(
                actor_obs, x_t_star, t_star_arr, params=grad_params, is_encoded=self._actor_is_encoded()
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

            if self.config.use_shortcut:
                # Shortcut targets with bootstrapping.
                def model_call_fn(obs, x_t, t, dt_base):
                    return self.actor_network.select('target_actor')(obs, x_t, t, dt_base)

                x_t, vel, t, dt_base, _ = get_shortcut_targets(
                    batch['observations'],
                    batch_actions,
                    model_call_fn,
                    targets_rng,
                    denoise_timesteps=self.config.shortcut_denoising_steps,
                    bootstrap_ratio=self.config.bootstrap_ratio,
                )
                pred = self.actor_network.select('actor')(
                    batch['observations'], x_t, t, dt_base, params=grad_params
                )
                loss_denoising = 0.0
            else:
                x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)
                loss_denoising = 0.0

                if self.config.use_denoiser:
                    # Denoiser branch for SDE-to-ODE correction.
                    rng, z_rng = jax.random.split(rng)
                    z = jax.random.normal(z_rng, x_t.shape)

                    sigma = self.config.constant_noise_std
                    x_t_noisy = x_t + sigma * z

                    pred, pred_z = self.actor_network.select('actor')(
                        actor_obs, x_t_noisy, t,
                        params=grad_params,
                        is_encoded=self._actor_is_encoded(),
                        return_denoiser=True
                    )

                    loss_denoising = apply_chunking_mask(
                        jnp.square(pred_z - z), valid_mask, batch_size,
                        self.config["horizon_length"], self.config["action_dim"],
                        self.config["action_chunking"]
                    )
                else:
                    pred = self.actor_network.select('actor')(
                        actor_obs, x_t, t,
                        params=grad_params,
                        is_encoded=self._actor_is_encoded()
                    )

            loss_vel = apply_chunking_mask(
                jnp.square(pred - vel), valid_mask, batch_size,
                self.config["horizon_length"], self.config["action_dim"],
                self.config["action_chunking"]
            )

            bc_loss = loss_vel + loss_denoising
            info = {
                'bc_loss': bc_loss,
                'bc_flow_loss_vel': loss_vel,
                'bc_flow_loss_denoising': loss_denoising
            }

        return bc_loss, info

    def one_step_distillation_loss(
        self,
        batch: dict,
        grad_params: flax.core.FrozenDict,
        rng: jnp.ndarray,
        use_q_loss = 1.0,  # 1.0 for online RL, 0.0 for BC training (critic untrained)
    ) -> Tuple[jnp.ndarray, dict]:
        """FQL one-step policy loss: Q-maximization + distillation from the flow policy (balance set by fql_distillation_coeff)."""
        if self.one_step_network is None:
            raise ValueError("one_step_network is None. Set use_one_step_policy=true in config.")

        batch_size = batch['observations'].shape[0]
        rng, noise_rng, ode_rng, q_rng = jax.random.split(rng, 4)

        batch_actions = preprocess_actions(batch, self.config.action_chunking)
        act_dim = batch_actions.shape[-1]

        images = batch.get('images')
        obs_encoded_actor = self._encode_for_actor(batch['observations'], images=images, params=grad_params)

        noise = jax.random.normal(noise_rng, (batch_size, act_dim))

        actions_one_step = self.one_step_network.select('one_step')(
            obs_encoded_actor, noise, is_encoded=True, params=grad_params
        )

        # Q-maximization (target encoder/critic for stability; zeroed via use_q_loss during BC).
        obs_encoded_critic = self._encode_for_critic(batch['observations'], images=images, use_target=True)
        is_enc = bool(self.config.encoder)

        q_values = self.critic_network.select('target_critic')(
            obs_encoded_critic, actions=actions_one_step, is_encoded=is_enc
        )  # (num_qs, batch_size)

        q_aggregated = aggregate_q_values(
            q_values,
            method=self.config.get('q_agg', 'mean'),
            rng=q_rng,
            num_qs=self.config.get('num_qs', 10),
        )  # (batch_size,)

        q_loss = -jnp.mean(q_aggregated) * use_q_loss

        # Distillation target from the flow-policy ODE (no gradients).
        def actor_fn(obs, actions, t, is_encoded=True):
            return self.actor_network.select('target_actor')(obs, actions, t, is_encoded=is_encoded)

        actions_flow_ode = sample_flow_actions_ode(
            actor_fn=actor_fn,
            observations=obs_encoded_actor,
            rng=ode_rng,
            flow_steps=self.config.flow_steps,
            act_dim=act_dim,
            act_min=self.config.act_min,
            act_max=self.config.act_max,
            clip_intermediate=self.config.get('clip_intermediate_actions', True),
            clip_value=self.config.get('denoised_clip_value', 1.0),
            noises=noise,  # same noise for fair comparison
            is_encoded=True,
        )

        actions_flow_ode = jax.lax.stop_gradient(actions_flow_ode)

        distill_loss = jnp.mean(jnp.square(actions_one_step - actions_flow_ode))

        # Per-component gradient norms (logging only) to help tune fql_distillation_coeff.
        def compute_q_loss_only(params):
            actions = self.one_step_network.select('one_step')(
                obs_encoded_actor, noise, is_encoded=True, params=params
            )
            q_vals = self.critic_network.select('target_critic')(
                obs_encoded_critic, actions=actions, is_encoded=is_enc
            )
            q_agg = aggregate_q_values(
                q_vals,
                method=self.config.get('q_agg', 'mean'),
                rng=q_rng,
                num_qs=self.config.get('num_qs', 10),
            )
            return -jnp.mean(q_agg) * use_q_loss

        def compute_distill_loss_only(params):
            actions = self.one_step_network.select('one_step')(
                obs_encoded_actor, noise, is_encoded=True, params=params
            )
            return jnp.mean(jnp.square(actions - actions_flow_ode))

        grad_q = jax.grad(compute_q_loss_only)(grad_params)
        grad_distill = jax.grad(compute_distill_loss_only)(grad_params)

        grad_q_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad_q)))
        grad_distill_norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grad_distill)))

        grad_distill_weighted_norm = grad_distill_norm * self.config.get('fql_distillation_coeff', 10.0)

        fql_distillation_coeff = self.config.get('fql_distillation_coeff', 10.0)
        total_loss = q_loss + fql_distillation_coeff * distill_loss

        info = {
            'one_step_total_loss': total_loss,
            'one_step_q_loss': q_loss,
            'one_step_distill_loss': distill_loss,
            'one_step_q_mean': q_aggregated.mean(),
            'one_step_q_std': q_aggregated.std(),
            'one_step_actions_mean': actions_one_step.mean(),
            'one_step_actions_std': actions_one_step.std(),
            'flow_ode_actions_mean': actions_flow_ode.mean(),
            'flow_ode_actions_std': actions_flow_ode.std(),
            'fql_distillation_coeff': fql_distillation_coeff,
            'one_step_grad_q_norm': grad_q_norm,
            'one_step_grad_distill_norm': grad_distill_norm,
            'one_step_grad_distill_weighted_norm': grad_distill_weighted_norm,
            'one_step_grad_ratio': grad_q_norm / (grad_distill_norm + 1e-8),
        }

        return total_loss, info

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False, images=None):
        """Computes the Q-value for a given state-action pair for logging."""
        if not is_encoded:
            observations = self._encode_for_critic(observations, images=images, use_target=True)
        return _get_q_values(
            critic_fn=self.critic_network.select('target_critic'),
            observations=observations,
            actions=actions,
            q_agg=self.config['q_agg'],
            is_encoded=self._critic_is_encoded(),
        )
    
    def _get_lr_info(self, actor_state=None, critic_state=None):
        """Get learning rate information and parameter difference norms for logging."""
        actor_state = actor_state or self.actor_network
        critic_state = critic_state or self.critic_network

        actor_norm = compute_param_norm(actor_state.params['modules_actor'])
        target_actor_norm = compute_param_norm(actor_state.params['modules_target_actor'])
        critic_norm = compute_param_norm(critic_state.params['modules_critic'])
        target_critic_norm = compute_param_norm(critic_state.params['modules_target_critic'])

        actor_diff_norm = compute_param_diff_norm(
            actor_state.params['modules_actor'],
            actor_state.params['modules_target_actor']
        )
        critic_diff_norm = compute_param_diff_norm(
            critic_state.params['modules_critic'],
            critic_state.params['modules_target_critic']
        )

        noise_info = {}
        if not self.config.use_constant_noise:
            noise_norm = compute_param_norm(actor_state.params['modules_noise_net'])
            target_noise_norm = compute_param_norm(actor_state.params['modules_target_noise_net'])
            noise_diff_norm = compute_param_diff_norm(
                actor_state.params['modules_noise_net'],
                actor_state.params['modules_target_noise_net']
            )
            noise_info = {
                'noise_param_norm': noise_norm,
                'target_noise_param_norm': target_noise_norm,
                'noise_target_diff_norm': noise_diff_norm,
            }

        return {
            'actor_lr': actor_state.opt_state.hyperparams['learning_rate'],
            'critic_lr': critic_state.opt_state.hyperparams['learning_rate'],
            'actor_step': actor_state.step,
            'critic_step': critic_state.step,
            'actor_param_norm': actor_norm,
            'target_actor_param_norm': target_actor_norm,
            'critic_param_norm': critic_norm,
            'target_critic_param_norm': target_critic_norm,
            'actor_target_diff_norm': actor_diff_norm,
            'critic_target_diff_norm': critic_diff_norm,
            **noise_info,
        }

    def reset_optimizers_with_lr(self,) -> 'OGPOAgent':
        """Reset optimizers with new LR for online RL phase."""
        if self.config.get('_freeze_encoder_for_bc', False):
            self.config['_freeze_encoder_for_bc'] = False
            print("[reset_optimizers] Unfreezing encoder for online RL")
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

        _has_trainable_encoder = (
            self.config.get('actor_obs') == 'image'
            and not self.config.get('_encoder_frozen', False)
            and self.config.get('encoder_lr', 0) > 0
        )

        if _has_trainable_encoder:
            raise NotImplementedError("Encoder-aware optimizer not available on this branch")
        else:
            new_actor_tx = create_optimizer(
                lr_schedule=actor_lr_schedule,
                use_muon=self.config.use_muon,
                clip_grad_norm=self.config.clip_grad_norm,
                weight_decay=self.config.actor_weight_decay,
                muon_beta=self.config.muon_beta,
                muon_ns_steps=self.config.muon_ns_steps,
                muon_nesterov=self.config.muon_nesterov,
            )
            opt_type = "Muon" if self.config.use_muon else "AdamW"
            print(f"Done: reset actor optimizer with {opt_type} (LR: {self.config.ppo_lr}, scheduler: {self.config.actor_scheduler})")

        new_critic_tx = create_optimizer(
            lr_schedule=critic_lr_schedule,
            use_muon=self.config.use_muon,
            clip_grad_norm=self.config.clip_grad_norm,
            weight_decay=self.config.critic_weight_decay,
            muon_beta=self.config.muon_beta,
            muon_ns_steps=self.config.muon_ns_steps,
            muon_nesterov=self.config.muon_nesterov,
        )
        print(f"Done: reset critic optimizer with {'Muon' if self.config.use_muon else 'AdamW'} "
              f"(LR: {self.config.critic_lr}, scheduler: {self.config.critic_scheduler})")

        new_actor_opt_state = new_actor_tx.init(self.actor_network.params)
        new_actor_network = self.actor_network.replace(tx=new_actor_tx, opt_state=new_actor_opt_state, step=1)

        new_critic_opt_state = new_critic_tx.init(self.critic_network.params)
        new_critic_network = self.critic_network.replace(tx=new_critic_tx, opt_state=new_critic_opt_state, step=1)

        # pi_slow: snapshot current actor as slow reference (shared by chi_po, kl_reg, fwd_kl_reg).
        # Drift init: 1.0 for chi_po (chi2_ratio), else 0.0 (log_ratio).
        ref_params = None
        chi_po_drift = None
        chi_po_on = self.config.get('chi_po', False)
        kl_reg_on = self.config.get('kl_reg', False)
        fwd_kl_reg_on = self.config.get('fwd_kl_reg', False)
        if chi_po_on or kl_reg_on or fwd_kl_reg_on:
            ref_params = jax.tree.map(lambda x: x.copy(), self.actor_network.params)
            chi_po_drift = jnp.float32(1.0 if chi_po_on else 0.0)
            tau_slow = self.config.get('tau_slow', self.config.get('chi_po_ref_tau', 0.0005))
            print(f"[reset_optimizers] Initialized pi_slow (tau_slow={tau_slow}) "
                  f"for chi_po={chi_po_on}, kl_reg={kl_reg_on}, fwd_kl_reg={fwd_kl_reg_on}")

        return self.replace(
            actor_network=new_actor_network,
            critic_network=new_critic_network,
            pi_slow_params=ref_params,
            chi_po_drift=chi_po_drift,
        )
    
    def restore_critic_params(self, critic_path):
        _restore_critic_params(self.critic_network, critic_path)

    def restore_actor_params(self, actor_path):
        _restore_actor_params(self.actor_network, actor_path, has_encoder=self._actor_is_encoded())

    def create_frozen_encode_fn(self):
        """Create a JIT-compiled encode function with the trained actor encoder's frozen weights."""
        from ogpo.networks.encoders import VisionProprioEncoder
        from ogpo.networks.encoders import vit_encoder_modules as _vit_modules

        vit = _vit_modules[self.config['vit_encoder']]()
        encoder = VisionProprioEncoder(
            vit=vit,
            state_proj_dim=self.config['vit_state_proj_dim'],
            img_proj_dim=self.config.get('vit_img_proj_dim', 0),
        )
        encoder_params = self.actor_network.params['modules_actor']['encoder']
        frozen_params = {'params': encoder_params}

        @jax.jit
        def encode_fn(observations, images):
            """Encode batched (observations, images) into flat feature vectors."""
            return encoder.apply(frozen_params, images, observations)

        return encode_fn

    @classmethod
    def create(cls, seed: int, ex_observations: jnp.ndarray, ex_actions: jnp.ndarray, config: Any, adv_clip_min=None, ex_images=None, ex_full_states=None) -> 'OGPOAgent':
        """Initialize a OGPOAgent with network states."""
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
        config['proprio_dim'] = ex_observations.shape[-1]
        if config['policy_type'] == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif config['policy_type'] == 'mip':
            assert 0.0 < config.mip_t_star < 1.0, "mip_t_star must be in (0, 1)"
            print(f"[MIP Mode] Using Minimal Iterative Policy with t*={config.mip_t_star}")
        else:
            print(f"[Flow Mode] Using Flow Matching with {config.flow_steps} steps")

        use_constant_noise = bool(config.get('use_constant_noise', False))
        use_tapered_noise = bool(config.get('use_tapered_noise', False))
        error_correct_sde_to_ode = bool(config.get('error_correct_sde_to_ode', False))
        error_correct_ode_to_sde = bool(config.get('error_correct_ode_to_sde', False))
        use_denoiser = bool(config.get('use_denoiser', False))
        assert not (use_constant_noise and use_tapered_noise), \
            "use_constant_noise and use_tapered_noise are mutually exclusive"
        if error_correct_ode_to_sde and not use_denoiser:
            raise ValueError("error_correct_ode_to_sde=True requires use_denoiser=True (legacy FPO path)")
        if error_correct_sde_to_ode and not (use_tapered_noise or use_constant_noise):
            print("[WARN] error_correct_sde_to_ode=True without tapered/constant noise: "
                  "score correction will use noise-net sigmas; (1-t) singularity at t=1 is unprotected.")
        if error_correct_sde_to_ode and config['policy_type'] == 'mip':
            print("[WARN] error_correct_sde_to_ode=True with policy_type=mip is a no-op "
                  "(MIP actor outputs a denoiser, not a velocity).")

        encoders = dict()
        # Vision path: VisionProprioEncoder when actor_obs/critic_obs == 'image'.
        if config.actor_obs == 'image' or config.critic_obs == 'image':
            from ogpo.networks.encoders import VisionProprioEncoder
            assert ex_images is not None, "ex_images required when actor_obs or critic_obs is 'image'"
            vit_img_h, vit_img_w, vit_num_ch = ex_images.shape  # (H, W, C)
            print(f"[Vision] ViT input: {vit_img_h}x{vit_img_w}, {vit_num_ch} channels")
        if config.actor_obs == 'image':
            actor_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['actor'] = VisionProprioEncoder(
                vit=actor_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
                freeze_backbone=config.get('freeze_vit_backbone', False),
            )
            config['encoder'] = 'vision'  # sentinel for existing boolean checks
        if config.critic_obs == 'image':
            critic_vit = vit_encoder_modules[config.vit_encoder](
                img_h=vit_img_h, img_w=vit_img_w, num_channel=vit_num_ch)
            encoders['critic'] = VisionProprioEncoder(
                vit=critic_vit,
                state_proj_dim=config.vit_state_proj_dim,
                img_proj_dim=config.vit_img_proj_dim,
                freeze_backbone=config.get('freeze_vit_backbone', False),
            )
            config['encoder'] = 'vision'
        # Legacy CNN/ViT encoder path.
        if config.encoder and config.actor_obs == 'state' and config.critic_obs == 'state':
            if 'vit' in config.encoder:
                encoder_module = vit_encoder_modules[config.encoder]
            else:
                encoder_module = encoder_modules[config.encoder]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
        # Resolve per-network backbone (useTF forces 'tf' for both).
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
            time_emb_module = SinusoidalTimeEmbedding(
                embed_dim=config.get('time_embedding_dim', 32),
            )
            print(f"[time_embedding] sinusoidal (dim={config.get('time_embedding_dim', 32)})")

        if config['policy_type'] == 'diffusion':
            raise NotImplementedError("Diffusion policy not available on this branch")
        elif a_backbone == 'tf':
            actor_def = ActorVectorFieldTF(
                hidden_dim=config.tf_pi_embed_dim,
                action_dim=action_dim,
                action_chunk_size=config.horizon_length if config.action_chunking else 1,
                layer_norm=False,
                num_layers=config.tf_pi_layers,
                num_heads=config.tf_pi_heads,
                dropout_rate=config.tf_pi_dropout,
                use_denoiser=config.use_denoiser,
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
                use_denoiser=config.use_denoiser,
                time_embedding=time_emb_module,
                obs_two_tier=config.get('obs_two_tier', False),
                two_tier_img_dim=config.get('_two_tier_img_dim', 0),
                two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
                two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
            )

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
        elif config.get('use_mip_q', False):
            if config.get('mip_q_ensemble', False):
                # Explicit ensemble: multiple networks with a single shared noise.
                from ogpo.networks import ValueMIPEnsemble
                critic_def = ValueMIPEnsemble(
                    hidden_dims=config['value_hidden_dims'],
                    num_ensembles=config.get('num_ensemble_members', 10),
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config['num_bins'],
                    q_min=config['q_min'],
                    q_max=config['q_max'],
                )
                print(f"[MIP-Q Ensemble Mode] Using EXPLICIT ensemble with {config.get('num_ensemble_members', 10)} networks, "
                      f"noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
            else:
                # Implicit ensemble: single network with multiple noise samples.
                from ogpo.networks import ValueMIP
                critic_def = ValueMIP(
                    hidden_dims=config['value_hidden_dims'],
                    mip_q_noise_scale=config.get('mip_q_noise_scale', 1.0),
                    mip_t_star=config.get('mip_q_t_star', 0.9),
                    layer_norm=config['layer_norm'],
                    encoder=encoders.get('critic'),
                    critic_loss_type=config['critic_loss_type'],
                    num_bins=config['num_bins'],
                    q_min=config['q_min'],
                    q_max=config['q_max'],
                )
                print(f"[MIP-Q Mode] Using IMPLICIT ensemble with {config.get('num_ensemble_members', 10)} noise samples, "
                      f"noise_scale={config.get('mip_q_noise_scale', 1.0)}, t*={config.get('mip_q_t_star', 0.9)}")
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
                obs_two_tier=config.get('obs_two_tier', False),
                two_tier_img_dim=config.get('_two_tier_img_dim', 0),
                two_tier_proprio_dim=config.get('_two_tier_proprio_dim', 0),
                two_tier_fused_dim=config.get('two_tier_fused_dim', 0),
            )

        ex_observations = ex_observations[None, ...]
        full_actions = full_actions[None, ...]
        ex_times = ex_times[None, ...]
        if ex_images is not None:
            ex_images = ex_images[None, ...]
        if ex_full_states is not None:
            ex_full_states = ex_full_states[None, ...]

        print(ex_observations.shape, full_actions.shape, ex_times.shape)
        # _original_critic_obs survives the frozen-encoder override.
        original_critic_obs = config.get('_original_critic_obs', config.get('critic_obs', 'state'))
        if ex_full_states is not None and original_critic_obs == 'state':
            print(f"[Mixed obs] actor_obs={config.actor_obs} ({ex_observations.shape[-1]}D), "
                  f"critic_obs={original_critic_obs} ({ex_full_states.shape[-1]}D full state)")

        actor_nets = {'actor': actor_def, 'target_actor': copy.deepcopy(actor_def)}

        actor_init_kw = dict(
            observations=ex_observations,
            actions=full_actions,
            times=ex_times,
            dt=jnp.ones_like(ex_times) if config.use_shortcut else None,
        )
        if config.use_denoiser:
            actor_init_kw['is_encoded'] = False
            actor_init_kw['return_denoiser'] = True
            if config['actor_backbone'] == 'tf':
                actor_init_kw['deterministic'] = True
        if config.actor_obs == 'image' and ex_images is not None:
            actor_init_kw['images'] = ex_images

        actor_args = {
            'actor': actor_init_kw,
            'target_actor': actor_init_kw,
        }

        critic_nets = {'critic': critic_def, 'target_critic': copy.deepcopy(critic_def)}
        # State-based critic uses full_states instead of proprio.
        ex_critic_obs = ex_full_states if (ex_full_states is not None and original_critic_obs == 'state') else ex_observations
        critic_init_kw = dict(observations=ex_critic_obs, actions=full_actions)
        if original_critic_obs == 'image' and ex_images is not None:
            critic_init_kw['images'] = ex_images
        critic_args = {'critic': critic_init_kw, 'target_critic': critic_init_kw}

        # State-only V(s) ensemble (reuses Value with actions=None at call time).
        if config.get('use_value_fn', False):
            v_ens_size = config.get('v_ensemble_size', 10)
            if config.get('v_pair_with_q', False) and v_ens_size != config['num_qs']:
                raise ValueError(
                    f"v_pair_with_q=True requires v_ensemble_size==num_qs, got "
                    f"v_ensemble_size={v_ens_size} and num_qs={config['num_qs']}"
                )
            value_def = Value(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                num_ensembles=v_ens_size,
                encoder=encoders.get('critic'),
                critic_loss_type='mse',
                q_min=config['q_min'],
                q_max=config['q_max'],
                action_repeat=False,
                use_film=False,
            )
            critic_nets['value'] = value_def
            critic_nets['target_value'] = copy.deepcopy(value_def)
            v_init_kw = dict(observations=ex_critic_obs)
            if original_critic_obs == 'image' and ex_images is not None:
                v_init_kw['images'] = ex_images
            critic_args['value'] = v_init_kw
            critic_args['target_value'] = v_init_kw

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
                if config.actor_obs == 'image':
                    # VisionProprioEncoder output dim
                    vit = encoders['actor'].vit
                    img_dim = config.vit_img_proj_dim if config.vit_img_proj_dim > 0 else (vit.num_patches * vit.embed_dim)
                    state_dim = config.vit_state_proj_dim if config.vit_state_proj_dim > 0 else config.proprio_dim
                    ex_noise_obs_shape = (img_dim + state_dim,)
                elif 'vit' in config.encoder:
                    ex_noise_obs_shape = (encoders['actor'].num_patches * encoders['actor'].embed_dim,)
                else:
                    ex_noise_obs_shape = (encoders['actor'].mlp_hidden_dims[-1],)
                ex_noise_obs = jnp.zeros((1, *ex_noise_obs_shape))
            else:
                ex_noise_obs = ex_observations
            actor_args['noise_net'] = (ex_noise_obs, ex_times)
            actor_args['target_noise_net'] = (ex_noise_obs, ex_times)

        actor_net_def = ModuleDict(actor_nets)
        critic_net_def = ModuleDict(critic_nets)

        # Coerce numeric config values to Python scalars (not arrays).
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
                return optax.adamw(learning_rate, weight_decay=weight_decay, eps=1e-8)
            actor_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=actor_lr_schedule, weight_decay=actor_weight_decay)
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                adam_optimizer(learning_rate=critic_lr_schedule, weight_decay=critic_weight_decay)
            )

        _has_trainable_encoder = (
            config.get('actor_obs') == 'image'
            and not config.get('_encoder_frozen', False)
            and config.get('encoder_lr', 0) > 0
        )

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        # MIP-Q needs rng for initialization.
        if config.get('use_mip_q', False):
            critic_init_rng, target_critic_init_rng = jax.random.split(critic_rng)
            critic_args['critic']['rng'] = critic_init_rng
            critic_args['target_critic']['rng'] = target_critic_init_rng

        if config.useSimBa:
            actor_variables = actor_net_def.init(actor_rng, **actor_args)
            critic_variables = critic_net_def.init(critic_rng, **critic_args)

            # Separate trainable params from batch_stats (pop handles cases without RSNorm).
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

            if _has_trainable_encoder:
                raise NotImplementedError("Encoder-aware optimizer not available on this branch")

            actor_state = TrainState.create(actor_net_def, actor_params, actor_tx)
            critic_state = TrainState.create(critic_net_def, critic_params, critic_tx)

        # Initialize target networks.
        actor_params = actor_state.params
        actor_params['modules_target_actor'] = actor_params['modules_actor']
        if not config.use_constant_noise:
            actor_params['modules_target_noise_net'] = actor_params['modules_noise_net']
        
        params = critic_state.params
        params['modules_target_critic'] = params['modules_critic']
        if config.get('use_value_fn', False):
            params['modules_target_value'] = params['modules_value']

        # Optional FQL one-step policy.
        one_step_state = None
        if config.get('use_one_step_policy', False):
            print("[One-Step Policy] Initializing for FQL distillation")
            one_step_def = OneStepPolicy(
                hidden_dims=config.get('one_step_hidden_dims', (512, 512, 512, 512)),
                action_dim=full_act_dim,
                layer_norm=config.get('one_step_layer_norm', False),
                encoder=encoders.get('actor'),
            )
            one_step_nets = {'one_step': one_step_def, 'target_one_step': copy.deepcopy(one_step_def)}
            one_step_net_def = ModuleDict(one_step_nets)

            ex_noise = jax.random.normal(init_rng, (1, full_act_dim))
            one_step_init_kw = dict(observations=ex_observations, noise=ex_noise, is_encoded=False)
            if config.actor_obs == 'image' and ex_images is not None:
                one_step_init_kw['images'] = ex_images
            one_step_args = {'one_step': one_step_init_kw, 'target_one_step': one_step_init_kw}

            # One-step optimizer defaults to the actor's config.
            one_step_lr = float(config.get('one_step_lr') or config['actor_lr'])
            one_step_scheduler = config.get('one_step_scheduler') or config.actor_scheduler
            one_step_warmup = int(config.get('one_step_warmup_steps') or config.actor_warmup_steps)
            one_step_decay = int(config.get('one_step_decay_steps') or config.actor_decay_steps)
            one_step_end = float(config.get('one_step_end_value') or config.actor_end_value)

            one_step_lr_schedule = create_lr_schedule(
                scheduler_type=one_step_scheduler,
                base_lr=one_step_lr,
                warmup_steps=one_step_warmup,
                decay_steps=one_step_decay,
                end_value=one_step_end
            )
            one_step_tx = create_optimizer(
                lr_schedule=one_step_lr_schedule,
                clip_grad_norm=clip_grad_norm,
                weight_decay=actor_weight_decay,
            )

            one_step_rng, init_rng = jax.random.split(init_rng)
            one_step_params = one_step_net_def.init(one_step_rng, **one_step_args)['params']
            one_step_state = TrainState.create(one_step_net_def, one_step_params, one_step_tx)

            one_step_state.params['modules_target_one_step'] = one_step_state.params['modules_one_step']

        config.ob_dims = ob_dims
        config.action_dim = action_dim
        return cls(rng, actor_state, critic_state, config, one_step_network=one_step_state)
