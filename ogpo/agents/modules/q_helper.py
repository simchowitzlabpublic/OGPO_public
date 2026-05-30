"""Q-learning utilities: TD targets, losses, ensemble aggregation, CalQL."""

from functools import partial
from typing import Callable, Tuple, Optional, Dict, Any

import chex
import jax
import jax.numpy as jnp
import optax

from ogpo.agents.modules.dist_helper import hl_gauss_target


def aggregate_q_values(
    q_values: jnp.ndarray,
    method: str = "mean",
    rng: Optional[jnp.ndarray] = None,
    num_qs: Optional[int] = None,
    subsample_size: int = 2,
) -> jnp.ndarray:
    """Aggregate Q-values across a critic ensemble (axis 0) via "mean", "min", or "subsample"."""
    if method == "min":
        return q_values.min(axis=0)
    elif method == "subsample":
        if rng is None or num_qs is None:
            raise ValueError("rng and num_qs required for subsample aggregation")
        subsample_idxs = jax.random.randint(rng, (subsample_size,), 0, num_qs)
        return q_values[subsample_idxs].min(axis=0)
    else:  # default to mean
        return q_values.mean(axis=0)


def reduce_q_over_samples(
    q_values: jnp.ndarray,
    method: str = "mean",
) -> jnp.ndarray:
    """ Reduce Q-values over a group of sampled actions (variance reduction). """
    if method == "mean":
        return q_values.mean(axis=0)
    elif method == "median":
        return jnp.median(q_values, axis=0)
    elif method == "trimmed_mean":
        sorted_q = jnp.sort(q_values, axis=0)
        G = q_values.shape[0]
        lo = G // 4
        hi = G - lo
        return sorted_q[lo:hi].mean(axis=0)
    else:
        raise ValueError(f"Unknown q_vr_reduction method: {method}")



def get_mip_q_targets(
    target_q: jnp.ndarray,
    rng: jnp.ndarray,
    noise_scale: float = 1.0,
    t_star: float = 0.9,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate MIP-Q targets for the two-term loss (regression at t=0, denoising at t=t*).

    Returns (z_0, z_t_star, t_0_arr, t_star_arr), each [batch_size, 1].
    """
    batch_size = target_q.shape[0]

    rng, noise_rng = jax.random.split(rng)
    z_0 = jax.random.uniform(
        noise_rng, (batch_size, 1),
        minval=-noise_scale,
        maxval=noise_scale
    )

    # Interpolate in Q-space: z_t* = (1 - t*) * z_0 + Q_target
    target_q_expanded = target_q[:, None]  # [batch_size, 1]
    z_t_star = (1 - t_star) * z_0 + target_q_expanded

    t_0_arr = jnp.zeros((batch_size, 1))
    t_star_arr = jnp.full((batch_size, 1), t_star)

    return z_0, z_t_star, t_0_arr, t_star_arr


def sample_mip_q_values(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    noise_samples: jnp.ndarray,
    mip_t_star: float = 0.9,
    is_encoded: bool = False,
) -> jnp.ndarray:
    """Two-step autoregressive MIP-Q sampling for inference/evaluation.

    vmaps over multiple noise samples for a single network. The refinement step
    uses the Step-1 prediction directly, unlike training which uses interpolated targets.

    Returns Q-values from all noise samples: [num_ensemble_members, batch_size].
    """
    batch_size = observations.shape[0]

    def eval_q_two_step(noise_i):
        t_0 = jnp.zeros((batch_size, 1))
        q_0 = critic_fn(
            observations,
            actions=actions,
            scalar_input=noise_i,
            time=t_0,
            is_encoded=is_encoded
        )

        t_star = jnp.full((batch_size, 1), mip_t_star)
        q_0_expanded = q_0[:, None]  # [batch_size, 1]
        q_final = critic_fn(
            observations,
            actions=actions,
            scalar_input=q_0_expanded,
            time=t_star,
            is_encoded=is_encoded
        )

        return q_final

    q_values = jax.vmap(eval_q_two_step)(noise_samples)  # [num_ensemble_members, batch_size]

    return q_values


def sample_mip_q_ensemble_values(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    noise_sample: jnp.ndarray,
    mip_t_star: float = 0.9,
    is_encoded: bool = False,
) -> jnp.ndarray:
    """Two-step MIP-Q sampling with an explicit ensemble (single shared noise, multiple networks).

    Unlike sample_mip_q_values(), the same noise is passed to an ensemble of networks
    (ValueMIPEnsemble); diversity comes from the networks rather than the noise. Used when
    mip_q_ensemble=True.

    Returns Q-values from all ensemble members: [num_ensembles, batch_size].
    """
    batch_size = observations.shape[0]

    t_0 = jnp.zeros((batch_size, 1))
    q_0 = critic_fn(
        observations,
        actions=actions,
        scalar_input=noise_sample,  # same noise for all ensemble members
        time=t_0,
        is_encoded=is_encoded
    )  # [num_ensembles, batch_size]

    t_star = jnp.full((batch_size, 1), mip_t_star)

    # Transpose [num_ensembles, batch] -> [batch, num_ensembles, 1] to feed each
    # ensemble member its own Step-1 prediction.
    q_0_expanded = q_0.T[..., None]  # [batch_size, num_ensembles, 1]

    q_final = critic_fn(
        observations,
        actions=actions,
        scalar_input=q_0_expanded,
        time=t_star,
        is_encoded=is_encoded
    )  # [num_ensembles, batch_size]

    return q_final


def compute_mip_q_predictions(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    target_q: jnp.ndarray,
    rng: jnp.ndarray,
    noise_scale: float = 1.0,
    mip_t_star: float = 0.9,
    num_ensemble_members: int = 10,
    is_encoded: bool = False,
    return_logits: bool = False,
) -> Tuple:
    """Compute MIP-Q training predictions (regression at t=0 and denoising at t=t*).

    vmaps over multiple noise samples for a single network. Returns
    (q_0, q_final) or, with return_logits, additionally (q_0_logits, q_final_logits).
    """
    batch_size = observations.shape[0]

    rng_keys = jax.random.split(rng, num_ensemble_members)

    def compute_q_for_noise(rng_i):
        z_0, z_t_star, t_0_arr, t_star_arr = get_mip_q_targets(
            target_q=target_q,
            rng=rng_i,
            noise_scale=noise_scale,
            t_star=mip_t_star
        )

        if return_logits:
            q_0, q_0_logits = critic_fn(
                observations,
                actions=actions,
                scalar_input=z_0,
                time=t_0_arr,
                is_encoded=is_encoded,
                return_logits=True
            )
        else:
            q_0 = critic_fn(
                observations,
                actions=actions,
                scalar_input=z_0,
                time=t_0_arr,
                is_encoded=is_encoded,
                return_logits=False
            )
            q_0_logits = None

        if return_logits:
            q_final, q_final_logits = critic_fn(
                observations,
                actions=actions,
                scalar_input=z_t_star,
                time=t_star_arr,
                is_encoded=is_encoded,
                return_logits=True
            )
        else:
            q_final = critic_fn(
                observations,
                actions=actions,
                scalar_input=z_t_star,
                time=t_star_arr,
                is_encoded=is_encoded,
                return_logits=False
            )
            q_final_logits = None

        if return_logits:
            return q_0, q_final, q_0_logits, q_final_logits
        else:
            return q_0, q_final

    if return_logits:
        outputs = jax.vmap(compute_q_for_noise)(rng_keys)
        q_0 = outputs[0]  # [num_ensemble_members, batch_size]
        q_final = outputs[1]  # [num_ensemble_members, batch_size]
        q_0_logits = outputs[2]  # [num_ensemble_members, batch_size, num_bins]
        q_final_logits = outputs[3]  # [num_ensemble_members, batch_size, num_bins]
        return q_0, q_final, q_0_logits, q_final_logits
    else:
        outputs = jax.vmap(compute_q_for_noise)(rng_keys)
        q_0 = outputs[0]  # [num_ensemble_members, batch_size]
        q_final = outputs[1]  # [num_ensemble_members, batch_size]
        return q_0, q_final


def compute_mip_q_ensemble_predictions(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    target_q: jnp.ndarray,
    rng: jnp.ndarray,
    noise_scale: float = 1.0,
    mip_t_star: float = 0.9,
    is_encoded: bool = False,
    return_logits: bool = False,
) -> Tuple:
    """Compute MIP-Q training predictions with an explicit ensemble.

    Unlike compute_mip_q_predictions(), samples one noise and passes it to an
    ensemble of networks (ValueMIPEnsemble). Returns (q_0, q_final) or, with
    return_logits, additionally (q_0_logits, q_final_logits).
    """
    z_0, z_t_star, t_0_arr, t_star_arr = get_mip_q_targets(
        target_q=target_q,
        rng=rng,
        noise_scale=noise_scale,
        t_star=mip_t_star
    )

    if return_logits:
        q_0, q_0_logits = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_0,
            time=t_0_arr,
            is_encoded=is_encoded,
            return_logits=True
        )  # [num_ensembles, batch_size], [num_ensembles, batch_size, num_bins]
    else:
        q_0 = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_0,
            time=t_0_arr,
            is_encoded=is_encoded,
            return_logits=False
        )  # [num_ensembles, batch_size]
        q_0_logits = None

    if return_logits:
        q_final, q_final_logits = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_t_star,
            time=t_star_arr,
            is_encoded=is_encoded,
            return_logits=True
        )  # [num_ensembles, batch_size], [num_ensembles, batch_size, num_bins]
    else:
        q_final = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_t_star,
            time=t_star_arr,
            is_encoded=is_encoded,
            return_logits=False
        )  # [num_ensembles, batch_size]
        q_final_logits = None

    if return_logits:
        return q_0, q_final, q_0_logits, q_final_logits
    else:
        return q_0, q_final


def compute_td_target(
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    next_q: jnp.ndarray,
    discount: float,
    horizon_length: int = 1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> jnp.ndarray:
    """Compute TD target: r + γ^H * mask * Q(s', a'). 2D rewards/masks use the last timestep."""
    if rewards.ndim > 1:
        rewards = rewards[..., -1]
    if masks.ndim > 1:
        masks = masks[..., -1]

    discount_factor = discount ** horizon_length
    target = rewards + discount_factor * masks * next_q

    if clip_min is not None or clip_max is not None:
        target = jnp.clip(target,
                          a_min=clip_min if clip_min is not None else -jnp.inf,
                          a_max=clip_max if clip_max is not None else jnp.inf)
    return target


def compute_td_loss(
    q_pred: jnp.ndarray,
    target_q: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
    loss_type: str = "mse",
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
    num_bins: int = 256,
    q_logits: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Compute TD loss (MSE or HLGauss cross-entropy)."""
    if valid_mask is None:
        valid_mask = jnp.ones(target_q.shape[0])

    if loss_type == "hlgauss":
        if q_logits is None or q_min is None or q_max is None:
            raise ValueError("q_logits, q_min, q_max required for hlgauss loss")

        critic_target_probs = hl_gauss_target(
            q_min=q_min,
            q_max=q_max,
            num_bins=num_bins,
            critic_value=target_q,
        )
        td_loss = optax.softmax_cross_entropy(q_logits, critic_target_probs)
        td_loss = jnp.mean(td_loss * valid_mask)

        stats = {
            'q_logit_mean': q_logits.mean(),
            'q_logit_max': q_logits.max(),
            'q_logit_min': q_logits.min(),
            'critic_target_probs': critic_target_probs.mean(),
            'critic_target_probs_max': critic_target_probs.max(),
            'critic_target_probs_min': critic_target_probs.min()
        }
    else:
        td_loss = (jnp.square(q_pred - target_q) * valid_mask).mean()
        stats = {}

    return td_loss, stats


def compute_mc_regression_loss(
    q_pred: jnp.ndarray,
    mc_returns: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """MSE regression loss between Q-predictions (num_qs, batch) and MC returns (batch,)."""
    mc_loss_per_q = jnp.square(q_pred - mc_returns[None])  # (num_qs, batch_size)
    mc_loss = mc_loss_per_q.mean(axis=0)  # (batch_size,)
    if valid_mask is not None:
        return (mc_loss * valid_mask).mean()
    return mc_loss.mean()


def compute_calql_regularizer(
    critic_fn: Callable,
    critic_params: Any,
    observations: jnp.ndarray,
    batch_actions: jnp.ndarray,
    mc_returns: jnp.ndarray,
    sample_fn: Callable,
    next_observations: jnp.ndarray,
    cql_n_actions: int,
    cql_temp: float,
    num_qs: int,
    calql_q_subsample: int,
    full_act_dim: int,
    rng: jnp.ndarray,
    obs_encoded: Optional[jnp.ndarray] = None,
    is_encoded: bool = False,
) -> jnp.ndarray:
    """Compute CalQL regularizer: lower-bounds OOD Q-values by MC returns to curb overestimation."""
    B = observations.shape[0]

    obs_input = obs_encoded if is_encoded else observations
    q_pred = critic_fn(obs_input, actions=batch_actions, params=critic_params, is_encoded=is_encoded)

    rng, action_rng = jax.random.split(rng)
    cql_random_actions = jax.random.uniform(
        action_rng,
        shape=(B, cql_n_actions, full_act_dim),
        minval=-1.0,
        maxval=1.0,
    )

    rng, current_a_rng = jax.random.split(rng)
    sample_rngs = jax.random.split(current_a_rng, cql_n_actions)
    vmapped_sample = jax.vmap(
        lambda obs, rng_i: sample_fn(obs, rng=rng_i),
        in_axes=(None, 0)
    )

    cql_current_actions = vmapped_sample(observations, sample_rngs)
    cql_current_actions = jnp.transpose(cql_current_actions, (1, 0, 2))
    cql_next_actions = vmapped_sample(next_observations, sample_rngs)
    cql_next_actions = jnp.transpose(cql_next_actions, (1, 0, 2))

    all_actions = jnp.concatenate([cql_random_actions, cql_current_actions, cql_next_actions], axis=1)

    vmapped_q_fn = jax.vmap(
        lambda a: critic_fn(obs_input, actions=a, params=critic_params, is_encoded=is_encoded),
        in_axes=1, out_axes=-1
    )
    cql_qs = vmapped_q_fn(all_actions)
    chex.assert_shape(cql_qs, (num_qs, B, cql_n_actions * 3))

    rng, subsample_key = jax.random.split(rng)
    subsample_idcs = jax.random.randint(subsample_key, (calql_q_subsample,), 0, num_qs)
    cql_qs = cql_qs[subsample_idcs]
    q_pred = q_pred[subsample_idcs]

    # CalQL lower bound: clamp OOD Q-values below by MC returns.
    n_actions_for_calql = cql_n_actions * 3
    mc_lower_bound = jnp.repeat(mc_returns.reshape(-1, 1), n_actions_for_calql, axis=1)
    cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
    cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)], axis=-1)
    cql_qs -= jnp.log(cql_qs.shape[-1]) * cql_temp

    cql_ood_values = jax.scipy.special.logsumexp(cql_qs / cql_temp, axis=-1) * cql_temp
    calql_regularizer = (cql_ood_values - q_pred).mean()

    return calql_regularizer


def compute_best_of_n_next_q(
    critic_fn: Callable,
    next_observations: jnp.ndarray,
    sample_fn: Callable,
    num_samples: int,
    q_agg: str,
    rng: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute next Q-value via Best-of-N: sample N actions, score all, pick the argmax Q.

    Returns (best_q_values [batch], best_actions [batch, action_dim]).
    """
    sample_keys = jax.random.split(rng, num_samples)
    next_actions_all = jnp.stack(
        [sample_fn.sample(seed=key) for key in sample_keys],
        axis=0
    )  # (num_samples, batch_size, action_dim)

    batch_obs = jnp.repeat(next_observations[None, ...], num_samples, axis=0)
    next_qs_all = critic_fn(batch_obs, actions=next_actions_all)  # (num_qs, num_samples, batch_size)

    if q_agg == "min":
        next_qs_all = next_qs_all.min(axis=0)  # (num_samples, batch_size)
    else:
        next_qs_all = next_qs_all.mean(axis=0)  # (num_samples, batch_size)

    best_idx = jnp.argmax(next_qs_all, axis=0)  # (batch_size,)
    best_q = jnp.take_along_axis(next_qs_all, best_idx[None, :], axis=0).squeeze(0)
    best_actions = jnp.take_along_axis(
        next_actions_all,
        best_idx[None, :, None],
        axis=0
    ).squeeze(0)

    return best_q, best_actions


def soft_target_update(
    params: Any,
    target_params: Any,
    tau: float,
) -> Any:
    """Polyak target update: target = tau * params + (1 - tau) * target."""
    return jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau),
        params,
        target_params,
    )


@partial(jax.jit, static_argnames=('critic_fn', 'is_encoded', 'q_agg'))
def get_q_values(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    q_agg: str = "mean",
    is_encoded: bool = False,
) -> jnp.ndarray:
    """Compute aggregated Q-values for evaluation/logging. "subsample" falls back to "min" (no rng)."""
    actions = jnp.reshape(actions, (actions.shape[0], -1))

    qs = critic_fn(observations, actions=actions, is_encoded=is_encoded)

    if q_agg == "min" or q_agg == "subsample":
        return qs.min(axis=0)
    else:
        return qs.mean(axis=0)


def compute_critic_loss_standard(
    critic_fn: Callable,
    target_critic_fn: Callable,
    critic_params: Any,
    batch: Dict[str, jnp.ndarray],
    next_actions: jnp.ndarray,
    config: Dict[str, Any],
    rng: jnp.ndarray,
    obs_encoded: Optional[jnp.ndarray] = None,
    next_obs_encoded: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """Compute standard TD loss (next-Q aggregation + TD target + MSE/HLGauss) used by OGPO, BPTT, QC.

    Returns (td_loss, q_pred, next_q, stats).
    """
    is_encoded = obs_encoded is not None

    if is_encoded:
        next_qs = target_critic_fn(next_obs_encoded, actions=next_actions, is_encoded=True)
    else:
        next_qs = target_critic_fn(batch['next_observations'], actions=next_actions)

    rng, agg_rng = jax.random.split(rng)
    next_q = aggregate_q_values(
        next_qs,
        method=config['q_agg'],
        rng=agg_rng,
        num_qs=config['num_qs'],
    )

    target_q = compute_td_target(
        rewards=batch['rewards'],
        masks=batch['masks'],
        next_q=next_q,
        discount=config['discount'],
        horizon_length=config['horizon_length'],
    )

    if config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]

    if config["critic_loss_type"] == "hlgauss":
        if is_encoded:
            q_pred, q_logits = critic_fn(
                obs_encoded, actions=batch_actions, params=critic_params,
                return_logits=True, is_encoded=True
            )
        else:
            q_pred, q_logits = critic_fn(
                batch['observations'], actions=batch_actions, params=critic_params,
                return_logits=True
            )

        valid = batch.get('valid')
        if valid is not None and valid.ndim > 1:
            valid = valid[..., -1]

        td_loss, stats = compute_td_loss(
            q_pred=q_pred,
            target_q=target_q,
            valid_mask=valid,
            loss_type="hlgauss",
            q_min=config['q_min'],
            q_max=config['q_max'],
            num_bins=config['num_bins'],
            q_logits=q_logits,
        )
    else:
        if is_encoded:
            q_pred = critic_fn(obs_encoded, actions=batch_actions, params=critic_params, is_encoded=True)
        else:
            q_pred = critic_fn(batch['observations'], actions=batch_actions, params=critic_params)

        valid = batch.get('valid')
        if valid is not None and valid.ndim > 1:
            valid = valid[..., -1]

        td_loss, stats = compute_td_loss(
            q_pred=q_pred,
            target_q=target_q,
            valid_mask=valid,
            loss_type="mse",
        )

    return td_loss, q_pred, next_q, stats
