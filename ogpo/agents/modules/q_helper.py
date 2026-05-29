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
    """Aggregate Q-values from critic ensemble.

    Args:
        q_values: Q-values from ensemble of shape (num_qs, batch_size) or
                  (num_qs, batch_size, ...).
        method: Aggregation method - "mean", "min", or "subsample".
        rng: Random key for subsampling. Required if method="subsample".
        num_qs: Total number of Q-functions in ensemble. Required if method="subsample".
        subsample_size: Number of Q-functions to subsample when method="subsample".

    Returns:
        Aggregated Q-values of shape (batch_size,) or (batch_size, ...).
    """
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
    """Generate MIP-Q targets for two-term loss (analogous to MIP actor).

    MIP-Q uses:
        Term 1: Regression at t=0 (predict Q_target from z_0)
        Term 2: Denoising at t=t* (predict Q_target from z_t*)

    Args:
        target_q: Target Q-values [batch_size]
        rng: Random key
        noise_scale: Scale for uniform noise sampling (samples from [-scale, +scale])
        t_star: Denoising time point (default 0.9)

    Returns:
        z_0: Initial noise samples [batch_size, 1]
        z_t_star: Interpolated scalars at t* [batch_size, 1]
        t_0_arr: Zero time array [batch_size, 1]
        t_star_arr: t* time array [batch_size, 1]
    """
    batch_size = target_q.shape[0]

    # Sample uniform noise: Uniform[-scale, +scale]
    rng, noise_rng = jax.random.split(rng)
    z_0 = jax.random.uniform(
        noise_rng, (batch_size, 1),
        minval=-noise_scale,
        maxval=noise_scale
    )

    # Interpolate in Q-space (analogous to action space interpolation in MIP actor)
    # z_t* = (1 - t*) * z_0 + Q_target (add scaled noise to target)
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

    This function encapsulates the standard MIP-Q sampling procedure used during:
    - TD target computation
    - Advantage computation
    - Best-of-N action selection

    The two-step process:
    1. Step 1 (t=0): Get initial Q estimate from noise
    2. Step 2 (t=t*): Refine estimate autoregressively using Step 1 output

    NOTE: This is DIFFERENT from training, where Step 2 uses interpolated targets.

    Args:
        critic_fn: Critic function. Should accept (obs, actions, scalar_input, time, is_encoded).
        observations: Observations [batch_size, obs_dim]
        actions: Actions [batch_size, act_dim]
        noise_samples: Noise samples [num_ensemble_members, batch_size, 1]
        mip_t_star: Time for second step (typically 0.9)
        is_encoded: Whether observations are already encoded

    Returns:
        Q-values from all noise samples: [num_ensemble_members, batch_size]
    """
    batch_size = observations.shape[0]

    def eval_q_two_step(noise_i):
        # Step 1: Initial estimate from noise at t=0
        t_0 = jnp.zeros((batch_size, 1))
        q_0 = critic_fn(
            observations,
            actions=actions,
            scalar_input=noise_i,
            time=t_0,
            is_encoded=is_encoded
        )

        # Step 2: Autoregressive refinement at t=t*
        # Uses Step 1 prediction (NOT interpolated target like in training)
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

    # Evaluate for all noise samples
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
    """Two-step MIP-Q sampling with EXPLICIT ensemble (single noise, multiple networks).

    Unlike sample_mip_q_values() which vmaps over multiple noise samples for a single network,
    this function passes the SAME noise to an ensemble of networks (ValueMIPEnsemble).
    The ensemble dimension comes from the critic_fn itself (vmapped networks).

    This is used when mip_q_ensemble=True, creating diversity through multiple networks
    instead of multiple noise samples.

    The two-step process:
    1. Step 1 (t=0): Get initial Q estimates from shared noise across all ensemble members
    2. Step 2 (t=t*): Refine estimates autoregressively using Step 1 outputs

    Args:
        critic_fn: Ensemble critic function. Should be a ValueMIPEnsemble that returns [num_ensembles, batch].
        observations: Observations [batch_size, obs_dim]
        actions: Actions [batch_size, act_dim]
        noise_sample: SINGLE noise sample [batch_size, 1] shared across all ensemble members
        mip_t_star: Time for second step (typically 0.9)
        is_encoded: Whether observations are already encoded

    Returns:
        Q-values from all ensemble members: [num_ensembles, batch_size]
    """
    batch_size = observations.shape[0]

    # Step 1: Initial estimate from shared noise at t=0
    # All ensemble members receive the SAME noise
    t_0 = jnp.zeros((batch_size, 1))
    q_0 = critic_fn(
        observations,
        actions=actions,
        scalar_input=noise_sample,  # Same noise for all ensemble members
        time=t_0,
        is_encoded=is_encoded
    )  # [num_ensembles, batch_size]

    # Step 2: Autoregressive refinement at t=t*
    # Each ensemble member uses its own Step 1 prediction
    t_star = jnp.full((batch_size, 1), mip_t_star)

    # q_0 is [num_ensembles, batch_size], need to pass each ensemble's q_0 back to itself
    # Transpose to [batch_size, num_ensembles] and expand to [batch_size, num_ensembles, 1]
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
    """Compute MIP-Q predictions for training (both t=0 and t=t* steps).

    This function encapsulates the MIP-Q training procedure:
    1. Sample noise and compute interpolated targets using get_mip_q_targets
    2. Compute Q predictions at t=0 (regression from noise)
    3. Compute Q predictions at t=t* (denoising from interpolated target)

    Args:
        critic_fn: Critic function with signature (obs, actions, scalar_input, time, params, is_encoded, return_logits)
        observations: Observations [batch_size, obs_dim]
        actions: Actions [batch_size, act_dim]
        target_q: TD target Q-values [batch_size]
        rng: Random key for noise sampling
        noise_scale: Scale for uniform noise sampling
        mip_t_star: Time for second step (typically 0.9)
        num_ensemble_members: Number of noise samples (implicit ensemble size)
        is_encoded: Whether observations are already encoded
        return_logits: Whether to return logits (for HL-Gauss loss)

    Returns:
        If return_logits=False:
            q_0: Q predictions at t=0 [num_ensemble_members, batch_size]
            q_final: Q predictions at t=t* [num_ensemble_members, batch_size]
        If return_logits=True:
            q_0: Q predictions at t=0 [num_ensemble_members, batch_size]
            q_final: Q predictions at t=t* [num_ensemble_members, batch_size]
            q_0_logits: Logits at t=0 [num_ensemble_members, batch_size, num_bins]
            q_final_logits: Logits at t=t* [num_ensemble_members, batch_size, num_bins]
    """
    batch_size = observations.shape[0]

    # Generate noise samples
    rng_keys = jax.random.split(rng, num_ensemble_members)

    def compute_q_for_noise(rng_i):
        # Use get_mip_q_targets to get noise and interpolated target
        z_0, z_t_star, t_0_arr, t_star_arr = get_mip_q_targets(
            target_q=target_q,
            rng=rng_i,
            noise_scale=noise_scale,
            t_star=mip_t_star
        )

        # Step 1: Regression at t=0 (predict from noise)
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

        # Step 2: Denoising at t=t* (predict from interpolated target)
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

    # Compute for all noise samples
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
    """Compute MIP-Q predictions for training with EXPLICIT ensemble.

    Unlike compute_mip_q_predictions() which vmaps over multiple noises for a single network,
    this function samples ONE noise and passes it to an ensemble of networks (ValueMIPEnsemble).

    The procedure:
    1. Sample noise ONCE and compute interpolated targets
    2. Compute Q predictions at t=0 across all ensemble members (regression from noise)
    3. Compute Q predictions at t=t* across all ensemble members (denoising from interpolated target)

    Args:
        critic_fn: Ensemble critic function (ValueMIPEnsemble) that returns [num_ensembles, batch]
        observations: Observations [batch_size, obs_dim]
        actions: Actions [batch_size, act_dim]
        target_q: TD target Q-values [batch_size]
        rng: Random key for noise sampling
        noise_scale: Scale for uniform noise sampling
        mip_t_star: Time for second step (typically 0.9)
        is_encoded: Whether observations are already encoded
        return_logits: Whether to return logits (for HL-Gauss loss)

    Returns:
        If return_logits=False:
            q_0: Q predictions at t=0 [num_ensembles, batch_size]
            q_final: Q predictions at t=t* [num_ensembles, batch_size]
        If return_logits=True:
            q_0: Q predictions at t=0 [num_ensembles, batch_size]
            q_final: Q predictions at t=t* [num_ensembles, batch_size]
            q_0_logits: Logits at t=0 [num_ensembles, batch_size, num_bins]
            q_final_logits: Logits at t=t* [num_ensembles, batch_size, num_bins]
    """
    # Sample noise ONCE and get interpolated targets
    z_0, z_t_star, t_0_arr, t_star_arr = get_mip_q_targets(
        target_q=target_q,
        rng=rng,
        noise_scale=noise_scale,
        t_star=mip_t_star
    )

    # Step 1: Regression at t=0 (predict from noise)
    # All ensemble members receive the SAME noise z_0
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

    # Step 2: Denoising at t=t* (predict from interpolated target)
    # All ensemble members receive the SAME interpolated target z_t_star
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
    """Compute TD target: r + γ^H * mask * Q(s', a').

    Args:
        rewards: Rewards of shape (batch_size,) or (batch_size, horizon_length).
                 If 2D, uses the last timestep reward.
        masks: Termination masks (1.0 if not terminal) of same shape as rewards.
               If 2D, uses the last timestep mask.
        next_q: Q-values for next state-action pairs, shape (batch_size,).
        discount: Discount factor γ.
        horizon_length: Horizon length H for action chunking (γ^H factor).
        clip_min: Optional lower bound for TD target (prevents Q-collapse).
        clip_max: Optional upper bound for TD target (prevents Q-explosion).

    Returns:
        TD targets of shape (batch_size,).
    """
    # Handle multi-dimensional rewards/masks from action chunking
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
        # MSE loss
        td_loss = (jnp.square(q_pred - target_q) * valid_mask).mean()
        stats = {}

    return td_loss, stats


def compute_mc_regression_loss(
    q_pred: jnp.ndarray,
    mc_returns: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """MSE regression loss between Q-predictions and Monte Carlo returns.

    Regresses the Q-ensemble towards MC returns for better calibration.

    Args:
        q_pred: Q-values from critic ensemble, shape (num_qs, batch_size).
        mc_returns: Monte Carlo returns, shape (batch_size,).
        valid_mask: Optional mask, shape (batch_size,). If None, all valid.

    Returns:
        Scalar MC regression loss (averaged over ensemble and batch).
    """
    # Broadcast mc_returns to match ensemble: (1, batch_size)
    mc_loss_per_q = jnp.square(q_pred - mc_returns[None])  # (num_qs, batch_size)
    mc_loss = mc_loss_per_q.mean(axis=0)  # (batch_size,) — average over ensemble
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
    """Compute CalQL regularization term.

    CalQL (Calibrated Q-Learning) regularizes Q-values to stay close to
    Monte Carlo return estimates, preventing overestimation.

    Args:
        critic_fn: Function to compute Q-values. Signature:
                   critic_fn(observations, actions, params, is_encoded) -> (num_qs, batch_size)
        critic_params: Parameters for the critic function.
        observations: Current observations, shape (batch_size, obs_dim).
        batch_actions: Actions from the dataset, shape (batch_size, action_dim).
        mc_returns: Monte Carlo returns for CalQL lower bound, shape (batch_size,).
        sample_fn: Function to sample actions from the policy.
                   Signature: sample_fn(observations, rng) -> (batch_size, action_dim)
        next_observations: Next observations, shape (batch_size, obs_dim).
        cql_n_actions: Number of random/policy actions to sample for CQL.
        cql_temp: CQL temperature for logsumexp.
        num_qs: Number of Q-functions in ensemble.
        calql_q_subsample: Number of Q-functions to subsample for CalQL.
        full_act_dim: Full action dimension (including chunking).
        rng: Random key.
        obs_encoded: Pre-encoded observations (optional).
        is_encoded: Whether observations are pre-encoded.

    Returns:
        CalQL regularization value (scalar).
    """
    B = observations.shape[0]

    # Get Q-values for batch actions
    obs_input = obs_encoded if is_encoded else observations
    q_pred = critic_fn(obs_input, actions=batch_actions, params=critic_params, is_encoded=is_encoded)

    # Sample random actions
    rng, action_rng = jax.random.split(rng)
    cql_random_actions = jax.random.uniform(
        action_rng,
        shape=(B, cql_n_actions, full_act_dim),
        minval=-1.0,
        maxval=1.0,
    )

    # Sample policy actions for current and next observations
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

    # Evaluate Q-values for all actions
    vmapped_q_fn = jax.vmap(
        lambda a: critic_fn(obs_input, actions=a, params=critic_params, is_encoded=is_encoded),
        in_axes=1, out_axes=-1
    )
    cql_qs = vmapped_q_fn(all_actions)
    chex.assert_shape(cql_qs, (num_qs, B, cql_n_actions * 3))

    # Subsample Q ensemble
    rng, subsample_key = jax.random.split(rng)
    subsample_idcs = jax.random.randint(subsample_key, (calql_q_subsample,), 0, num_qs)
    cql_qs = cql_qs[subsample_idcs]
    q_pred = q_pred[subsample_idcs]

    # Apply CalQL lower bound
    n_actions_for_calql = cql_n_actions * 3
    mc_lower_bound = jnp.repeat(mc_returns.reshape(-1, 1), n_actions_for_calql, axis=1)
    cql_qs = jnp.maximum(cql_qs, mc_lower_bound)
    cql_qs = jnp.concatenate([cql_qs, jnp.expand_dims(q_pred, -1)], axis=-1)
    cql_qs -= jnp.log(cql_qs.shape[-1]) * cql_temp

    # Compute logsumexp for CQL
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
    """Compute next Q-value using Best-of-N sampling.

    Samples N actions from the policy, evaluates Q-values for all,
    and selects the action with highest Q-value.

    Args:
        critic_fn: Function to compute Q-values. Signature:
                   critic_fn(observations, actions) -> (num_qs, num_samples, batch_size)
        next_observations: Next observations, shape (batch_size, obs_dim).
        sample_fn: Distribution object with sample(seed=key) method, or
                   a function that returns sampled actions when called.
        num_samples: Number of actions to sample (N in Best-of-N).
        q_agg: Q aggregation method ("min" or "mean").
        rng: Random key.

    Returns:
        Tuple of (best_q_values, best_actions):
        - best_q_values: shape (batch_size,)
        - best_actions: shape (batch_size, action_dim)
    """
    # Sample N actions
    sample_keys = jax.random.split(rng, num_samples)
    next_actions_all = jnp.stack(
        [sample_fn.sample(seed=key) for key in sample_keys],
        axis=0
    )  # (num_samples, batch_size, action_dim)

    # Evaluate Q-values for all samples
    batch_obs = jnp.repeat(next_observations[None, ...], num_samples, axis=0)
    next_qs_all = critic_fn(batch_obs, actions=next_actions_all)  # (num_qs, num_samples, batch_size)

    if q_agg == "min":
        next_qs_all = next_qs_all.min(axis=0)  # (num_samples, batch_size)
    else:
        next_qs_all = next_qs_all.mean(axis=0)  # (num_samples, batch_size)

    # Best-of-N: select action with highest Q-value
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
    """Soft update target network parameters using Polyak averaging.

    target_params = tau * params + (1 - tau) * target_params

    Args:
        params: Current network parameters.
        target_params: Target network parameters.
        tau: Update rate (typically 0.005).

    Returns:
        Updated target parameters.
    """
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
    """Compute Q-values for evaluation/logging.

    This is a convenience function for retrieving Q-values during evaluation.

    Args:
        critic_fn: Function to compute Q-values. Signature:
                   critic_fn(observations, actions, is_encoded) -> (num_qs, batch_size)
        observations: Observations, shape (batch_size, obs_dim).
        actions: Actions, shape (batch_size, action_dim) or (batch_size, horizon, action_dim).
        q_agg: Q aggregation method ("min", "mean", or "subsample").
               Note: "subsample" defaults to "min" behavior here since no rng.
        is_encoded: Whether observations are pre-encoded.

    Returns:
        Aggregated Q-values of shape (batch_size,).
    """
    # Flatten actions if needed (for action chunking)
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
    """Compute standard TD loss used by OGPO, BPTT, QC.

    This is a convenience function that combines:
    - Q aggregation for next state
    - TD target computation
    - TD loss computation (MSE or HLGauss)

    Args:
        critic_fn: Current critic function.
        target_critic_fn: Target critic function.
        critic_params: Gradient parameters for critic.
        batch: Batch dictionary with 'observations', 'next_observations',
               'rewards', 'masks', 'valid' keys.
        next_actions: Actions for next observations, shape (batch_size, action_dim).
        config: Config dictionary with 'q_agg', 'discount', 'horizon_length',
                'critic_loss_type', 'q_min', 'q_max', 'num_bins', 'num_qs' keys.
        rng: Random key (for subsample aggregation).
        obs_encoded: Pre-encoded current observations (optional).
        next_obs_encoded: Pre-encoded next observations (optional).

    Returns:
        Tuple of (td_loss, q_pred, next_q, stats).
    """
    is_encoded = obs_encoded is not None

    # Get next Q-values from target network
    if is_encoded:
        next_qs = target_critic_fn(next_obs_encoded, actions=next_actions, is_encoded=True)
    else:
        next_qs = target_critic_fn(batch['next_observations'], actions=next_actions)

    # Aggregate next Q-values
    rng, agg_rng = jax.random.split(rng)
    next_q = aggregate_q_values(
        next_qs,
        method=config['q_agg'],
        rng=agg_rng,
        num_qs=config['num_qs'],
    )

    # Compute TD target
    target_q = compute_td_target(
        rewards=batch['rewards'],
        masks=batch['masks'],
        next_q=next_q,
        discount=config['discount'],
        horizon_length=config['horizon_length'],
    )

    # Prepare actions for current state
    if config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]

    # Get current Q-values
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
