"""BC loss functions for flow matching policies."""

from typing import Tuple, Dict, Callable, Optional, Any

import jax
import jax.numpy as jnp


def get_flow_targets(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rng: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate standard flow matching targets.

    Samples x_0 ~ N(0, I) and t ~ U(0, 1), then computes:
        x_t = (1 - t) * x_0 + t * x_1
        v_target = x_1 - x_0

    Args:
        observations: Batch of observations (unused, kept for API consistency)
        actions: Batch of target actions (x_1)
        rng: Random key

    Returns:
        x_t: Interpolated states at time t
        v_target: Target velocities
        t: Time values (batch_size, 1)
    """
    batch_size = actions.shape[0]
    action_dim = actions.shape[-1]

    rng, x0_rng, t_rng = jax.random.split(rng, 3)

    x_0 = jax.random.normal(x0_rng, (batch_size, action_dim))
    x_1 = actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    v_target = x_1 - x_0

    return x_t, v_target, t


def get_mip_targets(
    actions: jnp.ndarray,
    rng: jnp.ndarray,
    t_star: float = 0.999,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate MIP (Masked Interpolant) targets for two-term loss.

    MIP mode uses:
        Term 1: Regression at t=0 (predict x_1 from x_0)
        Term 2: Denoising at t=t* (predict x_1 from x_t*)

    Args:
        actions: Batch of target actions (x_1)
        rng: Random key
        t_star: Denoising time point (default 0.999)

    Returns:
        x_0: Initial noise samples
        x_t_star: Interpolated states at t*
        t_0_arr: Zero time array (batch_size, 1)
        t_star_arr: t* time array (batch_size, 1)
    """
    batch_size = actions.shape[0]

    rng, x0_rng = jax.random.split(rng)
    x_0 = jax.random.normal(x0_rng, actions.shape)

    t_0_arr = jnp.zeros((batch_size, 1))
    t_star_arr = jnp.full((batch_size, 1), t_star)
    x_t_star = (1 - t_star) * x_0 + actions

    return x_0, x_t_star, t_0_arr, t_star_arr


def preprocess_actions(
    batch: Dict[str, jnp.ndarray],
    action_chunking: bool,
    action_key: str = "actions",
) -> jnp.ndarray:
    """Preprocess actions from batch for BC loss computation.

    Args:
        batch: Batch dictionary containing actions
        action_chunking: Whether action chunking is enabled
        action_key: Key to use for actions (default "actions", can be "true_actions")

    Returns:
        Flattened actions of shape (batch_size, action_dim) or (batch_size, horizon * action_dim)
    """
    actions = batch[action_key]
    if action_chunking:
        # Fold horizon_length together with action_dim: (B, H, A) -> (B, H*A)
        return jnp.reshape(actions, (actions.shape[0], -1))
    else:
        # Take the first timestep: (B, H, A) -> (B, A)
        return actions[..., 0, :]


def apply_chunking_mask(
    loss_per_element: jnp.ndarray,
    valid_mask: jnp.ndarray,
    batch_size: int,
    horizon_length: int,
    action_dim: int,
    action_chunking: bool,
) -> jnp.ndarray:
    """Apply valid mask for action chunking or compute mean directly"""
    if action_chunking:
        # Reshape to (B, H, A) and apply valid mask
        loss_reshaped = jnp.reshape(loss_per_element, (batch_size, horizon_length, action_dim))
        return jnp.mean(loss_reshaped * valid_mask[..., None])
    else:
        return jnp.mean(loss_per_element)


def compute_flow_bc_loss(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    action_chunking: bool,
    horizon_length: int,
    action_dim: int,
    action_key: str = "actions",
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute standard flow matching BC loss.

    This is the core BC loss shared by all algorithms:
        loss = E_t,x_0 [ || v_theta(x_t, t) - (x_1 - x_0) ||^2 ]
    """
    batch_actions = preprocess_actions(batch, action_chunking, action_key)
    batch_size = batch_actions.shape[0]

    rng, targets_rng = jax.random.split(rng)
    x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

    pred = model_fn(batch['observations'], x_t, t)

    loss_per_element = jnp.square(pred - vel)
    bc_loss = apply_chunking_mask(
        loss_per_element,
        batch.get("valid", jnp.ones((batch_size, horizon_length))),
        batch_size, horizon_length, action_dim, action_chunking
    )

    return bc_loss, {'bc_flow_loss': bc_loss}


def compute_mip_bc_loss(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    action_chunking: bool,
    horizon_length: int,
    action_dim: int,
    t_star: float = 0.999,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute MIP (Masked Interpolant) BC loss.

    MIP loss consists of two terms:
        loss_regression: || f_theta(x_0, t=0) - x_1 ||^2
        loss_denoising: || f_theta(x_t*, t*) - x_1 ||^2

    Args:
        batch: Batch dictionary with 'observations', 'actions', and optionally 'valid'
        rng: Random key
        model_fn: Function that takes (obs, x, t) and returns x_1 prediction
        action_chunking: Whether action chunking is enabled
        horizon_length: Number of timesteps in chunk
        action_dim: Action dimension per timestep
        t_star: Denoising time point (default 0.999)

    Returns:
        bc_loss: Combined scalar BC loss
        info: Dictionary with loss components
    """
    batch_actions = preprocess_actions(batch, action_chunking)
    batch_size = batch_actions.shape[0]

    x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(batch_actions, rng, t_star)

    # Term 1: Regression at t=0
    pred_0 = model_fn(batch['observations'], x_0, t_0_arr)
    loss_regression_per_element = jnp.square(pred_0 - batch_actions)

    # Term 2: Denoising at t=t*
    pred_t_star = model_fn(batch['observations'], x_t_star, t_star_arr)
    loss_denoising_per_element = jnp.square(pred_t_star - batch_actions)

    # Combined loss
    combined_loss_per_element = loss_regression_per_element + loss_denoising_per_element

    valid_mask = batch.get("valid", jnp.ones((batch_size, horizon_length)))

    bc_loss = apply_chunking_mask(
        combined_loss_per_element, valid_mask,
        batch_size, horizon_length, action_dim, action_chunking
    )
    loss_regression = apply_chunking_mask(
        loss_regression_per_element, valid_mask,
        batch_size, horizon_length, action_dim, action_chunking
    )
    loss_denoising = apply_chunking_mask(
        loss_denoising_per_element, valid_mask,
        batch_size, horizon_length, action_dim, action_chunking
    )

    info = {
        'bc_loss': bc_loss,
        'bc_mip_loss_regression': loss_regression,
        'bc_mip_loss_denoising': loss_denoising,
    }
    return bc_loss, info


def compute_flow_bc_loss_with_denoiser(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    action_chunking: bool,
    horizon_length: int,
    action_dim: int,
    noise_std: float = 0.01,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute flow matching BC loss with denoiser component (OGPO-specific).

    This adds stochastic interpolant training for SDE-to-ODE correction:
        1. Sample flow targets: x_t = (1-t)*x_0 + t*x_1
        2. Inject noise: x_t_noisy = x_t + sigma * z, where z ~ N(0, I)
        3. Predict velocity AND noise: (v_pred, z_pred) = f_theta(x_t_noisy, t)
        4. Loss = velocity_loss + denoiser_loss

    Args:
        batch: Batch dictionary with 'observations', 'actions', and optionally 'valid'
        rng: Random key
        model_fn: Function that takes (obs, x_t, t, return_denoiser=True)
                  and returns (velocity_pred, noise_pred)
        action_chunking: Whether action chunking is enabled
        horizon_length: Number of timesteps in chunk
        action_dim: Action dimension per timestep
        noise_std: Standard deviation of injected noise (sigma)

    Returns:
        bc_loss: Combined scalar BC loss (velocity + denoiser)
        info: Dictionary with loss components
    """
    batch_actions = preprocess_actions(batch, action_chunking)
    batch_size = batch_actions.shape[0]

    rng, targets_rng, z_rng = jax.random.split(rng, 3)
    x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

    # Sample noise for stochastic interpolant
    z = jax.random.normal(z_rng, x_t.shape)

    # Inject noise into flow state
    x_t_noisy = x_t + noise_std * z

    # Forward pass requesting denoiser output
    pred_vel, pred_z = model_fn(batch['observations'], x_t_noisy, t, return_denoiser=True)

    # Velocity loss
    loss_vel_per_element = jnp.square(pred_vel - vel)

    # Denoiser loss (predict injected noise)
    loss_denoiser_per_element = jnp.square(pred_z - z)

    valid_mask = batch.get("valid", jnp.ones((batch_size, horizon_length)))

    loss_vel = apply_chunking_mask(
        loss_vel_per_element, valid_mask,
        batch_size, horizon_length, action_dim, action_chunking
    )
    loss_denoiser = apply_chunking_mask(
        loss_denoiser_per_element, valid_mask,
        batch_size, horizon_length, action_dim, action_chunking
    )

    bc_loss = loss_vel + loss_denoiser

    info = {
        'bc_loss': bc_loss,
        'bc_flow_loss_vel': loss_vel,
        'bc_flow_loss_denoising': loss_denoiser,
    }
    return bc_loss, info


def compute_flow_bc_loss_online(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    action_chunking: bool,
    horizon_length: int,
    action_dim: int,
    bc_coeff: float = 1.0,
    action_key: str = "true_actions",
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute flow matching BC loss for online RL success buffer.

    Similar to standard flow BC but:
    - Uses bc_coeff weighting for regularization
    - Treats all timesteps as valid (no masking needed)
    - Uses 'true_actions' by default (actual executed actions, not noise)

    Args:
        batch: Batch dictionary with observations and actions
        rng: Random key
        model_fn: Function that takes (obs, x_t, t) and returns velocity prediction
        action_chunking: Whether action chunking is enabled
        horizon_length: Number of timesteps in chunk
        action_dim: Action dimension per timestep
        bc_coeff: Weighting coefficient for regularization
        action_key: Key for actions (default "true_actions" for online)

    Returns:
        bc_loss: Weighted scalar BC loss
        info: Dictionary with loss statistics
    """
    batch_actions = preprocess_actions(batch, action_chunking, action_key)
    batch_size = batch_actions.shape[0]

    rng, targets_rng = jax.random.split(rng)
    x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

    pred = model_fn(batch['observations'], x_t, t)

    loss_per_element = jnp.square(pred - vel)

    # For online data, all timesteps are valid
    if action_chunking:
        loss_reshaped = jnp.reshape(loss_per_element, (batch_size, horizon_length, action_dim))
        bc_loss_unweighted = jnp.mean(loss_reshaped)
    else:
        bc_loss_unweighted = jnp.mean(loss_per_element)

    bc_loss = bc_coeff * bc_loss_unweighted

    return bc_loss, {
        'bc_flow_loss': bc_loss,
        'bc_flow_loss_unweighted': bc_loss_unweighted,
    }


def compute_bc_loss(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    config: Any,
    model_fn_denoiser: Optional[Callable] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Unified BC loss dispatcher that routes to appropriate loss function.

    This is the main entry point for BC loss computation. Routes based on config:
    - policy_type == 'mip': Use MIP loss
    - use_denoiser == True: Use flow + denoiser loss (OGPO-specific)
    - Otherwise: Standard flow matching loss

    Args:
        batch: Batch dictionary
        rng: Random key
        model_fn: Standard model forward function (obs, x_t, t) -> pred
        config: Configuration object with:
            - action_chunking: bool
            - horizon_length: int
            - action_dim: int
            - policy_type: str (optional, 'mip' or 'flow')
            - use_denoiser: bool (optional)
            - mip_t_star: float (optional)
            - constant_noise_std: float (optional)
        model_fn_denoiser: Optional function for denoiser mode
                           (obs, x_t, t, return_denoiser=True) -> (vel, z)

    Returns:
        bc_loss: Scalar BC loss
        info: Dictionary with loss statistics
    """
    action_chunking = config["action_chunking"]
    horizon_length = config["horizon_length"]
    action_dim = config["action_dim"]
    policy_type = config["policy_type"]
    use_denoiser = config["use_denoiser"]

    if policy_type == 'mip':
        t_star = config["mip_t_star"]
        return compute_mip_bc_loss(
            batch, rng, model_fn,
            action_chunking, horizon_length, action_dim,
            t_star=t_star
        )
    elif use_denoiser and model_fn_denoiser is not None:
        noise_std = config["constant_noise_std"]
        return compute_flow_bc_loss_with_denoiser(
            batch, rng, model_fn_denoiser,
            action_chunking, horizon_length, action_dim,
            noise_std=noise_std
        )
    else:
        return compute_flow_bc_loss(
            batch, rng, model_fn,
            action_chunking, horizon_length, action_dim
        )
