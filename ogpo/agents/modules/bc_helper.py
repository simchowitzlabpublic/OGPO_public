"""BC loss functions for flow matching policies."""

from typing import Tuple, Dict, Callable, Optional, Any

import jax
import jax.numpy as jnp


def get_flow_targets(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    rng: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Standard flow matching targets: x_t = (1-t)*x_0 + t*x_1, v_target = x_1 - x_0.

    Returns (x_t, v_target, t) with x_0 ~ N(0,I), t ~ U(0,1). observations unused (API parity).
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
    """MIP targets for the two-term loss (regression at t=0, denoising at t=t*).

    Returns (x_0, x_t_star, t_0_arr, t_star_arr).
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
    """Flatten batch actions for BC loss: (B,H,A)->(B,H*A) if chunking else first timestep (B,A)."""
    actions = batch[action_key]
    if action_chunking:
        return jnp.reshape(actions, (actions.shape[0], -1))
    else:
        return actions[..., 0, :]


def apply_chunking_mask(
    loss_per_element: jnp.ndarray,
    valid_mask: jnp.ndarray,
    batch_size: int,
    horizon_length: int,
    action_dim: int,
    action_chunking: bool,
) -> jnp.ndarray:
    """Apply the valid mask under action chunking, else take the plain mean."""
    if action_chunking:
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
    """Standard flow matching BC loss: E_{t,x_0} || v_theta(x_t, t) - (x_1 - x_0) ||^2."""
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
    """MIP BC loss: ||f(x_0, 0) - x_1||^2 (regression) + ||f(x_t*, t*) - x_1||^2 (denoising)."""
    batch_actions = preprocess_actions(batch, action_chunking)
    batch_size = batch_actions.shape[0]

    x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(batch_actions, rng, t_star)

    pred_0 = model_fn(batch['observations'], x_0, t_0_arr)
    loss_regression_per_element = jnp.square(pred_0 - batch_actions)

    pred_t_star = model_fn(batch['observations'], x_t_star, t_star_arr)
    loss_denoising_per_element = jnp.square(pred_t_star - batch_actions)

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
    """Flow matching BC loss with denoiser term (OGPO-specific) for SDE-to-ODE correction.

    Adds stochastic-interpolant noise z ~ N(0,I) to x_t and trains the model to predict
    both velocity and z; loss = velocity_loss + denoiser_loss.
    """
    batch_actions = preprocess_actions(batch, action_chunking)
    batch_size = batch_actions.shape[0]

    rng, targets_rng, z_rng = jax.random.split(rng, 3)
    x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

    z = jax.random.normal(z_rng, x_t.shape)
    x_t_noisy = x_t + noise_std * z

    pred_vel, pred_z = model_fn(batch['observations'], x_t_noisy, t, return_denoiser=True)

    loss_vel_per_element = jnp.square(pred_vel - vel)
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
    """Flow matching BC loss for the online RL success buffer.

    Like standard flow BC but bc_coeff-weighted, treats all timesteps as valid,
    and defaults to executed 'true_actions'.
    """
    batch_actions = preprocess_actions(batch, action_chunking, action_key)
    batch_size = batch_actions.shape[0]

    rng, targets_rng = jax.random.split(rng)
    x_t, vel, t = get_flow_targets(batch['observations'], batch_actions, targets_rng)

    pred = model_fn(batch['observations'], x_t, t)

    loss_per_element = jnp.square(pred - vel)

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
    """Dispatch BC loss by config: MIP (policy_type=='mip'), flow+denoiser (use_denoiser), else flow."""
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
