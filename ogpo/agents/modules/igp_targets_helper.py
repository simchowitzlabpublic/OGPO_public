"""Bootstrap target generation for shortcut flow policies.

Follows the original shortcut models:
https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py
"""

import jax
import jax.numpy as jnp
import numpy as np


def get_shortcut_targets(
    observations,
    actions,
    model_fn,
    rng,
    denoise_timesteps=32,
    bootstrap_ratio=0.25,
):
    """Generate bootstrap targets for shortcut policy training.

    Returns (x_t, v_target, t, dt_base, is_bootstrap); dt_base is log2-encoded step size.
    denoise_timesteps must be a power of 2.
    """
    batch_size = observations.shape[0]
    action_dim = actions.shape[-1]

    bootstrap_batchsize = int(batch_size * bootstrap_ratio)
    bst_size_data = batch_size - bootstrap_batchsize

    rng, x0_rng, time_rng, dt_rng = jax.random.split(rng, 4)

    x_0 = jax.random.normal(x0_rng, (batch_size, action_dim))
    x_1 = actions

    # Logarithmic step size sampling
    log2_sections = int(np.log2(denoise_timesteps))  # e.g., log2(32) = 5
    dt_base = jnp.repeat(
        log2_sections - 1 - jnp.arange(log2_sections),
        bootstrap_batchsize // log2_sections
    )
    dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize-dt_base.shape[0],)])

    # dt_base [4, 3, 2, 1, 0] -> dt [1/16, 1/8, 1/4, 1/2, 1]
    dt = 1.0 / (2.0 ** dt_base)

    # Stratified time sampling based on step size
    dt_sections = jnp.power(2, dt_base).astype(jnp.int32)  # [16, 8, 4, 2, 1]
    t_bootstrap = jax.random.randint(
        time_rng, (bootstrap_batchsize,), minval=0, maxval=dt_sections
    ).astype(jnp.float32)
    t_bootstrap = t_bootstrap / dt_sections.astype(jnp.float32)  # normalize to [0, 1]

    t_bootstrap = t_bootstrap.reshape(-1, 1)
    dt_bootstrap = dt.reshape(-1, 1)
    dt_base_bootstrap = dt_base.reshape(-1, 1)

    # Flow matching with epsilon term for stability
    epsilon = 1e-5
    t_full_bootstrap = t_bootstrap
    x_t_bootstrap = (1 - (1 - epsilon) * t_full_bootstrap) * x_0[:bootstrap_batchsize] + t_full_bootstrap * x_1[:bootstrap_batchsize]

    if bootstrap_batchsize > 0:
        dt_base_bootstrap_half = dt_base_bootstrap + 1  # higher resolution for bootstrap
        dt_bootstrap_half = dt_bootstrap / 2

        # First half-step
        v_b1 = model_fn(
            observations[:bootstrap_batchsize],
            x_t_bootstrap,
            t_bootstrap,
            dt_base_bootstrap_half,
        )

        t2 = t_bootstrap + dt_bootstrap_half
        x_t2 = x_t_bootstrap + dt_bootstrap_half * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)  # stability clipping

        # Second half-step
        v_b2 = model_fn(
            observations[:bootstrap_batchsize],
            x_t2,
            t2,
            dt_base_bootstrap_half,
        )

        # Bootstrap target: average of the two half-step predictions
        v_bootstrap = (v_b1 + v_b2) / 2

        # Flow matching targets for the remaining batch
        if bst_size_data > 0:
            rng, flow_time_rng = jax.random.split(rng)
            t_flow = jax.random.uniform(flow_time_rng, (bst_size_data, 1))

            x_t_flow = (1 - (1 - epsilon) * t_flow) * x_0[bootstrap_batchsize:] + t_flow * x_1[bootstrap_batchsize:]
            v_flow = x_1[bootstrap_batchsize:] - (1 - epsilon) * x_0[bootstrap_batchsize:]

            dt_base_flow = jnp.full((bst_size_data, 1), log2_sections - 1)  # smallest dt_base

            x_t = jnp.concatenate([x_t_bootstrap, x_t_flow], axis=0)
            v_target = jnp.concatenate([v_bootstrap, v_flow], axis=0)
            t = jnp.concatenate([t_bootstrap, t_flow], axis=0)
            dt_base_combined = jnp.concatenate([dt_base_bootstrap, dt_base_flow], axis=0)
        else:
            x_t = x_t_bootstrap
            v_target = v_bootstrap
            t = t_bootstrap
            dt_base_combined = dt_base_bootstrap

        is_bootstrap = jnp.concatenate([
            jnp.ones(bootstrap_batchsize, dtype=bool),
            jnp.zeros(bst_size_data, dtype=bool)
        ])

    else:
        # Pure flow matching mode
        rng, flow_time_rng = jax.random.split(rng)
        t = jax.random.uniform(flow_time_rng, (batch_size, 1))
        x_t = (1 - (1 - epsilon) * t) * x_0 + t * x_1
        v_target = x_1 - (1 - epsilon) * x_0
        dt_base_combined = jnp.full((batch_size, 1), log2_sections - 1)
        is_bootstrap = jnp.zeros(batch_size, dtype=bool)
    
    return x_t, v_target, t, dt_base_combined, is_bootstrap


def get_flow_targets(observations, actions, rng,):
    """Standard flow matching targets; returns (x_t, v_target = x_1 - x_0, t)."""
    batch_size = observations.shape[0]
    action_dim = actions.shape[-1]
    
    rng, x0_rng, t_rng = jax.random.split(rng, 3)
    
    x_0 = jax.random.normal(x0_rng, (batch_size, action_dim))
    x_1 = actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    v_target = x_1 - x_0
    
    return x_t, v_target, t