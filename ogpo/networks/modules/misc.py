"""Miscellaneous neural network utilities."""

from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax


def default_init(scale=1.0):
    """Default weight initialization using variance scaling."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def orthogonal_init(scale: float = 1.0, dtype=jnp.float32):
    """Orthogonal weight initialization."""
    def _init(key, shape, dtype=dtype):
        # Flatten to 2D for orthogonalization, then reshape back.
        if len(shape) < 2:
            raise ValueError("Orthogonal initializer requires at least 2D shape")
        n_rows = 1
        for d in shape[:-1]:
            n_rows *= d
        n_cols = shape[-1]
        flat_shape = (n_rows, n_cols)
        a = jax.random.normal(key, flat_shape, dtype=dtype)
        q, r = jnp.linalg.qr(a)
        d = jnp.sign(jnp.diag(r))  # enforce uniform sign
        q = q * d
        q = q.reshape(shape)
        return (scale * q).astype(dtype)
    return _init


zeros_init = nn.initializers.zeros


class Identity(nn.Module):
    """Identity layer that returns its input unchanged."""

    def __call__(self, x):
        return x


class LogParam(nn.Module):
    """Learnable positive scalar stored in log space for stability."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class NoiseInjectionNetwork(nn.Module):
    """Noise injection network for ReinFlow.

    Predicts state- and time-dependent noise scales to make flow policies
    stochastic for policy gradient training.
    """

    hidden_dims: Sequence[int] = (256, 256)
    action_dim: int = 1
    layer_norm: bool = True
    min_noise_std: float = 0.001
    max_noise_std: float = 1.0

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 time_steps: jnp.ndarray) -> jnp.ndarray:
        """Predict noise standard deviation from observations and time."""
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        x = jnp.concatenate([observations, time_steps], axis=-1)

        for size in self.hidden_dims:
            x = nn.Dense(features=size, kernel_init=default_init())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        logvar = nn.Dense(features=self.action_dim,
                          kernel_init=default_init(0.1))(x)

        # Squash to [-1, 1] then linearly map to [log(min²), log(max²)].
        logvar = jnp.tanh(logvar)
        logvar_min = jnp.log(self.min_noise_std ** 2)
        logvar_max = jnp.log(self.max_noise_std ** 2)
        logvar = logvar_min + (logvar_max - logvar_min) * (logvar + 1.0) / 2.0

        noise_std = jnp.exp(0.5 * logvar)  # σ = exp(0.5 * logσ²)
        return noise_std
