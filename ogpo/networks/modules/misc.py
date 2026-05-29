"""Miscellaneous neural network utilities."""

from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax


def default_init(scale=1.0):
    """Default weight initialization using variance scaling.

    Args:
        scale: Scaling factor for initialization.

    Returns:
        Initialization function.
    """
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def orthogonal_init(scale: float = 1.0, dtype=jnp.float32):
    """Orthogonal weight initialization.

    Args:
        scale: Scaling factor for initialization.
        dtype: Data type of the initialized weights.

    Returns:
        Initialization function.
    """
    def _init(key, shape, dtype=dtype):
        # Flatten to 2D for orthogonalization, then reshape back
        if len(shape) < 2:
            raise ValueError("Orthogonal initializer requires at least 2D shape")
        n_rows = 1
        for d in shape[:-1]:
            n_rows *= d
        n_cols = shape[-1]
        flat_shape = (n_rows, n_cols)
        a = jax.random.normal(key, flat_shape, dtype=dtype)
        # QR decomposition
        q, r = jnp.linalg.qr(a)
        # Enforce uniform sign
        d = jnp.sign(jnp.diag(r))
        q = q * d
        q = q.reshape(shape)
        return (scale * q).astype(dtype)
    return _init


# Convenience reference to zeros initializer
zeros_init = nn.initializers.zeros


class Identity(nn.Module):
    """Identity layer that returns its input unchanged."""

    def __call__(self, x):
        return x


class LogParam(nn.Module):
    """Learnable scalar parameter with log scale.

    This module stores a parameter in log space for numerical stability
    and positive constraint. Useful for learning standard deviations,
    scales, and other positive quantities.

    Attributes:
        init_value: Initial value of the parameter (before log transform).
    """

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        """Returns the exponentiated parameter value.

        Returns:
            Scalar parameter value (positive).
        """
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class NoiseInjectionNetwork(nn.Module):
    """Noise injection network for ReinFlow.

    This network learns to inject appropriate noise into flow policies
    to make them stochastic for policy gradient training. It predicts
    state and time-dependent noise scales.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        min_noise_std: Minimum noise standard deviation.
        max_noise_std: Maximum noise standard deviation.
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
        """Predict noise standard deviation based on observations and time.

        Args:
            observations: Observation tensor.
            time_steps: Time step tensor.

        Returns:
            Noise standard deviation tensor.
        """
        # Concatenate inputs
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        x = jnp.concatenate([observations, time_steps], axis=-1)

        # MLP hidden layers
        for size in self.hidden_dims:
            x = nn.Dense(features=size, kernel_init=default_init())(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Predict log-variance (log σ²)
        logvar = nn.Dense(features=self.action_dim,
                          kernel_init=default_init(0.1))(x)

        # Squash into [-1,1]
        logvar = jnp.tanh(logvar)

        # Linearly map to [log(min²), log(max²)]
        logvar_min = jnp.log(self.min_noise_std ** 2)
        logvar_max = jnp.log(self.max_noise_std ** 2)
        logvar = logvar_min + (logvar_max - logvar_min) * (logvar + 1.0) / 2.0

        # Convert to std: σ = exp(0.5 * logσ²)
        noise_std = jnp.exp(0.5 * logvar)
        return noise_std
