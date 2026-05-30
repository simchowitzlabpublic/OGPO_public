"""Distribution classes and custom bijectors."""

from typing import Optional
import jax.numpy as jnp
import distrax


class RescaleFromTanhBijector(distrax.Bijector):
    """Rescaling bijector mapping (-1, 1) to (low, high).

    Replaces distrax.Lambda to avoid JAX 0.6.0+ incompatibility with the
    removed jax.core.Var.
    """

    def __init__(self, low: jnp.ndarray, high: jnp.ndarray):
        super().__init__(event_ndims_in=1, event_ndims_out=1)
        self._low = low
        self._high = high

    def forward_and_log_det(self, x):
        """Transform from (-1, 1) to (low, high)."""
        # (-1, 1) => (0, 1) => (low, high)
        y = (x + 1) / 2
        y = y * (self._high - self._low) + self._low
        high_ = jnp.broadcast_to(self._high, x.shape)
        low_ = jnp.broadcast_to(self._low, x.shape)
        log_det = jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)
        return y, log_det

    def inverse_and_log_det(self, y):
        """Transform from (low, high) to (-1, 1)."""
        # (low, high) => (0, 1) => (-1, 1)
        x = (y - self._low) / (self._high - self._low)
        x = 2 * x - 1
        high_ = jnp.broadcast_to(self._high, y.shape)
        low_ = jnp.broadcast_to(self._low, y.shape)
        log_det = -jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)
        return x, log_det


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Multivariate normal with tanh squashing, optionally rescaled to (low, high).

    Commonly used in SAC and other continuous control RL algorithms.
    """

    def __init__(self,
                 distribution: distrax.Distribution,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        layers = []
        if not (low is None or high is None):
            layers.append(RescaleFromTanhBijector(low=low, high=high))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        """Compute the mode of the distribution."""
        return self.bijector.forward(self.distribution.mode())


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution that adds a mode() method."""

    def mode(self):
        """Compute the mode of the distribution."""
        return self.bijector.forward(self.distribution.mode())
