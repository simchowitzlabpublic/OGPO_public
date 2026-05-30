"""One-step policy network for the FQL algorithm: maps (state, noise) to actions without ODE solving."""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ogpo.networks.modules import MLP


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class OneStepPolicy(nn.Module):
    """One-step FQL policy mapping (observations, noise) -> action in a single forward pass.

    Learns via distillation from the BC flow policy and Q-value maximization.
    """
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    action_dim: int = None
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        noise: jnp.ndarray,
        is_encoded: bool = False,
        params: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Forward pass mapping (observations, noise) -> actions."""
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        x = jnp.concatenate([observations, noise], axis=-1)
        actions = self.mlp(x)

        return actions

    def encode(self, observations, images=None):
        """Encode observations using the encoder, if any."""
        if self.encoder is None:
            return observations

        if images is not None:
            return self.encoder(observations, images=images)
        return self.encoder(observations)
