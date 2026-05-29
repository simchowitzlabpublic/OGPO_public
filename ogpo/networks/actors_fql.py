"""One-step policy network for FQL algorithm.

This module contains the OneStepPolicy network that directly maps
(state, noise) to actions without ODE solving, used for FQL's
fast inference and RL training with distillation.
"""

from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp

from ogpo.networks.modules import MLP


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class OneStepPolicy(nn.Module):
    """One-step policy for FQL: μω(s, z) → a.

    This policy directly maps observations and noise to actions in a single
    forward pass, without iterative ODE solving. It learns via distillation
    from the BC flow policy and Q-value maximization.

    Key differences from ActorVectorField:
    - ActorVectorField: (obs, action, time) → velocity (requires ODE solving)
    - OneStepPolicy: (obs, noise) → action (direct mapping)

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP.
        action_dim: Dimension of the action space.
        layer_norm: Whether to apply layer normalization.
        final_fc_init_scale: Initialization scale for final layer.
        encoder: Optional encoder module for observations.
    """
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    action_dim: int = None
    layer_norm: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        """Initialize the network layers."""
        # Single MLP: same architecture as ActorVectorField for better distillation
        # [hidden_dims..., action_dim] with no activation on output
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,  # No activation on output (like ActorVectorField)
            layer_norm=self.layer_norm,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        noise: jnp.ndarray,
        is_encoded: bool = False,
        params: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Forward pass: (observations, noise) → actions.

        Args:
            observations: State observations (batch_size, obs_dim).
            noise: Gaussian noise samples (batch_size, action_dim).
            is_encoded: Whether observations are already encoded.
            params: Optional parameters (for compatibility with TrainState).

        Returns:
            actions: Predicted actions (batch_size, action_dim).
        """
        # Encode observations if needed
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)

        # Concatenate observations and noise
        x = jnp.concatenate([observations, noise], axis=-1)

        # Pass through MLP (outputs actions directly, no final activation)
        actions = self.mlp(x)

        return actions

    def encode(self, observations, images=None):
        """Encode observations using the encoder.

        Args:
            observations: Raw observations.
            images: Optional images (for vision-based policies).

        Returns:
            Encoded observations.
        """
        if self.encoder is None:
            return observations

        if images is not None:
            return self.encoder(observations, images=images)
        return self.encoder(observations)
