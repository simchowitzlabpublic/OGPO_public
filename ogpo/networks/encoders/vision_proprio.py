"""Vision + Proprioception encoder.

Wraps a ViT image encoder and concatenates its output with proprioceptive state.
Used when actor_obs='image' or critic_obs='image'.
"""

import jax
import flax.linen as nn
import jax.numpy as jnp


class VisionProprioEncoder(nn.Module):
    """Encode images via ViT and concatenate with proprioceptive state.

    Attributes:
        vit: A MinViT (or compatible) image encoder module.
        state_proj_dim: If > 0, project proprioception through a Dense layer
                        before concatenation. If 0, concatenate raw state.
        img_proj_dim: If > 0, project ViT features through Dense + LayerNorm
                      before concatenation. Reduces 15k+ ViT dims to manageable size.
    """
    vit: nn.Module
    state_proj_dim: int = 0
    img_proj_dim: int = 0
    freeze_backbone: bool = False

    @nn.compact
    def __call__(self, images, state):
        """Encode images and state into a single flat feature vector.

        Args:
            images: Image observations, shape (B, H, W, C).
            state: Proprioceptive state, shape (B, state_dim).

        Returns:
            Flat feature vector, shape (B, img_dim + state_dim).
        """
        img_features = self.vit(images)  # (B, num_patches * embed_dim) when flatten=True
        if self.freeze_backbone:
            img_features = jax.lax.stop_gradient(img_features)
        if self.img_proj_dim > 0:
            img_features = nn.Dense(self.img_proj_dim)(img_features)
            img_features = nn.LayerNorm()(img_features)
            img_features = nn.relu(img_features)
        if self.state_proj_dim > 0:
            state = nn.Dense(self.state_proj_dim)(state)
            state = nn.relu(state)
        return jnp.concatenate([img_features, state], axis=-1)
