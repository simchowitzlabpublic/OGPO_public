"""Vision + Proprioception encoder.

Wraps a ViT image encoder and concatenates its output with proprioceptive state.
Used when actor_obs='image' or critic_obs='image'.
"""

import jax
import flax.linen as nn
import jax.numpy as jnp


class VisionProprioEncoder(nn.Module):
    """Encode images via ViT and concatenate with proprioceptive state.

    state_proj_dim / img_proj_dim, when > 0, project the respective modality
    before concatenation (img projection also reduces the large ViT feature dim).
    """
    vit: nn.Module
    state_proj_dim: int = 0
    img_proj_dim: int = 0
    freeze_backbone: bool = False

    @nn.compact
    def __call__(self, images, state):
        """Encode images (B, H, W, C) and state (B, state_dim) into a flat vector."""
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
