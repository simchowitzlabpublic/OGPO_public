"""Two-tier observation encoder for image + proprio fusion.

For frozen vision-encoder runs the actor/critic receive a flat vector
`concat(img_features, proprio)` (e.g. PaliGemma 2048-D + 23-D full_state).
Feeding that directly into a variance-scaling Dense violates the unit-variance
input assumption — the image side can drown out proprio.

This module:
  * L2-normalizes the image features (bounded scale; well-conditioned for Dense init)
  * Projects each modality through its own Dense -> LayerNorm -> GELU into fused_dim/2
  * Concatenates the two halves into `fused_dim`

The two towers each get equal capacity in the fused representation regardless
of input-dim disparity.
"""

import flax.linen as nn
import jax.numpy as jnp


class TwoTierObsEncoder(nn.Module):
    """Split image-feature + proprio observation into two pre-normalized projections.

    Expects observations of shape (..., img_feat_dim + proprio_dim), with
    image features in the leading slice and proprio in the trailing slice
    (matches `_pre_encode_dataset`'s `concat([img_features, proprio])` layout).

    Attributes:
        img_feat_dim: Dimensionality of the (frozen) image feature slice.
        proprio_dim:  Dimensionality of the proprio slice.
        fused_dim:    Output dimensionality; split 50/50 between the two towers.
    """

    img_feat_dim: int
    proprio_dim: int
    fused_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        if self.fused_dim % 2 != 0:
            raise ValueError(f"fused_dim must be even, got {self.fused_dim}")
        half = self.fused_dim // 2

        img = observations[..., : self.img_feat_dim]
        prop = observations[..., self.img_feat_dim : self.img_feat_dim + self.proprio_dim]

        # L2-normalize image features so the projection sees unit-norm input.
        img_norm = img / (jnp.linalg.norm(img, axis=-1, keepdims=True) + 1e-6)

        # Image tower
        img_h = nn.Dense(half, name='img_proj')(img_norm)
        img_h = nn.LayerNorm(name='img_ln')(img_h)
        img_h = nn.gelu(img_h)

        # Proprio tower
        prop_h = nn.Dense(half, name='prop_proj')(prop)
        prop_h = nn.LayerNorm(name='prop_ln')(prop_h)
        prop_h = nn.gelu(prop_h)

        return jnp.concatenate([img_h, prop_h], axis=-1)
