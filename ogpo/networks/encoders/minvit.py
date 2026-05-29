"""MinViT vision encoder implementation.

This module contains the MinViT (Minimal Vision Transformer) encoder,
a lightweight Vision Transformer architecture suitable for reinforcement learning
and other vision tasks where computational efficiency is important.
"""

from typing import Callable, TypeVar

from flax import linen as nn
import jax.numpy as jnp


T = TypeVar("T")


def normalize_images(img, img_norm_type="default"):
    """Normalize images based on the specified normalization type.

    Args:
        img: Input image array with pixel values in [0, 255].
        img_norm_type: Type of normalization to apply.
            - "default": Normalize to [-1, 1]
            - "imagenet": Apply ImageNet mean/std normalization
            - "minvit": Normalize to [-0.5, 0.5] (MinViT-specific)

    Returns:
        Normalized image array.
    """
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.astype(jnp.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.astype(jnp.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # tile mean and std (to account for stacked early_fusion images)
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = jnp.tile(mean, num_tile)
        std_tile = jnp.tile(std, num_tile)

        # tile the mean/std, normalize image, and return
        return (img - mean_tile) / std_tile
    elif img_norm_type == "minvit":
        # Match PyTorch MinVit normalization: obs / 255.0 - 0.5
        return img.astype(jnp.float32) / 255.0 - 0.5
    raise ValueError(f"Unknown img_norm_type: {img_norm_type}")


class PatchEmbed2JAX(nn.Module):
    """JAX/Flax version of PyTorch PatchEmbed2 that exactly matches the implementation.

    This module implements a two-stage convolutional patch embedding, which provides
    better feature extraction compared to a single large-stride convolution.

    Attributes:
        embed_dim: Embedding dimension for patch features.
        use_norm: Whether to apply group normalization.
        num_channel: Number of input channels.
        img_h: Input image height.
        img_w: Input image width.
    """

    embed_dim: int
    use_norm: bool
    num_channel: int = 3
    img_h: int = 96
    img_w: int = 96

    def setup(self):
        # Calculate number of patches exactly like PyTorch version
        import math
        H1 = math.ceil((self.img_h - 8) / 4) + 1
        W1 = math.ceil((self.img_w - 8) / 4) + 1
        H2 = math.ceil((H1 - 3) / 2) + 1
        W2 = math.ceil((W1 - 3) / 2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = self.embed_dim

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # First conv: kernel=8, stride=4
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
        )(x)

        # Optional GroupNorm
        if self.use_norm:
            x = nn.GroupNorm(num_groups=self.embed_dim)(x)

        # ReLU
        x = nn.relu(x)

        # Second conv: kernel=3, stride=2
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )(x)

        return x


class TransformerLayer(nn.Module):
    """Transformer encoder layer with pre-layer normalization.

    Implements a standard transformer encoder block with multi-head self-attention
    and a feed-forward MLP, both with residual connections.

    Attributes:
        embed_dim: Dimension of the embedding/hidden states.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim.
        dropout: Dropout probability.
        attn_kernel_init: Initializer for attention layer kernels.
        dense_kernel_init: Initializer for dense layer kernels.
    """

    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0
    attn_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    dense_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        # Pre-LN Self-Attention
        h = nn.LayerNorm(name="ln1")(x)
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=self.dropout,
            deterministic=not train,
            kernel_init=self.attn_kernel_init,
            name="attn",
        )(h)
        x = x + h  # residual

        # Pre-LN MLP
        h = nn.LayerNorm(name="ln2")(x)
        h = nn.Dense(
            self.embed_dim * self.mlp_ratio,
            kernel_init=self.dense_kernel_init,
            name="fc1",
        )(h)
        h = nn.gelu(h)
        h = nn.Dense(
            self.embed_dim,
            kernel_init=self.dense_kernel_init,
            name="fc2",
        )(h)
        if self.dropout > 0.0:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        x = x + h  # residual
        return x


class MinViT(nn.Module):
    """Minimal Vision Transformer (MinViT) encoder.

    JAX/Flax implementation that exactly mirrors the PyTorch MinViT.
    A lightweight ViT architecture optimized for efficiency while maintaining
    good performance on visual tasks.

    Attributes:
        embed_style: Patch embedding style ('embed1' for single conv, 'embed2' for two-stage).
        embed_dim: Embedding dimension.
        embed_norm: Whether to use normalization in embed2 (0=False, 1=True).
        num_heads: Number of attention heads.
        depth: Number of transformer layers.
        dropout: Dropout probability.
        num_channel: Number of input image channels.
        img_h: Input image height.
        img_w: Input image width.
        flatten: Whether to flatten the output patches.
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim.
        mlp_hidden_dims: Hidden dimensions for optional output MLP.
        pos_kernel_init: Initializer for positional embeddings.
        attn_kernel_init: Initializer for attention layers.
        dense_kernel_init: Initializer for dense layers.
    """

    embed_style: str = "embed2"      # 'embed1' or 'embed2'
    embed_dim: int = 128
    embed_norm: int = 0              # whether to use norm in embed2 (0=False, 1=True)
    num_heads: int = 4
    depth: int = 1
    dropout: float = 0.0
    num_channel: int = 3
    img_h: int = 96
    img_w: int = 96
    flatten: bool = False
    mlp_ratio: int = 4
    mlp_hidden_dims: tuple = (768,)

    # Initializers to match ViT-style
    pos_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    attn_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    dense_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)

    @property
    def num_patches(self):
        """Calculate num_patches based on embed_style exactly like PyTorch."""
        import math
        if self.embed_style == "embed1":
            return math.ceil(self.img_h / 8) * math.ceil(self.img_w / 8)
        elif self.embed_style == "embed2":
            H1 = math.ceil((self.img_h - 8) / 4) + 1
            W1 = math.ceil((self.img_w - 8) / 4) + 1
            H2 = math.ceil((H1 - 3) / 2) + 1
            W2 = math.ceil((W1 - 3) / 2) + 1
            return H2 * W2
        else:
            raise ValueError(f"Unknown embed_style: {self.embed_style}")

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *, train: bool = True, cond_var=None):
        x = normalize_images(observations, "minvit")

        if self.embed_style == "embed1":
            x = nn.Conv(
                features=self.embed_dim,
                kernel_size=(8, 8),
                strides=(8, 8),
                padding="VALID",
                name="patch_embed_conv",
            )(x)
        elif self.embed_style == "embed2":
            x = PatchEmbed2JAX(
                embed_dim=self.embed_dim,
                use_norm=bool(self.embed_norm),
                num_channel=self.num_channel,
                img_h=self.img_h,
                img_w=self.img_w,
                name="patch_embed",
            )(x)
        else:
            raise ValueError(f"Unknown embed_style: {self.embed_style}")

        if len(x.shape) == 3:
            h, w, c = x.shape
            x = jnp.expand_dims(x, axis=0)
            b = 1
            was_unbatched = True
        else:
            b, h, w, c = x.shape
            was_unbatched = False

        assert c == self.embed_dim, f"Expected channels={self.embed_dim}, got {c}"
        t = h * w
        x = jnp.reshape(x, (b, t, c))

        assert t == self.num_patches, f"Expected {self.num_patches} patches, got {t}"

        pos_embed = self.param(
            "pos_embed",
            self.pos_kernel_init,
            (1, self.num_patches, self.embed_dim),
        )
        x = x + pos_embed

        for i in range(self.depth):
            x = TransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attn_kernel_init=self.attn_kernel_init,
                dense_kernel_init=self.dense_kernel_init,
                name=f"block_{i}",
            )(x, train=train)

        x = nn.LayerNorm(name="norm")(x)

        if self.flatten:
            x = jnp.reshape(x, (b, self.num_patches * self.embed_dim))

        if was_unbatched:
            x = jnp.squeeze(x, axis=0)
        return x


# Alias for compatibility
MinVitEncoder = MinViT
