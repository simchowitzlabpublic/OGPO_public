"""MinViT (Minimal Vision Transformer) encoder, a lightweight ViT for RL vision tasks."""

from typing import Callable, TypeVar

from flax import linen as nn
import jax.numpy as jnp


T = TypeVar("T")


def normalize_images(img, img_norm_type="default"):
    """Normalize images by the given type: 'default', 'imagenet', or 'minvit'."""
    if img_norm_type == "default":
        # put pixels in [-1, 1]
        return img.astype(jnp.float32) / 127.5 - 1.0
    elif img_norm_type == "imagenet":
        # put pixels in [0,1]
        img = img.astype(jnp.float32) / 255
        assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

        mean = jnp.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = jnp.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))

        # Tile mean/std to cover stacked early_fusion images.
        num_tile = (1, 1, 1, int(img.shape[-1] / 3))
        mean_tile = jnp.tile(mean, num_tile)
        std_tile = jnp.tile(std, num_tile)

        return (img - mean_tile) / std_tile
    elif img_norm_type == "minvit":
        # Match PyTorch MinVit normalization: obs / 255.0 - 0.5
        return img.astype(jnp.float32) / 255.0 - 0.5
    raise ValueError(f"Unknown img_norm_type: {img_norm_type}")


class PatchEmbed2JAX(nn.Module):
    """Two-stage convolutional patch embedding (port of PyTorch PatchEmbed2)."""

    embed_dim: int
    use_norm: bool
    num_channel: int = 3
    img_h: int = 96
    img_w: int = 96

    def setup(self):
        import math
        H1 = math.ceil((self.img_h - 8) / 4) + 1
        W1 = math.ceil((self.img_w - 8) / 4) + 1
        H2 = math.ceil((H1 - 3) / 2) + 1
        W2 = math.ceil((W1 - 3) / 2) + 1
        self.num_patch = H2 * W2
        self.patch_dim = self.embed_dim

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
        )(x)

        if self.use_norm:
            x = nn.GroupNorm(num_groups=self.embed_dim)(x)

        x = nn.relu(x)

        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
        )(x)

        return x


class TransformerLayer(nn.Module):
    """Pre-norm transformer encoder block (self-attention + MLP, residual)."""

    embed_dim: int
    num_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0
    attn_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    dense_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)

    @nn.compact
    def __call__(self, x, *, train: bool = True):
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
    """Minimal Vision Transformer encoder (JAX/Flax port of PyTorch MinViT)."""

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

    pos_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    attn_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)
    dense_kernel_init: nn.initializers.Initializer = nn.initializers.truncated_normal(stddev=0.02)

    @property
    def num_patches(self):
        """Number of patches for the configured embed_style."""
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


MinVitEncoder = MinViT
