"""Attention and transformer components (DiT-style AdaLN-Zero)."""

import math
from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange


# Xavier uniform — matches PyTorch nn.Linear default and DiT.
def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def zero_init():
    return nn.initializers.constant(0.0)


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP (DiT-style)."""
    hidden_dim: int
    frequency_embedding_size: int = 256

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        t = jnp.ravel(t)  # ensure (N,)
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(t_freq)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        return x


class ConditioningMLP(nn.Module):
    """2-layer MLP with SiLU for conditioning projection (DiT-style)."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        head_dim = self.hidden_dim // self.num_heads

        qkv = nn.Dense(3 * self.hidden_dim, kernel_init=default_init())(x)
        q, k, v = rearrange(
            qkv, 'b l (three h d) -> three b h l d',
            three=3, h=self.num_heads,
        )

        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('b h q d, b h k d -> b h q k', q, k) * scale

        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum('b h q k, b h k d -> b h q d', attn_weights, v)
        output = rearrange(attn_output, 'b h l d -> b l (h d)')

        output = nn.Dense(self.hidden_dim, kernel_init=default_init())(output)
        return output


class FeedForward(nn.Module):
    """Feedforward network."""
    hidden_dim: int
    ff_dim: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.ff_dim, kernel_init=default_init())(x)
        x = self.activation(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        return x


class AdaLNLayer(nn.Module):
    """AdaLN-Zero transformer block.

    Zero-initializes the adaLN modulation layer so that at init
    each block is an identity function.
    """
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        cond_processed = nn.silu(cond)
        adaLN_params = nn.Dense(
            6 * self.hidden_dim,
            kernel_init=nn.initializers.constant(0.0),
        )(cond_processed)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            jnp.split(adaLN_params, 6, axis=-1)
        )

        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        modulated_x = modulate(norm_x, shift_msa, scale_msa)
        attn_output = MultiHeadAttention(
            self.hidden_dim, self.num_heads, self.dropout_rate
        )(modulated_x, mask, deterministic)
        x = x + gate_msa[:, None] * attn_output

        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        modulated_x = modulate(norm_x, shift_mlp, scale_mlp)
        ff_output = FeedForward(
            self.hidden_dim, self.ff_dim, self.dropout_rate,
            self.activation
        )(modulated_x, deterministic)
        x = x + gate_mlp[:, None] * ff_output

        return x


class AdaLNFinalLayer(nn.Module):
    """Final layer with adaLN + zero-initialized output projection."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x, cond):
        cond_processed = nn.silu(cond)
        adaLN_params = nn.Dense(
            2 * self.hidden_dim,
            kernel_init=nn.initializers.constant(0.0),
        )(cond_processed)

        shift, scale = jnp.split(adaLN_params, 2, axis=-1)
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        x = modulate(norm_x, shift, scale)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.constant(0.0),
        )(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention: Q from x, K/V from context."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, context, deterministic=True):
        head_dim = self.hidden_dim // self.num_heads

        q = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        kv = nn.Dense(2 * self.hidden_dim, kernel_init=default_init())(context)
        k, v = rearrange(
            kv, 'b s (two h d) -> two b h s d',
            two=2, h=self.num_heads,
        )
        q = rearrange(q, 'b q (h d) -> b h q d', h=self.num_heads)

        scale = jax.lax.rsqrt(jnp.float32(head_dim))
        attn_weights = jnp.einsum('b h q d, b h k d -> b h q k', q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate)(attn_weights, deterministic=deterministic)

        attn_output = jnp.einsum('b h q k, b h k d -> b h q d', attn_weights, v)
        output = rearrange(attn_output, 'b h l d -> b l (h d)')
        return nn.Dense(self.hidden_dim, kernel_init=default_init())(output)


class CrossAttnLayer(nn.Module):
    """Pre-norm transformer layer with self-attention + cross-attention conditioning.

    Three sub-layers: causal self-attention, cross-attention to context, FFN.
    """
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, context, mask=None, deterministic=True):
        norm_x = nn.LayerNorm(epsilon=1e-6)(x)
        attn_output = MultiHeadAttention(
            self.hidden_dim, self.num_heads, self.dropout_rate
        )(norm_x, mask, deterministic)
        x = x + attn_output

        norm_x = nn.LayerNorm(epsilon=1e-6)(x)
        cross_output = CrossAttention(
            self.hidden_dim, self.num_heads, self.dropout_rate
        )(norm_x, context, deterministic)
        x = x + cross_output

        norm_x = nn.LayerNorm(epsilon=1e-6)(x)
        ff_output = FeedForward(
            self.hidden_dim, self.ff_dim, self.dropout_rate
        )(norm_x, deterministic)
        x = x + ff_output

        return x


class CrossAttnFinalLayer(nn.Module):
    """Final layer for cross-attention transformer: LayerNorm + zero-init Dense."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=zero_init(),
            bias_init=zero_init(),
        )(x)
        return x


class AdaLNTransformer(nn.Module):
    """Complete DiT-style transformer with adaLN-Zero conditioning."""
    hidden_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    output_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, cond, mask=None, deterministic=True):
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        cond = nn.Dense(self.hidden_dim, kernel_init=default_init())(cond)

        for _ in range(self.num_layers):
            x = AdaLNLayer(
                self.hidden_dim, self.num_heads, self.ff_dim, self.dropout_rate
            )(x, cond, mask, deterministic)

        x = AdaLNFinalLayer(self.hidden_dim, self.output_dim)(x, cond)
        return x
