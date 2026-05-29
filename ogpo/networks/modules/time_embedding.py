"""Sinusoidal time embedding for flow matching timesteps.

Encodes scalar t ∈ [0, 1] into a rich feature vector using sinusoidal
positional embeddings followed by a small MLP, matching the standard
approach used in DPPO, DiT, Pi0/Pi0.5, and rectified flow literature.
"""

import math
import flax.linen as nn
import jax.numpy as jnp


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar timesteps.

    Pipeline: scalar t → sinusoidal features → Linear → GELU → Linear

    Attributes:
        embed_dim: Output embedding dimension (must be even).
        max_freq_log: Log of maximum frequency (DPPO uses log(10000) ≈ 9.21).
    """
    embed_dim: int = 32
    max_freq_log: float = math.log(10000)

    @nn.compact
    def __call__(self, t):
        """Embed scalar timestep(s).

        Args:
            t: Timestep tensor, shape (..., 1) or (...,).

        Returns:
            Embedding tensor, shape (..., embed_dim).
        """
        # Flatten trailing dim if shape is (..., 1)
        if t.shape[-1] == 1:
            t = t[..., 0]

        half_dim = self.embed_dim // 2
        freqs = jnp.exp(
            jnp.arange(half_dim, dtype=jnp.float32) * -(self.max_freq_log / (half_dim - 1))
        )
        # (...,) x (half_dim,) -> (..., half_dim)
        angles = t[..., None] * freqs
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

        # Small MLP: Linear → GELU → Linear (matches DPPO/DiT convention)
        emb = nn.Dense(self.embed_dim * 2)(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.embed_dim)(emb)
        return emb
