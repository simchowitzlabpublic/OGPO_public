"""MLP variants and related components."""

from typing import Any, Sequence, Optional
import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


def default_init(scale=1.0):
    """Default weight initialization using variance scaling."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class MLPCond(nn.Module):
    """MLP that concatenates a conditioning vector to the input at each layer."""

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, cond):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(jnp.concatenate([x, cond], axis=-1))
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class MLPResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation."""
    dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        residual = x

        ortho_init = initializers.orthogonal()
        zeros_init = initializers.constant(0.0)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * 4,
                     kernel_init=ortho_init,
                     bias_init=zeros_init)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim,
                     kernel_init=ortho_init,
                     bias_init=zeros_init)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        return x + residual


class MLPWithFiLM(nn.Module):
    """MLP with FiLM conditioning and sinusoidal time embedding for diffusion models."""
    act_dim: int
    Ta: int
    obs_dim: int
    To: int
    emb_dim: int = 512
    n_layers: int = 6
    timestep_emb_dim: int = 128
    max_freq: float = 100.0
    disable_time_embedding: bool = False
    dropout: float = 0.1

    def setup(self):
        if self.timestep_emb_dim % 2 != 0:
            raise ValueError("timestep_emb_dim must be even")
        num_frequencies = self.timestep_emb_dim // 2

        self.frequencies = jnp.linspace(0.0, self.max_freq, num_frequencies)
        time_emb_size = 0 if self.disable_time_embedding else 2 * self.timestep_emb_dim
        input_dim = self.act_dim * self.Ta + time_emb_size + self.obs_dim * self.To
        output_dim = self.act_dim * self.Ta

        ortho_init = initializers.orthogonal()
        zeros_init = initializers.constant(0.0)

        self.input_proj = nn.Dense(self.emb_dim, kernel_init=ortho_init, bias_init=zeros_init)
        self.input_norm = nn.LayerNorm()

        self.residual_blocks = [MLPResidualBlock(self.emb_dim, self.dropout) for _ in range(self.n_layers)]
        self.final_norm = nn.LayerNorm()

        self.main_output = nn.Dense(output_dim, kernel_init=ortho_init, bias_init=zeros_init)

        self.scalar_output = nn.Dense(1, kernel_init=zeros_init, bias_init=zeros_init)

    def _embed_time(self, t: jnp.ndarray) -> jnp.ndarray:
        """Generate sinusoidal time embeddings."""
        t_unsqueezed = t[..., None]
        angles = t_unsqueezed * self.frequencies[None, :]
        sin_embed = jnp.sin(angles)
        cos_embed = jnp.cos(angles)
        embedded = jnp.concatenate([sin_embed, cos_embed], axis=-1)
        return embedded

    def __call__(
        self,
        x: jnp.ndarray,
        s: jnp.ndarray,
        t: jnp.ndarray,
        condition: Optional[jnp.ndarray],
        deterministic: bool
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_flat = x.reshape(x.shape[0], -1)  # (b, Ta * act_dim)

        if condition is not None:
            condition_flat = condition.reshape(condition.shape[0], -1)  # (b, To * obs_dim)
        else:
            batch_size = x.shape[0]
            condition_flat = jnp.zeros((batch_size, self.To * self.obs_dim), dtype=x.dtype)

        if not self.disable_time_embedding:
            s_embedded = self._embed_time(s)
            t_embedded = self._embed_time(t)
            input_data = jnp.concatenate([x_flat, s_embedded, t_embedded, condition_flat], axis=-1)
        else:
            input_data = jnp.concatenate([x_flat, condition_flat], axis=-1)

        features = self.input_proj(input_data)
        features = self.input_norm(features)
        features = nn.gelu(features)

        for block in self.residual_blocks:
            features = block(features, deterministic=deterministic)

        features = self.final_norm(features)
        main_output_flat = self.main_output(features)  # (b, Ta * act_dim)
        scalar_output = self.scalar_output(features)     # (b, 1)

        main_output = main_output_flat.reshape(x.shape[0], self.Ta, self.act_dim)
        return main_output, scalar_output


class RSNorm(nn.Module):
    """Running Statistics Normalization layer with train/eval modes."""
    epsilon: float = 1e-8
    momentum: float = 0.99

    @nn.compact
    def __call__(self, x, use_running_average: bool = False):
        running_mean = self.variable('batch_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        running_var = self.variable('batch_stats', 'var', lambda: jnp.ones(x.shape[-1]))

        if use_running_average:
            mean = running_mean.value
            var = running_var.value
        else:
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)

            if self.is_mutable_collection('batch_stats'):
                running_mean.value = self.momentum * running_mean.value + (1 - self.momentum) * mean
                running_var.value = self.momentum * running_var.value + (1 - self.momentum) * var

        return (x - mean) / jnp.sqrt(var + self.epsilon)


class SimBaInternalMLP(nn.Module):
    """Internal MLP for a SimBa block with an inverted bottleneck."""
    hidden_dim: int
    kernel_init: Any = default_init

    @nn.compact
    def __call__(self, x):
        intermediate_dim = self.hidden_dim * 4
        y = nn.Dense(intermediate_dim, kernel_init=self.kernel_init)(x)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(y)
        return y


class SimBaMLP(nn.Module):
    """SimBa architecture (RSNorm + residual feedforward blocks).

    Accepts a `hidden_dims` sequence like a standard MLP; the last element is
    the output size and intermediate blocks all use the width of the first.
    """
    hidden_dims: Sequence[int]
    activations: Any = nn.silu
    activate_final: bool = False
    kernel_init: Any = default_init
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8

    @nn.compact
    def __call__(self, x, train: bool = True):
        if not self.hidden_dims:
            raise ValueError("hidden_dims cannot be empty.")

        main_body_dims = self.hidden_dims[:-1]
        output_dim = self.hidden_dims[-1]

        x = RSNorm(
            momentum=self.rs_norm_momentum,
            epsilon=self.rs_norm_epsilon,
            name='RSNorm'
        )(x, use_running_average=not train)

        if not main_body_dims:
            x = nn.Dense(output_dim, kernel_init=self.kernel_init)(x)
            if self.activate_final:
                x = self.activations(x)
            return x

        # All blocks share the width of the first hidden dimension.
        hidden_dim = main_body_dims[0]
        num_blocks = len(main_body_dims)

        x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)

        for i in range(num_blocks):
            residual = x
            x_norm = nn.LayerNorm(name=f'pre_ln_{i}')(x)
            mlp_output = SimBaInternalMLP(
                hidden_dim=hidden_dim,
                kernel_init=self.kernel_init
            )(x_norm)
            x = residual + mlp_output

        x = nn.LayerNorm(name='post_ln')(x)

        self.sow('intermediates', 'feature', x)

        x = nn.Dense(output_dim, kernel_init=self.kernel_init)(x)

        if self.activate_final:
            x = self.activations(x)

        return x
