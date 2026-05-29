"""MLP variants and related components."""

from typing import Any, Sequence, Optional
import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers


def default_init(scale=1.0):
    """Default weight initialization using variance scaling.

    Args:
        scale: Scaling factor for initialization.

    Returns:
        Initialization function.
    """
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

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
    """MLP with conditioning input repeated for each layer.

    This MLP concatenates a conditioning vector to the input at each layer,
    allowing the network to be modulated by external information.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

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
    """Residual block with LayerNorm and GELU activation.

    Attributes:
        dim: Dimension of the block.
        dropout: Dropout rate.
    """
    dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """
        Args:
            x: The input tensor.
            deterministic: If True, dropout is disabled.
        """
        residual = x

        # Define initializers
        ortho_init = initializers.orthogonal()
        zeros_init = initializers.constant(0.0)

        # First sub-layer
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * 4,
                     kernel_init=ortho_init,
                     bias_init=zeros_init)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        # Second sub-layer
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim,
                     kernel_init=ortho_init,
                     bias_init=zeros_init)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        return x + residual


class MLPWithFiLM(nn.Module):
    """MLP with Feature-wise Linear Modulation (FiLM) conditioning.

    This network is designed for diffusion models and includes time embedding
    and conditioning support.

    Attributes:
        act_dim: Action dimension.
        Ta: Action sequence length.
        obs_dim: Observation dimension.
        To: Observation sequence length.
        emb_dim: Embedding dimension.
        n_layers: Number of residual blocks.
        timestep_emb_dim: Dimension of timestep embedding.
        max_freq: Maximum frequency for sinusoidal time embedding.
        disable_time_embedding: Whether to disable time embedding.
        dropout: Dropout rate.
    """
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
        """
        Args:
            x: (b, Ta, act_dim)
            s: (b, )
            t: (b, )
            condition: (b, To, obs_dim) or None.
            deterministic: If True, dropout is disabled.

        Returns:
            output_data: (b, Ta, act_dim)
            scalar_output: (b, 1)
        """
        x_flat = x.reshape(x.shape[0], -1)  # (b, Ta * act_dim)

        # Handle condition - use zeros if None
        if condition is not None:
            condition_flat = condition.reshape(condition.shape[0], -1) # (b, To * obs_dim)
        else:
            # Create zeros for missing condition
            batch_size = x.shape[0]
            condition_flat = jnp.zeros((batch_size, self.To * self.obs_dim), dtype=x.dtype)

        if not self.disable_time_embedding:
            s_embedded = self._embed_time(s)
            t_embedded = self._embed_time(t)
            input_data = jnp.concatenate([x_flat, s_embedded, t_embedded, condition_flat], axis=-1)
        else:
            # Skip time embeddings when disabled
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
    """Running Statistics Normalization (RSNorm) layer.

    This layer correctly handles state updates and distinguishes between
    training and evaluation modes.

    Attributes:
        epsilon: Small constant for numerical stability.
        momentum: Momentum for running statistics update.
    """
    epsilon: float = 1e-8
    momentum: float = 0.99

    @nn.compact
    def __call__(self, x, use_running_average: bool = False):
        """
        Applies running statistics normalization.

        Args:
            x: The input tensor.
            use_running_average: If True, uses stored running statistics for
                normalization (evaluation mode). If False, uses the current batch's
                statistics and updates the running statistics (training mode).
        """
        # Define variables for running statistics in the 'batch_stats' collection.
        running_mean = self.variable('batch_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        running_var = self.variable('batch_stats', 'var', lambda: jnp.ones(x.shape[-1]))

        if use_running_average:
            # In evaluation mode, use the stored running averages.
            mean = running_mean.value
            var = running_var.value
        else:
            # In training mode, use the statistics of the current batch.
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)

            # Update the running statistics if the 'batch_stats' collection is mutable.
            if self.is_mutable_collection('batch_stats'):
                running_mean.value = self.momentum * running_mean.value + (1 - self.momentum) * mean
                running_var.value = self.momentum * running_var.value + (1 - self.momentum) * var

        return (x - mean) / jnp.sqrt(var + self.epsilon)


class SimBaInternalMLP(nn.Module):
    """The internal MLP for a SimBa block with an inverted bottleneck.

    Attributes:
        hidden_dim: Hidden dimension.
        kernel_init: Kernel initializer.
    """
    hidden_dim: int
    kernel_init: Any = default_init

    @nn.compact
    def __call__(self, x):
        # Expands the hidden dimension to 4*d_h and applies ReLU
        intermediate_dim = self.hidden_dim * 4
        y = nn.Dense(intermediate_dim, kernel_init=self.kernel_init)(x)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_dim, kernel_init=self.kernel_init)(y)
        return y


class SimBaMLP(nn.Module):
    """SimBa architecture that accepts a `hidden_dims` sequence like a standard MLP.

    SimBa (Simple Baselines) is an architecture that uses running statistics normalization
    and residual feedforward blocks for improved performance.

    Attributes:
        hidden_dims: Sequence of layer dimensions. The last element is the output size.
                     All intermediate blocks will use the width of the first element.
        activations: Activation function for the final layer (if activate_final is True).
        activate_final: Whether to apply activation to the final output layer.
        kernel_init: Kernel initializer for dense layers.
        rs_norm_momentum: Momentum for the RSNorm layer.
        rs_norm_epsilon: Epsilon for the RSNorm layer.
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

        # The main body of the network is defined by all but the last dimension
        main_body_dims = self.hidden_dims[:-1]
        output_dim = self.hidden_dims[-1]

        # 1. Running Statistics Normalization on input
        x = RSNorm(
            momentum=self.rs_norm_momentum,
            epsilon=self.rs_norm_epsilon,
            name='RSNorm'
        )(x, use_running_average=not train)

        # If there are no hidden layers specified, just project to the output
        if not main_body_dims:
            x = nn.Dense(output_dim, kernel_init=self.kernel_init)(x)
            if self.activate_final:
                x = self.activations(x)
            return x

        # Use the first dimension as the constant hidden dimension for all blocks
        hidden_dim = main_body_dims[0]
        num_blocks = len(main_body_dims)

        # 2. Initial Linear Embedding Layer
        x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)

        # 3. Sequence of Residual Feedforward Blocks
        for i in range(num_blocks):
            residual = x
            x_norm = nn.LayerNorm(name=f'pre_ln_{i}')(x)
            mlp_output = SimBaInternalMLP(
                hidden_dim=hidden_dim,
                kernel_init=self.kernel_init
            )(x_norm)
            x = residual + mlp_output

        # 4. Final Post-Layer Normalization
        x = nn.LayerNorm(name='post_ln')(x)

        # Sow intermediates here, consistent with the original MLP's logic
        self.sow('intermediates', 'feature', x)

        # 5. Final projection to the output dimension
        x = nn.Dense(output_dim, kernel_init=self.kernel_init)(x)

        if self.activate_final:
            x = self.activations(x)

        return x
