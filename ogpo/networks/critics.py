"""Critic (Q-function / value) network architectures with ensemble and HL-Gauss distributional support."""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    AdaLNLayer,
    CrossAttnLayer,
    TwoTierObsEncoder,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Value(nn.Module):
    """MLP critic with optional FiLM conditioning, ensembling, and HL-Gauss distributional output."""
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    action_repeat: bool = False
    use_film: bool = False
    # Two-tier obs encoder (frozen-encoder image runs): see TwoTierObsEncoder.
    obs_two_tier: bool = False
    two_tier_img_dim: int = 0
    two_tier_proprio_dim: int = 0
    two_tier_fused_dim: int = 0  # if 0, defaults to hidden_dims[0]

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        if self.obs_two_tier:
            fused = self.two_tier_fused_dim if self.two_tier_fused_dim > 0 else self.hidden_dims[0]
            self.two_tier_encoder = TwoTierObsEncoder(
                img_feat_dim=self.two_tier_img_dim,
                proprio_dim=self.two_tier_proprio_dim,
                fused_dim=fused,
            )

        if self.use_film:
            mlp_class = MLPWithFiLM
        else:
            mlp_class = MLPCond if self.action_repeat else MLP

        if self.num_ensembles > 1:
            if self.use_film:
                # FiLM takes two inputs (x, cond)
                mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=(None, None), out_axes=0)
            else:
                mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.value_net = value_net

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        """Forward pass returning Q-values, optionally with HL-Gauss logits."""
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        if self.obs_two_tier:
            observations = self.two_tier_encoder(observations)

        if self.use_film:
            # FiLM: observations condition the network, actions are the input
            assert actions is not None, "FiLM critic requires actions"
            if self.critic_loss_type == 'hlgauss':
                q_logits = self.value_net(actions, observations)
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            else:
                v = self.value_net(actions, observations).squeeze(-1)
        else:
            inputs = [observations]
            if actions is not None and not self.action_repeat:
                inputs.append(actions)
            inputs = jnp.concatenate(inputs, axis=-1)

            if self.critic_loss_type == 'hlgauss':
                if self.action_repeat:
                    q_logits = self.value_net(inputs, actions)
                else:
                    q_logits = self.value_net(inputs)
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            elif self.critic_loss_type == 'mse':
                if self.action_repeat:
                    v = self.value_net(inputs, actions).squeeze(-1)
                else:
                    v = self.value_net(inputs).squeeze(-1)
            else:
                raise ValueError(f"Unknown critic loss type: {self.critic_loss_type}")

        return v


class ValueTF(nn.Module):
    """Transformer critic over sequential action chunks (bidirectional self-attention, no positional embeddings)."""
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    conditioning_type: str = 'adaln'  # 'adaln' or 'cross_attn'

    def setup(self):
        assert self.hidden_dim % self.num_heads == 0

        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        class TransformerCritic(nn.Module):
            hidden_dim: int
            num_layers: int
            num_heads: int
            dropout_rate: float
            num_output: int
            conditioning_type: str = 'adaln'

            @nn.compact
            def __call__(self, observations, actions, deterministic=True):
                x = nn.Dense(self.hidden_dim, kernel_init=default_init())(actions)
                cond = nn.Dense(self.hidden_dim, kernel_init=default_init())(observations)

                if self.conditioning_type == 'adaln':
                    for _ in range(self.num_layers):
                        x = AdaLNLayer(
                            self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                            self.dropout_rate
                        )(x, cond, deterministic=deterministic)
                else:  # cross_attn
                    # state-only conditioning wrapped as a single context token (B, 1, D)
                    context = cond[:, None, :]
                    for _ in range(self.num_layers):
                        x = CrossAttnLayer(
                            self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                            self.dropout_rate
                        )(x, context, deterministic=deterministic)

                # Pool over sequence: (B, T, hidden_dim) -> (B, hidden_dim)
                x = jnp.mean(x, axis=1)
                return nn.Dense(self.num_output, kernel_init=default_init())(x)

        self.value_net = ensemblize(TransformerCritic, self.num_ensembles,
            in_axes=(None, None),
            out_axes=0,
        )(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, num_output=num_output,
            conditioning_type=self.conditioning_type,
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, deterministic=True, images=None):
        """Forward pass through the transformer critic."""
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Reshape flat actions (B, T*a_dim) -> (B, T, a_dim)
        if actions.ndim == 2:
            actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)

        output = self.value_net(observations, actions, deterministic=deterministic)

        if self.critic_loss_type == 'hlgauss':
            q_logits = output
            v = jnp.sum(jax.nn.softmax(q_logits, axis=-1) * jnp.linspace(self.q_min, self.q_max, self.num_bins), axis=-1)
            if return_logits:
                return v, q_logits
        else:
            v = output.squeeze(-1)
        return v


class ValueSimBa(nn.Module):
    """Critic using the SimBa architecture (residual blocks + running-stat normalization)."""
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        mlp_class = SimBaMLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        self.value_net = mlp_class(
            hidden_dims=(*self.hidden_dims, num_output),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False,
                 is_encoded=False, images=None):
        """Forward pass returning Q-values, optionally with HL-Gauss logits."""
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)

        q_values = self.value_net(x)

        if self.critic_loss_type == 'hlgauss':
            if self.num_ensembles > 1:
                v_list = []
                for q_logits in q_values:
                    v = jnp.sum(
                        jax.nn.softmax(q_logits, axis=-1) *
                        jnp.linspace(self.q_min, self.q_max, self.num_bins),
                        axis=-1
                    )
                    v_list.append(v)
                v = jnp.stack(v_list, axis=0)
            else:
                v = jnp.sum(
                    jax.nn.softmax(q_values, axis=-1) *
                    jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1
                )

            if return_logits:
                return v, q_values
        else:
            v = q_values.squeeze(-1)

        return v


class ValueMIP(nn.Module):
    """MIP-parameterized Q-function with noise-based implicit ensembles.

    Uses a single network conditioned on random scalar noise z ~ Uniform[-u, u]
    to create diversity, in place of explicit separate ensemble networks.
    """
    hidden_dims: Sequence[int]
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None

    def setup(self):
        # Single MLP for both MIP steps: [obs, action, scalar_value, t] -> Q-value,
        # where scalar_value is noise z_0 at t=0 or the Q_0 prediction at t=t*.
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1
        self.mlp = MLP(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        """Forward pass returning Q-values.

        scalar_input is either noise z_0 (t=0) or the Q_0 prediction (t=t*); if None,
        noise is sampled internally from Uniform[-scale, scale] using rng.
        """
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Supports 2D (batch, dim) and 3D (batch, chunk, dim) action-chunking inputs.
        obs_shape = observations.shape
        batch_size = obs_shape[0]

        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            if scalar_input.shape[1] == 1:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)

        if self.critic_loss_type == 'hlgauss':
            q_logits = q_output
            q_values = jnp.sum(
                jax.nn.softmax(q_logits, axis=-1)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )
            if return_logits:
                return q_values, q_logits
        else:
            q_values = q_output.squeeze(-1)

        return q_values


class ValueMIPEnsemble(nn.Module):
    """Explicit ensemble of MIP-Q networks (separate params, shared noise per call).

    Diversity comes from independent network parameters rather than per-member noise.
    """
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        mlp_class = MLP
        # in_axes=0 maps over the leading ensemble dimension of the concatenated input.
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0, out_axes=0)

        self.mlp = mlp_class(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        """Forward pass returning Q-values [num_ensembles, B].

        scalar_input is [B, 1] (noise shared across members) or [B, num_ensembles, 1]
        (per-member q_0); if None, noise is sampled internally via rng.
        """
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Supports 2D (batch, dim) and 3D (batch, chunk, dim) action-chunking inputs.
        obs_shape = observations.shape
        batch_size = obs_shape[0]

        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            if scalar_input.ndim == 2 and scalar_input.shape[1] == 1:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        # Reshape all inputs to [num_ensembles, B, ...] for the vmapped MLP (in_axes=0).
        if scalar_input.ndim == 3:
            if len(obs_shape) == 3:
                # action chunking: [B, chunk, 1] -> [num_ensembles, B, chunk, 1]
                scalar_input = jnp.tile(scalar_input[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            elif scalar_input.shape[1] == self.num_ensembles:
                # per-member q_0: [B, num_ensembles, 1] -> [num_ensembles, B, 1]
                scalar_input = jnp.transpose(scalar_input, (1, 0, 2))
            else:
                raise ValueError(f"Unexpected scalar_input shape: {scalar_input.shape}")
        elif scalar_input.ndim == 2:
            # shared noise: [B, 1] -> [num_ensembles, B, 1]
            scalar_input = jnp.tile(scalar_input[None, :, :], (self.num_ensembles, 1, 1))
        else:
            raise ValueError(f"scalar_input must be 2D or 3D, got shape: {scalar_input.shape}")

        if observations.ndim == 2:
            observations = jnp.tile(observations[None, :, :], (self.num_ensembles, 1, 1))
            actions = jnp.tile(actions[None, :, :], (self.num_ensembles, 1, 1))
            time = jnp.tile(time[None, :, :], (self.num_ensembles, 1, 1))
        else:
            observations = jnp.tile(observations[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            actions = jnp.tile(actions[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            time = jnp.tile(time[None, :, :, :], (self.num_ensembles, 1, 1, 1))

        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)  # [num_ensembles, B, num_output]

        if self.critic_loss_type == 'hlgauss':
            q_logits = q_output  # [num_ensembles, B, num_bins]
            q_values = jnp.sum(
                jax.nn.softmax(q_logits, axis=-1)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )  # [num_ensembles, B]
            if return_logits:
                return q_values, q_logits
        else:
            q_values = q_output.squeeze(-1)  # [num_ensembles, B]

        return q_values


__all__ = [
    'default_init',
    'ensemblize',
    'Value',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
]
