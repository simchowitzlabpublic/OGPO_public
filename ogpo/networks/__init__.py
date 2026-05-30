"""JAX/Flax network architectures: actors, critics, encoders, and building blocks."""

from ogpo.networks.actors import (
    Actor,
    ActorVectorField,
    ActorVectorFieldTF,
    ActorVectorFieldSimBa,
    EditPolicy,
    EditActor,
)

from ogpo.networks.actors_fql import (
    OneStepPolicy,
)

from ogpo.networks.critics import (
    Value,
    ValueTF,
    ValueSimBa,
    ValueMIP,
    ValueMIPEnsemble,
    ensemblize,
)

from ogpo.networks.modules.misc import (
    NoiseInjectionNetwork,
    LogParam,
    Identity,
    orthogonal_init,
    zeros_init,
)

from ogpo.networks.encoders import (
    encoder_modules,
    ImpalaEncoder,
    MinViT,
    MinVitEncoder,
)

__all__ = [
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
    'EditPolicy',
    'EditActor',
    'OneStepPolicy',
    'Value',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
    'ensemblize',
    'NoiseInjectionNetwork',
    'LogParam',
    'Identity',
    'orthogonal_init',
    'zeros_init',
    'encoder_modules',
    'ImpalaEncoder',
    'MinViT',
    'MinVitEncoder',
]
