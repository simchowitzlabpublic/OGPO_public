"""Vision encoders for the OGPO network architecture.

This package provides various vision encoder architectures including:
- IMPALA-style convolutional encoders (ResNet-based)
- MinViT (Minimal Vision Transformer) encoder
"""

import functools as ft

from ogpo.networks.encoders.impala import (
    ImpalaEncoder,
    ResnetStack,
    Encoder,
    encoder_modules as impala_encoder_modules,
)
from ogpo.networks.encoders.minvit import (
    MinViT,
    MinVitEncoder,
    normalize_images,
)
from ogpo.networks.encoders.vision_proprio import VisionProprioEncoder
from ogpo.networks.encoders.paligemma import load_paligemma_encoder


# Consolidated encoder modules registry
encoder_modules = {
    # IMPALA encoders
    'impala': ImpalaEncoder,
    'impala_debug': impala_encoder_modules['impala_debug'],
    'impala_small': impala_encoder_modules['impala_small'],
    'impala_large': impala_encoder_modules['impala_large'],
    'small': Encoder,

    # MinViT encoder with default configuration
    'minvit': ft.partial(
        MinViT,
        embed_style="embed2",
        embed_dim=128,
        embed_norm=0,  # False
        num_heads=8,
        depth=3,
        dropout=0.0,
        num_channel=3,
        img_h=96,
        img_w=96,
        flatten=True,
    ),
}


# ViT encoder modules registry (separate from CNN-based encoders)
vit_encoder_modules = {
    'minvit': ft.partial(
        MinViT,
        embed_style="embed2",
        embed_dim=128,
        embed_norm=0,  # False
        num_heads=8,
        depth=3,
        dropout=0.0,
        num_channel=3,
        img_h=96,
        img_w=96,
        flatten=True,
    ),
    'minvit_small': ft.partial(
        MinViT,
        embed_style="embed2",
        embed_dim=64,
        embed_norm=0,
        num_heads=4,
        depth=2,
        dropout=0.0,
        num_channel=3,
        img_h=96,
        img_w=96,
        flatten=True,
    ),
    'minvit_large': ft.partial(
        MinViT,
        embed_style="embed2",
        embed_dim=256,
        embed_norm=0,
        num_heads=8,
        depth=6,
        dropout=0.0,
        num_channel=3,
        img_h=96,
        img_w=96,
        flatten=True,
    ),
}


__all__ = [
    # IMPALA encoders
    'ImpalaEncoder',
    'ResnetStack',
    'Encoder',

    # MinViT encoder
    'MinViT',
    'MinVitEncoder',
    'normalize_images',

    # Vision + Proprio encoder
    'VisionProprioEncoder',

    # PaliGemma frozen encoder
    'load_paligemma_encoder',

    # Encoder registries
    'encoder_modules',
    'vit_encoder_modules',
]
