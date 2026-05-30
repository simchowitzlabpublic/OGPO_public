import jax
import jax.numpy as jnp
import functools
from functools import partial


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image."""
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


def rgb_to_hsv(r, g, b):
    """Convert R, G, B values to H, S, V values.

    Only input values in [0, 1] are guaranteed to work properly.
    """
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.)
    norm = jnp.where(range_ != 0, 1. / (6. * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2. / 6.
    hb = norm * (r - g) + 4. / 6.

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)

    return hue, sat, vv


def hsv_to_rgb(h, s, v):
    """Convert H, S, V values to an (R, G, B) tuple.

    Only input values in [0, 1] are guaranteed to work properly.
    """
    c = s * v
    m = v - c
    dh = (h % 1.) * 6.
    fmodu = dh % 2.
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.int32)
    rr = jnp.where(
        (hcat == 0) | (hcat == 5), c, jnp.where(
            (hcat == 1) | (hcat == 4), x, 0)) + m
    gg = jnp.where(
        (hcat == 1) | (hcat == 2), c, jnp.where(
            (hcat == 0) | (hcat == 3), x, 0)) + m
    bb = jnp.where(
        (hcat == 3) | (hcat == 4), c, jnp.where(
            (hcat == 2) | (hcat == 5), x, 0)) + m
    return rr, gg, bb


def adjust_brightness(rgb_tuple, delta):
    return jax.tree_util.tree_map(lambda x: x + delta, rgb_tuple)


def adjust_contrast(image, factor):
    def _adjust_contrast_channel(channel):
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return factor * (channel - mean) + mean
    return jax.tree_util.tree_map(_adjust_contrast_channel, image)


def adjust_saturation(h, s, v, factor):
    return h, jnp.clip(s * factor, 0., 1.), v


def adjust_hue(h, s, v, delta):
    # Note: this method exactly matches TF"s adjust_hue (combined with the hsv/rgb
    # conversions) when running on GPU. When running on CPU, the results will be
    # different if all RGB values for a pixel are outside of the [0, 1] range.
    return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, rng, max_delta):
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, rng, max_delta):
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
    grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[..., jnp.newaxis]
    return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.


def _color_transform_single_image(image, rng, brightness, contrast, saturation,
                                  hue, to_grayscale_prob, color_jitter_prob,
                                  apply_prob, shuffle):
    """Applies color jittering to a single image."""
    apply_rng, transform_rng = jax.random.split(rng)
    perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
        transform_rng, 7)

    should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
    should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
    should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

    def _make_cond(fn, idx):

        def identity_fn(x, unused_rng, unused_param):
            return x

        def cond_fn(args, i):
            def clip(args):
                return jax.tree_util.tree_map(lambda arg: jnp.clip(arg, 0., 1.), args)
            out = jax.lax.cond(should_apply & should_apply_color & (i == idx), args,
                         lambda a: clip(fn(*a)), args,
                         lambda a: identity_fn(*a))
            return jax.lax.stop_gradient(out)

        return cond_fn

    random_brightness_cond = _make_cond(_random_brightness, idx=0)
    random_contrast_cond = _make_cond(_random_contrast, idx=1)
    random_saturation_cond = _make_cond(_random_saturation, idx=2)
    random_hue_cond = _make_cond(_random_hue, idx=3)

    def _color_jitter(x):
        rgb_tuple = tuple(jax.tree_util.tree_map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
        if shuffle:
            order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
        else:
            order = range(4)
        for idx in order:
            if brightness > 0:
                rgb_tuple = random_brightness_cond((rgb_tuple, b_rng, brightness), idx)
            if contrast > 0:
                rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
            if saturation > 0:
                rgb_tuple = random_saturation_cond((rgb_tuple, s_rng, saturation), idx)
            if hue > 0:
                rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
        return jnp.stack(rgb_tuple, axis=-1)

    out_apply = _color_jitter(image)
    out_apply = jax.lax.cond(should_apply & should_apply_gs, out_apply,
                           _to_grayscale, out_apply, lambda x: x)
    return jnp.clip(out_apply, 0., 1.)


def _random_flip_single_image(image, rng):
    _, flip_rng = jax.random.split(rng)
    should_flip_lr = jax.random.uniform(flip_rng, shape=()) <= 0.5
    image = jax.lax.cond(should_flip_lr, image, jnp.fliplr, image, lambda x: x)
    return image


def random_flip(images, rng):
    rngs = jax.random.split(rng, images.shape[0])
    return jax.vmap(_random_flip_single_image)(images, rngs)


@partial(jax.jit, static_argnames=('brightness', 'contrast', 'saturation', 'hue', 'color_jitter_prob', 'to_grayscale_prob', 'apply_prob', 'shuffle'))
def color_transform(rng,            
                    images,
                    brightness=0.2,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.03,
                    color_jitter_prob=0.8,
                    to_grayscale_prob=0.0,
                    apply_prob=1.0,
                    shuffle=True):
    """Apply color jittering and/or grayscaling to a batch of NHWC images (C=3)."""
    rngs = jax.random.split(rng, images.shape[0])
    jitter_fn = functools.partial(
        _color_transform_single_image,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        color_jitter_prob=color_jitter_prob,
        to_grayscale_prob=to_grayscale_prob,
        apply_prob=apply_prob,
        shuffle=shuffle)
    augmented_images = jax.vmap(jitter_fn)(images, rngs)
    return augmented_images


