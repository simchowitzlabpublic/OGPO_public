"""PaliGemma frozen VL encoder for OGPO (pure JAX/Flax).

Loads SigLIP So400m/14 vision tower + Gemma 2B language trunk from a Pi0/Pi0.5
checkpoint and uses them together as a frozen vision-language encoder. For each
sample we run:

    image -> SigLIP -> 256 image tokens (2048-D)
    text  -> Gemma embedder -> T text tokens (2048-D)
    [image_tokens, text_tokens] -> Gemma trunk (18 layers, MQA, RoPE)
    mean-pool over all (256+T) positions -> (B, 2048)

The text prompt is fixed per robomimic task (see ROBOMIMIC_PROMPTS). The
encoded vector is concatenated with proprio state to feed the policy/critic.

Default checkpoint is the base Pi0.5 model (pi05_base), which keeps the
PaliGemma SigLIP+Gemma trunk pretrained on the open-vocabulary corpus rather
than the LIBERO-task-finetuned variant.

Usage:
    encode_fn = load_paligemma_encoder(
        checkpoint_path="/data/.../pi05_base/params",
        env_name="square-mh-image",
    )
    encoded = encode_fn(observations, images)  # numpy in, numpy out
"""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


# Per-task text instructions, keyed by env name prefix (env_name.split("-")[0]).
ROBOMIMIC_PROMPTS = {
    "lift": "pick up the small cube from the table",
    "can": "pick up the coke can and place it in the bin",
    "square": "pick up the square nut and insert it into the peg",
    "transport": "unlid the source bin with the hammer and move the red object from the target bin to the other bin. Then transport the hammer from the source bin to the target bin",
    "tool_hang": "pick up the needle, insert into the hole, and then hang the wrench on the hook",
    "toolhang": "pick up the needle, insert into the hole, and then hang the wrench on the hook",
    "pusht": "push the T-shaped block to the target",
}


def _prompt_for_env(env_name: str) -> str:
    task = env_name.split("-")[0]
    if task not in ROBOMIMIC_PROMPTS:
        raise KeyError(
            f"No PaliGemma prompt registered for task '{task}' (env '{env_name}'). "
            f"Add an entry to ROBOMIMIC_PROMPTS in {__file__}."
        )
    return ROBOMIMIC_PROMPTS[task]


# SigLIP architecture (hand-port of openpi siglip.py).

def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    y, x = jnp.mgrid[:h, :w]
    assert width % 4 == 0
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }
        _, _, d = x.shape
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class SigLIPEncoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class SigLIPViT(nn.Module):
    """SigLIP So400m/14 Vision Transformer."""
    num_classes: int | None = None
    patch_size: tuple = (14, 14)
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    posemb: str = "learn"
    pool_type: str = "none"
    scan: bool = True
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        image = jnp.asarray(image, jnp.float32)

        x = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        x = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        n, _, c = x.shape
        x = x.astype(self.dtype_mm)

        x, _ = SigLIPEncoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            scan=self.scan,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)

        if self.num_classes:
            x = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head")(x)

        return x  # (B, 256, num_classes)


# Gemma 2B language trunk (Flax port keyed to the Pi0/Pi0.5 checkpoint layout
# under params/PaliGemma/llm; identical between pi05_base and pi05_libero).
#
# Param-tree contract (after stripping the orbax /value suffix):
#   embedder/input_embedding              (vocab, hidden)
#   final_norm/scale                       (hidden,)
#   layers/attn/q_einsum/w                 (depth, H, hidden, head_dim)
#   layers/attn/kv_einsum/w                (depth, 2, num_kv, hidden, head_dim)
#   layers/attn/attn_vec_einsum/w          (depth, H, head_dim, hidden)
#   layers/mlp/gating_einsum               (depth, 2, hidden, mlp_hidden)
#   layers/mlp/linear                       (depth, mlp_hidden, hidden)
#   layers/pre_attention_norm/scale        (depth, hidden)
#   layers/pre_ffw_norm/scale              (depth, hidden)
# (the *_1 / final_norm_1 keys belong to the Pi0.5 action expert; ignored.)

GEMMA_CONFIG = dict(
    depth=18,
    hidden=2048,
    num_heads=8,
    num_kv_heads=1,
    head_dim=256,
    mlp_hidden=16384,
    vocab=257152,
    rope_theta=10_000.0,
    rms_eps=1e-6,
)


class _ParamHolder(nn.Module):
    """Tiny wrapper so a param key path can include an extra level (e.g. `q_einsum/w`)."""
    shape: tuple
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self):
        return self.param("w", nn.initializers.zeros, self.shape, self.dtype)


class GemmaRMSNorm(nn.Module):
    """Gemma RMSNorm: y = x * rsqrt(mean(x**2) + eps) * (1 + scale)."""
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.zeros, (self.dim,), jnp.float32)
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x.astype(jnp.float32) * jax.lax.rsqrt(var + self.eps)
        return (normed * (1.0 + scale)).astype(x.dtype)


def _apply_rope(x: jnp.ndarray, theta: float) -> jnp.ndarray:
    """Apply rotary positional embedding along the last axis.

    x: (B, T, H, D). Positions are implicit 0..T-1.
    Uses the "split-half" RoPE convention: rotate [d0..d/2-1] against [d/2..d-1].
    """
    T = x.shape[1]
    head_dim = x.shape[-1]
    half = head_dim // 2
    freq_idx = jnp.arange(half, dtype=jnp.float32)
    inv_freq = theta ** (-freq_idx / half)
    positions = jnp.arange(T, dtype=jnp.float32)
    angles = positions[:, None] * inv_freq[None, :]                # (T, half)
    sin = jnp.sin(angles)[None, :, None, :].astype(x.dtype)        # (1, T, 1, half)
    cos = jnp.cos(angles)[None, :, None, :].astype(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return jnp.concatenate([rot1, rot2], axis=-1)


class GemmaAttention(nn.Module):
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float = 10_000.0

    @nn.compact
    def __call__(self, x):
        w_q = _ParamHolder(
            shape=(self.num_heads, self.hidden, self.head_dim), name="q_einsum"
        )()
        q = jnp.einsum("btd,hde->bthe", x, w_q)

        # kv_einsum w shape (2, num_kv, D, head_dim); index 0=K, 1=V.
        w_kv = _ParamHolder(
            shape=(2, self.num_kv_heads, self.hidden, self.head_dim), name="kv_einsum"
        )()
        kv = jnp.einsum("btd,sNde->sbtNe", x, w_kv)
        k, v = kv[0], kv[1]  # each (B, T, num_kv, head_dim)

        # RoPE positions 0..T-1 are implicit in _apply_rope.
        q = _apply_rope(q, self.rope_theta)
        k = _apply_rope(k, self.rope_theta)

        q = q * (self.head_dim ** -0.5)

        # MQA / GQA broadcast: repeat K,V from num_kv to num_heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        # Full bidirectional attention over the (image+text) prefix.
        attn = jnp.einsum("bthd,bshd->bhts", q, k)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhts,bshd->bthd", attn, v)  # (B, T, H, head_dim)

        w_o = _ParamHolder(
            shape=(self.num_heads, self.head_dim, self.hidden), name="attn_vec_einsum"
        )()
        return jnp.einsum("bthe,hed->btd", out, w_o)


class GemmaMLP(nn.Module):
    hidden: int
    mlp_hidden: int

    @nn.compact
    def __call__(self, x):
        # gating_einsum shape (2, hidden, mlp_hidden); index 0=gate, 1=up.
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.zeros,
            (2, self.hidden, self.mlp_hidden),
            jnp.float32,
        )
        gate = jnp.einsum("btd,di->bti", x, w_gating[0])
        up = jnp.einsum("btd,di->bti", x, w_gating[1])
        # Gemma uses GeGLU with tanh-approx GELU.
        hidden = nn.gelu(gate, approximate=True) * up
        w_down = self.param(
            "linear",
            nn.initializers.zeros,
            (self.mlp_hidden, self.hidden),
            jnp.float32,
        )
        return jnp.einsum("bti,id->btd", hidden, w_down)


class GemmaBlock(nn.Module):
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    mlp_hidden: int
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        h = GemmaRMSNorm(self.hidden, self.rms_eps, name="pre_attention_norm")(x)
        h = GemmaAttention(
            hidden=self.hidden,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            name="attn",
        )(h)
        x = x + h

        h = GemmaRMSNorm(self.hidden, self.rms_eps, name="pre_ffw_norm")(x)
        h = GemmaMLP(hidden=self.hidden, mlp_hidden=self.mlp_hidden, name="mlp")(h)
        return x + h, None  # (carry, per-iter ys) for nn.scan


class GemmaTrunk(nn.Module):
    """Stack of GemmaBlocks via nn.scan, matching the checkpoint's depth-stacked layout."""
    depth: int = 18
    hidden: int = 2048
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    mlp_hidden: int = 16384
    vocab: int = 257152
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6

    @nn.compact
    def __call__(self, input_embeds):
        ScanBlock = nn.scan(
            GemmaBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=self.depth,
        )
        x, _ = ScanBlock(
            hidden=self.hidden,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            mlp_hidden=self.mlp_hidden,
            rope_theta=self.rope_theta,
            rms_eps=self.rms_eps,
            name="layers",
        )(input_embeds)
        return GemmaRMSNorm(self.hidden, self.rms_eps, name="final_norm")(x)


# Separate sub-Module so its param key is `embedder/input_embedding`,
# matching the checkpoint layout.
class GemmaEmbedder(nn.Module):
    vocab: int
    hidden: int

    @nn.compact
    def __call__(self, token_ids):
        emb = self.param(
            "input_embedding",
            nn.initializers.zeros,
            (self.vocab, self.hidden),
            jnp.float32,
        )
        # Gemma scales embeddings by sqrt(hidden).
        return emb[token_ids] * jnp.sqrt(self.hidden).astype(jnp.float32)


def preprocess_images(images, target_size=224):
    """Resize and normalize (B, H, W, C) images for SigLIP, splitting multi-camera channels.

    Returns (B * num_cams, target_size, target_size, 3) float32 in [-1, 1].
    C must be a multiple of 3.
    """
    from PIL import Image as PILImage

    images = np.asarray(images)
    B, H, W, C = images.shape
    if C % 3 != 0:
        raise ValueError(f"preprocess_images expects C divisible by 3 (got C={C})")
    num_cams = C // 3

    if num_cams > 1:
        images = images.reshape(B, H, W, num_cams, 3)
        images = images.transpose(0, 3, 1, 2, 4)
        images = images.reshape(B * num_cams, H, W, 3)

    if H != target_size or W != target_size:
        imgs_uint8 = (
            images if images.dtype == np.uint8
            else np.clip(images, 0, 255).astype(np.uint8)
        )
        n_imgs = imgs_uint8.shape[0]
        resized = np.empty((n_imgs, target_size, target_size, 3), dtype=np.uint8)
        for i in range(n_imgs):
            resized[i] = np.array(
                PILImage.fromarray(imgs_uint8[i]).resize(
                    (target_size, target_size), PILImage.BILINEAR
                )
            )
        images = resized

    images = images.astype(np.float32) / 255.0
    images = (images - 0.5) / 0.5
    return images


def _load_paligemma_params(checkpoint_path):
    """Load img + llm subtrees from a Pi0/Pi0.5 orbax checkpoint.

    Returns (img_params, llm_params), each with the orbax /value suffix stripped
    so the keys line up with the Flax module names declared above.
    """
    import orbax.checkpoint as ocp
    import gc
    from flax import traverse_util

    print(f"[PaliGemma] Loading checkpoint: {checkpoint_path}")
    ckptr = ocp.PyTreeCheckpointer()

    metadata = ckptr.metadata(checkpoint_path)
    for attr in ("item_metadata", "metadata", "tree"):
        inner = getattr(metadata, attr, None)
        if inner is not None:
            metadata = inner
            break

    cpu_device = jax.devices("cpu")[0]
    cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)

    def _is_array_leaf(x):
        return hasattr(x, "shape") and hasattr(x, "dtype")

    def _to_abstract(leaf):
        if _is_array_leaf(leaf):
            return jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=cpu_sharding
            )
        return leaf

    target = jax.tree.map(_to_abstract, metadata, is_leaf=_is_array_leaf)
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    result = ckptr.restore(checkpoint_path, restore_args=restore_args)

    img_params = result["params"]["PaliGemma"]["img"]
    llm_params = result["params"]["PaliGemma"]["llm"]

    # Drop everything else (action expert, etc.) so it gets GC'd.
    del result, target, restore_args, metadata
    gc.collect()

    # Strip the orbax NNX "value" suffix from both subtrees.
    def _strip_value(params):
        flat = traverse_util.flatten_dict(params)
        if all(kp[-1] == "value" for kp in flat):
            flat = {kp[:-1]: v for kp, v in flat.items()}
            params = traverse_util.unflatten_dict(flat)
        return params

    img_params = _strip_value(img_params)
    llm_params = _strip_value(llm_params)

    # Drop the Pi0.5 action-expert keys inside `llm` (everything with a `_1` suffix
    # plus `final_norm_1`). We only want the main VL trunk.
    def _drop_action_expert(params):
        flat = traverse_util.flatten_dict(params)
        kept = {}
        for k, v in flat.items():
            if any(seg.endswith("_1") for seg in k):
                continue
            kept[k] = v
        return traverse_util.unflatten_dict(kept)

    llm_params = _drop_action_expert(llm_params)
    return img_params, llm_params


# Optional baked-in location for the PaliGemma SentencePiece tokenizer. If you
# run in a container or air-gapped environment, place a copy of the Gemma
# tokenizer (`unsloth/gemma-2-2b`'s non-gated `tokenizer.model`, identical IDs
# to `gs://big_vision/paligemma_tokenizer.model` for natural-language prompts)
# here so the language trunk can tokenize task prompts without runtime HF access.
_DOCKER_TOKENIZER_PATH = "/opt/paligemma_tokenizer/tokenizer.model"


def _load_paligemma_tokenizer(tokenizer_path: str | None):
    """Load a Gemma-compatible tokenizer.

    PaliGemma's full vocab (257152) extends Gemma's (256000) with special
    `<image>`/`<loc>`/`<seg>` tokens we don't use for natural-language prompts,
    so any Gemma SentencePiece tokenizer produces the right ids for our text.

    Resolution order:
      1. Explicit `tokenizer_path` argument (str), if provided.
      2. `$PALIGEMMA_TOKENIZER_PATH` env var (handy for containers).
      3. Baked-in path `/opt/paligemma_tokenizer/tokenizer.model` if present.
      4. HF fallback: `unsloth/gemma-2-2b` (non-gated), then gated `google/*`.

    Accepted forms for any of the above:
      - HF repo id or local snapshot dir (anything `AutoTokenizer.from_pretrained` accepts)
      - Path to a raw `tokenizer.model` SentencePiece file
      - Directory containing a `tokenizer.model` file
    """
    # 1-3) Resolve a local/baked tokenizer first so training never relies on
    #      runtime HF outbound access.
    resolved = tokenizer_path
    if not resolved:
        env_path = os.environ.get("PALIGEMMA_TOKENIZER_PATH")
        if env_path:
            resolved = env_path
    if not resolved and os.path.exists(_DOCKER_TOKENIZER_PATH):
        resolved = _DOCKER_TOKENIZER_PATH

    # Accept either a `.model` file directly or a dir containing one.
    if resolved and os.path.isdir(resolved):
        candidate = os.path.join(resolved, "tokenizer.model")
        if os.path.exists(candidate):
            resolved = candidate

    if resolved and resolved.endswith(".model") and os.path.exists(resolved):
        import sentencepiece as spm
        print(f"[PaliGemma] Loading SentencePiece tokenizer: {resolved}")
        sp = spm.SentencePieceProcessor()
        sp.Load(resolved)

        class _SpmAdapter:
            """Minimal HF-tokenizer-like interface (only what _tokenize_prompt needs)."""
            def __init__(self, sp):
                self.sp = sp

            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = self.sp.EncodeAsIds(text)
                if add_special_tokens:
                    bos = self.sp.bos_id()
                    if bos >= 0:
                        ids = [bos] + ids
                arr = np.asarray(ids, dtype=np.int32)[None]
                return {"input_ids": arr}

        return _SpmAdapter(sp)

    from transformers import AutoTokenizer

    # Default fallback list. `unsloth/gemma-2-2b` is a non-gated mirror of
    # Gemma's SentencePiece vocab (256000 tokens) — identical IDs to PaliGemma
    # for plain natural-language prompts; PaliGemma's extra 1152 special tokens
    # (`<image>`/`<loc>`/`<seg>`) are never produced from our task instructions.
    # The two `google/*` repos are gated and used only when HF auth is set.
    candidates = [resolved] if resolved else [
        "unsloth/gemma-2-2b",
        "google/paligemma-3b-pt-224",
        "google/gemma-2-2b-it",
    ]
    last_err = None
    for name in candidates:
        if not name:
            continue
        try:
            print(f"[PaliGemma] Loading tokenizer: {name}")
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[PaliGemma]   failed ({type(e).__name__}); trying next candidate")
    raise RuntimeError(
        "Could not load any PaliGemma-compatible tokenizer. Pass `tokenizer_path=` "
        "as a local snapshot dir or a tokenizer.model SentencePiece file, set "
        "PALIGEMMA_TOKENIZER_PATH, or bake one into the image at "
        f"{_DOCKER_TOKENIZER_PATH}."
    ) from last_err


def _tokenize_prompt(tokenizer, prompt: str, max_len: int = 64) -> np.ndarray:
    """Tokenize a fixed task prompt to a 1-D int32 array.

    Uses the underlying SentencePiece tokenizer with a leading BOS and a trailing
    newline, mirroring PaliGemma's prefix-LM format. Padded/truncated to `max_len`.
    Truncation prints a warning — silent truncation drops words from the task
    instruction.
    """
    text = f"{prompt}\n"
    ids = tokenizer(text, return_tensors="np", add_special_tokens=True)["input_ids"][0]
    ids = np.asarray(ids, dtype=np.int32)
    if ids.shape[0] > max_len:
        print(
            f"[PaliGemma] WARNING: prompt tokenized to {ids.shape[0]} ids but "
            f"prompt_max_len={max_len}; truncating. Bump prompt_max_len to keep "
            f"the full instruction."
        )
        ids = ids[:max_len]
    elif ids.shape[0] < max_len:
        pad = np.zeros(max_len - ids.shape[0], dtype=np.int32)
        ids = np.concatenate([ids, pad])
    return ids


def load_paligemma_encoder(
    checkpoint_path: str = os.path.expanduser("~/checkpoints/pi05_base/params"),
    env_name: str | None = None,
    img_proj_dim: int = 0,
    state_proj_dim: int = 0,
    target_size: int = 224,
    encoder_device: int | None = None,
    tokenizer_path: str | None = None,
    prompt_max_len: int = 64,
):
    """Load PaliGemma SigLIP + Gemma LM and return a frozen VL encode function.

    The returned encode_fn(observations, images) yields a numpy array of shape
    (B, num_cams * 2048 + state_dim). `env_name` selects the task prompt from
    ROBOMIMIC_PROMPTS.
    """
    if env_name is None:
        raise ValueError(
            "load_paligemma_encoder now requires env_name to look up the task prompt. "
            "Pass it from main.py (env_config['name'])."
        )
    prompt = _prompt_for_env(env_name)
    print(f"[PaliGemma] Task prompt for '{env_name}': {prompt!r}")

    img_params, llm_params = _load_paligemma_params(checkpoint_path)

    # Tokenize the (fixed) prompt once on host.
    tokenizer = _load_paligemma_tokenizer(tokenizer_path)
    text_ids_np = _tokenize_prompt(tokenizer, prompt, max_len=prompt_max_len)
    print(f"[PaliGemma] Prompt token ids (len={len(text_ids_np)}): {text_ids_np.tolist()}")

    if encoder_device is not None:
        devices = jax.devices("gpu")
        if encoder_device < len(devices):
            enc_device = devices[encoder_device]
            print(f"[PaliGemma] Using GPU {encoder_device} for encoding")
        else:
            enc_device = jax.devices()[0]
            print(f"[PaliGemma] GPU {encoder_device} not available, using default device")
    else:
        enc_device = jax.devices()[0]

    img_params = jax.device_put(img_params, enc_device)
    llm_params = jax.device_put(llm_params, enc_device)
    text_ids_jax = jax.device_put(jnp.asarray(text_ids_np), enc_device)

    siglip = SigLIPViT(
        num_classes=2048,
        patch_size=(14, 14),
        width=1152,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        posemb="learn",
        pool_type="none",
        scan=True,
        dtype_mm="float32",
    )
    embedder = GemmaEmbedder(vocab=GEMMA_CONFIG["vocab"], hidden=GEMMA_CONFIG["hidden"])
    trunk = GemmaTrunk(
        depth=GEMMA_CONFIG["depth"],
        hidden=GEMMA_CONFIG["hidden"],
        num_heads=GEMMA_CONFIG["num_heads"],
        num_kv_heads=GEMMA_CONFIG["num_kv_heads"],
        head_dim=GEMMA_CONFIG["head_dim"],
        mlp_hidden=GEMMA_CONFIG["mlp_hidden"],
        vocab=GEMMA_CONFIG["vocab"],
        rope_theta=GEMMA_CONFIG["rope_theta"],
        rms_eps=GEMMA_CONFIG["rms_eps"],
    )

    siglip_params = {"params": img_params}
    embedder_params = {"params": llm_params["embedder"]}
    # The trunk uses the rest of `llm` minus the embedder.
    trunk_params_raw = {k: v for k, v in llm_params.items() if k != "embedder"}
    trunk_params = {"params": trunk_params_raw}

    @functools.partial(jax.jit, device=enc_device)
    def _encode_vl(siglip_p, emb_p, trunk_p, pixel_values, text_ids):
        img_tokens = siglip.apply(siglip_p, pixel_values, train=False)  # (B*ncams, 256, 2048)

        text_emb = embedder.apply(emb_p, text_ids)  # (T, 2048)
        text_emb = jnp.broadcast_to(
            text_emb[None], (img_tokens.shape[0], text_emb.shape[0], text_emb.shape[1])
        )

        seq = jnp.concatenate([img_tokens, text_emb], axis=1)  # (B*ncams, 256+T, 2048)
        seq = trunk.apply(trunk_p, seq)
        return seq.mean(axis=1)  # mean-pool image+text -> (B*ncams, 2048)

    dummy_img = jax.device_put(jnp.zeros((1, target_size, target_size, 3)), enc_device)
    try:
        out = _encode_vl(siglip_params, embedder_params, trunk_params, dummy_img, text_ids_jax)
        output_dim = int(out.shape[-1])
        print(f"[PaliGemma] VL output: {out.shape} (after mean-pool over image+text)")
    except Exception as e:
        print(f"[PaliGemma] Warning: shape verification failed: {e}")
        output_dim = GEMMA_CONFIG["hidden"]

    print(f"[PaliGemma] Final encoded dim per camera: {output_dim} (+ proprio at end)")

    def encode_fn(observations, images):
        """Encode (images, proprio) -> flat VL feature vector.

        Multi-camera (C = 3·num_cams) is run per-camera and the per-camera
        features are concatenated along the feature axis, then proprio appended.
        """
        squeeze = images.ndim == 3
        if squeeze:
            images = images[np.newaxis]
            observations = observations[np.newaxis]
        B = images.shape[0]
        num_cams = images.shape[-1] // 3

        pixel_values = preprocess_images(images, target_size=target_size)
        pixel_jax = jax.device_put(jnp.asarray(pixel_values), enc_device)

        feats = np.asarray(
            _encode_vl(siglip_params, embedder_params, trunk_params, pixel_jax, text_ids_jax)
        )  # (B*ncams, hidden)

        if num_cams > 1:
            feats = feats.reshape(B, num_cams * feats.shape[-1])

        encoded = np.concatenate([feats, observations], axis=-1).astype(np.float32)
        if squeeze:
            encoded = encoded[0]
        return encoded

    return encode_fn
