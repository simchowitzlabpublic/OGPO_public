# Run Scripts

Each script here is a **single training job**: a self-contained config block
followed by one `python ogpo/main.py ...` call. They run locally as-is:

```bash
bash scripts/ogpo/square.sh
```

Layout is `scripts/<algorithm>/<env>.sh`:

| Dir | `--algo` | Notes |
|-----|----------|-------|
| `ogpo/` | `ogpo` | Main method. Also hosts image/value variants (`*_image.sh`, `*_value.sh`, `*_pg_libero.sh`, `*_image_paligemma.sh`). |
| `bptt/` | `bptt` | Backprop-through-time baseline. |
| `qc/` | `qc` | Action-chunked FQL (ACFQL). |
| `dsrl/` | `dsrl` | Noise-space SAC over a frozen flow. |
| `expo/` | `expo` | Expressive policy with a learned edit head. |
| `dsrl_plus_expo/` | `dsrl_plus_expo` | DSRL + EXPO edit layer. |
| `fql/` | `ogpo` | FQL = OGPO with `--agent.use_one_step_policy=true`. |
| `ogpo_awr/`, `ogpo_fpo/`, `mipq/` | `ogpo` | OGPO ablations toggled via `--agent.*` flags. |

---

## Anatomy of a script

Every script has three parts:

1. **Config block** — plain shell variables (`seed=1`, `online_steps=2000000`, …).
2. **CLI override parser** — turns any `--key=value` arg into an override of those
   variables (`eval "$key=\"$value\""`).
3. **The launch** — one `python ogpo/main.py` call mapping those variables onto
   the structured config namespaces (`--experiment.* --env.* --agent.* …`).

Outputs go to `SAVE_DIR` (defaults to `exp/`). Checkpoints default to
`./checkpoints/<algo>/<seed>_<env>`; override the root with
`OGPO_CHECKPOINT_ROOT=/path/to/scratch`.

---

## Composing experiments (override anything)

You do **not** need a new script per experiment. Every config variable is
overridable from the command line — compose them like building blocks:

```bash
# Change the seed
bash scripts/ogpo/square.sh --seed=42

# Longer run, different W&B grouping
bash scripts/ogpo/square.sh --seed=42 --online_steps=5000000 --run_group=square_long

# Swap the env (env-specific HPs like discount/horizon live in the script — copy
# the closest env's script and override, or pass them inline)
bash scripts/ogpo/transport.sh --seed=7

# Toggle an algorithmic ablation (these are just --agent.* flags)
bash scripts/ogpo/square.sh --chi_po=true --q_variance_reduction=true
bash scripts/ogpo/square.sh --adv_strategy=awr --awr_beta=10.0

# Sweep a hyperparameter from your shell
for s in 1 2 3; do
  for lr in 1e-4 3e-4 1e-3; do
    bash scripts/ogpo/square.sh --seed=$s --actor_lr=$lr --run_group=lr_sweep_$lr
  done
done
```

The override names are exactly the shell variable names at the top of each
script (`seed`, `actor_lr`, `discount`, `horizon_length`, `online_steps`,
`bc_pi_steps`, `chi_po`, `adv_strategy`, …). Open any script to see its full set.

### Calling `ogpo/main.py` directly

If you'd rather skip the shell wrapper, the underlying call uses dotted config
namespaces:

```bash
python ogpo/main.py \
  --algo=ogpo \
  --env.name=square-mh-low_dim \
  --experiment.seed=1 \
  --experiment.project=OGPO --experiment.run_group=square \
  --training.bc_pi_steps=500000 --training.online_steps=2000000 \
  --agent.actor_lr=3e-4 --agent.discount=0.99 --agent.horizon_length=4
```

Namespaces: `experiment`, `env`, `training`, `buffer`, `eval`, `intervals`,
`bc`, `success_buffer`, `agent`, `dataset`, `checkpoint`. Defaults live in
`ogpo/configs/algos/common.yaml` (+ per-algo YAML overrides).

### Environments

Low-dim: `square-mh-low_dim`, `transport-mh-low_dim`, `tool_hang-ph-low_dim`,
`kitchen-mixed-v2`, `pusht-keypoints-v0`, `AdroitHand*-v1`.
Image (frozen encoder): `square-mh-image`, `transport-mh-image`,
`tool_hang-ph-image` (see the `*_image*.sh` / `*_pg_libero.sh` scripts).

---

## Adapting to your cluster (SLURM / SageMaker / other)

These scripts are intentionally **cluster-agnostic** — no scheduler directives,
no site-specific paths. To run them on a managed cluster, wrap them. The
recipes below are written so you can paste them (and the script you want to run)
into an LLM coding assistant and have it generate the wrapper for your site.

### SLURM

A SLURM wrapper just prepends `#SBATCH` directives and (optionally) activates
your environment, then calls the script unchanged. Minimal pattern:

```bash
#!/bin/bash
#SBATCH --job-name=ogpo_square
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<N>
#SBATCH --mem=<MEM>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=<YOUR_PARTITION>     # site-specific
# #SBATCH --constraint=<YOUR_GPU_TYPE>   # if your site uses feature constraints

mkdir -p logs
# source <your-env-activation>           # conda/uv/venv as appropriate
bash scripts/ogpo/square.sh --seed=$SLURM_ARRAY_TASK_ID   # e.g. for array jobs
```

> Prompt you can hand to an assistant:
> *"Here is a training script (paste `scripts/ogpo/square.sh`). Write a SLURM
> sbatch wrapper that requests 1 GPU, 20 CPUs, 80G RAM, 48h, on partition X with
> GPU constraint Y, activates my uv venv, writes logs to `logs/`, and passes
> through any `--key=value` args. Make seed a job-array index."*

Fill in `<...>` with your site's partition names, GPU constraints, and account —
the OGPO scripts themselves need no changes.

### SageMaker / other managed "towers"

The scripts are plain processes, so any orchestrator that can run a container or
a shell command can run them. General recipe:

1. **Image**: build a CUDA 12/13 image, `uv sync --extra cuda12 --extra robomimic`
   (add your own entrypoint). Bake or mount datasets (set
   `ROBOMIMIC_DATASET_ROOT`) and, for image runs, the frozen encoder checkpoint.
2. **Entrypoint**: have the container run `bash scripts/<algo>/<env>.sh "$@"`,
   forwarding overrides.
3. **Secrets**: pass `WANDB_API_KEY` via your platform's secret mechanism (never
   commit it). With no key, training logs to W&B offline automatically.
4. **Multi-GPU**: set `--gpu_mode` / `--training.gpu_mode` per your device count
   (the scripts default to single-GPU).

> Prompt you can hand to an assistant:
> *"I have a training repo whose entrypoint is `bash scripts/<algo>/<env>.sh`.
> Write a `<SageMaker / Ray / k8s / …>` launcher that builds the image, mounts my
> dataset at `$ROBOMIMIC_DATASET_ROOT`, injects `WANDB_API_KEY` from a secret,
> requests <N> GPUs, and submits one job per (seed, env)."*

This keeps your cluster's account IDs, queue names, bucket names, and IAM roles
in **your** infra config — none of that belongs in this repo.
