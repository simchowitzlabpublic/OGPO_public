# OGPO

**Off-Policy Generative Policy Optimization** — online RL fine-tuning of
flow-matching policies, with a suite of baseline algorithms, on Robomimic,
D4RL/Kitchen, Adroit, Push-T, and OGBench environments.

Policies are flow-matching generators rather than simple Gaussians, enabling
multimodal action distributions for manipulation. Every method follows the same
pipeline — **offline BC pretraining → online RL fine-tuning** — and differs in
how it does the RL part.

## Install

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# CUDA 12
uv sync --extra cuda12 --extra robomimic

# or CUDA 13
uv sync --extra cuda13 --extra robomimic
```

Drop `--extra robomimic` if you are not running the Robomimic environments
(square, transport, tool_hang). `make sync` is a shortcut for the CUDA 12 line.

## Quickstart

```bash
# OGPO on square (low-dim)
bash scripts/ogpo/square.sh

# Override any hyperparameter inline
bash scripts/ogpo/square.sh --seed=42 --online_steps=500000

# Other algorithms
bash scripts/bptt/square.sh
bash scripts/qc/square.sh
bash scripts/dsrl/square.sh
bash scripts/expo/square.sh
bash scripts/fql/square.sh
```

Runs log to Weights & Biases. Set `WANDB_API_KEY` in your environment to log
online; without it, W&B falls back to offline mode. Outputs are written under
`exp/` and checkpoints under `./checkpoints/` (override the latter with
`OGPO_CHECKPOINT_ROOT`).

## Algorithms

| Script dir | Method | Idea |
|------------|--------|------|
| `ogpo/` | **OGPO** | PPO-style policy gradient on SDE-fied flow policies via advantages computed using GRPO-style mean(Q) value subtraction over a group of sampled actions. |
| `bptt/` | **BPTT** | Differentiate Q through the ODE sampled actions into the actor. |
| `qc/` | **QC / ACFQL** | Action-chunked FQL; BON + SFT from Q. |
| `dsrl/` | **DSRL** | SAC in the initial noise space of a frozen pretrained flow policy. |
| `expo/` | **EXPO** | Learned Gaussian residuals on top of a frozen pretrained flow policy. |
| `dsrl_plus_expo/` | **DSRL+EXPO** | DSRL noise actor + EXPO edit layer. |
| `fql/` | **FQL** | OGPO with one-step distillation (`--agent.use_one_step_policy=true`). |
| `ogpo_awr/`, `ogpo_fpo/`, `mipq/` | OGPO ablations | Advantage-weighting (AWR, AW-OGPO, AW-OGPO-NN) / Offpolicy FPO/FPO++ (OFPO, OFPO++) / MIP Q variants via `--agent.*` flags. |

## Composing experiments

You rarely need to edit a script. Every config variable at the top of a script
is overridable from the command line, so experiments compose as argument
patterns:

```bash
bash scripts/ogpo/square.sh --seed=42 --actor_lr=4.5e-5 --adv_strategy=conservative \
    --run_group=square_conservative_adv
```

See [scripts/README.md](scripts/README.md) for the full override pattern, the
environment list, how to call `ogpo/main.py` directly, and **micro-tutorials for
wrapping these scripts for your own SLURM / SageMaker / container cluster**.

## Repository layout

```
ogpo/        core package — agents, networks, runners, configs, training entry point
envs/        environment + dataset wrappers (Robomimic, D4RL, Adroit, Push-T, OGBench)
scripts/     one runnable training script per (algorithm, environment)
```

Configuration defaults live in `ogpo/configs/algos/` (`common.yaml` + per-algo
overrides); the training entry point is `ogpo/main.py` (`ogpo-train`).

## Image observations (frozen VL encoder)

Image-based runs use a frozen PaliGemma (SigLIP vision tower + Gemma language
trunk) encoder loaded from a Pi0/Pi0.5 checkpoint. Point
`--checkpoint.restore_encoder_path` at your local checkpoint directory (the
`*_pg_libero.sh` / `*_image_paligemma.sh` scripts show the expected layout and a
`gcloud storage` download command). State defaults to robot proprioception only
(`--agent.use_state=proprio`).

## Citation

If you use this code, please cite:

```bibtex
@article{patil2026ogpo,
  title={OGPO: Sample Efficient Full-Finetuning of Generative Control Policies},
  author={Patil, Sarvesh and Nakamoto, Mitsuhiko and Agarwal, Manan and Saxena, Shashwat and Zhang, Jesse and Anantharaman, Giri and Winston, Cleah and Pan, Chaoyi and Chen, Douglas and Huang, Nai-Chieh and others},
  journal={arXiv preprint arXiv:2605.03065},
  year={2026}
}
```

## License

MIT (see `pyproject.toml`).
