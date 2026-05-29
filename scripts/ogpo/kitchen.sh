#!/bin/bash

set -e

export MUJOCO_GL=egl
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# === Config ===
seed=49
device=0
jax_frac=0.45
env_name=mixed
post_success_steps=0
run_name="ogpo_kitchen"
wandb_name="ogpo_kitchen"
project="OGPO"
run_group="kitchen"

# Training phases
bc_pi_steps=500000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=2000000
gpu_mode=single

# Agent config
horizon_length=4
discount=0.99
tau=0.05
actor_tau=0.005
best_of_n=8
num_qs=10
q_agg=subsample
subsample_bon=true
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal       # scalar | sinusoidal
time_embedding_dim=32
bc_coeff=1.0
adv_strategy=vanilla
awr_beta=10

# UTD and buffer
utd_warmup=1
utd_online=1
offline_ratio=0.0
start_training=10000

# Evaluation
n_eval_envs=32
eval_episodes=2
log_interval=5000
eval_interval=20000
eval_interval_bc=20000
save_interval=500000

# BC settings
clip_bc=false
clip_bc_threshold=0.45
clip_bc_wrt=SDE                 # low_dim OGPO legacy: gate BC early-stop on SDE eval

# Success buffer
use_success_buffer=true
use_success_buffer_q=false

# Architecture backbone (mlp or tf, independently for actor/critic)
actor_backbone=mlp
critic_backbone=mlp

# OGPO-specific
ppo_batch_size=256
clip_epsilon=0.01
entropy_coeff=0.0
group_num_samples=32
use_bc_regularization=true
# Legacy denoiser-trained correction during ODE sampling (FPO path); off by default.
error_correct_ode_to_sde=false
use_denoiser=false
# Noise schedule: tapered (sigma_t = sigma * sqrt(1-t)) is the default.
# Mutually exclusive with use_constant_noise.
use_constant_noise=false
use_tapered_noise=true
# Score correction during SDE sampling (Theorem 17): preserves marginal of BC ODE.
error_correct_sde_to_ode=true
constant_noise_std=0.01
min_noise_std=0.01
max_noise_std=0.01

# Paths
SAVE_DIR="exp/ogpo"
restore_actor_path="null"
restore_critic_path="null"
ep_resume=0

# Logging
log=true
plot_q_vs_mc=false

# === Parse CLI overrides ===
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"
      eval "$key=\"$value\""
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES=$device
export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac

env_name="D4RL/kitchen/$env_name-v2"

# === Run ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== OGPO Kitchen | seed=$seed ==="

uv run python ogpo/main.py \
    --algo=ogpo \
    --experiment.project=$project \
    --experiment.run_group=$run_group \
    --experiment.run_name=$run_name \
    --experiment.wandb_name=$wandb_name \
    --experiment.seed=$seed \
    --experiment.log=$log \
    --experiment.save_dir=$SAVE_DIR \
    --env.name=$env_name \
    --env.sparse=false \
    --env.post_success_steps=$post_success_steps \
    --training.bc_pi_steps=$bc_pi_steps \
    --training.calql_steps=$calql_steps \
    --training.q_warmup_steps=$q_warmup_steps \
    --training.bc_refine_steps=$bc_refine_steps \
    --training.online_steps=$online_steps \
    --training.bc_only=false \
    --training.gpu_mode=$gpu_mode \
    --buffer.start_training=$start_training \
    --buffer.utd_warmup=$utd_warmup \
    --buffer.utd_online=$utd_online \
    --buffer.offline_ratio=$offline_ratio \
    --eval.n_eval_envs=$n_eval_envs \
    --eval.eval_episodes=$eval_episodes \
    --eval.plot_q_vs_mc=$plot_q_vs_mc \
    --intervals.log_interval=$log_interval \
    --intervals.eval_interval=$eval_interval \
    --intervals.eval_interval_bc=$eval_interval_bc \
    --intervals.save_interval=$save_interval \
    --bc.clip_bc=$clip_bc \
    --bc.clip_bc_threshold=$clip_bc_threshold \
    --bc.clip_bc_wrt=$clip_bc_wrt \
    --success_buffer.use_success_buffer=$use_success_buffer \
    --agent.use_success_buffer_q=$use_success_buffer_q \
    --checkpoint.restore_actor_path=$restore_actor_path \
    --checkpoint.restore_critic_path=$restore_critic_path \
    --checkpoint.ep_resume=$ep_resume \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.actor_tau=$actor_tau \
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.q_agg=$q_agg \
    --agent.subsample_bon=$subsample_bon \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.time_embedding=$time_embedding \
    --agent.time_embedding_dim=$time_embedding_dim \
    --agent.bc_coeff=$bc_coeff \
    --agent.adv_strategy=$adv_strategy \
    --agent.awr_beta=$awr_beta \
    --agent.ppo_batch_size=$ppo_batch_size \
    --agent.clip_epsilon=$clip_epsilon \
    --agent.entropy_coeff=$entropy_coeff \
    --agent.group_num_samples=$group_num_samples \
    --agent.use_bc_regularization=$use_bc_regularization \
    --agent.error_correct_ode_to_sde=$error_correct_ode_to_sde \
    --agent.use_denoiser=$use_denoiser \
    --agent.use_constant_noise=$use_constant_noise \
    --agent.use_tapered_noise=$use_tapered_noise \
    --agent.error_correct_sde_to_ode=$error_correct_sde_to_ode \
    --agent.constant_noise_std=$constant_noise_std \
    --agent.min_noise_std=$min_noise_std \
    --agent.max_noise_std=$max_noise_std \
    --agent.actor_backbone=$actor_backbone \
    --agent.critic_backbone=$critic_backbone
    