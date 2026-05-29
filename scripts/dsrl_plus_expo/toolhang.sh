#!/bin/bash

set -e

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# === Config ===
seed=1
device=0
env_name="tool_hang-ph-low_dim"
run_name="dsrl_expo_toolhang"
wandb_name="dsrl_expo_toolhang"
project="DSRL_EXPO"
run_group="toolhang"

# Training phases
bc_pi_steps=500000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=4000000
gpu_mode=single

# Agent config
horizon_length=8
discount=0.999
tau=0.005
best_of_n=1
num_qs=10
q_agg="mean"
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal       # scalar | sinusoidal
time_embedding_dim=32
noise_chunk_length=-1

# Stability hyperparameters (match original DSRL repo)
action_magnitude=1.5
target_entropy=0.0
noise_critic_grad_steps=10
clip_grad_norm=1.0

# DSRL+EXPO specific
use_edit_actor=true
edit_action_scale=0.5
edit_start_step=100000
n_edit_samples=8

# UTD and buffer
utd_warmup=1
utd_online=20
offline_ratio=0.0
start_training=10000

# Evaluation
n_eval_envs=32
eval_episodes=2
log_interval=5000
eval_interval=20000
eval_interval_bc=5000
save_interval=500000

# BC settings
clip_bc=false

# Success buffer
use_success_buffer=false

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

# === Run ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== DSRL+EXPO Toolhang | seed=$seed ==="

uv run python ogpo/main.py \
    --algo=dsrl_plus_expo \
    --experiment.project=$project \
    --experiment.run_group=$run_group \
    --experiment.run_name=$run_name \
    --experiment.wandb_name=$wandb_name \
    --experiment.seed=$seed \
    --experiment.log=$log \
    --experiment.save_dir=$SAVE_DIR \
    --env.name=$env_name \
    --env.sparse=false \
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
    --success_buffer.use_success_buffer=$use_success_buffer \
    --checkpoint.restore_actor_path=$restore_actor_path \
    --checkpoint.restore_critic_path=$restore_critic_path \
    --checkpoint.ep_resume=$ep_resume \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.q_agg=$q_agg \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.time_embedding=$time_embedding \
    --agent.time_embedding_dim=$time_embedding_dim \
    --agent.noise_chunk_length=$noise_chunk_length \
    --agent.use_edit_actor=$use_edit_actor \
    --agent.edit_action_scale=$edit_action_scale \
    --agent.edit_start_step=$edit_start_step \
    --agent.n_edit_samples=$n_edit_samples \
    --agent.action_magnitude=$action_magnitude \
    --agent.target_entropy=$target_entropy \
    --agent.noise_critic_grad_steps=$noise_critic_grad_steps \
    --agent.clip_grad_norm=$clip_grad_norm
