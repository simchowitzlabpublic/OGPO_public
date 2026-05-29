#!/bin/bash

set -e

export MUJOCO_GL=egl

# === Config ===
seed=82723
device=0
jax_frac=0.45
env_name="pusht-keypoints-v0"
post_success_steps=0
run_name="expo_pusht"
wandb_name="expo_pusht"
project="EXPO"
run_group="pusht"
force_jax_sync=true
gpu_mode=single

# Training phases
bc_pi_steps=500000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=2000000

# Agent config
lr=3e-4
actor_lr=3e-4
critic_lr=3e-4
clip_grad_norm=1000.0
horizon_length=4
discount=0.95
tau=0.005
actor_tau=0.005
best_of_n=8
num_qs=10
num_min_qs=2
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal
time_embedding_dim=32
constant_noise_std=0.01
clip_intermediate_actions=true

# Edit policy (EXPO-specific)
edit_policy_bound=0.05
n_action_samples=8
n_edit_samples=8
edit_lr=3e-4
edit_hidden_dims="[256,256,256]"
edit_layer_norm=true
edit_weight_decay=0.0
alpha_lr=3e-4
init_temperature=1.0
target_entropy=null

# BC regularization (EXPO+, off by default)
use_bc_regularization=false
bc_coeff=0.1

# UTD and buffer
buffer_size=2000000
utd_warmup=1
utd_online=4
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
clip_bc_threshold=0.45
use_constant_scheduler_for_bc=false

# LR schedules
actor_scheduler=cosine
actor_warmup_steps=2000
actor_decay_steps=50000
actor_end_value=2e-5
actor_weight_decay=0.0
critic_scheduler=constant
critic_warmup_steps=500
critic_decay_steps=5000
critic_end_value=1e-8
critic_weight_decay=1e-5

# Data augmentation
p_aug=0.0

# Success buffer
use_success_buffer=false
use_success_buffer_q=false

# Architecture
actor_backbone=mlp
critic_backbone=mlp
actor_layer_norm=false
layer_norm=true
actor_hidden_dims="[512,512,512,512]"
value_hidden_dims="[512,512,512,512]"

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
if [ "$(echo "$jax_frac == -1" | bc)" -eq 1 ]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
else
    export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac
fi

# === Run ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== EXPO PushT | seed=$seed ==="

uv run python ogpo/main.py \
    --algo=expo \
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
    --training.force_jax_sync=$force_jax_sync \
    --training.bc_only=false \
    --training.gpu_mode=$gpu_mode \
    --buffer.size=$buffer_size \
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
    --success_buffer.use_success_buffer=$use_success_buffer \
    --agent.use_success_buffer_q=$use_success_buffer_q \
    --checkpoint.restore_actor_path=$restore_actor_path \
    --checkpoint.restore_critic_path=$restore_critic_path \
    --checkpoint.ep_resume=$ep_resume \
    --agent.lr=$lr \
    --agent.actor_lr=$actor_lr \
    --agent.critic_lr=$critic_lr \
    --agent.clip_grad_norm=$clip_grad_norm \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.actor_tau=$actor_tau \
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.num_min_qs=$num_min_qs \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.time_embedding=$time_embedding \
    --agent.time_embedding_dim=$time_embedding_dim \
    --agent.constant_noise_std=$constant_noise_std \
    --agent.clip_intermediate_actions=$clip_intermediate_actions \
    --agent.edit_policy_bound=$edit_policy_bound \
    --agent.n_action_samples=$n_action_samples \
    --agent.n_edit_samples=$n_edit_samples \
    --agent.edit_lr=$edit_lr \
    --agent.edit_hidden_dims=$edit_hidden_dims \
    --agent.edit_layer_norm=$edit_layer_norm \
    --agent.edit_weight_decay=$edit_weight_decay \
    --agent.alpha_lr=$alpha_lr \
    --agent.init_temperature=$init_temperature \
    --agent.target_entropy=$target_entropy \
    --agent.use_bc_regularization=$use_bc_regularization \
    --agent.bc_coeff=$bc_coeff \
    --agent.use_constant_scheduler_for_bc=$use_constant_scheduler_for_bc \
    --agent.actor_scheduler=$actor_scheduler \
    --agent.actor_warmup_steps=$actor_warmup_steps \
    --agent.actor_decay_steps=$actor_decay_steps \
    --agent.actor_end_value=$actor_end_value \
    --agent.actor_weight_decay=$actor_weight_decay \
    --agent.critic_scheduler=$critic_scheduler \
    --agent.critic_warmup_steps=$critic_warmup_steps \
    --agent.critic_decay_steps=$critic_decay_steps \
    --agent.critic_end_value=$critic_end_value \
    --agent.critic_weight_decay=$critic_weight_decay \
    --agent.actor_backbone=$actor_backbone \
    --agent.critic_backbone=$critic_backbone \
    --agent.actor_layer_norm=$actor_layer_norm \
    --agent.layer_norm=$layer_norm \
    --agent.actor_hidden_dims=$actor_hidden_dims \
    --agent.value_hidden_dims=$value_hidden_dims \
    --dataset.p_aug=$p_aug
