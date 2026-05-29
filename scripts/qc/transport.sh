#!/bin/bash

set -e

export MUJOCO_GL=egl

# === Config ===
seed=1
device=0
jax_frac=0.45
env_name="transport-mh-low_dim"
post_success_steps=0
run_name="qc_transport"
wandb_name="qc_transport"
project="QC"
run_group="transport"

# Training phases
bc_pi_steps=1000000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=6000000
force_jax_sync=true
gpu_mode=single

# Agent config
lr=3e-4
actor_lr=3e-4
critic_lr=3e-4
clip_grad_norm=1000.0
horizon_length=8
discount=0.999
tau=0.005
best_of_n=8
num_qs=10
q_agg="mean"
subsample_bon=false
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal       # scalar | sinusoidal
time_embedding_dim=32
alpha=500
actor_num_samples=32

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
eval_interval_bc=5000
save_interval=500000

# BC settings
clip_bc=false
clip_bc_threshold=0.45
use_constant_scheduler_for_bc=false

# LR schedules (cosine warmup-decay for actor and critic)
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

# Architecture backbone (mlp or tf, independently for actor/critic)
actor_backbone=mlp
critic_backbone=mlp
actor_layer_norm=false
layer_norm=true
actor_hidden_dims="[512,512,512,512]"
value_hidden_dims="[512,512,512,512]"
# Transformer params (used when backbone=tf)
tf_pi_layers=8
tf_pi_embed_dim=256
tf_pi_heads=4
tf_q_layers=8
tf_q_embed_dim=256
tf_q_heads=4

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

echo "=== QC Transport | seed=$seed ==="

python ogpo/main.py \
    --algo=qc \
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
    --training.force_jax_sync=$force_jax_sync \
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
    --agent.best_of_n=$best_of_n \
    --agent.num_qs=$num_qs \
    --agent.q_agg=$q_agg \
    --agent.subsample_bon=$subsample_bon \
    --agent.flow_steps=$flow_steps \
    --agent.ft_flow_steps=$ft_flow_steps \
    --agent.time_embedding=$time_embedding \
    --agent.time_embedding_dim=$time_embedding_dim \
    --agent.alpha=$alpha \
    --agent.actor_num_samples=$actor_num_samples \
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
    --agent.tf_pi_layers=$tf_pi_layers \
    --agent.tf_pi_embed_dim=$tf_pi_embed_dim \
    --agent.tf_pi_heads=$tf_pi_heads \
    --agent.tf_q_layers=$tf_q_layers \
    --agent.tf_q_embed_dim=$tf_q_embed_dim \
    --agent.tf_q_heads=$tf_q_heads \
    --dataset.p_aug=$p_aug
