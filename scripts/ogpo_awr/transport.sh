#!/bin/bash

set -e
export MUJOCO_GL=egl

# === Config ===
seed=1
device=0
jax_frac=0.45
env_name="transport-mh-low_dim"
post_success_steps=0
run_name="ogpo_awr_transport"
wandb_name="ogpo_awr_transport"
project="OGPO_AWR"
run_group="transport"

# Training phases
bc_pi_steps=500000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=3000000
force_jax_sync=true
gpu_mode=single

# Agent config
lr=3e-4
actor_lr=3e-4
critic_lr=3e-4
ppo_lr=4.5e-5
clip_grad_norm=1000.0
horizon_length=4
discount=0.999
tau=0.05
actor_tau=0.005
best_of_n=8
num_qs=10
q_agg=subsample
subsample_bon=true
flow_steps=10
ft_flow_steps=10
time_embedding=sinusoidal
time_embedding_dim=32
bc_coeff=1.0
adv_strategy=awr

# AWR-specific
use_awr=false
awr_beta=10.0
awr_n_mc=16
awr_weight_max=20.0
awr_mode=symmetric

# chi2-Pessimistic regularization (optional)
chi_po=false
chi_po_beta_base=0.1
chi_po_ref_tau=0.0005
chi_po_q_std_target=1.0
chi_po_ensemble_alpha=5.0
chi_po_R_max=100.0

# UTD and buffer
buffer_size=2000000
utd_warmup=1
utd_online=1
utd_q=1
utd_pi=1
offline_ratio=0.0
start_training=10000

# Evaluation
n_eval_envs=32
eval_episodes=2
log_interval=5000
eval_interval=10000
eval_interval_bc=5000
save_interval=500000

# BC settings
clip_bc=true
clip_bc_threshold=0.45
use_constant_scheduler_for_bc=true

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
use_success_buffer=true
use_success_buffer_q=true

# Architecture
actor_backbone=mlp
critic_backbone=mlp
actor_layer_norm=false
layer_norm=true
actor_hidden_dims="[512,512,512,512]"
value_hidden_dims="[512,512,512,512]"

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

# Q-target variance reduction
q_variance_reduction=false
q_vr_num_samples=8
q_vr_reduction=median

# MC regression
mc_regression=false
mc_regression_coeff=1.0

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

echo "=== OGPO-AWR | env=$env_name | seed=$seed ==="

python ogpo/main.py \
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
    --training.bc_q_steps=0 \
    --training.calql_steps=$calql_steps \
    --training.q_warmup_steps=$q_warmup_steps \
    --training.bc_refine_steps=$bc_refine_steps \
    --training.online_steps=$online_steps \
    --training.bc_only=false \
    --training.gpu_mode=$gpu_mode \
    --training.force_jax_sync=$force_jax_sync \
    --buffer.size=$buffer_size \
    --buffer.start_training=$start_training \
    --buffer.utd_warmup=$utd_warmup \
    --buffer.utd_online=$utd_online \
    --buffer.utd_q=$utd_q \
    --buffer.utd_pi=$utd_pi \
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
    --agent.ppo_lr=$ppo_lr \
    --agent.clip_grad_norm=$clip_grad_norm \
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
    --agent.use_awr=$use_awr \
    --agent.awr_beta=$awr_beta \
    --agent.awr_n_mc=$awr_n_mc \
    --agent.awr_weight_max=$awr_weight_max \
    --agent.awr_mode=$awr_mode \
    --agent.chi_po=$chi_po \
    --agent.chi_po_beta_base=$chi_po_beta_base \
    --agent.chi_po_ref_tau=$chi_po_ref_tau \
    --agent.chi_po_q_std_target=$chi_po_q_std_target \
    --agent.chi_po_ensemble_alpha=$chi_po_ensemble_alpha \
    --agent.chi_po_R_max=$chi_po_R_max \
    --agent.ppo_batch_size=$ppo_batch_size \
    --agent.clip_epsilon=$clip_epsilon \
    --agent.entropy_coeff=$entropy_coeff \
    --agent.group_num_samples=$group_num_samples \
    --agent.q_variance_reduction=$q_variance_reduction \
    --agent.q_vr_num_samples=$q_vr_num_samples \
    --agent.q_vr_reduction=$q_vr_reduction \
    --agent.mc_regression=$mc_regression \
    --agent.mc_regression_coeff=$mc_regression_coeff \
    --agent.use_constant_scheduler_for_bc=$use_constant_scheduler_for_bc \
    --agent.actor_scheduler=$actor_scheduler \
    --agent.actor_warmup_steps=$actor_warmup_steps \
    --agent.actor_decay_steps=$actor_decay_steps \
    --agent.actor_end_value=$actor_end_value \
    --agent.critic_scheduler=$critic_scheduler \
    --agent.critic_warmup_steps=$critic_warmup_steps \
    --agent.critic_decay_steps=$critic_decay_steps \
    --agent.critic_end_value=$critic_end_value \
    --agent.actor_weight_decay=$actor_weight_decay \
    --agent.critic_weight_decay=$critic_weight_decay \
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
    --agent.critic_backbone=$critic_backbone \
    --agent.actor_layer_norm=$actor_layer_norm \
    --agent.layer_norm=$layer_norm \
    --agent.actor_hidden_dims=$actor_hidden_dims \
    --agent.value_hidden_dims=$value_hidden_dims \
    --dataset.p_aug=$p_aug
