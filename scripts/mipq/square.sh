#!/bin/bash

set -e

export MUJOCO_GL=egl

# === Config ===
seed=1
device=0
jax_frac=0.45
env_name="square-mh-low_dim"
run_name="mipq_square"
wandb_name="mipq_square"
project="MIPQ"
run_group="square"

# Training phases
offline_steps=500000
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=2000000
force_jax_sync=true
gpu_mode=single

# Agent config
horizon_length=4
discount=0.99
tau=0.05
actor_tau=0.005
best_of_n=8
q_agg=subsample
subsample_bon=true
flow_steps=10
time_embedding=sinusoidal       # scalar | sinusoidal
time_embedding_dim=32
bc_coeff=1.0
adv_strategy=vanilla
awr_beta=10
awr_mode=symmetric
awr_weight_max=20.0

# FPO settings (toggle with use_fpo=true)
use_fpo=false
fpo_n_mc=16
fpo_per_sample=true
fpo_use_aspo=false
fpo_cfm_loss_clamp=4.0
fpo_cfm_diff_clamp=5.0
fpo_use_huber=true
fpo_huber_delta=1.0

# chi2-Pessimistic (toggle with chi_po=true)
chi_po=false
chi_po_beta_base=0.1
chi_po_ref_tau=0.005
chi_po_q_std_target=1.0
chi_po_ensemble_alpha=1.0
chi_po_R_max=100.0

# MIP-Q specific (NEW)
use_mip_q=true
mip_q_ensemble=true
mip_q_noise_scale=1.0
mip_q_t_star=0.9
num_ensemble_members=10

# UTD and buffer
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
eval_interval=20000
eval_interval_bc=5000
save_interval=500000

# BC settings
clip_bc=true
clip_bc_threshold=0.45

# Success buffer
use_success_buffer=true

# Architecture backbone (mlp or tf, independently for actor/critic)
actor_backbone=mlp
critic_backbone=mlp
actor_layer_norm=true
layer_norm=true                         # critic layer norm
actor_hidden_dims="[512,512,512,512]"
value_hidden_dims="[512,512,512,512]"
# Transformer params (used when backbone=tf)
tf_pi_layers=3
tf_pi_embed_dim=128
tf_pi_heads=4
tf_q_layers=3
tf_q_embed_dim=128
tf_q_heads=4

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
SAVE_DIR="exp/mipq"
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

echo "=== MIPQ Square | seed=$seed ==="

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
    --training.offline_steps=$offline_steps \
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
    --checkpoint.restore_actor_path=$restore_actor_path \
    --checkpoint.restore_critic_path=$restore_critic_path \
    --checkpoint.ep_resume=$ep_resume \
    --agent.horizon_length=$horizon_length \
    --agent.discount=$discount \
    --agent.tau=$tau \
    --agent.actor_tau=$actor_tau \
    --agent.best_of_n=$best_of_n \
    --agent.q_agg=$q_agg \
    --agent.subsample_bon=$subsample_bon \
    --agent.flow_steps=$flow_steps \
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
    --agent.use_mip_q=$use_mip_q \
    --agent.mip_q_ensemble=$mip_q_ensemble \
    --agent.mip_q_noise_scale=$mip_q_noise_scale \
    --agent.mip_q_t_star=$mip_q_t_star \
    --agent.awr_mode=$awr_mode \
    --agent.awr_weight_max=$awr_weight_max \
    --agent.use_fpo=$use_fpo \
    --agent.fpo_n_mc=$fpo_n_mc \
    --agent.fpo_per_sample=$fpo_per_sample \
    --agent.fpo_use_aspo=$fpo_use_aspo \
    --agent.fpo_cfm_loss_clamp=$fpo_cfm_loss_clamp \
    --agent.fpo_cfm_diff_clamp=$fpo_cfm_diff_clamp \
    --agent.fpo_use_huber=$fpo_use_huber \
    --agent.fpo_huber_delta=$fpo_huber_delta \
    --agent.chi_po=$chi_po \
    --agent.chi_po_beta_base=$chi_po_beta_base \
    --agent.chi_po_ref_tau=$chi_po_ref_tau \
    --agent.chi_po_q_std_target=$chi_po_q_std_target \
    --agent.chi_po_ensemble_alpha=$chi_po_ensemble_alpha \
    --agent.chi_po_R_max=$chi_po_R_max \
    --agent.num_ensemble_members=$num_ensemble_members
