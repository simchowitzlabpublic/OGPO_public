#!/bin/bash

set -e

export MUJOCO_GL=egl

# === Config ===
seed=82723
device=0
encoder_device=1                # GPU for PaliGemma pre-encoding (separate from training)
jax_frac=0.90
env_name="tool_hang-ph-image"
run_name="ogpo_tool_hang_paligemma"
wandb_name="ogpo_tool_hang_paligemma"
project="OGPO"
run_group="tool_hang_paligemma"

# Training phases
bc_pi_steps=1000000
bc_q_steps=0
calql_steps=0
q_warmup_steps=0
bc_refine_steps=0
online_steps=3000000
force_jax_sync=true
gpu_mode="reserve:1"

# UTD and buffer
buffer_size=2000000
utd_warmup=1
utd_online=1
utd_q=1
utd_pi=1
offline_ratio=0.0
start_training=20000

# Evaluation
n_eval_envs=1
eval_episodes=20
log_interval=5000
eval_interval=20000
eval_interval_bc=20000
save_interval=500000

# Agent config
lr=3e-4
actor_lr=3e-4
critic_lr=3e-4
ppo_lr=4.5e-5
clip_grad_norm=1000.0
horizon_length=8
post_success_steps=8
discount=0.999
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

# BC settings
clip_bc=true
clip_bc_threshold=0.45
clip_bc_wrt=SDE                 # match low_dim OGPO: gate BC early-stop on SDE eval
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
critic_weight_decay=1e-6

# Data augmentation - disabled for frozen encoder
p_aug=0.0

# Success buffer
use_success_buffer=true
use_success_buffer_q=true

# Architecture
actor_backbone=mlp
critic_backbone=mlp
actor_layer_norm=false
layer_norm=true
actor_hidden_dims="[512,512,512,512,512]"
value_hidden_dims="[512,512,512,512,512]"
tf_pi_layers=5
tf_pi_embed_dim=128
tf_pi_heads=4
tf_q_layers=5
tf_q_embed_dim=128
tf_q_heads=4
tf_pi_conditioning=adaln
tf_q_conditioning=adaln

# Vision config: PaliGemma frozen backbone (LIBERO finetuned)
actor_obs=image
critic_obs=image
vit_encoder=paligemma
vit_img_proj_dim=0          # 0 = use full 2048-dim PaliGemma features
vit_state_proj_dim=0
freeze_vision_encoder=true
# Two-tier obs encoder: L2-normalize image features + separate Dense->LN->GELU
# towers for image and proprio; concat into a fused representation before the trunk.
obs_two_tier=true
two_tier_fused_dim=0        # 0 = use actor_hidden_dims[0]
# State source for the proprio tower (matches dataset pre-encoding):
#   full    -> robot proprio + object pose
#   proprio -> robot proprio only (9D)
# Two-tier head structure is unchanged; only the proprio tower's input dim shifts.
use_state="proprio"

# Q-target variance reduction
q_variance_reduction=true
q_vr_num_samples=8
q_vr_reduction=mean

# MC regression
mc_regression=false
mc_regression_coeff=1.0

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
constant_noise_std=0.1
min_noise_std=0.01
max_noise_std=0.01

# chi2-Pessimistic OGPO
chi_po=false
chi_po_beta_base=0.1
tau_slow=0.0005
chi_po_q_std_target=1.0
chi_po_ensemble_alpha=1.0
chi_po_R_max=100.0

# KL-Pessimistic OGPO (stacks with chi_po; share tau_slow + chi_po_q_std_target)
kl_reg=false
kl_reg_beta_base=0.1
kl_reg_ensemble_alpha=5.0
kl_reg_R_max=100.0

# Forward-KL regularization (BC flow-matching against slow ref samples)
fwd_kl_reg=false
fwd_kl_bc_coeff=0.1

# Paths
SAVE_DIR="exp/ogpo"
restore_actor_path="null"
restore_critic_path="null"
# PaliGemma vision encoder: path to Pi0/Pi0.5 orbax checkpoint
# pi05_libero: Pi0.5 backbone finetuned on LIBERO (SigLIP So400m/14 → 2048-dim output)
restore_encoder_path="$HOME/checkpoints/pi05_libero/params"
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

# Make both training and encoder GPUs visible
export CUDA_VISIBLE_DEVICES=$device,$encoder_device
export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac

# === Run ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"
[ -f "$PROJECT_ROOT/.venv/bin/activate" ] && source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== OGPO Square PaliGemma | seed=$seed | encoder=$vit_encoder | model=$restore_encoder_path ==="

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
    --training.bc_q_steps=$bc_q_steps \
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
    --checkpoint.restore_encoder_path=$restore_encoder_path \
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
    --agent.awr_beta=$awr_beta \
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
    --agent.tf_pi_layers=$tf_pi_layers \
    --agent.tf_pi_embed_dim=$tf_pi_embed_dim \
    --agent.tf_pi_heads=$tf_pi_heads \
    --agent.tf_q_layers=$tf_q_layers \
    --agent.tf_q_embed_dim=$tf_q_embed_dim \
    --agent.tf_q_heads=$tf_q_heads \
    --agent.tf_pi_conditioning=$tf_pi_conditioning \
    --agent.tf_q_conditioning=$tf_q_conditioning \
    --agent.actor_obs=$actor_obs \
    --agent.critic_obs=$critic_obs \
    --agent.vit_encoder=$vit_encoder \
    --agent.vit_img_proj_dim=$vit_img_proj_dim \
    --agent.vit_state_proj_dim=$vit_state_proj_dim \
    --agent.freeze_vision_encoder=$freeze_vision_encoder \
    --agent.obs_two_tier=$obs_two_tier \
    --agent.two_tier_fused_dim=$two_tier_fused_dim \
    --agent.use_state=$use_state \
    --agent.encoder_device=$encoder_device \
    --agent.chi_po=$chi_po \
    --agent.chi_po_beta_base=$chi_po_beta_base \
    --agent.tau_slow=$tau_slow \
    --agent.chi_po_q_std_target=$chi_po_q_std_target \
    --agent.chi_po_ensemble_alpha=$chi_po_ensemble_alpha \
    --agent.chi_po_R_max=$chi_po_R_max \
    --agent.kl_reg=$kl_reg \
    --agent.kl_reg_beta_base=$kl_reg_beta_base \
    --agent.kl_reg_ensemble_alpha=$kl_reg_ensemble_alpha \
    --agent.kl_reg_R_max=$kl_reg_R_max \
    --agent.fwd_kl_reg=$fwd_kl_reg \
    --agent.fwd_kl_bc_coeff=$fwd_kl_bc_coeff \
    --dataset.p_aug=$p_aug
