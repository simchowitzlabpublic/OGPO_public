"""LR scheduling, optimizer creation, checkpoint restore, param logging."""

import pickle
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import optax


def create_lr_schedule(
    scheduler_type: str,
    base_lr: float,
    warmup_steps: int = 5000,
    decay_steps: int = 10000,
    end_value: float = 0.0
) -> Union[float, callable]:
    """Create learning rate schedule based on scheduler type."""
    if scheduler_type == 'constant':
        return base_lr
    elif scheduler_type == 'cosine':
        base_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=end_value,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )
        period = decay_steps
        def repeating_schedule(count):
            return base_schedule(count % period)
        return repeating_schedule
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'constant' or 'cosine'.")


def compute_param_norm(params: Any) -> jnp.ndarray:
    """Compute L2 norm of parameters."""
    flat_params, _ = jax.tree_util.tree_flatten(params)
    return jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params))


def compute_param_diff_norm(params1: Any, params2: Any) -> jnp.ndarray:
    """Compute L2 norm of parameter difference."""
    diff_tree = jax.tree_util.tree_map(lambda p1, p2: p1 - p2, params1, params2)
    flat_diff, _ = jax.tree_util.tree_flatten(diff_tree)
    return jnp.sqrt(sum(jnp.sum(p**2) for p in flat_diff))


def get_lr_info(
    actor_state: Any,
    critic_state: Any,
    actor_key: str = 'modules_actor',
    target_actor_key: str = 'modules_target_actor',
    critic_key: str = 'modules_critic',
    target_critic_key: str = 'modules_target_critic',
) -> Dict[str, float]:
    """Get learning rate information and parameter norms for logging."""
    actor_norm = compute_param_norm(actor_state.params[actor_key])
    target_actor_norm = compute_param_norm(actor_state.params[target_actor_key])
    critic_norm = compute_param_norm(critic_state.params[critic_key])
    target_critic_norm = compute_param_norm(critic_state.params[target_critic_key])

    actor_diff_norm = compute_param_diff_norm(
        actor_state.params[actor_key],
        actor_state.params[target_actor_key]
    )
    critic_diff_norm = compute_param_diff_norm(
        critic_state.params[critic_key],
        critic_state.params[target_critic_key]
    )

    return {
        'actor_norm': float(actor_norm),
        'target_actor_norm': float(target_actor_norm),
        'critic_norm': float(critic_norm),
        'target_critic_norm': float(target_critic_norm),
        'actor_diff_norm': float(actor_diff_norm),
        'critic_diff_norm': float(critic_diff_norm),
    }


def restore_critic_params(
    critic_network: Any,
    critic_path: str,
    agent_key: str = "agent",
    network_key: str = "critic_network",
) -> None:
    """Restore critic parameters from checkpoint."""
    with open(critic_path, 'rb') as f:
        load_dict = pickle.load(f)
    critic_network.params['modules_critic'] = load_dict[agent_key][network_key]["params"]["modules_critic"]
    critic_network.params['modules_target_critic'] = load_dict[agent_key][network_key]["params"]["modules_target_critic"]
    print(f"Restored critic from {critic_path}")


def restore_actor_params(
    actor_network: Any,
    actor_path: str,
    has_encoder: bool = False,
    agent_key: str = "agent",
) -> None:
    """Restore actor parameters from checkpoint.

    Handles multiple checkpoint formats:
    - actor_network format (from OGPO/BPTT training)
    - network format with modules_actor_bc_flow (from BC pretraining)
    """
    with open(actor_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent_dict = load_dict[agent_key]

    if "actor_network" in agent_dict:
        src_params = agent_dict["actor_network"]["params"]
        assert jax.tree_util.tree_structure(actor_network.params['modules_actor']) == \
               jax.tree_util.tree_structure(src_params["modules_actor"])
        actor_network.params['modules_actor'] = src_params["modules_actor"]
        actor_network.params['modules_target_actor'] = src_params["modules_target_actor"]
        print(f"Restored actor from {actor_path}")

    elif "network" in agent_dict and "modules_actor_bc_flow" in agent_dict["network"]["params"]:
        src_params = agent_dict["network"]["params"]

        if has_encoder:
            assert jax.tree_util.tree_structure(src_params["modules_actor_bc_flow_encoder"]) == \
                   jax.tree_util.tree_structure(actor_network.params['modules_actor']['encoder'])
            actor_network.params['modules_actor']['encoder'] = src_params["modules_actor_bc_flow_encoder"]
            actor_network.params['modules_target_actor']['encoder'] = src_params["modules_actor_bc_flow_encoder"]

            assert jax.tree_util.tree_structure(src_params["modules_actor_bc_flow"]["mlp"]) == \
                   jax.tree_util.tree_structure(actor_network.params['modules_actor']['mlp'])
            actor_network.params['modules_actor']['mlp'] = src_params["modules_actor_bc_flow"]["mlp"]
            actor_network.params['modules_target_actor']['mlp'] = src_params["modules_actor_bc_flow"]["mlp"]
        else:
            assert jax.tree_util.tree_structure(actor_network.params['modules_actor']) == \
                   jax.tree_util.tree_structure(src_params["modules_actor_bc_flow"])
            actor_network.params['modules_actor'] = src_params["modules_actor_bc_flow"]
            actor_network.params['modules_target_actor'] = src_params.get(
                "modules_target_actor_bc_flow",
                src_params["modules_actor_bc_flow"]
            )
        print(f"Restored actor from {actor_path}")
    else:
        raise ValueError(f"Unknown checkpoint format in {actor_path}")


def create_optimizer(
    lr_schedule: Union[float, callable],
    use_muon: bool = False,
    clip_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    muon_beta: float = 0.95,
    muon_ns_steps: int = 5,
    muon_nesterov: bool = True,
) -> optax.GradientTransformation:
    """Create optimizer with optional Muon or AdamW."""
    if use_muon:
        return optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.contrib.muon(
                learning_rate=lr_schedule,
                beta=muon_beta,
                ns_steps=muon_ns_steps,
                weight_decay=weight_decay,
                nesterov=muon_nesterov,
            )
        )
    else:
        @optax.inject_hyperparams
        def adam_optimizer(learning_rate, weight_decay):
            return optax.adamw(learning_rate, weight_decay=weight_decay)
        return optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            adam_optimizer(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
        )
