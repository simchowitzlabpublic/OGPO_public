"""BC and CalQL training runners."""

import glob
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple
import jax
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.evaluation import evaluate, evaluate_parallel
from ogpo.utils.main_helper import LoggingHelper
from ogpo.utils.mem_monitor import log_memory
from ogpo.utils.sharding import shard_batch_flat


def _wrap_eval_for_frozen_encoder(fn, enc_fn, actor_uses_state=False, use_full_state=True):
    """Wrap an actor function to encode images before calling (frozen encoder mode)."""
    def wrapped(observations, images=None, **kwargs):
        full_state = kwargs.pop('full_state', None)
        if actor_uses_state and full_state is not None:
            return fn(observations=full_state, **kwargs)
        if images is not None:
            state_for_enc = full_state if (use_full_state and full_state is not None) else observations
            observations = enc_fn(state_for_enc, images)
        return fn(observations=observations, **kwargs)
    return wrapped


def _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name):
    if n_eval_envs <= 1:
        return None
    return SimpleNamespace(
        n_eval_envs=n_eval_envs, eval_episodes=eval_episodes,
        discount=discount, run_name=run_name, env_name=env_name,
    )


def _run_evaluation(agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot, encode_fn=None):
    if eval_flags is None:
        eval_info, _, _ = evaluate(
            agent=agent, env=eval_env, global_step=log_step,
            action_dim=action_dim, num_eval_episodes=eval_episodes,
            num_video_episodes=video_episodes, video_frame_skip=video_frame_skip,
            actor_fn=actor_fn, env_name=run_name, plot=plot, encode_fn=encode_fn,
        )
    else:
        eval_info = evaluate_parallel(
            eval_envs, agent, eval_flags, action_dim, log_step,
            actor_fn=actor_fn, plot=plot, encode_fn=encode_fn,
        )
    return eval_info


def _log_step_data(logger, batch, phase, log_step):
    try:
        logger.log({
            'phase': phase,
            'batch_obs_mean': float(batch['observations'].mean()),
            'batch_obs_std': float(batch['observations'].std()),
            'batch_action_mean': float(batch['actions'].mean()),
            'batch_action_std': float(batch['actions'].std()),
        }, "step_data", step=log_step)
    except Exception:
        pass


_LOSS_KEYS = ('actor/bc_flow_loss', 'bc_flow_loss', 'actor_loss', 'bc_loss', 'loss')


def _find_loss_value(info):
    for k in _LOSS_KEYS:
        if k in info:
            return float(info[k])
    return None


def run_bc_training(
    agent: Any,
    train_dataset: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    bc_pi_steps: int,
    horizon_length: int,
    discount: float,
    batch_size: int,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    eval_interval_bc: int = 10000,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'bc',
    log_interval: int = 5000,
    save_interval_bc: int = -1,
    step_data_log_interval: int = 10,
    clip_bc: bool = False,
    clip_bc_threshold: float = 0.45,
    clip_bc_wrt: str = "ODE",
    env_name: str = '',
    start_step: int = 0,
    log_to_file: Optional[str] = None,
    encode_fn: Optional[Any] = None,
    mesh=None,
) -> Tuple[Any, int, Dict]:
    log_step = start_step
    eval_step = 0
    action_dim = train_dataset.sample(())['actions'].shape[-1]
    dataset_idx = 0

    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    should_log_step_data = step_data_log_interval > 0
    test_log_data = {'losses': [], 'eval_success': [], 'steps': []}
    eval_info = {}
    bc_mem_interval = max(eval_interval_bc, 50000) if eval_interval_bc > 0 else 50000

    def _maybe_shard(b):
        return shard_batch_flat(b, mesh) if (mesh is not None and b is not None) else b

    log_memory("bc_start", step=log_step, logger=logger)

    sample_fn = lambda: train_dataset.sample_sequence(
        batch_size, sequence_length=horizon_length, discount=discount)

    try:
        for i in tqdm.tqdm(range(1, bc_pi_steps + 1), dynamic_ncols=True, desc="BC Training"):
            log_step += 1

            batch = sample_fn()
            agent, offline_info = agent.bc_update(_maybe_shard(batch))

            if should_log_step_data and i % step_data_log_interval == 0:
                _log_step_data(logger, batch, 'bc', log_step)

            if i % log_interval == 0:
                logger.log(offline_info, "offline_agent", step=log_step)
                if log_to_file is not None:
                    loss_val = _find_loss_value(offline_info)
                    if loss_val is not None:
                        test_log_data['losses'].append(loss_val)
                        test_log_data['steps'].append(log_step)

            if i == bc_pi_steps or (eval_interval_bc != 0 and i % eval_interval_bc == 0):
                jax.block_until_ready(agent)
                eval_step += 1

                # ODE eval
                eval_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
                if encode_fn is not None:
                    _actor_uses_state = agent.config.get('_original_actor_obs', 'state') == 'state'
                    _use_full_state = agent.config.get('use_state', 'proprio') == 'full'
                    eval_actor_fn = _wrap_eval_for_frozen_encoder(
                        eval_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                eval_info = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    eval_actor_fn, log_step, eval_episodes, video_episodes,
                    video_frame_skip, run_name, plot=False,
                )
                ode_success = eval_info.get('success', 0.0)
                print(f"[EVAL ODE] step={i:6d} | success={ode_success:.4f} | return={eval_info.get('return', 0.0):.4f}", flush=True)
                logger.log(eval_info, "eval", step=log_step)

                # SDE eval (BON or sample_actions), matching the online RL dual eval
                sde_success = 0.0
                best_of_n = config.get('best_of_n', 1)
                if best_of_n > 1 and hasattr(agent, 'sample_actions_BON'):
                    sde_actor_fn = agent.sample_actions_BON
                elif hasattr(agent, 'sample_actions'):
                    sde_actor_fn = agent.sample_actions
                else:
                    sde_actor_fn = None

                if sde_actor_fn is not None:
                    if encode_fn is not None:
                        sde_actor_fn = _wrap_eval_for_frozen_encoder(
                            sde_actor_fn, encode_fn,
                            actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                    eval_info_sde = _run_evaluation(
                        agent, eval_env, eval_envs, eval_flags, action_dim,
                        sde_actor_fn, log_step, eval_episodes, 0,
                        video_frame_skip, run_name, plot=False,
                    )
                    sde_success = eval_info_sde.get('success', 0.0)
                    print(f"[EVAL SDE] step={i:6d} | success={sde_success:.4f} | return={eval_info_sde.get('return', 0.0):.4f}", flush=True)
                    try:
                        logger.log(eval_info_sde, "eval_sde", step=log_step)
                    except Exception:
                        pass

                if config.get('use_one_step_policy', False) and agent.one_step_network is not None:
                    one_step_actor_fn = agent.sample_actions_one_step_BON if best_of_n > 1 else agent.sample_actions_one_step
                    # Frozen-encoder runs: wrap so the env's raw proprio gets combined
                    # with image features into the same vector the one-step MLP was
                    # init'd with, else apply raises ScopeParamShapeError.
                    if encode_fn is not None:
                        one_step_actor_fn = _wrap_eval_for_frozen_encoder(
                            one_step_actor_fn, encode_fn,
                            actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                    eval_info_one_step = _run_evaluation(
                        agent, eval_env, eval_envs, eval_flags, action_dim,
                        one_step_actor_fn, log_step, eval_episodes, video_episodes,
                        video_frame_skip, run_name, plot=False,
                    )
                    one_step_success = eval_info_one_step.get('success', 0.0)
                    print(f"[EVAL-OneStep] step={i:6d} | success={one_step_success:.4f} | return={eval_info_one_step.get('return', 0.0):.4f}", flush=True)
                    logger.log(eval_info_one_step, "eval_one_step", step=log_step)

                # clip_bc gates on ODE success by default; OGPO low_dim runs pass
                # clip_bc_wrt="SDE" to preserve their legacy SDE-based gating.
                use_sde = clip_bc_wrt.upper() == "SDE" and sde_actor_fn is not None
                clip_success = sde_success if use_sde else ode_success
                clip_source = "SDE" if use_sde else "ODE"
                if log_to_file is not None:
                    test_log_data['eval_success'].append(float(clip_success))

                if clip_bc and clip_success >= clip_bc_threshold:
                    print(f"Breaking due to BC clipping ({clip_source} success {clip_success:.4f} >= {clip_bc_threshold})")
                    save_agent(agent, save_dir, log_step)
                    break

            if save_interval_bc > 0 and i % save_interval_bc == 0:
                save_agent(agent, save_dir, log_step)

            if i % bc_mem_interval == 0:
                log_memory("bc", step=log_step, logger=logger)
    finally:
        pass

    log_memory("bc_end", step=log_step, logger=logger)

    if log_to_file is not None:
        import json
        with open(log_to_file, 'w') as f:
            json.dump(test_log_data, f, indent=2)
        print(f"Test log written to {log_to_file}")

    return agent, log_step, eval_info


def run_calql_training(
    agent: Any,
    train_dataset: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    calql_steps: int,
    horizon_length: int,
    discount: float,
    batch_size: int,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    eval_interval_bc: int = 10000,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'calql',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    env_name: str = '',
    start_step: int = 0,
    encode_fn: Optional[Any] = None,
    mesh=None,
) -> Tuple[Any, int, Dict]:
    if calql_steps <= 0:
        return agent, start_step, {}

    log_step = start_step
    eval_step = 0
    action_dim = train_dataset.sample(())['actions'].shape[-1]
    dataset_idx = 0
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    should_log_step_data = step_data_log_interval > 0
    calql_log_interval = max(1, log_interval // 5)
    eval_info = {}

    def _maybe_shard(b):
        return shard_batch_flat(b, mesh) if (mesh is not None and b is not None) else b

    for i in tqdm.tqdm(range(1, calql_steps + 1), dynamic_ncols=True, desc="CalQL Training"):
        log_step += 1

        batch = train_dataset.sample_sequence(
            batch_size, sequence_length=horizon_length, discount=discount,
            ret_next_act=True, ret_mc=True)
        agent, offline_info = agent.calql_update(_maybe_shard(batch))

        if should_log_step_data and i % step_data_log_interval == 0:
            _log_step_data(logger, batch, 'calql', log_step)

        if i % calql_log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        if i == calql_steps or (eval_interval_bc != 0 and i % eval_interval_bc == 0):
            eval_step += 1
            eval_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            if encode_fn is not None:
                _actor_uses_state = agent.config.get('_original_actor_obs', 'state') == 'state'
                _use_full_state = agent.config.get('use_state', 'proprio') == 'full'
                eval_actor_fn = _wrap_eval_for_frozen_encoder(
                    eval_actor_fn, encode_fn,
                    actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
            eval_info = _run_evaluation(
                agent, eval_env, eval_envs, eval_flags, action_dim,
                eval_actor_fn, log_step, eval_episodes, video_episodes,
                video_frame_skip, run_name, plot=False,
            )
            print(f"CalQL Step {i}: Success Rate = {eval_info.get('success', 'N/A')}")
            logger.log(eval_info, "eval", step=log_step)

    return agent, log_step, eval_info


def run_bc_q_training(
    agent: Any,
    train_dataset: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    bc_q_steps: int,
    horizon_length: int,
    discount: float,
    batch_size: int,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    eval_interval_bc: int = 10000,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'bc_q',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    env_name: str = '',
    start_step: int = 0,
    mesh=None,
) -> Tuple[Any, int, Dict]:
    """Train critic with TD + MC regression loss on offline dataset."""
    if bc_q_steps <= 0:
        return agent, start_step, {}

    log_step = start_step
    eval_step = 0
    action_dim = train_dataset.sample(())['actions'].shape[-1]
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)
    should_log_step_data = step_data_log_interval > 0
    bc_q_log_interval = max(1, log_interval // 5)
    eval_info = {}

    def _maybe_shard(b):
        return shard_batch_flat(b, mesh) if (mesh is not None and b is not None) else b

    for i in tqdm.tqdm(range(1, bc_q_steps + 1), dynamic_ncols=True, desc="BC-Q Training"):
        log_step += 1

        batch = train_dataset.sample_sequence(
            batch_size, sequence_length=horizon_length, discount=discount,
            ret_next_act=True, ret_mc=True)
        agent, offline_info = agent.bc_q_update(_maybe_shard(batch))

        if should_log_step_data and i % step_data_log_interval == 0:
            _log_step_data(logger, batch, 'bc_q', log_step)

        if i % bc_q_log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

    print("Done BC-Q Pretraining.")
    return agent, log_step, eval_info
