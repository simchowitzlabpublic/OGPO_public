"""Online RL training runner for all algorithms."""

import gc
import inspect
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.evaluation import evaluate, evaluate_parallel, _split_obs
from ogpo.utils.main_helper import LoggingHelper, save_rolling_checkpoint, create_success_buffer_batch
from ogpo.utils.mem_monitor import log_memory
from ogpo.utils.sharding import shard_batch

ALGORITHMS_WITH_BATCH_BC = {'ogpo', 'bptt', 'expo'}
ALGORITHMS_PURE_ONLINE = {'dsrl', 'dsrl_bon', 'dsrl_plus_expo'}
ALGORITHMS_WITH_PHASES = {'ogpo', 'bptt'}
ALGORITHMS_DUAL_EVAL = {'ogpo', 'expo'}


def _wrap_actor_for_frozen_encoder(fn, enc_fn, actor_uses_state=False, use_full_state=True):
    """Wrap an actor function to encode images before calling (frozen encoder mode).

    `use_full_state` mirrors agent.use_state and must match the dataset
    pre-encoding (full_state vs robot proprio only). `full_state` is popped (not
    just read) so it isn't forwarded to the underlying agent method, which is
    often a baseline taking only (observations, rng).
    """
    def wrapped(observations, images=None, **kwargs):
        full_state = kwargs.pop('full_state', None)
        if actor_uses_state and full_state is not None:
            return fn(observations=full_state, **kwargs)
        if images is not None:
            state_for_enc = full_state if (use_full_state and full_state is not None) else observations
            observations = enc_fn(state_for_enc, images)
        return fn(observations=observations, **kwargs)
    return wrapped


def _actor_supported_kwargs(actor_fn):
    """Return the set of kwarg names `actor_fn` accepts (None if it has **kwargs).

    Used to filter OGPO-only kwargs (images / full_state / is_encoded) before
    calling baselines whose sample_actions takes only (observations, rng).
    """
    try:
        sig = inspect.signature(actor_fn)
    except (TypeError, ValueError):
        return None  # builtin/C-impl — assume permissive, don't filter
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in sig.parameters.values())
    if accepts_var_kw:
        return None
    return {
        name for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }


def _filter_actor_kwargs(kwargs, supported):
    """Drop kwargs the actor doesn't accept (no-op when supported is None)."""
    if supported is None:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in supported}


def _state_for_encoder(observation, full_state, use_full_state):
    """Pick the state vector to feed the frozen image encoder, mirroring
    agent.use_state so encode_fn call sites match the dataset pre-encoding."""
    if use_full_state and full_state is not None:
        return full_state
    return observation


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

def _reshape_for_utd(batch, utd_ratio, batch_size):
    return jax.tree.map(lambda x: x.reshape((utd_ratio, batch_size) + x.shape[1:]), batch)

def _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                        horizon_length, discount, offline_ratio):
    """Sample a mixed online/offline batch and reshape for UTD."""
    offline_batch_size = int(batch_size * offline_ratio)
    online_batch_size = batch_size - offline_batch_size

    offline_batch = None
    if offline_batch_size > 0:
        offline_batch = train_dataset.sample_sequence(
            offline_batch_size * utd_ratio,
            sequence_length=horizon_length, discount=discount)
        offline_batch = _reshape_for_utd(offline_batch, utd_ratio, offline_batch_size)

    online_batch = None
    if online_batch_size > 0:
        min_online_size = online_batch_size * utd_ratio * horizon_length
        if replay_buffer.size >= min_online_size:
            online_batch = replay_buffer.sample_sequence(
                online_batch_size * utd_ratio,
                sequence_length=horizon_length, discount=discount)
            online_batch = _reshape_for_utd(online_batch, utd_ratio, online_batch_size)
        else:
            online_batch_size = 0

    if offline_batch is not None and online_batch is not None:
        common_keys = set(offline_batch.keys()) & set(online_batch.keys())
        return jax.tree.map(
            lambda off, on: np.concatenate([off, on], axis=1),
            {k: offline_batch[k] for k in common_keys},
            {k: online_batch[k] for k in common_keys},
        )
    return offline_batch if offline_batch is not None else online_batch


def _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount):
    """Sample a pure-online batch and reshape for UTD."""
    batch = replay_buffer.sample_sequence(
        batch_size * utd_ratio, sequence_length=horizon_length, discount=discount)
    return _reshape_for_utd(batch, utd_ratio, batch_size)


def _sample_success_batch(replay_buffer, success_buffer_batch_size, utd_ratio,
                          horizon_length, discount):
    """Sample from success buffer. Returns (batch, True) or (None, False)."""
    num_successful = replay_buffer._traj_success_mask[:replay_buffer.size].sum()
    min_required = success_buffer_batch_size * utd_ratio + horizon_length
    if num_successful < min_required:
        return None, False

    batch = create_success_buffer_batch(
        replay_buffer, success_buffer_batch_size * utd_ratio,
        horizon_length, discount)
    batch = _reshape_for_utd(batch, utd_ratio, success_buffer_batch_size)
    return batch, True


def _q_warmup_collect(
    agent, env, replay_buffer, *,
    q_warmup_steps, action_dim, has_images, discount, seed,
    use_success_buffer, mc_regression, encode_fn,
):
    """Phase A of phase-split q_warmup: env collection with the base actor (no BoN).

    Runs `q_warmup_steps` env steps; the final incomplete episode is discarded so
    the buffer only holds fully-bootstrapped trajectories.
    """
    _use_full_state = agent.config.get('use_state', 'proprio') == 'full'
    online_rng = jax.random.PRNGKey(seed + 40000 + 1)
    initial_buffer_size = replay_buffer.size
    episode_counter = 0
    raw_ob, _ = env.reset(seed=seed + 10000)
    ob, ob_images, ob_full_state = _split_obs(raw_ob)
    if encode_fn is not None and ob_images is not None:
        ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
        ob_images = None
    _qwarmup_actor_supported = _actor_supported_kwargs(agent.sample_actions)
    trajectory_buffer = []
    action_queue = []

    for _ in tqdm.tqdm(range(q_warmup_steps), dynamic_ncols=True, desc="Q-warmup collect"):
        online_rng, key = jax.random.split(online_rng)
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            if ob_full_state is not None:
                actor_kwargs['full_state'] = ob_full_state
            actor_kwargs = _filter_actor_kwargs(actor_kwargs, _qwarmup_actor_supported)
            action = agent.sample_actions(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        next_ob, next_ob_images, next_ob_full_state = _split_obs(raw_next_ob)
        if encode_fn is not None and next_ob_images is not None:
            next_ob = np.asarray(encode_fn(_state_for_encoder(next_ob, next_ob_full_state, _use_full_state)[None], next_ob_images[None])[0])
            next_ob_images = None
        done = terminated or truncated

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        # Attach full_states only when not pre-encoding: the encoded obs already
        # folds full_state in and the buffer schema has no full_states keys.
        if ob_full_state is not None and encode_fn is None:
            transition['full_states'] = ob_full_state
            transition['next_full_states'] = next_ob_full_state
        trajectory_buffer.append(transition)

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9
            if mc_regression:
                G = 0.0
                for t_idx in range(len(trajectory_buffer) - 1, -1, -1):
                    G = float(trajectory_buffer[t_idx]['rewards']) + discount * G
                    trajectory_buffer[t_idx]['mc_returns'] = G
            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
            trajectory_buffer = []
            episode_counter += 1
            raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
            ob, ob_images, ob_full_state = _split_obs(raw_ob)
            if encode_fn is not None and ob_images is not None:
                ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
                ob_images = None
            action_queue = []
        else:
            ob = next_ob
            ob_images = next_ob_images
            ob_full_state = next_ob_full_state

    discarded = len(trajectory_buffer)
    collected = replay_buffer.size - initial_buffer_size
    num_success = int(replay_buffer._traj_success_mask[:replay_buffer.size].sum())
    print(f"[q_warmup] collect: +{collected} transitions, {episode_counter} complete episodes, "
          f"{discarded} dangling steps discarded. Success transitions in RB: {num_success}.")


def _q_warmup_train(
    agent, replay_buffer, train_dataset, logger, *,
    q_warmup_steps, utd_warmup, batch_size, horizon_length, discount,
    use_success_buffer, success_buffer_batch_size,
    needs_offline_data, offline_ratio,
    log_interval, log_step, mesh, force_jax_sync,
):
    """Phase B of phase-split q_warmup: run q_warmup_steps * utd_warmup critic updates.

    Each iteration runs one `batch_q_warmup_update` on a fresh RB batch, plus one
    on a success-buffer batch when enough successes are present.
    """
    def _maybe_shard(b):
        return shard_batch(b, mesh) if (mesh is not None and b is not None) else b

    sb_active = 0
    rb_q_running, sb_q_running = 0.0, 0.0
    rb_cnt, sb_cnt = 0, 0
    for upd_i in tqdm.tqdm(range(q_warmup_steps), dynamic_ncols=True, desc="Q-warmup train"):
        if needs_offline_data:
            batch = _sample_mixed_batch(replay_buffer, train_dataset, batch_size,
                                        utd_warmup, horizon_length, discount, offline_ratio)
        else:
            batch = _sample_online_batch(replay_buffer, batch_size, utd_warmup,
                                         horizon_length, discount)
        agent, info_rb = agent.batch_q_warmup_update(_maybe_shard(batch))

        q_mean_rb = info_rb.get('critic/q_mean')
        if q_mean_rb is not None:
            rb_q_running += float(q_mean_rb); rb_cnt += 1

        if use_success_buffer:
            batch_sb, sb_flag = _sample_success_batch(
                replay_buffer, success_buffer_batch_size, utd_warmup,
                horizon_length, discount)
            if batch_sb is not None:
                agent, info_sb = agent.batch_q_warmup_update(_maybe_shard(batch_sb))
                q_mean_sb = info_sb.get('critic/q_mean')
                if q_mean_sb is not None:
                    sb_q_running += float(q_mean_sb); sb_cnt += 1
                sb_active += 1

        if force_jax_sync:
            jax.block_until_ready(agent)

        if log_interval > 0 and (upd_i + 1) % log_interval == 0:
            log_payload = {}
            if rb_cnt > 0:
                log_payload['rb_q_mean'] = rb_q_running / rb_cnt
            if sb_cnt > 0:
                log_payload['sb_q_mean'] = sb_q_running / sb_cnt
            if log_payload:
                logger.log(log_payload, "q_warmup", step=log_step + upd_i + 1)
            rb_q_running, sb_q_running = 0.0, 0.0
            rb_cnt, sb_cnt = 0, 0

    print(f"[q_warmup] train: {q_warmup_steps} calls x utd_warmup={utd_warmup} = "
          f"{q_warmup_steps * utd_warmup} critic updates. SB-active iters: {sb_active}/{q_warmup_steps}.")
    return agent, log_step + q_warmup_steps

def run_online_rl(
    agent: Any,
    env: Any,
    eval_env: Any,
    eval_envs: Optional[Any],
    train_dataset: Dataset,
    config: Dict,
    logger: LoggingHelper,
    save_dir: str,
    algo_name: str,
    online_steps: int,
    start_training: int,
    batch_size: int,
    horizon_length: int,
    discount: float,
    utd_warmup: int = 1,
    utd_online: int = 1,
    utd_q: Optional[int] = None,
    utd_pi: Optional[int] = None,
    utd_v: Optional[int] = None,
    offline_ratio: float = 0.0,
    bc_refine_steps: int = 0,
    q_warmup_steps: int = 0,
    use_success_buffer: bool = False,
    success_buffer_batch_size: int = 256,
    best_of_n: int = 1,
    eval_interval: int = 100000,
    n_eval_envs: int = 32,
    eval_episodes: int = 2,
    video_episodes: int = 0,
    video_frame_skip: int = 3,
    plot_q_vs_mc: bool = False,
    run_name: str = 'online_rl',
    env_name: str = '',
    log_interval: int = 5000,
    step_data_log_interval: int = 10,
    save_interval: int = 500000,
    preemption_backup_interval: int = -1,
    log_enabled: bool = True,
    sparse: bool = False,
    p_aug: Optional[float] = None,
    buffer_size: int = 2000000,
    seed: int = 0,
    is_resumed: bool = False,
    resumed_online_step: int = 0,
    resumed_replay_buffer: Optional[ReplayBuffer] = None,
    start_step: int = 0,
    wandb_run_id: Optional[str] = None,
    force_jax_sync: bool = True,
    encode_fn: Optional[Any] = None,
    mc_regression: bool = False,
    mesh=None,
) -> Tuple[Any, int, Dict, ReplayBuffer]:
    if algo_name in ALGORITHMS_PURE_ONLINE and offline_ratio != 0.0:
        raise ValueError(f"{algo_name} requires offline_ratio=0.0, got {offline_ratio}")

    _use_full_state = agent.config.get('use_state', 'proprio') == 'full'

    example_batch = train_dataset.sample(())
    action_dim = example_batch['actions'].shape[-1]
    has_images = 'images' in example_batch

    needs_offline_data = offline_ratio > 0.0
    if not needs_offline_data:
        train_dataset = None
        import gc; gc.collect()

    if is_resumed and resumed_replay_buffer is not None:
        replay_buffer = resumed_replay_buffer
    else:
        rb_template = dict(
            observations=example_batch['observations'],
            actions=example_batch['actions'],
            rewards=np.array(0.0), terminals=np.array(0.0),
            masks=np.array(1.0), next_observations=example_batch['observations'],
        )
        if has_images:
            rb_template['images'] = example_batch['images']
            rb_template['next_images'] = example_batch['images']
        if 'full_states' in example_batch:
            rb_template['full_states'] = example_batch['full_states']
            rb_template['next_full_states'] = example_batch['full_states']
        if mc_regression:
            rb_template['mc_returns'] = np.array(0.0)
        replay_buffer = ReplayBuffer.create(
            rb_template, buffer_size, numpy_rng=np.random.default_rng(seed + 30000))

    if p_aug is not None:
        replay_buffer.p_aug = p_aug

    episode_counter = 0
    raw_ob, _ = env.reset(seed=seed + 10000)
    ob, ob_images, ob_full_state = _split_obs(raw_ob)
    if encode_fn is not None and ob_images is not None:
        ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
        ob_images = None
    trajectory_buffer = []
    action_queue = []

    if not is_resumed and hasattr(agent, 'reset_optimizers_with_lr'):
        agent = agent.reset_optimizers_with_lr()
        print("Reset optimizer with online learning rate")

    # Separate Q/Pi/V UTD ratios (default to utd_online)
    eff_utd_q = utd_q if utd_q is not None else utd_online
    eff_utd_pi = utd_pi if utd_pi is not None else utd_online
    eff_utd_v = utd_v if utd_v is not None else eff_utd_q
    eff_utd_v_extra = max(0, eff_utd_v - eff_utd_q) if hasattr(agent, 'batch_v_only_update') else 0
    if eff_utd_v_extra > 0:
        print(f"[UTD] V extra updates: running {eff_utd_v_extra} V-only updates per env step "
              f"(utd_v={eff_utd_v}, utd_q={eff_utd_q})")
    split_utd = eff_utd_q != eff_utd_pi
    if split_utd:
        has_critic_only = hasattr(agent, 'batch_critic_only_update')
        if eff_utd_pi > eff_utd_q:
            print(f"[UTD] WARNING: utd_pi={eff_utd_pi} > utd_q={eff_utd_q} not supported, "
                  f"using max={eff_utd_pi} for both")
            split_utd = False
        elif not has_critic_only:
            print(f"[UTD] WARNING: {algo_name} has no batch_critic_only_update, "
                  f"using max(utd_q={eff_utd_q}, utd_pi={eff_utd_pi}) for both")
            split_utd = False
        else:
            print(f"[UTD] Split updates: critic={eff_utd_q}, actor={eff_utd_pi} per env step")
    elif eff_utd_q != utd_online:
        print(f"[UTD] critic={eff_utd_q}, actor={eff_utd_pi} per env step")

    needs_batch_bc = algo_name in ALGORITHMS_WITH_BATCH_BC
    has_phases = algo_name in ALGORITHMS_WITH_PHASES
    needs_dual_eval = algo_name in ALGORITHMS_DUAL_EVAL
    should_log_step_data = step_data_log_interval > 0

    log_step = start_step
    eval_step = 0
    update_info = {}
    online_rng = jax.random.PRNGKey(seed + 40000)
    eval_flags = _build_eval_flags(n_eval_envs, eval_episodes, discount, run_name, env_name)

    # Phase-split q_warmup: a dedicated collect-then-train phase before the main
    # online loop (collect env steps with the base actor, then run critic updates
    # back-to-back). Only safe when bc_refine is not set, otherwise it would
    # reverse the intended BC-refine-then-q_warmup ordering.
    do_phase_split_q_warmup = (
        has_phases and q_warmup_steps > 0 and bc_refine_steps == 0 and not is_resumed
    )
    if do_phase_split_q_warmup:
        print(f"\n=== Phase-split Q-warmup: collect {q_warmup_steps} env steps -> "
              f"{q_warmup_steps * utd_warmup} critic updates ===")
        _q_warmup_collect(
            agent=agent, env=env, replay_buffer=replay_buffer,
            q_warmup_steps=q_warmup_steps, action_dim=action_dim,
            has_images=has_images, discount=discount, seed=seed,
            use_success_buffer=use_success_buffer, mc_regression=mc_regression,
            encode_fn=encode_fn,
        )
        agent, log_step = _q_warmup_train(
            agent=agent, replay_buffer=replay_buffer, train_dataset=train_dataset,
            logger=logger,
            q_warmup_steps=q_warmup_steps, utd_warmup=utd_warmup,
            batch_size=batch_size, horizon_length=horizon_length, discount=discount,
            use_success_buffer=use_success_buffer,
            success_buffer_batch_size=success_buffer_batch_size,
            needs_offline_data=needs_offline_data, offline_ratio=offline_ratio,
            log_interval=log_interval, log_step=log_step,
            mesh=mesh, force_jax_sync=force_jax_sync,
        )
        # Reset env for the main loop and zero out the warmup budget there
        raw_ob, _ = env.reset(seed=seed + 10000 + 999999)
        ob, ob_images, ob_full_state = _split_obs(raw_ob)
        if encode_fn is not None and ob_images is not None:
            ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
            ob_images = None
        trajectory_buffer = []
        action_queue = []
        episode_counter = 0
        q_warmup_steps = 0
        start_training = 0

    q_warmup_end = start_training + bc_refine_steps + q_warmup_steps
    eval_start = q_warmup_end if has_phases else start_training

    # Select the policy used for online rollouts
    if config.get('use_one_step_for_rollouts', False):
        if best_of_n > 1 and hasattr(agent, 'sample_actions_one_step_BON'):
            _actor_method_name = 'sample_actions_one_step_BON'
            print(f"Using one-step BON policy (N={best_of_n}) for online rollouts")
        elif hasattr(agent, 'sample_actions_one_step'):
            _actor_method_name = 'sample_actions_one_step'
            print("Using one-step policy for online rollouts")
        else:
            _actor_method_name = 'sample_actions'
    elif best_of_n > 1 and hasattr(agent, 'sample_actions_BON'):
        _actor_method_name = 'sample_actions_BON'
    else:
        _actor_method_name = 'sample_actions'

    _rollout_actor_supported = _actor_supported_kwargs(getattr(agent, _actor_method_name))

    start_online_step = 1
    if is_resumed and resumed_online_step > 0:
        start_online_step = resumed_online_step + 1
        print(f"Resuming online loop at step {start_online_step}")

    eval_info = {}
    mem_log_interval = max(eval_interval, 20000)  # Log memory at least every eval

    def _maybe_shard(b):
        return shard_batch(b, mesh) if (mesh is not None and b is not None) else b

    log_memory("online_start", step=log_step, logger=logger)

    for i in tqdm.tqdm(range(start_online_step, online_steps + 1), dynamic_ncols=True, desc="Online RL", initial=start_online_step - 1, total=online_steps,):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)

        # Action chunking
        if len(action_queue) == 0:
            actor_kwargs = dict(observations=ob, rng=key)
            if ob_images is not None:
                actor_kwargs['images'] = ob_images
            if ob_full_state is not None:
                actor_kwargs['full_state'] = ob_full_state
            actor_kwargs = _filter_actor_kwargs(actor_kwargs, _rollout_actor_supported)
            action = getattr(agent, _actor_method_name)(**actor_kwargs)
            for a in np.array(action).reshape(-1, action_dim):
                action_queue.append(a)
        action = action_queue.pop(0)

        raw_next_ob, int_reward, terminated, truncated, info = env.step(action)
        next_ob, next_ob_images, next_ob_full_state = _split_obs(raw_next_ob)
        if encode_fn is not None and next_ob_images is not None:
            next_ob = np.asarray(encode_fn(_state_for_encoder(next_ob, next_ob_full_state, _use_full_state)[None], next_ob_images[None])[0])
            next_ob_images = None
        done = terminated or truncated

        env_info = {k: v for k, v in info.items() if k.startswith("distance")}
        if env_info:
            try:
                logger.log(env_info, "env", step=log_step)
            except Exception:
                pass

        transition = dict(
            observations=ob, actions=action, rewards=int_reward,
            terminals=float(done), masks=1.0 - float(terminated),
            next_observations=next_ob,
        )
        if has_images:
            transition['images'] = ob_images
            transition['next_images'] = next_ob_images
        # Attach full_states only when not pre-encoding: the encoded obs already
        # folds full_state in and the buffer schema has no full_states keys.
        if ob_full_state is not None and encode_fn is None:
            transition['full_states'] = ob_full_state
            transition['next_full_states'] = next_ob_full_state
        trajectory_buffer.append(transition)

        if should_log_step_data and i % step_data_log_interval == 0:
            try:
                logger.log({
                    'phase': 'online', 'reward': float(int_reward), 'done': float(done),
                    'batch_obs_mean': float(np.mean(ob)), 'batch_obs_std': float(np.std(ob)),
                    'batch_action_mean': float(np.mean(action)), 'batch_action_std': float(np.std(action)),
                }, "step_data", step=log_step)
            except Exception:
                pass

        if done:
            is_success = float(info.get('success', 0.0))
            score = is_success if is_success != 0 else 1e-9

            # MC returns via a backward pass over the trajectory
            if mc_regression:
                G = 0.0
                for t_idx in range(len(trajectory_buffer) - 1, -1, -1):
                    G = float(trajectory_buffer[t_idx]['rewards']) + discount * G
                    trajectory_buffer[t_idx]['mc_returns'] = G

            for t in trajectory_buffer:
                if use_success_buffer:
                    t['is_success'] = is_success
                    t['score'] = score
                replay_buffer.add_transition(t)
            trajectory_buffer = []
            episode_counter += 1
            raw_ob, _ = env.reset(seed=seed + 10000 + episode_counter)
            ob, ob_images, ob_full_state = _split_obs(raw_ob)
            if encode_fn is not None and ob_images is not None:
                ob = np.asarray(encode_fn(_state_for_encoder(ob, ob_full_state, _use_full_state)[None], ob_images[None])[0])
                ob_images = None
            action_queue = []
        else:
            ob = next_ob
            ob_images = next_ob_images
            ob_full_state = next_ob_full_state

        if i < start_training:
            continue

        is_warmup = has_phases and (i < bc_refine_steps or i < q_warmup_end)
        utd_ratio = utd_warmup if is_warmup else max(eff_utd_q, eff_utd_pi)

        if replay_buffer.size < batch_size * utd_ratio * horizon_length:
            continue

        if needs_offline_data:
            batch = _sample_mixed_batch(replay_buffer, train_dataset, batch_size, utd_ratio,
                                        horizon_length, discount, offline_ratio)
        else:
            batch = _sample_online_batch(replay_buffer, batch_size, utd_ratio, horizon_length, discount)

        # Sample success buffer at utd_ratio (= max(utd_q, utd_pi)) so batch_update can slice per phase
        batch_success = None
        success_flag = False
        if use_success_buffer:
            batch_success, success_flag = _sample_success_batch(replay_buffer, success_buffer_batch_size,
                                                                utd_ratio, horizon_length, discount)

        if has_phases and i < bc_refine_steps:
            if batch_success is None:
                batch_success = batch
            agent, update_info["online_agent"] = agent.batch_bc_refine_update(_maybe_shard(batch_success))
        elif has_phases and i < q_warmup_end:
            agent, update_info["online_agent"] = agent.batch_q_warmup_update(_maybe_shard(batch))
            if batch_success is not None:
                agent, update_info["online_agent"] = agent.batch_q_warmup_update(_maybe_shard(batch_success))
        else:
            if batch_success is None:
                batch_success = batch
            if split_utd:
                batch_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch)
                bs_joint = jax.tree.map(lambda x: x[:eff_utd_pi], batch_success)
                agent, update_info["online_agent"] = agent.batch_update(
                    _maybe_shard(batch_joint), _maybe_shard(bs_joint), success_flag)
                batch_extra = jax.tree.map(lambda x: x[eff_utd_pi:eff_utd_q], batch)
                agent, critic_info = agent.batch_critic_only_update(_maybe_shard(batch_extra))
                update_info["online_agent"].update(critic_info)
            else:
                agent, update_info["online_agent"] = agent.batch_update(
                    _maybe_shard(batch), _maybe_shard(batch_success), success_flag)

            if eff_utd_v_extra > 0:
                v_batch = _sample_online_batch(replay_buffer, batch_size, eff_utd_v_extra,
                                               horizon_length, discount)
                agent, v_info = agent.batch_v_only_update(_maybe_shard(v_batch))
                update_info["online_agent"].update(v_info)

        if force_jax_sync:
            jax.block_until_ready(agent)

        if i % log_interval == 0 and update_info:
            for key, info_dict in update_info.items():
                logger.log(info_dict, key, step=log_step)
            update_info = {}

        if i >= eval_start and (i == online_steps or (eval_interval > 0 and i % eval_interval == 0)):
            eval_step += 1
            gc.collect()

            ode_actor_fn = getattr(agent, 'compute_flow_actions_corrected', agent.compute_flow_actions)
            if encode_fn is not None:
                _actor_uses_state = agent.config.get('_original_actor_obs', 'state') == 'state'
                ode_actor_fn = _wrap_actor_for_frozen_encoder(
                    ode_actor_fn, encode_fn,
                    actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
            eval_info = _run_evaluation(
                agent, eval_env, eval_envs, eval_flags, action_dim,
                ode_actor_fn, log_step, eval_episodes, video_episodes,
                video_frame_skip, run_name, plot=(eval_step % 2 == 0) and plot_q_vs_mc,
                encode_fn=encode_fn,
            )
            rb_success_pct = 100.0 * replay_buffer._traj_success_mask[:replay_buffer.size].sum() / max(replay_buffer.size, 1)
            print(f"[EVAL ODE] step={i} | success={float(eval_info.get('success', 0.0)):.4f} | rb_success={rb_success_pct:.1f}% ({replay_buffer.size} total)")
            logger.log(eval_info, "eval", step=log_step)

            if needs_dual_eval:
                sde_actor_fn = getattr(agent, _actor_method_name)
                if encode_fn is not None:
                    sde_actor_fn = _wrap_actor_for_frozen_encoder(
                        sde_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                eval_info_sde = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    sde_actor_fn, log_step, eval_episodes, 0,
                    video_frame_skip, run_name, plot=False,
                    encode_fn=encode_fn,
                )
                print(f"[EVAL SDE] step={i} | success={float(eval_info_sde.get('success', 0.0)):.4f}")
                try:
                    logger.log(eval_info_sde, "eval_sde", step=log_step)
                except Exception:
                    pass

            if hasattr(agent, 'one_step_network') and agent.one_step_network is not None:
                one_step_actor_fn = agent.sample_actions_one_step_BON if best_of_n > 1 else agent.sample_actions_one_step
                # Frozen-encoder runs: wrap so the env's raw obs gets re-encoded
                # to match the dim the one-step MLP was init'd with.
                if encode_fn is not None:
                    one_step_actor_fn = _wrap_actor_for_frozen_encoder(
                        one_step_actor_fn, encode_fn,
                        actor_uses_state=_actor_uses_state, use_full_state=_use_full_state)
                eval_info_one_step = _run_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    one_step_actor_fn, log_step, eval_episodes, 0,
                    video_frame_skip, run_name, plot=False,
                )
                print(f"[EVAL OneStep] step={i} | success={float(eval_info_one_step.get('success', 0.0)):.4f}")
                try:
                    logger.log(eval_info_one_step, "eval_one_step", step=log_step)
                except Exception:
                    pass

        if i % mem_log_interval == 0:
            gc.collect()
            log_memory("online", step=log_step, logger=logger)

        if log_enabled and save_interval > 0 and i % save_interval == 0:
            save_agent(agent, save_dir, log_step)

        if preemption_backup_interval > 0 and i % preemption_backup_interval == 0:
            print(f"Saving rolling checkpoint at step {log_step}")
            save_rolling_checkpoint(agent, replay_buffer, log_step, i, seed, env_name, wandb_run_id=wandb_run_id, algo_name=algo_name)

    save_agent(agent, save_dir, log_step)
    print(f"Online RL training complete. Final step: {log_step}")
    return agent, log_step, eval_info, replay_buffer
