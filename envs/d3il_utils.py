import gymnasium as gym
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

from ogpo.utils.datasets import Dataset
import gym_inserting
import gym_stacking


def quat2euler(quat):
    """Convert quaternion to euler angles."""
    if len(quat) == 4:
        if abs(quat[0]) > 0.9:  # Likely [w, x, y, z] format
            w, x, y, z = quat
        else:  # Likely [x, y, z, w] format
            x, y, z, w = quat

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        return np.array([roll, pitch, yaw])
    return quat


def flatten_obs_dict(obs):
    """Flatten observation into a single numpy array (D3IL obs is already flat)."""
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(np.atleast_1d(obs[key]))
        return np.concatenate(parts, axis=-1)
    else:
        return obs


class D3ILWrapper(gym.Wrapper):
    """Wrapper for D3IL environments to match the Franka Kitchen interface."""
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        # Handle both old gym (returns obs) and new gym (returns obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        info['success'] = 0.0

        return obs, info

    def step(self, action):
        result = self.env.step(action)

        # Handle both old gym (4-tuple) and new gym (5-tuple) returns
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result

        if 'success' not in info:
            info['success'] = float(terminated and reward > 0.5)

        return obs, reward, terminated, truncated, info


def make_env(env_name: str, seed: int = 0, **kwargs) -> gym.Env:
    """Create a D3IL environment and apply the wrapper."""
    if '-state' in env_name:
        task_name = env_name.replace('-state', '')
    elif '-image' in env_name:
        task_name = env_name.replace('-image', '')
        raise NotImplementedError("Image observation mode not yet implemented")
    else:
        task_name = env_name
    
    gym_id_map = {
        'gate_insertion': 'gate_insertion-v0',
        'stacking': 'stacking-v0',
    }
    
    gym_id = gym_id_map.get(task_name)
    if gym_id is None:
        raise ValueError(f"Unknown task: {task_name}")
    
    env_kwargs = {
        'render': False,
        'max_steps_per_episode': kwargs.pop('max_steps_per_episode', 1000),
    }
    
    if 'stacking' in gym_id:
        env_kwargs['if_vision'] = False
    
    env_kwargs.update(kwargs)
    
    try:
        env = gym.make(gym_id, **env_kwargs)
    except gym.error.UnregisteredEnv:
        raise ImportError(
            f"Environment '{gym_id}' not registered. "
            f"Install it with:\n"
            f"  cd d3il/environments/d3il/envs/gym_{task_name}_env\n"
            f"  pip install -e ."
        )
    
    env = env.unwrapped
    if hasattr(env, 'start'):
        env.start()

    env = D3ILWrapper(env)
    
    return env


def build_observation_stacking(episode_data: Dict, t: int) -> np.ndarray:
    """Build the stacking-task observation matching the env's get_observation().

    Layout: j_state, robot_c_pos, robot_c_quat, then per-box (pos[3],
    orient[2] = cos/sin(yaw)) for red, green, blue boxes.
    """
    robot_data = episode_data['robot']

    obs_parts = []

    j_pos = robot_data['j_pos'][t]
    j_vel = robot_data['j_vel'][t]
    gripper = robot_data['gripper_width'][t]
    if np.isscalar(gripper):
        gripper = np.array([gripper])
    j_state = np.concatenate([j_pos, gripper])
    obs_parts.append(j_state)

    robot_c_pos = robot_data['c_pos'][t]
    robot_c_quat = robot_data['c_quat'][t]
    obs_parts.append(robot_c_pos)
    obs_parts.append(robot_c_quat)

    red_box_pos = episode_data['red-box']['pos'][t]
    red_box_quat_full = episode_data['red-box']['quat'][t]
    red_box_euler = quat2euler(red_box_quat_full)
    red_box_yaw = red_box_euler[-1]
    red_box_orient = np.array([np.cos(red_box_yaw), np.sin(red_box_yaw)])
    obs_parts.append(red_box_pos)
    obs_parts.append(red_box_orient)

    green_box_pos = episode_data['green-box']['pos'][t]
    green_box_quat_full = episode_data['green-box']['quat'][t]
    green_box_euler = quat2euler(green_box_quat_full)
    green_box_yaw = green_box_euler[-1]
    green_box_orient = np.array([np.cos(green_box_yaw), np.sin(green_box_yaw)])
    obs_parts.append(green_box_pos)
    obs_parts.append(green_box_orient)

    blue_box_pos = episode_data['blue-box']['pos'][t]
    blue_box_quat_full = episode_data['blue-box']['quat'][t]
    blue_box_euler = quat2euler(blue_box_quat_full)
    blue_box_yaw = blue_box_euler[-1]
    blue_box_orient = np.array([np.cos(blue_box_yaw), np.sin(blue_box_yaw)])
    obs_parts.append(blue_box_pos)
    obs_parts.append(blue_box_orient)
    
    return np.concatenate(obs_parts).astype(np.float32)


def build_observation_insertion(episode_data: Dict, t: int) -> np.ndarray:
    """Build the insertion-task observation matching the env's get_observation().

    Layout: robot j_pos[:2], then per-box (pos[:2], orient[2] = cos/sin(yaw))
    for boxes 1-3, then the three target positions (xy only).
    """
    robot_data = episode_data['robot']

    obs_parts = []

    robot_pos = robot_data['j_pos'][t][:2]
    obs_parts.append(robot_pos)

    box1_pos = episode_data['box-1']['pos'][t][:2]  # only x,y
    box1_quat_full = episode_data['box-1']['quat'][t]
    box1_euler = quat2euler(box1_quat_full)
    box1_yaw = box1_euler[-1]
    box1_orient = np.array([np.cos(box1_yaw), np.sin(box1_yaw)])
    obs_parts.append(box1_pos)
    obs_parts.append(box1_orient)

    box2_pos = episode_data['box-2']['pos'][t][:2]
    box2_quat_full = episode_data['box-2']['quat'][t]
    box2_euler = quat2euler(box2_quat_full)
    box2_yaw = box2_euler[-1]
    box2_orient = np.array([np.cos(box2_yaw), np.sin(box2_yaw)])
    obs_parts.append(box2_pos)
    obs_parts.append(box2_orient)

    box3_pos = episode_data['box-3']['pos'][t][:2]
    box3_quat_full = episode_data['box-3']['quat'][t]
    box3_euler = quat2euler(box3_quat_full)
    box3_yaw = box3_euler[-1]
    box3_orient = np.array([np.cos(box3_yaw), np.sin(box3_yaw)])
    obs_parts.append(box3_pos)
    obs_parts.append(box3_orient)

    target1_pos = episode_data['box-target-1']['pos'][t][:2]
    target2_pos = episode_data['box-target-2']['pos'][t][:2]
    target3_pos = episode_data['box-target-3']['pos'][t][:2]
    obs_parts.append(target1_pos)
    obs_parts.append(target2_pos)
    obs_parts.append(target3_pos)
    
    return np.concatenate(obs_parts).astype(np.float32)


def get_dataset(env_name: str, data_dir: Optional[str] = None) -> Dataset:
    """Load and process the D3IL dataset into an OGPO-compatible Dataset."""
    if '-state' in env_name:
        task_name = env_name.replace('-state', '')
    elif '-image' in env_name:
        task_name = env_name.replace('-image', '')
    else:
        task_name = env_name

    task_to_data_dir = {
        'gate_insertion': 'inserting',
        'stacking': 'stacking',
    }

    if data_dir is None:
        data_subdir = task_to_data_dir.get(task_name, task_name)
        data_dir = f'./data/{data_subdir}/'

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    pkl_files = sorted(data_path.rglob('*.pkl'))

    if len(pkl_files) == 0:
        raise ValueError(f"No pickle files found in {data_path}")

    print(f"Loading D3IL dataset '{task_name}' from {data_path}")
    print(f"Found {len(pkl_files)} episode files")

    total_steps = 0
    valid_files = []
    for filepath in pkl_files:
        episode_data = np.load(filepath, allow_pickle=True)
        print(type(episode_data))
        if isinstance(episode_data, dict):
            print(episode_data.keys())
            if episode_data.get('robot', -1) != -1:
                traj_len = len(episode_data['robot']['j_pos'])
                if traj_len > 0:
                    total_steps += traj_len
                    valid_files.append(filepath)
    
    if total_steps == 0:
        raise ValueError(f"No valid episodes found in {data_path}")
    
    print(f"Found {len(valid_files)} valid episodes with {total_steps} total timesteps")
    
    first_episode = np.load(valid_files[0], allow_pickle=True)

    if 'stacking' in task_name:
        sample_obs = build_observation_stacking(first_episode, 0)
    else:  # gate_insertion
        sample_obs = build_observation_insertion(first_episode, 0)

    robot_data = first_episode['robot']
    action_parts = []
    action_parts.append(robot_data['des_j_pos'][0])
    if 'gripper_width' in robot_data:
        gripper = robot_data['gripper_width'][0]
        if np.isscalar(gripper):
            action_parts.append(np.array([gripper]))
        else:
            action_parts.append(gripper)
    sample_action = np.concatenate(action_parts, axis=-1)

    all_obs = np.zeros((total_steps, sample_obs.shape[0]), dtype=np.float32)
    all_actions = np.zeros((total_steps, sample_action.shape[0]), dtype=np.float32)
    all_rewards = np.zeros(total_steps, dtype=np.float32)
    all_terminals = np.zeros(total_steps, dtype=np.float32)

    current_idx = 0
    for filepath in tqdm(valid_files, desc=f"Processing '{task_name}'"):
        try:
            episode_data = np.load(filepath, allow_pickle=True)
            robot_data = episode_data['robot']
            traj_len = len(robot_data['j_pos'])
            
            if traj_len == 0:
                continue
            
            idx_slice = slice(current_idx, current_idx + traj_len)

            obs_list = []
            for t in range(traj_len):
                if 'stacking' in task_name:
                    obs = build_observation_stacking(episode_data, t)
                else:  # gate_insertion
                    obs = build_observation_insertion(episode_data, t)
                obs_list.append(obs)
            
            episode_obs = np.array(obs_list, dtype=np.float32)

            action_list = []
            for t in range(traj_len):
                action_parts = []
                action_parts.append(robot_data['des_j_pos'][t])

                if 'gripper_width' in robot_data:
                    gripper = robot_data['gripper_width'][t]
                    if np.isscalar(gripper):
                        action_parts.append(np.array([gripper]))
                    else:
                        action_parts.append(gripper)
                
                action_list.append(np.concatenate(action_parts, axis=-1))
            
            episode_actions = np.array(action_list, dtype=np.float32)

            obs_mean = np.mean(episode_obs, axis=0)
            obs_std = np.std(episode_obs, axis=0)
            action_mean = np.mean(episode_actions, axis=0)
            action_std = np.std(episode_actions, axis=0)
            
            print(f"\n[Episode {filepath.name}]")
            print(f"  Obs shape: {episode_obs.shape}, Action shape: {episode_actions.shape}")
            print(f"  Obs mean: {obs_mean[:5]}... (showing first 5)")
            print(f"  Obs std:  {obs_std[:5]}... (showing first 5)")
            print(f"  Action mean: {action_mean}")
            print(f"  Action std:  {action_std}")

            if not np.all(np.isfinite(episode_obs)):
                nan_count_obs = np.sum(np.isnan(episode_obs))
                inf_count_obs = np.sum(np.isinf(episode_obs))
                print(f"  WARNING: Found {nan_count_obs} NaN and {inf_count_obs} Inf values in observations!")
            if not np.all(np.isfinite(episode_actions)):
                nan_count_act = np.sum(np.isnan(episode_actions))
                inf_count_act = np.sum(np.isinf(episode_actions))
                print(f"  WARNING: Found {nan_count_act} NaN and {inf_count_act} Inf values in actions!")

            # Sparse reward: 1.0 at the last step if the demonstration succeeded.
            episode_rewards = np.zeros(traj_len, dtype=np.float32)
            if 'context' in episode_data:
                context = episode_data['context']
                if isinstance(context, dict) and context.get('success', True):
                    episode_rewards[-1] = 1.0
            else:
                episode_rewards[-1] = 1.0

            episode_terminals = np.zeros(traj_len, dtype=np.float32)
            episode_terminals[-1] = 1.0

            all_obs[idx_slice] = episode_obs
            all_actions[idx_slice] = episode_actions
            all_rewards[idx_slice] = episode_rewards
            all_terminals[idx_slice] = episode_terminals

            current_idx += traj_len

        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
            continue

    if current_idx < total_steps:
        all_obs = all_obs[:current_idx]
        all_actions = all_actions[:current_idx]
        all_rewards = all_rewards[:current_idx]
        all_terminals = all_terminals[:current_idx]
    
    print(f"\n{'='*60}")
    print(f"Dataset loaded with {current_idx} timesteps")
    print(f"Observation shape: {all_obs.shape}")
    print(f"Action shape: {all_actions.shape}")

    print(f"\n[Overall Dataset Statistics]")
    print(f"  Observations:")
    print(f"    Mean: {np.mean(all_obs, axis=0)[:5]}... (showing first 5 dims)")
    print(f"    Std:  {np.std(all_obs, axis=0)[:5]}... (showing first 5 dims)")
    print(f"    Min:  {np.min(all_obs, axis=0)[:5]}... (showing first 5 dims)")
    print(f"    Max:  {np.max(all_obs, axis=0)[:5]}... (showing first 5 dims)")
    
    print(f"  Actions:")
    print(f"    Mean: {np.mean(all_actions, axis=0)}")
    print(f"    Std:  {np.std(all_actions, axis=0)}")
    print(f"    Min:  {np.min(all_actions, axis=0)}")
    print(f"    Max:  {np.max(all_actions, axis=0)}")

    total_non_finite_obs = np.sum(~np.isfinite(all_obs))
    total_non_finite_act = np.sum(~np.isfinite(all_actions))
    if total_non_finite_obs > 0:
        print(f"\n  ⚠️  WARNING: Found {total_non_finite_obs} non-finite (NaN/Inf) values in observations!")
    if total_non_finite_act > 0:
        print(f"  ⚠️  WARNING: Found {total_non_finite_act} non-finite (NaN/Inf) values in actions!")

    if total_non_finite_obs == 0 and total_non_finite_act == 0:
        print(f"\n  ✓ No non-finite (NaN/Inf) values detected in dataset")
    
    print(f"{'='*60}\n")

    next_observations = np.roll(all_obs, -1, axis=0)
    episode_ends = np.where(all_terminals == 1.0)[0]
    for end_idx in episode_ends:
        if end_idx < len(all_terminals) - 1:
            next_observations[end_idx] = all_obs[end_idx]

    masks = 1.0 - all_terminals
    
    return Dataset.create(
        observations=all_obs,
        actions=all_actions,
        rewards=all_rewards,
        terminals=all_terminals,
        masks=masks,
        next_observations=next_observations,
    )