import gymnasium as gym
import numpy as np
import minari
from tqdm import tqdm

from ogpo.utils.datasets import Dataset

def flatten_obs_dict(obs_dict):
    """Flatten a dict observation into a single numpy array (live env wrapper)."""
    goal_keys = sorted(obs_dict['desired_goal'].keys())
    
    desired_goal_parts = [np.atleast_1d(obs_dict['desired_goal'][key]) for key in goal_keys]
    desired_goal = np.concatenate(desired_goal_parts, axis=-1)
    
    achieved_goal_parts = [np.atleast_1d(obs_dict['achieved_goal'][key]) for key in goal_keys]
    achieved_goal = np.concatenate(achieved_goal_parts, axis=-1)
    
    return np.concatenate([obs_dict['observation'], desired_goal, achieved_goal], axis=-1)

class FrankaKitchenWrapper(gym.Wrapper):
    """Flattens FrankaKitchen-v1 dict observations into a single vector."""
    def __init__(self, env):
        super().__init__(env)
        sample_obs, _ = env.reset()
        flattened_sample = flatten_obs_dict(sample_obs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=flattened_sample.shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset(**kwargs)
        info['tasks_completed'] = 0
        info['success'] = 0.0
        return flatten_obs_dict(obs_dict), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        reward = reward - 7
        if reward == -3:
            reward = 0
        info['reward'] = reward
        info['tasks_completed'] = len(info.get('episode_task_completions', []))
        info['success'] = float(terminated)
        return flatten_obs_dict(obs_dict), reward, terminated, truncated, info

def make_env(env_name: str, **kwargs):
    """Creates the FrankaKitchen-v1 environment and applies the flattening wrapper."""
    dataset = minari.load_dataset(env_name, download=True)
    env = dataset.recover_environment(terminate_on_tasks_completed=True, max_episode_steps=600)
    env = FrankaKitchenWrapper(env)
    return env

def get_dataset(env_name: str):
    """Load and process the Franka Kitchen Minari dataset into a Dataset."""
    dataset = minari.load_dataset(env_name)
    total_steps = dataset.total_steps
    print(f"Loading Minari dataset '{env_name}' with {total_steps} timesteps...")

    # Find the first valid episode for shape inference (some episodes are empty).
    first_valid_episode = None
    for episode in dataset:
        if len(episode.actions) > 0 and len(episode.observations['observation']) > 0:
            first_valid_episode = episode
            break
    if first_valid_episode is None:
        raise ValueError("Dataset contains no valid episodes.")

    achieved_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['achieved_goal'].items()
    }
    desired_goal_t0 = {
        key: val[0] for key, val in first_valid_episode.observations['desired_goal'].items()
    }

    sample_obs_dict = {
        'observation': first_valid_episode.observations['observation'][0],
        'achieved_goal': achieved_goal_t0,
        'desired_goal': desired_goal_t0,
    }
    
    sample_flat_obs = flatten_obs_dict(sample_obs_dict)
    sample_action = first_valid_episode.actions[0]
    
    all_obs = np.zeros((total_steps, sample_flat_obs.shape[0]), dtype=np.float32)
    all_actions = np.zeros((total_steps, sample_action.shape[0]), dtype=np.float32)
    all_rewards = np.zeros(total_steps, dtype=np.float32)
    all_terminals = np.zeros(total_steps, dtype=np.float32)
    
    current_idx = 0
    for episode in tqdm(dataset, desc=f"Processing '{env_name}'"):
        if len(episode.actions) == 0 or len(episode.observations['observation']) == 0:
            continue

        num_steps = min(len(episode.actions), len(episode.observations['observation']))
        idx_slice = slice(current_idx, current_idx + num_steps)

        obs_part = episode.observations['observation'][:num_steps]

        goal_keys = sorted(episode.observations['desired_goal'].keys())

        desired_goal_parts = [episode.observations['desired_goal'][key][:num_steps] for key in goal_keys]
        desired_goal_flat = np.concatenate(desired_goal_parts, axis=1)

        achieved_goal_parts = [episode.observations['achieved_goal'][key][:num_steps] for key in goal_keys]
        achieved_goal_flat = np.concatenate(achieved_goal_parts, axis=1)

        episode_obs_flat = np.concatenate([obs_part, desired_goal_flat, achieved_goal_flat], axis=1)

        all_obs[idx_slice] = episode_obs_flat.astype(np.float32)
        all_actions[idx_slice] = episode.actions[:num_steps].astype(np.float32)
        all_rewards[idx_slice] = episode.rewards[:num_steps].astype(np.float32)
        dones = np.logical_or(episode.terminations, episode.truncations)
        all_terminals[idx_slice] = dones[:num_steps].astype(np.float32)
        current_idx += num_steps

    if current_idx < total_steps:
        all_obs = all_obs[:current_idx]
        all_actions = all_actions[:current_idx]
        all_rewards = all_rewards[:current_idx]
        all_terminals = all_terminals[:current_idx]

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