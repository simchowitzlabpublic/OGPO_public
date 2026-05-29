# envs/gymnasium_robotics_utils.py (Corrected)

import gymnasium
import minari
import numpy as np
from tqdm import tqdm

from ogpo.utils.datasets import Dataset

# A mapping from the environment name to its corresponding Minari dataset ID.
MINARI_DATASET_MAPPING = {
    'AdroitHandDoor-v1': 'D4RL/door/expert-v2',
    'AdroitHandHammer-v1': 'D4RL/hammer/expert-v2',
    'AdroitHandPen-v1': 'D4RL/pen/expert-v2',
    'AdroitHandRelocate-v1': 'D4RL/relocate/expert-v2',
}

def make_env(env_name: str, seed: int = 0, post_success_steps: int = 0):
    """
    Creates a Gymnasium Robotics environment.

    `post_success_steps` is accepted for signature parity with the robomimic
    adapter (called from make_vectorized_env) but ignored — Adroit episodes
    don't have a meaningful follow-through window after success.
    """
    sparse_name = env_name.replace('-v1', 'Sparse-v1')
    env = gymnasium.make(sparse_name)
    return env

# In envs/gymnasium_robotics_utils.py

def get_dataset(env_name: str):
    """
    Loads a Minari dataset for a given environment and converts it into the
    project's custom `Dataset` format.
    """
    dataset_id = MINARI_DATASET_MAPPING.get(env_name)
    if not dataset_id:
        raise ValueError(f"No Minari dataset mapping found for env: {env_name}")
    
    minari.download_dataset(dataset_id)
    dataset = minari.load_dataset(dataset_id)

    # Use the dataset's total_steps attribute for efficient pre-allocation.
    total_steps = dataset.total_steps
    print(f"Loading Minari dataset '{dataset_id}' with {total_steps} timesteps...")
    
    # Get shape information from the first episode's data.
    sample_obs = dataset[0].observations
    sample_action = dataset[0].actions
    
    # Pre-allocate arrays for performance.
    all_obs = np.zeros((total_steps, sample_obs.shape[1]), dtype=np.float32)
    all_actions = np.zeros((total_steps, sample_action.shape[1]), dtype=np.float32)
    all_rewards = np.zeros(total_steps, dtype=np.float32)
    all_terminals = np.zeros(total_steps, dtype=np.float32)
    
    current_idx = 0
    for episode in tqdm(dataset, desc=f"Processing '{dataset_id}'"):
        # --- FIX 1: Get the number of steps from the length of the actions array. ---
        num_steps = len(episode.actions)
        
        if num_steps == 0:
            continue
            
        idx_slice = slice(current_idx, current_idx + num_steps)
        
        # --- FIX 2: Slice observations to exclude the final, post-termination observation. ---
        # We need N observations for N transitions (actions).
        all_obs[idx_slice] = episode.observations[:-1]
        
        all_actions[idx_slice] = episode.actions
        all_rewards[idx_slice] = episode.rewards
        
        dones = np.logical_or(episode.terminations, episode.truncations)
        all_terminals[idx_slice] = dones.astype(np.float32)
        
        current_idx += num_steps

    # Create next_observations by shifting observations.
    # This works because all_obs now correctly contains the sequence of
    # starting observations for every transition in the dataset.
    next_observations = np.roll(all_obs, -1, axis=0)
    
    # At terminal states, the 'next_observation' should be the same as the current one.
    episode_ends = np.where(all_terminals == 1.0)[0]
    for end_idx in episode_ends:
        if end_idx < total_steps - 1:
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