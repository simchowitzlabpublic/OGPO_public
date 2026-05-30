from os.path import expanduser
import os

import numpy as np
import gym
from gym.spaces import Box, Dict
import imageio
import h5py
import roboverse
from ogpo.utils.datasets import Dataset
from tqdm import tqdm

def _process_image(obs):
    obs = (obs * 255).astype(np.uint8)
    obs = np.reshape(obs, (3, 128, 128))
    return np.transpose(obs, (1, 2, 0))


class Roboverse(gym.ObservationWrapper):

    def __init__(self, 
                 env,
                 max_episode_length=40
                 ):
        super().__init__(env)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.observation_space = Dict(obs_dict)
        self.max_episode_length = max_episode_length
        self.t = 0
        self.n_episodes = 0
        self.env_step = 0

    def observation(self, observation):
        pixels = _process_image(observation['image'])
        return pixels

    def reset(self, **kwargs):
        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1

        obs = self.env.reset(**kwargs)
        return self.observation(obs), {}
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.observation(raw_obs)
        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1
        if reward > 0.:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0
        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info
    
    
def make_env(env_name, seed=0):
    env = roboverse.make(env_name, transpose_image=True)
    env = Roboverse(env)
    dataset = get_dataset(env, env_name)
    return env

def get_dataset(env, env_name):
    dataset_path = os.path.join(
        expanduser("~/.roboverse"),
        "pickplacedata_noise0.1",
        env_name,
        "train",
        "out40.npy"
    )
    demos = np.load(dataset_path, allow_pickle=True)
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    for ep in tqdm(demos):
        a = np.array(ep['actions'])
        dones = np.array(ep['terminals'])
        r = np.array(ep['rewards'])
        m = 1.0 - dones
        obs = np.array([o['image'] for o in ep['observations']]) # (T, 128, 128, 3)
        next_obs = np.array([o['image'] for o in ep['next_observations']]) # (T, 128, 128, 3)
        observations.append(obs)
        next_observations.append(next_obs)
        actions.append(a)
        terminals.append(dones)
        rewards.append(r)
        masks.append(m)
        
    num_transition = np.concatenate(actions, axis=0).shape[0]
    print(f"the number of demos is {len(observations)}")
    print(f"the number of transitions is {num_transition}")
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        masks=np.concatenate(masks, axis=0),
    )


if __name__ == "__main__":
    env = make_env("PutBallintoBowl-v0")
    dataset = get_dataset(env, "PutBallintoBowl-v0")
    print(dataset)
    