from os.path import expanduser
import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY
from ogpo.utils.datasets import Dataset


# Per-task observation key configuration
# image_size: (H, W) matching --camera_height/--camera_width used in dataset_states_to_obs.py
TASK_CONFIG = {
    'lift': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'can': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'square': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'tool_hang': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['sideview_image'],
        'image_size': (240, 240),
    },
    'transport': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos'],
        'image_keys': ['shouldercamera0_image', 'shouldercamera1_image'],
        'image_size': (96, 96),
    },
}


def _get_task_config(env_name):
    """Get per-task observation key configuration."""
    task = env_name.split("-")[0]
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Supported: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task]


def is_robomimic_env(env_name):
    """Determine whether an env name refers to a robomimic environment."""
    try:
        if ("low_dim" not in env_name) and ("image" not in env_name):
            return False
        task, dataset_type, hdf5_type = env_name.split("-")
        return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in ("mh", "ph")
    except ValueError:
        return False

# Default initialization for backward compatibility (overridden per-task in make_env/get_dataset)
low_dim_keys = {"low_dim": ('robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object')}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


def make_env(env_name, seed=0, frame_stack=1, post_success_steps=0):
    """Create a robomimic env. Call get_dataset() first so metadata is downloaded."""
    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)
    task_cfg = _get_task_config(env_name)

    if "image" in env_name:
        image_keys = task_cfg['image_keys']
        proprio_keys = tuple(task_cfg['proprio_keys'])
        img_h, img_w = task_cfg['image_size']

        shape_meta = {
            'obs': {
                'rgb': {'shape': (3, img_h, img_w)},
            },
        }

        obs_keys = {"low_dim": list(proprio_keys), 'rgb': image_keys}
        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_keys)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=True,
        )
        env = RobomimicImageWrapper(
            env, shape_meta=shape_meta, image_keys=image_keys,
            proprio_keys=proprio_keys,
            low_dim_keys=tuple(task_cfg['low_dim_keys']),
            max_episode_length=max_episode_length, num_stack=frame_stack,
            post_success_steps=post_success_steps,
        )
    else:
        task_low_dim_keys = task_cfg['low_dim_keys']
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": task_low_dim_keys})

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
        )
        env = RobomimicLowdimWrapper(env, low_dim_keys=task_low_dim_keys, max_episode_length=max_episode_length, post_success_steps=post_success_steps)
    env.seed(seed)
    env.env.hard_reset = False
    return env

def _check_dataset_exists(env_name):
    task, dataset_type, hdf5_type = env_name.split("-")
    if hdf5_type == "image":
        file_name = "image_v15.hdf5"
    elif dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = f"low_dim_v15.hdf5"
    # Set ROBOMIMIC_DATASET_ROOT to point at your downloaded robomimic datasets;
    # otherwise we look under ~/.robomimic.
    dataset_root = os.environ.get("ROBOMIMIC_DATASET_ROOT")
    if dataset_root is None:
        dataset_root = os.path.join(expanduser("~"), ".robomimic")

    dataset_path = os.path.join(
        dataset_root,
        task,
        dataset_type,
        file_name,
    )

    return dataset_path

def get_dataset(env, env_name):
    dataset_path = _check_dataset_exists(env_name)
    task_cfg = _get_task_config(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")

    is_image_env = "image" in env_name

    image_keys = task_cfg['image_keys']
    proprio_keys = task_cfg['proprio_keys']
    task_low_dim_keys = task_cfg['low_dim_keys']

    num_stack = 1
    if is_image_env:
        current_env = env
        while hasattr(current_env, 'env'):
            if hasattr(current_env, 'num_stack'):
                num_stack = current_env.num_stack
                break
            current_env = current_env.env
        if hasattr(current_env, 'num_stack'):
            num_stack = current_env.num_stack

    # data holders — small fields use lists; large image fields are preallocated
    # below (uint8, written per-episode) so we never hold list + concatenated copy
    # simultaneously. For tool_hang (240x240) the float32 list+concat path peaked
    # at ~190 GB and OOM-killed the process.
    observations = []      # always proprioceptive state
    actions = []
    next_observations = []  # always proprioceptive state
    terminals = []
    rewards = []
    masks = []
    all_full_states = []    # full low-dim state for image envs (includes object pose)
    all_next_full_states = []

    images_arr = None       # preallocated (num_timesteps, H, W, 3*num_cams*num_stack) uint8
    next_images_arr = None
    img_cursor = 0

    for ep in demos:
        a = np.array(rm_dataset["data/{}/actions".format(ep)])
        dones = np.array(rm_dataset["data/{}/dones".format(ep)])
        r = np.array(rm_dataset["data/{}/rewards".format(ep)])
        r = r - 1.0  # Transform rewards from [0, 1] to [-1, 0].

        if is_image_env:
            # Keep images as uint8; encoders normalize /255 internally.
            img_obs_parts = [np.asarray(rm_dataset[f"data/{ep}/obs/{key}"]) for key in image_keys]
            img_next_parts = [np.asarray(rm_dataset[f"data/{ep}/next_obs/{key}"]) for key in image_keys]
            img_obs = np.concatenate(img_obs_parts, axis=3) if len(img_obs_parts) > 1 else img_obs_parts[0]
            img_next = np.concatenate(img_next_parts, axis=3) if len(img_next_parts) > 1 else img_next_parts[0]
            del img_obs_parts, img_next_parts

            if num_stack > 1:
                all_frames = np.concatenate([img_obs, img_next[-1:]], axis=0)
                pad = np.repeat(all_frames[:1], num_stack - 1, axis=0)
                padded_frames = np.concatenate([pad, all_frames], axis=0)
                stacks = [padded_frames[i : i + img_obs.shape[0] + 1] for i in range(num_stack)]
                stacked_frames = np.concatenate(stacks, axis=3)
                img_obs = stacked_frames[:-1]
                img_next = stacked_frames[1:]
                del all_frames, pad, padded_frames, stacks, stacked_frames

            n_ep = img_obs.shape[0]
            if images_arr is None:
                images_arr = np.empty((num_timesteps,) + img_obs.shape[1:], dtype=img_obs.dtype)
                next_images_arr = np.empty((num_timesteps,) + img_next.shape[1:], dtype=img_next.dtype)
            images_arr[img_cursor:img_cursor + n_ep] = img_obs
            next_images_arr[img_cursor:img_cursor + n_ep] = img_next
            img_cursor += n_ep
            del img_obs, img_next

            state_obs = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/obs/{k}"]) for k in proprio_keys], axis=-1)
            state_next = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]) for k in proprio_keys], axis=-1)
            observations.append(state_obs.astype(np.float32))
            next_observations.append(state_next.astype(np.float32))

            # Full low-dim state for the state-based critic (includes object pose).
            full_state_obs = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/obs/{k}"]) for k in task_low_dim_keys], axis=-1)
            full_state_next = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]) for k in task_low_dim_keys], axis=-1)
            all_full_states.append(full_state_obs.astype(np.float32))
            all_next_full_states.append(full_state_next.astype(np.float32))
        else:
            obs, next_obs = [], []
            for k in task_low_dim_keys:
                obs.append(np.array(rm_dataset[f"data/{ep}/obs/{k}"]))
            for k in task_low_dim_keys:
                next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]))
            obs = np.concatenate(obs, axis=-1)
            next_obs = np.concatenate(next_obs, axis=-1)
            observations.append(obs.astype(np.float32))
            next_observations.append(next_obs.astype(np.float32))

        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))

    dataset_fields = dict(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )
    if is_image_env:
        assert img_cursor == num_timesteps, f"image count {img_cursor} != {num_timesteps}"
        dataset_fields['images'] = images_arr
        dataset_fields['next_images'] = next_images_arr
        dataset_fields['full_states'] = np.concatenate(all_full_states, axis=0)
        dataset_fields['next_full_states'] = np.concatenate(all_next_full_states, axis=0)

    return Dataset.create(**dataset_fields)


class RobomimicLowdimWrapper(gym.Env):
    """Wrapper for Robomimic environments with state observations.

    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        post_success_steps=0,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.post_success_steps = post_success_steps
        self.env_step = 0
        self.n_episodes = 0
        self.t_succ = None

        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # Spaces use [-1, 1].
        low = np.full(env.action_dimension, fill_value=-1.)
        high = np.full(env.action_dimension, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self):
        raw_obs = self.env.get_observation()
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        self.t = 0
        self.t_succ = None
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if options is None:
            options = {}

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            self.env.reset()

        return self.get_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        # Transform reward from [0, 1] to [-1, 0]
        reward = reward - 1.0
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(raw_obs)
        else:
            obs = raw_obs

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        # Success / post-success reward shaping
        is_success_step = reward > -0.5

        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            # Post-success phase
            info["success"] = 1
            steps_since_succ = self.t - self.t_succ
            if is_success_step and steps_since_succ <= self.post_success_steps:
                reward += 5.0
            if steps_since_succ >= self.post_success_steps or not is_success_step:
                done = True

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}



class RobomimicImageWrapper(gym.Env):
    """Robomimic image environment wrapper.

    Returns dict observations: {'state': proprio_vec, 'image': image_array}
    so that the pipeline can independently route state and image to actor/critic.
    """
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        proprio_keys=(
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ),
        low_dim_keys=None,
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        num_stack=1,
        post_success_steps=0,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.post_success_steps = post_success_steps
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.t_succ = None
        self.num_stack = num_stack
        self.frame_stack = deque(maxlen=num_stack)
        self.proprio_keys = proprio_keys
        self.low_dim_keys = low_dim_keys or list(proprio_keys)

        # set up normalization for actions only
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.image_keys = image_keys

        # Image observation space
        rgb_shape = shape_meta["obs"]["rgb"]["shape"]  # (3, 96, 96) as (C, H, W)
        channels, height, width = rgb_shape
        num_cameras = len(image_keys)
        full_image_shape = (height, width, channels * num_cameras * num_stack)

        # Proprioception and full state dimensions from a dummy obs
        raw_obs = self.env.get_observation()
        proprio_dim = sum(raw_obs[k].shape[0] for k in self.proprio_keys if k in raw_obs)
        full_state_dim = sum(raw_obs[k].shape[0] for k in self.low_dim_keys if k in raw_obs)

        self.observation_space = Dict({
            'state': Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
            'image': Box(low=0, high=255, shape=full_image_shape, dtype=np.float32),
            'full_state': Box(low=-np.inf, high=np.inf, shape=(full_state_dim,), dtype=np.float32),
        })

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_single_observation(self, raw_obs):
        """Extract image, proprioception, and full low-dim state from a raw obs.

        Returns a dict with 'state' (proprio vec), 'image' (HWC float32), and
        'full_state' (full low-dim state including object pose).
        """
        img_parts = []
        for key in self.image_keys:
            if key in raw_obs:
                img_parts.append(raw_obs[key].astype(np.float32))  # (H, W, C)
        image = np.concatenate(img_parts, axis=2) if len(img_parts) > 1 else img_parts[0]

        # Proprioception
        state_parts = [raw_obs[k].astype(np.float32) for k in self.proprio_keys if k in raw_obs]
        state = np.concatenate(state_parts, axis=0)

        # Full low-dim state (includes object pose for state-based critic)
        full_state_parts = [raw_obs[k].astype(np.float32) for k in self.low_dim_keys if k in raw_obs]
        full_state = np.concatenate(full_state_parts, axis=0)

        return {'state': state, 'image': image, 'full_state': full_state}

    def _get_stacked_observation(self):
        assert len(self.frame_stack) == self.num_stack
        frames = list(self.frame_stack)
        stacked_image = np.concatenate([f['image'] for f in frames], axis=-1)
        current_state = frames[-1]['state']  # latest proprioception
        current_full_state = frames[-1]['full_state']  # full low-dim state
        return {'state': current_state, 'image': stacked_image, 'full_state': current_full_state}

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        if options is None:
            options = {}

        self.t = 0
        self.t_succ = None
        self.episode_return, self.episode_length = 0, 0
        self.n_episodes += 1

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            # random reset
            raw_obs = self.env.reset()
            
        # Initialize frame stack
        current_obs = self.get_single_observation(raw_obs)
        for _ in range(self.num_stack):
            self.frame_stack.append(current_obs)
            
        return self._get_stacked_observation(), {}

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        # Transform reward from [0, 1] to [-1, 0]
        reward = reward - 1.0

        # Update stack
        current_obs = self.get_single_observation(raw_obs)
        self.frame_stack.append(current_obs)
        obs = self._get_stacked_observation()

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)


        # Success / post-success reward shaping
        is_success_step = reward > -0.5

        if self.t_succ is None:
            if is_success_step:
                self.t_succ = self.t
                info["success"] = 1
                reward += 5.0
                if self.post_success_steps <= 0:
                    done = True
            else:
                info["success"] = 0
        else:
            # Post-success phase
            info["success"] = 1
            steps_since_succ = self.t - self.t_succ
            if is_success_step and steps_since_succ <= self.post_success_steps:
                reward += 5.0
            if steps_since_succ >= self.post_success_steps or not is_success_step:
                done = True

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

if __name__ == "__main__":
    # for testing
    import sys
    env_name = sys.argv[1] if len(sys.argv) > 1 else "square-mh-image"
    print(f"Testing: {env_name}")
    task_cfg = _get_task_config(env_name)
    print(f"Task config: {task_cfg}")
    env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    print(f"Observations shape: {dataset['observations'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
    if 'images' in dataset._dict:
        print(f"Images shape: {dataset['images'].shape}")
    print("OK")
