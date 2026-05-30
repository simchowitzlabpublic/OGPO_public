from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from ogpo.utils.augmentations import batched_random_crop, color_transform

eps = 1e-9

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, numpy_rng=None, **fields):
        """Create a dataset from the given fields."""
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data, numpy_rng=numpy_rng)

    def __init__(self, *args, numpy_rng=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Set outside the class.

        if numpy_rng is None:
            self.numpy_rng = np.random.default_rng(42)
        else:
            self.numpy_rng = numpy_rng

        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        # JAX RNG for color augmentation, derived from numpy_rng for reproducibility.
        self.rng = jax.random.PRNGKey(int(self.numpy_rng.integers(2**31)))

        # Per-trajectory success -> scores (1 for successful traj, eps otherwise).
        if 'is_success' in self._dict:
            traj_success_mask = np.zeros(self.size, dtype=np.bool_)
            for i in range(len(self.terminal_locs)):
                s = self.initial_locs[i]
                e = self.terminal_locs[i]
                succ = np.any(self['is_success'][s:e+1] > 0)
                traj_success_mask[s:e+1] = succ
            new_scores = np.where(traj_success_mask, 1.0, eps).astype(np.float32)
            self._dict['score'] = new_scores
            self._traj_success_mask = traj_success_mask
        else:
            self._traj_success_mask = np.zeros(self.size, dtype=np.bool_)

    def compute_mc_returns(self, discount):
        mc_returns = np.zeros_like(self['rewards'])
        for i in range(len(self.terminal_locs)):
            start_idx = self.initial_locs[i]
            end_idx = self.terminal_locs[i]

            G = 0.0
            for t in range(end_idx, start_idx - 1, -1):
                G = self['rewards'][t] + discount * G
                mc_returns[t] = G
        return mc_returns

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` uniformly random indices."""
        return self.numpy_rng.integers(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None, success_only=False):
        """Sample a batch of transitions.

        Uniform over the buffer by default; score-based sampling is handled
        explicitly in `sample_sequence` via `by_score` so that online RL
        batches are not silently success-weighted.
        """
        if idxs is None:
            if success_only:
                success_mask = self._traj_success_mask[:self.size]
                success_idxs = np.where(success_mask)[0]
                if len(success_idxs) < batch_size:
                    raise ValueError(f"Not enough successful transitions: {len(success_idxs)} < {batch_size}")
                idxs = self.numpy_rng.choice(success_idxs, size=batch_size, replace=False)
            else:
                idxs = self.numpy_rng.integers(self.size, size=batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # [ob[t - frame_stack + 1], ..., ob[t]]
            next_obs = []  # [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]]
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None and batch['observations'].ndim > 3:
            # Image-as-observation path.
            if self.numpy_rng.random() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        # Separate image fields.
        if self.p_aug is not None and 'images' in batch and batch['images'].ndim > 3:
            if self.numpy_rng.random() < self.p_aug:
                self.augment(batch, ['images', 'next_images'])
        if 'mc_returns' in self._dict and 'mc_returns' not in batch:
            batch['mc_returns'] = self['mc_returns'][idxs]
        return batch

    def sample_sequence(self, batch_size, sequence_length, discount, ret_next_act=False, ret_mc=False, filter_fn=None, success_only=None, by_score=False):
        """Sample length-T sequences.

        success_only: True -> only successful trajectories, False -> only
            unsuccessful, None -> no success filter.
        filter_fn(data_dict, num_starts) -> boolean mask over valid starts.
        by_score: sample start indices proportionally to their score values.
        """
        B = batch_size
        T = sequence_length
        max_start = self.size - T
        if max_start < 0:
            raise ValueError(f"Sequence length too long for dataset size. Need at least {T} transitions, but have {self.size}")

        starts = np.arange(max_start + 1, dtype=np.int64)
        mask = np.ones(max_start + 1, dtype=bool)

        if success_only is not None:
            succ_mask = self._traj_success_mask[:max_start + 1]
            mask &= (succ_mask if success_only else ~succ_mask)

        if filter_fn is not None:
            extra = filter_fn(self._dict, max_start + 1)
            if extra is None or len(extra) != max_start + 1:
                raise ValueError("filter_fn must return a boolean mask of length max_start + 1.")
            mask &= extra

        valid_indices = starts[mask]
        if len(valid_indices) == 0:
            raise ValueError("No transitions found matching filter")

        # Importance sample start indices proportional to score (by_score only).
        probs = None  # None -> uniform sampling
        if by_score:
            scores = self._dict.get('score', None)
            if scores is not None:
                w = np.asarray(scores[valid_indices], dtype=np.float64)
                total = w.sum()
                if np.isfinite(total) and total > 0:
                    probs = w / total

        idxs = self.numpy_rng.choice(valid_indices, size=B, p=probs)

        offs = np.arange(T, dtype=np.int64)[None, :]          # (1, T)
        seq_idxs = idxs[:, None].astype(np.int64) + offs      # (B, T)

        obs_seq         = self['observations'][seq_idxs]      # (B, T, obs_dim)
        next_obs_seq    = self['next_observations'][seq_idxs] # (B, T, obs_dim)
        actions_seq     = self['actions'][seq_idxs]           # (B, T, act_dim)
        rewards_seq     = self['rewards'][seq_idxs]           # (B, T)
        masks_seq       = self['masks'][seq_idxs]             # (B, T)
        terminals_seq   = self['terminals'][seq_idxs]         # (B, T)

        masks_prefix     = np.minimum.accumulate(masks_seq, axis=1)
        terminals_prefix = np.maximum.accumulate(terminals_seq, axis=1)

        # valid: 1 at i==0; for i>0 it's 1 - terminals_prefix at i-1.
        valid = np.ones_like(masks_seq, dtype=np.float32)
        valid[:, 1:] = 1.0 - terminals_prefix[:, :-1]

        # Prefix discounted return (no stop-at-terminal masking).
        rdtype = rewards_seq.dtype
        disc_pows = (discount ** np.arange(T)).astype(rdtype, copy=False)  # (T,)
        rewards_prefix = np.cumsum(rewards_seq * disc_pows[None, :], axis=1)

        first_obs = self['observations'][idxs]  # (B, obs_dim)
        last_obs = next_obs_seq[:, -1, ...]

        if self.p_aug is not None and first_obs.ndim > 3 and self.numpy_rng.random() < self.p_aug:
            tmp = {'observations': first_obs, 'next_observations': last_obs}
            self.augment(tmp, ['observations', 'next_observations'])
            first_obs = tmp['observations']
            last_obs = tmp['next_observations']

        result = dict(
            observations=first_obs.copy(),
            actions=actions_seq,
            masks=masks_prefix,
            rewards=rewards_prefix,
            terminals=terminals_prefix,
            valid=valid,
            next_observations=last_obs,
        )

        # Gather separate image fields when present.
        if 'images' in self._dict:
            result['images'] = self['images'][idxs]                 # (B, H, W, C)
            result['next_images'] = self['next_images'][seq_idxs[:, -1]]  # (B, H, W, C)
            if self.p_aug is not None and self.numpy_rng.random() < self.p_aug:
                self.augment(result, ['images', 'next_images'])

        if 'full_states' in self._dict:
            result['full_states'] = self['full_states'][idxs]
            result['next_full_states'] = self['next_full_states'][seq_idxs[:, -1]]

        if 'is_success' in self._dict:
            result['is_success'] = self['is_success'][seq_idxs]
    
        if ret_next_act:
            result['next_actions'] = self['actions'][np.minimum(seq_idxs + 1, self.size - 1)]

        if 'mc_returns' in self._dict:
            result['mc_returns'] = self['mc_returns'][idxs]
        return result

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = self.numpy_rng.integers(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: batched_random_crop(arr, crop_froms, padding) if len(arr.shape) == 4 else arr,
                batch[key],
            )
            self.rng, aug_rng = jax.random.split(self.rng)
            batch[key] = (color_transform(aug_rng, batch[key] / 255.0) * 255.0).astype(jnp.uint8)


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size, numpy_rng=None):
        """Create a replay buffer from an example transition."""

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        if 'is_success' not in buffer_dict:
            buffer_dict['is_success'] = np.zeros(size, dtype=np.float32)
        if 'score' not in buffer_dict:
            buffer_dict['score'] = np.zeros(size, dtype=np.float32)
        return cls(buffer_dict, numpy_rng=numpy_rng)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size, numpy_rng=None):
        """Create a replay buffer pre-filled from an initial dataset."""

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        init_size = get_size(init_dataset)

        if 'is_success' not in buffer_dict:
            buffer_dict['is_success'] = np.zeros(size, dtype=np.float32)
            if 'is_success' in init_dataset:
                buffer_dict['is_success'][:init_size] = init_dataset['is_success']
        if 'score' not in buffer_dict:
            buffer_dict['score'] = np.zeros(size, dtype=np.float32)
            if 'score' in init_dataset:
                buffer_dict['score'][:init_size] = init_dataset['score']
            elif 'is_success' in init_dataset:
                buffer_dict['score'][:init_size] = init_dataset['is_success']

        dataset = cls(buffer_dict, numpy_rng=numpy_rng)
        dataset.size = dataset.pointer = init_size
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

        self._traj_success_mask = np.zeros(self.max_size, dtype=np.bool_)

    def __reduce__(self):
        # Bypass __init__ on unpickling so our get/setstate fully control state.
        return (self.__class__.__new__, (self.__class__,), self.__getstate__())

    def __getstate__(self):
        """Support pickling."""
        state = self.__dict__.copy()
        # FrozenDict content might not be in __dict__.
        if '_dict' not in state and hasattr(self, '_dict'):
             state['_dict'] = self._dict
        # numpy_rng itself is unpicklable; save its state instead.
        if 'numpy_rng' in state:
            state['numpy_rng_state'] = state['numpy_rng'].__getstate__()
            del state['numpy_rng']
        return state

    def __setstate__(self, state):
        """Support unpickling."""
        if 'numpy_rng_state' in state:
            numpy_rng_state = state.pop('numpy_rng_state')
            state['numpy_rng'] = np.random.default_rng()
            state['numpy_rng'].__setstate__(numpy_rng_state)

        for key, value in state.items():
            object.__setattr__(self, key, value)

        object.__setattr__(self, '_hash', None)

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""
        transition = transition.copy()
        if 'is_success' in transition and 'score' not in transition:
            # eps (not 0) for unsuccessful steps to keep score-based sampling well-defined.
            transition['score'] = 1.0 if float(transition['is_success']) > 0 else eps
        elif 'is_success' not in transition:
            transition['is_success'] = 0.0
            transition['score'] = eps

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self._traj_success_mask[self.pointer] = (float(transition['is_success']) > 0)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
