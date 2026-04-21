"""
Mixed random + PPO trajectory sampler for LeWM training.

Buffers use ``ledata.collect_crafter_data`` pickle format:
``{"trajectories": [{"obs": [...], "actions": [...]}, ...], ...}``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_buffer(path: str | Path) -> dict[str, Any]:
    """Load a Crafter offline buffer pickle."""
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def trajectories_from_buffer_dict(data: dict[str, Any]) -> list:
    """Normalize ``ledata`` trajectories or flat random-rollout format to a list of trajectories.

    Flat format (from ``collect_random_rollouts``)::

        obs: (N, H, W, C) uint8, actions: (N,) int64,
        episode_ends: (E,) int64 — cumulative exclusive end indices (same as
        ``np.cumsum(episode_lengths)``).
    """
    if "trajectories" in data:
        return data["trajectories"]
    obs = np.asarray(data["obs"])
    actions = np.asarray(data["actions"], dtype=np.int64)
    ends = np.asarray(data["episode_ends"], dtype=np.int64)
    if obs.shape[0] != len(actions):
        raise ValueError("obs and actions length mismatch")
    trajs: list[dict[str, Any]] = []
    start = 0
    for end in ends:
        trajs.append(
            {
                "obs": obs[start:end],
                "actions": actions[start:end],
            }
        )
        start = int(end)
    if start != len(obs):
        raise ValueError(
            f"episode_ends last value {start} != N={len(obs)}"
        )
    return trajs


def _traj_len(traj: dict[str, Any]) -> int:
    o = traj["obs"]
    if isinstance(o, np.ndarray) and o.ndim == 4:
        return int(o.shape[0])
    return len(o)


def _slice_traj(traj: dict[str, Any], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    o = traj["obs"]
    a = traj["actions"]
    if isinstance(o, np.ndarray) and o.ndim == 4:
        obs_chunk = o[start:end]
    else:
        obs_chunk = np.stack(o[start:end], axis=0)
    act_chunk = np.asarray(a[start:end], dtype=np.int64)
    return obs_chunk, act_chunk


class MixedTransitionSampler:
    """
    Samples training sub-trajectories from two buffers with a fixed ratio.

    Default: 70% random, 30% PPO. Random data provides broad dynamics coverage
    including no-op crafting transitions; PPO data provides achievement coverage.

    Returns:
        obs: (B, T, H, W, C) uint8
        actions: (B, T) int64
    """

    def __init__(
        self,
        random_buffer_path: str | None = None,
        ppo_buffer_path: str | None = None,
        seq_len: int = 16,
        random_ratio: float = 0.7,
        seed: int = 0,
        random_trajs: list | None = None,
        ppo_trajs: list | None = None,
    ):
        self.seq_len = seq_len
        self.random_ratio = random_ratio
        self.rng = np.random.default_rng(seed)

        if random_trajs is None:
            if random_buffer_path is None:
                raise ValueError("Provide random_buffer_path or random_trajs")
            random_trajs = trajectories_from_buffer_dict(load_buffer(random_buffer_path))
        if ppo_trajs is None:
            if ppo_buffer_path is None:
                raise ValueError("Provide ppo_buffer_path or ppo_trajs")
            ppo_trajs = trajectories_from_buffer_dict(load_buffer(ppo_buffer_path))

        self.random_trajs = self._filter_long_enough(random_trajs, seq_len)
        self.ppo_trajs = self._filter_long_enough(ppo_trajs, seq_len)
        if not self.random_trajs:
            raise ValueError("No random trajectories long enough for seq_len")
        if not self.ppo_trajs:
            raise ValueError("No PPO trajectories long enough for seq_len")

    @staticmethod
    def _filter_long_enough(trajs: list, seq_len: int) -> list:
        return [t for t in trajs if _traj_len(t) >= seq_len]

    def _sample_one(self, trajs: list) -> tuple[np.ndarray, np.ndarray]:
        """Return (obs, actions) with shapes (T,H,W,C) and (T,)."""
        ti = int(self.rng.integers(0, len(trajs)))
        traj = trajs[ti]
        T = _traj_len(traj)
        start = int(self.rng.integers(0, T - self.seq_len + 1))
        end = start + self.seq_len
        obs, actions = _slice_traj(traj, start, end)
        return obs, actions

    def _sample_subtrajectories(
        self, trajs: list, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        obs_list: list[np.ndarray] = []
        act_list: list[np.ndarray] = []
        for _ in range(n):
            o, a = self._sample_one(trajs)
            obs_list.append(o)
            act_list.append(a)
        obs = np.stack(obs_list, axis=0)
        actions = np.stack(act_list, axis=0)
        return obs, actions

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        n_random = int(batch_size * self.random_ratio)
        n_ppo = batch_size - n_random
        parts_o: list[np.ndarray] = []
        parts_a: list[np.ndarray] = []
        if n_random > 0:
            o, a = self._sample_subtrajectories(self.random_trajs, n_random)
            parts_o.append(o)
            parts_a.append(a)
        if n_ppo > 0:
            o, a = self._sample_subtrajectories(self.ppo_trajs, n_ppo)
            parts_o.append(o)
            parts_a.append(a)
        obs = np.concatenate(parts_o, axis=0)
        actions = np.concatenate(parts_a, axis=0)
        perm = self.rng.permutation(batch_size)
        return obs[perm], actions[perm]
