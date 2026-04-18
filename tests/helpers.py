"""
Standalone test helpers (no pytest imports) — importable by all test modules.
"""

import pickle
from pathlib import Path

import numpy as np

ACTION_DIM = 17
IMG_H = IMG_W = 64


def make_synthetic_pkl(
    path: Path,
    n_trajectories: int = 3,
    traj_len: int = 40,
    action_dim: int = ACTION_DIM,
    collection_method: str = "heuristic",
) -> dict:
    """Build a minimal Crafter-format pickle without running any environment."""
    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n_trajectories):
        obs = [rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
               for _ in range(traj_len)]
        actions = [int(rng.integers(0, action_dim)) for _ in range(traj_len)]
        achievements = [[] for _ in range(traj_len)]
        trajectories.append({"obs": obs, "actions": actions, "achievements": achievements})

    data = {
        "trajectories": trajectories,
        "goal_library": {},
        "action_dim": action_dim,
        "collection_method": collection_method,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data
