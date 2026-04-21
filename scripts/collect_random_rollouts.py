#!/usr/bin/env python3
"""
Collect Crafter rollouts with uniform-random actions.

Saves a pickle compatible with ``hwm.data.mixed_sampler.trajectories_from_buffer_dict``::

    obs: (N, H, W, C) uint8
    actions: (N,) int64
    episode_ends: (E,) int64 — cumulative exclusive end indices (np.cumsum of episode lengths)

Default output: data/crafter/random_rollouts/random_500k.pkl

Usage::

    python scripts/collect_random_rollouts.py --steps 500000 --output data/crafter/random_rollouts/random_500k.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import crafter
except ImportError as e:
    raise SystemExit("Install crafter: pip install crafter") from e


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument(
        "--output",
        type=str,
        default="data/crafter/random_rollouts/random_500k.pkl",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    env = crafter.Env()

    obs_list: list[np.ndarray] = []
    act_list: list[int] = []
    episode_lengths: list[int] = []

    obs = env.reset()
    ep_steps = 0
    for _ in range(args.steps):
        a = int(env.action_space.sample())
        obs_list.append(np.asarray(obs, dtype=np.uint8))
        act_list.append(a)
        ep_steps += 1

        obs, _reward, done, _info = env.step(a)
        if done:
            episode_lengths.append(ep_steps)
            obs = env.reset()
            ep_steps = 0

    if ep_steps > 0:
        episode_lengths.append(ep_steps)

    obs_arr = np.stack(obs_list, axis=0)
    actions_arr = np.asarray(act_list, dtype=np.int64)
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64))

    assert obs_arr.shape[0] == len(actions_arr) == int(episode_ends[-1])

    out = {
        "obs": obs_arr,
        "actions": actions_arr,
        "episode_ends": episode_ends,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    total = len(actions_arr)
    counts = np.bincount(actions_arr, minlength=env.action_space.n).astype(np.float64)
    freqs = counts / max(1, total)
    print(f"Saved {total} transitions to {out_path.resolve()}")
    print(f"Episodes: {len(episode_lengths)}  steps min/mean/max: "
          f"{min(episode_lengths)} / {np.mean(episode_lengths):.1f} / {max(episode_lengths)}")
    print("Action frequency (min should stay > 0.005 with uniform sampling):")
    for i in range(env.action_space.n):
        print(f"  action {i:2d}: {freqs[i]:.4f}")
    if freqs.min() < 0.005:
        print("WARNING: at least one action < 0.5% frequency (unexpected for uniform policy).")


if __name__ == "__main__":
    main()
