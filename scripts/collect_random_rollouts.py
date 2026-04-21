#!/usr/bin/env python3
"""
Collect Crafter rollouts with uniform-random actions.

Saves a pickle compatible with ``hwm.data.mixed_sampler.trajectories_from_buffer_dict``::

    obs: (N, H, W, C) uint8
    actions: (N,) int64
    episode_ends: (E,) int64 — cumulative exclusive end indices (np.cumsum of episode lengths)

**GPU:** Crafter's simulator is CPU-only; ``env.step`` does not run on CUDA. This script
speeds up collection by (1) pre-allocated NumPy buffers, and (2) optional ``--workers``
to run several independent envs in parallel processes (multi-core CPU).

Default output: data/crafter/random_rollouts/random_500k.pkl

Usage::

    python scripts/collect_random_rollouts.py --steps 500000 --output data/crafter/random_rollouts/random_500k.pkl
    python scripts/collect_random_rollouts.py --steps 500000 --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np

try:
    import crafter
except ImportError as e:
    raise SystemExit("Install crafter: pip install crafter") from e


def _split_steps(total: int, n_workers: int) -> list[int]:
    """Partition ``total`` into ``n_workers`` nonnegative integers that sum to ``total``."""
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    base = total // n_workers
    rem = total % n_workers
    return [base + (1 if i < rem else 0) for i in range(n_workers)]


def _collect_segment(
    n_steps: int,
    seed: int,
    log_every: int = 0,
    t0: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run one env for ``n_steps`` transitions; return obs, actions, episode length list."""
    rng = np.random.default_rng(seed)
    env = crafter.Env()
    obs0 = env.reset()
    h, w, c = np.asarray(obs0).shape
    n_act = int(env.action_space.n)

    obs_buf = np.empty((n_steps, h, w, c), dtype=np.uint8)
    act_buf = np.empty((n_steps,), dtype=np.int64)
    episode_lengths: list[int] = []

    t0 = t0 if t0 is not None else time.perf_counter()
    obs = obs0
    ep_steps = 0
    for step in range(n_steps):
        a = int(rng.integers(0, n_act))
        obs_buf[step] = np.asarray(obs, dtype=np.uint8)
        act_buf[step] = a
        ep_steps += 1

        obs, _reward, done, _info = env.step(a)
        if done:
            episode_lengths.append(ep_steps)
            obs = env.reset()
            ep_steps = 0

        if log_every and (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step + 1}/{n_steps}  "
                f"elapsed {elapsed:.1f}s  {(step + 1) / max(elapsed, 1e-6):.0f} steps/s",
                flush=True,
            )

    if ep_steps > 0:
        episode_lengths.append(ep_steps)

    return obs_buf, act_buf, episode_lengths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument(
        "--output",
        type=str,
        default="data/crafter/random_rollouts/random_500k.pkl",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Print progress every N steps (0 = quiet). Single-worker only.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel processes (each own env). CPU-only sim; uses multiple cores. Default 1.",
    )
    args = p.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    t0 = time.perf_counter()

    if args.workers == 1:
        print(
            f"Collecting {args.steps} steps (1 worker, pre-allocated buffers). "
            f"Crafter sim is CPU-only — GPU does not speed env.step. "
            f"Use --workers N for multi-core.",
            flush=True,
        )
        log_ev = args.log_every if args.log_every > 0 else 0
        obs_arr, actions_arr, episode_lengths = _collect_segment(
            args.steps,
            args.seed,
            log_every=log_ev,
            t0=t0,
        )
    else:
        sizes = _split_steps(args.steps, args.workers)
        print(
            f"Collecting {args.steps} steps across {args.workers} workers "
            f"sizes={sizes}. Crafter is CPU-only.",
            flush=True,
        )
        with mp.Pool(processes=args.workers) as pool:
            seeds = [args.seed + 100_003 * (i + 1) for i in range(args.workers)]
            chunks = pool.starmap(
                _collect_segment,
                [(sizes[i], seeds[i]) for i in range(args.workers)],
            )

        obs_arr = np.concatenate([c[0] for c in chunks], axis=0)
        actions_arr = np.concatenate([c[1] for c in chunks], axis=0)
        episode_lengths = []
        for c in chunks:
            episode_lengths.extend(c[2])

        if obs_arr.shape[0] != args.steps:
            raise RuntimeError(
                f"internal error: got {obs_arr.shape[0]} steps, expected {args.steps}"
            )

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
    env = crafter.Env()
    counts = np.bincount(actions_arr, minlength=env.action_space.n).astype(np.float64)
    freqs = counts / max(1, total)
    print(
        f"Done in {time.perf_counter() - t0:.1f}s. "
        f"Saved {total} transitions to {out_path.resolve()}",
        flush=True,
    )
    print(
        f"Episodes: {len(episode_lengths)}  steps min/mean/max: "
        f"{min(episode_lengths)} / {np.mean(episode_lengths):.1f} / {max(episode_lengths)}",
        flush=True,
    )
    print("Action frequency (min should stay > 0.005 with uniform sampling):")
    for i in range(env.action_space.n):
        print(f"  action {i:2d}: {freqs[i]:.4f}")
    if freqs.min() < 0.005:
        print("WARNING: at least one action < 0.5% frequency (unexpected for uniform policy).")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
