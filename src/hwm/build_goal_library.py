"""Build goal_library.npz and trajectory_dataset.npz from human playthrough NPZ files.

Outputs
-------
data/goal_library.npz
    goal_frames          (22, 64, 64, 3) uint8  — one canonical frame per achievement
    goal_names           (22,) str              — achievement names (ACHIEVEMENT_NAMES order)
    goal_achievement_steps (22,) int64          — timestep of first unlock in best trajectory
    goal_source_files    (22,) str              — which NPZ file the frame came from

data/trajectory_dataset.npz
    obs                  (N, 64, 64, 3) uint8   — concatenated frames across all episodes
    actions              (N,)          int64    — action at each frame
    trajectory_boundaries (100,)       int64    — start index in obs[] for each episode

Usage
-----
    python src/hwm/build_goal_library.py
    python src/hwm/build_goal_library.py --npz_dir data/human_crafter --out_dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make src/ importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    ACHIEVEMENT_NAMES,
    DATA_OUT,
    GOAL_LIBRARY,
    NPZ_DIR,
    TRAJ_DATASET,
    ach_key,
)


def _load_npz(path: str) -> dict:
    """Load an NPZ file and return it as a plain dict of numpy arrays."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_goal_library(npz_dir: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")
    print(f"Found {len(npz_files)} trajectory files.")

    # ── Pass 1: find best (earliest first-unlock) frame per achievement ───────
    # best[name] = {'frame': np.ndarray, 'step': int, 'file': str}
    best: dict[str, dict] = {}

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    boundaries: list[int] = []
    total = 0

    for fpath in npz_files:
        d = _load_npz(str(fpath))

        images  = d["image"]   # (T, 64, 64, 3) uint8
        actions = d["action"]  # (T,) int64
        T = len(images)

        boundaries.append(total)
        all_obs.append(images)
        all_actions.append(actions)
        total += T

        for name in ACHIEVEMENT_NAMES:
            col = d[ach_key(name)]                   # (T,) cumulative count
            diffs = np.diff(col.astype(np.int64), prepend=0)  # (T,)
            event_steps = np.where(diffs > 0)[0]

            if len(event_steps) == 0:
                continue  # achievement never reached in this trajectory

            t = int(event_steps[0])  # first unlock timestep

            # Keep this trajectory's frame if it's the earliest seen so far
            if name not in best or t < best[name]["step"]:
                # Goal frame = the frame at the unlock moment (boundary-safe)
                frame_idx = min(max(0, t), T - 1)
                best[name] = {
                    "frame": images[frame_idx].copy(),
                    "step":  t,
                    "file":  fpath.name,
                }

    print(f"\nTotal frames: {total}")
    print(f"Achievements with at least one unlock: {len(best)}/{len(ACHIEVEMENT_NAMES)}")

    # ── Build goal_library arrays ─────────────────────────────────────────────
    H, W = 64, 64
    goal_frames = np.zeros((len(ACHIEVEMENT_NAMES), H, W, 3), dtype=np.uint8)
    goal_steps  = np.full(len(ACHIEVEMENT_NAMES), -1, dtype=np.int64)
    goal_files  = np.array([""] * len(ACHIEVEMENT_NAMES), dtype=object)
    goal_names  = np.array(ACHIEVEMENT_NAMES, dtype=object)

    for i, name in enumerate(ACHIEVEMENT_NAMES):
        if name in best:
            goal_frames[i] = best[name]["frame"]
            goal_steps[i]  = best[name]["step"]
            goal_files[i]  = best[name]["file"]
            print(f"  {name:30s}: t={best[name]['step']:4d}  ({best[name]['file']})")
        else:
            print(f"  {name:30s}: NOT FOUND — leaving frame blank (all zeros)")

    goal_lib_path = out_path / "goal_library.npz"
    np.savez(
        goal_lib_path,
        goal_frames=goal_frames,
        goal_names=goal_names,
        goal_achievement_steps=goal_steps,
        goal_source_files=goal_files,
    )
    print(f"\nSaved goal library → {goal_lib_path}")

    # ── Build trajectory_dataset arrays ──────────────────────────────────────
    obs_concat     = np.concatenate(all_obs, axis=0)      # (N, 64, 64, 3)
    actions_concat = np.concatenate(all_actions, axis=0)  # (N,)
    boundaries_arr = np.array(boundaries, dtype=np.int64)

    traj_path = out_path / "trajectory_dataset.npz"
    np.savez(
        traj_path,
        obs=obs_concat,
        actions=actions_concat,
        trajectory_boundaries=boundaries_arr,
    )
    print(f"Saved trajectory dataset → {traj_path}")
    print(f"  obs shape:    {obs_concat.shape}")
    print(f"  actions shape:{actions_concat.shape}")
    print(f"  n_episodes:   {len(boundaries_arr)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build goal_library.npz and trajectory_dataset.npz from human NPZ files."
    )
    parser.add_argument("--npz_dir", default=NPZ_DIR,
                        help="Directory containing human playthrough .npz files")
    parser.add_argument("--out_dir", default=DATA_OUT,
                        help="Output directory for generated .npz files")
    args = parser.parse_args()
    build_goal_library(args.npz_dir, args.out_dir)


if __name__ == "__main__":
    main()
