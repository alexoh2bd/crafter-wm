"""Convert human NPZ playthroughs to a CrafterDataset-compatible pickle.

Reads the sorted list of .npz files from npz_dir, skips the eval episodes
defined by EVAL_EP_INDICES, and writes a single pkl containing the 95 train
trajectories in the format expected by ledata.CrafterDataset:

    {
        "trajectories": [
            {"obs": [np.ndarray(64,64,3) uint8, ...], "actions": [int, ...]},
            ...
        ],
        "action_dim": 17,
    }

Usage
-----
    python src/hwm/build_human_pkl.py
    python src/hwm/build_human_pkl.py --npz_dir data/crafter/human \
        --out_path data/crafter/human/crafter_human_train.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import ACTION_DIM, EVAL_EP_INDICES, NPZ_DIR


def build_human_pkl(
    npz_dir: str,
    out_path: str,
    eval_ep_indices: list[int] | tuple[int, ...] = (),
) -> None:
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    eval_set = set(eval_ep_indices)
    train_files = [(i, f) for i, f in enumerate(npz_files) if i not in eval_set]

    print(f"Found {len(npz_files)} NPZ files total.")
    print(f"Holding out eval episodes: {sorted(eval_set)}")
    print(f"Building pkl from {len(train_files)} train episodes...")

    trajectories = []
    total_frames = 0

    for ep_i, fpath in train_files:
        d = np.load(str(fpath), allow_pickle=True)
        images  = d["image"]   # (T, 64, 64, 3) uint8
        actions = d["action"]  # (T,) int
        T = len(images)

        trajectories.append({
            "obs":     list(images),           # list of (64, 64, 3) uint8 arrays
            "actions": list(actions.astype(int)),
        })
        total_frames += T

    print(f"Total trajectories: {len(trajectories)}")
    print(f"Total frames:       {total_frames}")
    print(f"Mean frames/ep:     {total_frames / max(len(trajectories), 1):.1f}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "trajectories": trajectories,
        "action_dim":   ACTION_DIM,
    }
    with open(out, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    default_out = str(
        Path(NPZ_DIR) / "crafter_human_train.pkl"
    )
    parser = argparse.ArgumentParser(
        description="Convert human NPZ episodes to CrafterDataset pkl (train split only)."
    )
    parser.add_argument("--npz_dir",  default=NPZ_DIR,
                        help="Directory containing human playthrough .npz files")
    parser.add_argument("--out_path", default=default_out,
                        help="Output path for the pickle file")
    parser.add_argument("--eval_ep_indices", type=int, nargs="*",
                        default=EVAL_EP_INDICES,
                        help="0-based episode indices to exclude (eval holdout)")
    args = parser.parse_args()
    build_human_pkl(args.npz_dir, args.out_path,
                    eval_ep_indices=args.eval_ep_indices)


if __name__ == "__main__":
    main()
