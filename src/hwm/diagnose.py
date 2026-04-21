"""
Diagnostic: measure per-action latent delta magnitudes from a trained LeWM.

Thesis being tested:
    Training data is dominated by movement actions. Crafting actions are rare.
    If the predictor is miscalibrated by this imbalance, it will produce
    disproportionately large latent deltas for rare (crafting) actions relative
    to common (movement) actions, despite rare actions often being environment
    no-ops.

Outputs:
    - Per-action mean/median/std of ||z_{t+1} - z_t||_2, averaged over N sampled states.
    - Training-set action frequency for the same actions.
    - Ratio of (latent delta magnitude) / (training frequency), which quantifies
      the miscalibration: high ratio = action has outsized predicted effect
      relative to how often the model has seen it.
    - A bar chart saved to `action_magnitudes.png` (or `--plot` path).

Interpretation:
    - Uniform latent delta magnitudes across actions → predictor is well-calibrated,
      the planning failure has a different cause.
    - Crafting actions produce ~5x+ the delta of movement actions → hypothesis
      confirmed, retrain with balanced data or uncertainty head.

Run (from repo root; ``src/`` is added to ``sys.path`` automatically)::

    python src/hwm/diagnose.py \\
        --checkpoint data/crafter/world_model/lewm_human_ft/best.pt \\
        --trajectory_dataset data/crafter/wm_cache/trajectory_dataset.npz

Or from ``src/`` with a ledata pickle buffer::

    python hwm/diagnose.py --checkpoint ../data/.../best.pt --buffer crafter_data.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow imports whether cwd is repo root or src/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hwm.constants import ACTION_DIM, CRAFTER_ACTIONS, TRAJ_DATASET
from hwm.models import load_lewm

ACTION_NAMES = list(CRAFTER_ACTIONS)

ACTION_GROUPS = {
    "movement": [0, 1, 2, 3, 4],
    "interact": [5],
    "sleep": [6],
    "place": [7, 8, 9, 10],
    "craft": [11, 12, 13, 14, 15, 16],
}


def load_crafter_buffer(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load offline Crafter buffer from ledata.collect_crafter_data pickle.

    Returns:
        obs: (T, C, H, W) uint8
        actions: (T,) int64
    """
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    obs_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []
    for traj in data["trajectories"]:
        obs_parts.append(np.stack(traj["obs"], axis=0))
        act_parts.append(np.array(traj["actions"], dtype=np.int64))
    obs = np.concatenate(obs_parts, axis=0)
    actions = np.concatenate(act_parts, axis=0)
    obs = np.transpose(obs, (0, 3, 1, 2))
    return obs, actions


def load_trajectory_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``trajectory_dataset.npz`` from ``build_goal_library.py``.

    Returns:
        obs: (N, C, H, W) uint8
        actions: (N,) int64
    """
    path = Path(path)
    d = np.load(path)
    obs = np.asarray(d["obs"])
    actions = np.asarray(d["actions"], dtype=np.int64)
    if obs.ndim != 4:
        raise ValueError(f"Expected obs (N,H,W,C), got shape {obs.shape}")
    # Stored as (N, 64, 64, 3) in wm_cache
    if obs.shape[-1] == 3:
        obs = np.transpose(obs, (0, 3, 1, 2))
    return obs, actions


def compute_action_frequencies(actions: np.ndarray) -> np.ndarray:
    """actions: (T,) int array. Returns (ACTION_DIM,) fraction-of-dataset per action."""
    counts = Counter(actions.tolist())
    total = len(actions)
    return np.array([counts.get(a, 0) / total for a in range(ACTION_DIM)], dtype=np.float64)


@torch.no_grad()
def measure_per_action_deltas(
    lewm: torch.nn.Module,
    states: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    For each of N states and each of ACTION_DIM actions, predict z_{t+1} and
    measure ||z_{t+1} - z_t||_2. Returns (N, ACTION_DIM) array.
    """
    lewm.eval()
    N = states.shape[0]
    deltas = np.zeros((N, ACTION_DIM), dtype=np.float32)

    z_curr = lewm.encode(states.to(device))
    if z_curr.dim() == 3:
        z_curr = z_curr[:, -1]

    for a in range(ACTION_DIM):
        a_oh = torch.zeros(N, 1, ACTION_DIM, device=device)
        a_oh[:, 0, a] = 1.0
        z_curr_seq = z_curr.unsqueeze(1)
        z_next = lewm.predictor(z_curr_seq, a_oh).squeeze(1)
        delta = (z_next - z_curr).norm(dim=-1)
        deltas[:, a] = delta.cpu().numpy()

    return deltas


def _resolve_data_source(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.buffer and args.trajectory_dataset:
        raise SystemExit("Use only one of --buffer or --trajectory_dataset.")
    if args.buffer:
        return load_crafter_buffer(args.buffer)
    if args.trajectory_dataset:
        return load_trajectory_npz(args.trajectory_dataset)
    traj = Path(TRAJ_DATASET)
    if traj.is_file():
        print(f"No data flag given; using default trajectory_dataset:\n  {traj}")
        return load_trajectory_npz(traj)
    raise SystemExit(
        "Provide --buffer (pickle) or --trajectory_dataset (.npz), or place "
        f"trajectory_dataset at:\n  {traj}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-action LeWM latent delta diagnostic")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="LeWM checkpoint (.pt) from letrain.py or HWM pipeline",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default=None,
        help="crafter_data.pkl (ledata.collect_crafter_data format)",
    )
    parser.add_argument(
        "--trajectory_dataset",
        type=str,
        default=None,
        help="trajectory_dataset.npz (obs, actions) from build_goal_library.py",
    )
    parser.add_argument("--n-states", type=int, default=256, help="Number of random states to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", type=str, default="action_magnitudes.png", help="Output bar chart path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lewm, _ = load_lewm(args.checkpoint, device=device)

    obs_all, actions_all = _resolve_data_source(args)
    rng = np.random.default_rng(args.seed)
    n = min(args.n_states, len(obs_all))
    idx = rng.choice(len(obs_all), size=n, replace=False)
    states = torch.from_numpy(obs_all[idx].astype(np.float32))
    if states.max() > 1.5:
        states = states / 255.0

    deltas = measure_per_action_deltas(lewm, states, device)
    freqs = compute_action_frequencies(actions_all)

    mean_d = deltas.mean(axis=0)
    median_d = np.median(deltas, axis=0)
    std_d = deltas.std(axis=0)
    miscal = mean_d / np.maximum(freqs, 1e-4)

    print(f"\n{'action':<24}{'freq':>10}{'mean Δz':>12}{'median Δz':>12}{'std Δz':>10}{'miscal':>10}")
    print("-" * 78)
    for a in range(ACTION_DIM):
        print(
            f"{ACTION_NAMES[a]:<24}"
            f"{freqs[a]:>10.4f}"
            f"{mean_d[a]:>12.3f}"
            f"{median_d[a]:>12.3f}"
            f"{std_d[a]:>10.3f}"
            f"{miscal[a]:>10.1f}"
        )

    print(f"\n{'group':<12}{'mean Δz':>12}{'mean freq':>12}{'mean miscal':>14}")
    print("-" * 50)
    for name, idxs in ACTION_GROUPS.items():
        g_delta = float(mean_d[idxs].mean())
        g_freq = float(freqs[idxs].mean())
        g_mis = float(miscal[idxs].mean())
        print(f"{name:<12}{g_delta:>12.3f}{g_freq:>12.4f}{g_mis:>14.1f}")

    x = np.arange(ACTION_DIM)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax0.bar(x, mean_d, color="steelblue", edgecolor="none")
    ax0.set_ylabel(r"mean $\|\Delta z\|_2$")
    ax0.set_title("Predicted one-step latent delta magnitude per action")
    ax1.bar(x, miscal, color="coral", edgecolor="none")
    ax1.set_ylabel(r"mean $\|\Delta z\|_2$ / freq")
    ax1.set_xlabel("action")
    ax1.set_title("Miscalibration ratio (higher = larger delta vs how often seen in buffer)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote plot to {Path(args.plot).resolve()}")


if __name__ == "__main__":
    main()
