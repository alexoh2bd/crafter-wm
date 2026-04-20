"""Achievement probe cost functions for planning.

Trains one logistic regression classifier per Crafter achievement using
pre-encoded LeWM latents (Z) from latents.npz and per-frame achievement
labels extracted from the raw human playthrough NPZ files.

The probe probability replaces the broken image-matching L1 cost:
    cost(z) = 1 - P(achievement_active | z)

Usage
-----
Fit probes (also serves as a diagnostic — print per-achievement accuracy):

    python src/hwm/probe.py
    python src/hwm/probe.py --npz_dir data/human_crafter \
                             --latents data/latents.npz \
                             --save data/probes.pkl

Then use in planning via evaluate.py --cost probe.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    ACHIEVEMENT_NAMES,
    DATA_OUT,
    LATENTS_CACHE,
    NPZ_DIR,
    N_ACHIEVEMENTS,
    ach_key,
)


# ── Label construction ────────────────────────────────────────────────────────

def build_achievement_labels(
    npz_dir: str,
    Z: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Build a binary (N, 22) label matrix from raw human playthrough NPZ files.

    Label[t, a] = 1 if achievement a has been completed at least once by step t
    within that episode (cumulative count > 0).

    Files are loaded in sorted() order — identical to build_goal_library.py —
    so episode i in Z corresponds to sorted(npz_files)[i].

    Args:
        npz_dir:    Directory containing human playthrough .npz files.
        Z:          (N, D) pre-encoded latents from latents.npz.
        boundaries: (K,) episode start indices into Z.

    Returns:
        labels: (N, 22) float32 binary label matrix.
    """
    npz_files = sorted(Path(npz_dir).glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    N = len(Z)
    labels = np.zeros((N, N_ACHIEVEMENTS), dtype=np.float32)

    n_eps = len(boundaries)
    ep_ends = np.concatenate([boundaries[1:], [N]])

    for ep_i, fpath in enumerate(npz_files):
        if ep_i >= n_eps:
            break
        start = int(boundaries[ep_i])
        end   = int(ep_ends[ep_i])
        T     = end - start

        d = np.load(fpath, allow_pickle=True)

        for a_idx, name in enumerate(ACHIEVEMENT_NAMES):
            col_key = ach_key(name)
            if col_key not in d:
                continue
            counts = d[col_key].astype(np.int64)  # (T_file,) cumulative
            # Truncate/align to the window we have in Z
            T_file = min(len(counts), T)
            labels[start : start + T_file, a_idx] = (counts[:T_file] > 0).astype(np.float32)

    print(f"Built achievement labels: {labels.shape}  "
          f"mean positive rate = {labels.mean():.4f}")
    return labels


# ── Probe fitting ─────────────────────────────────────────────────────────────

def fit_probes(
    Z: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    C: float = 1.0,
    test_frac: float = 0.2,
    seed: int = 42,
) -> dict:
    """Fit one logistic regression per achievement and save to save_path.

    Prints a per-achievement accuracy table.  Target >70% per achievement
    to confirm the LeWM encoder captures task-relevant information.

    Args:
        Z:          (N, D) latent matrix.
        labels:     (N, 22) binary label matrix.
        save_path:  Where to pickle the probe dict.
        C:          Inverse regularization strength for LogisticRegression.
        test_frac:  Fraction of data held out for accuracy reporting.
        seed:       Random seed for train/test split.

    Returns:
        probe_dict: {'probes': {ach_name: fitted_clf}, 'ach_names': [...]}
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(Z))
    rng.shuffle(idx)
    split = int(len(idx) * (1 - test_frac))
    train_idx, test_idx = idx[:split], idx[split:]

    Z_train, Z_test = Z[train_idx], Z[test_idx]

    probes: dict[str, LogisticRegression] = {}
    print(f"\n{'Achievement':<30}  {'Pos%':>6}  {'Acc':>6}  {'Note'}")
    print("-" * 62)

    for a_idx, name in enumerate(ACHIEVEMENT_NAMES):
        y_train = labels[train_idx, a_idx]
        y_test  = labels[test_idx,  a_idx]

        pos_rate = y_train.mean()

        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                 class_weight="balanced", random_state=seed)
        clf.fit(Z_train, y_train)

        acc = clf.score(Z_test, y_test)

        note = ""
        if pos_rate < 0.01:
            note = "too few positives"
        elif acc < 0.60:
            note = "<<< WARNING: below 60%"
        elif acc < 0.70:
            note = "borderline"

        print(f"  {name:<28}  {pos_rate:>6.2%}  {acc:>6.2%}  {note}")
        probes[name] = clf

    print("-" * 62)

    probe_dict = {"probes": probes, "ach_names": list(ACHIEVEMENT_NAMES)}
    save = Path(save_path)
    save.parent.mkdir(parents=True, exist_ok=True)
    with open(save, "wb") as f:
        pickle.dump(probe_dict, f)
    print(f"\nSaved probe dict → {save}")
    return probe_dict


# ── Load ──────────────────────────────────────────────────────────────────────

def load_probes(save_path: str) -> dict:
    """Load a probe dict saved by fit_probes."""
    with open(save_path, "rb") as f:
        return pickle.load(f)


# ── Cost function (used inside CEM) ──────────────────────────────────────────

def probe_cost_batch(
    probes: dict,
    Z_batch_np: np.ndarray,
    ach_name: str,
) -> np.ndarray:
    """Compute probe-based planning cost for a batch of latents.

    Args:
        probes:      {'probes': {name: clf}, ...} dict from fit_probes/load_probes.
        Z_batch_np:  (n_samples, D) float32 latent batch (numpy).
        ach_name:    Name of the target achievement.

    Returns:
        costs: (n_samples,) float32 — lower is better (= 1 - P(achievement active)).
    """
    clf = probes["probes"][ach_name]
    # predict_proba returns (n_samples, n_classes); class 1 = achievement active
    proba = clf.predict_proba(Z_batch_np)[:, 1]  # (n_samples,)
    return (1.0 - proba).astype(np.float32)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit per-achievement linear probes on LeWM latents"
    )
    parser.add_argument("--npz_dir",  default=NPZ_DIR,
                        help="Directory of human playthrough .npz files")
    parser.add_argument("--latents",  default=LATENTS_CACHE,
                        help="Path to latents.npz")
    parser.add_argument("--save",     default=str(Path(DATA_OUT) / "probes.pkl"),
                        help="Where to save the fitted probe dict")
    parser.add_argument("--C",        type=float, default=1.0)
    parser.add_argument("--test_frac",type=float, default=0.2)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    d = np.load(args.latents)
    Z          = d["Z"].astype(np.float32)
    boundaries = d["trajectory_boundaries"]
    print(f"Loaded latents: Z={Z.shape}  boundaries={boundaries.shape}")

    labels = build_achievement_labels(args.npz_dir, Z, boundaries)
    fit_probes(Z, labels, save_path=args.save, C=args.C,
               test_frac=args.test_frac, seed=args.seed)


if __name__ == "__main__":
    main()
