"""Condition B — Linear regression (Ridge) dynamics + CEM planner.

Uses the frozen LeWM encoder to produce latent states, then fits a linear
dynamics model:
    z_{t+1} ≈ W_z @ z_t + W_a @ one_hot(a_t) + b

The Ridge model is fit once (offline) from the pre-encoded trajectory
dataset and cached in data/ridge_model.pkl.

At planning time, the rollout function is H sequential matrix multiplications
(numpy only — no GPU needed), and the same CEM structure as Condition A is
used to select the best action sequence.

Standalone usage:
    python src/hwm/plan_linear.py --achievement collect_wood --n_episodes 3
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    ACTION_DIM,
    CHECKPOINT,
    GOAL_LIBRARY,
    LATENT_DIM,
    LATENTS_CACHE,
    RIDGE_MODEL,
)
from hwm.models import load_lewm


# ── Ridge model fit / load ─────────────────────────────────────────────────────

class LinearDynamics:
    """Thin wrapper around a fitted Ridge regression for latent dynamics.

    z_{t+1} = W @ [z_t || one_hot(a_t)] + b
    """

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self.W = W    # (LATENT_DIM, LATENT_DIM + ACTION_DIM)
        self.b = b    # (LATENT_DIM,)

    def step(self, z: np.ndarray, a: int) -> np.ndarray:
        """Single-step prediction.

        Args:
            z: (D,) float32 latent state.
            a: int action index.

        Returns:
            z_next: (D,) float32 predicted next latent.
        """
        a_oh = np.zeros(ACTION_DIM, dtype=np.float32)
        a_oh[a] = 1.0
        x = np.concatenate([z, a_oh])  # (D + A,)
        return self.W @ x + self.b     # (D,)

    def rollout(self, z0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Roll out H steps.

        Args:
            z0:      (D,) initial latent.
            actions: (H,) int action sequence.

        Returns:
            z_traj: (H+1, D) trajectory including z0.
        """
        traj = [z0]
        z = z0.copy()
        for a in actions:
            z = self.step(z, int(a))
            traj.append(z)
        return np.stack(traj, axis=0)  # (H+1, D)


def fit_linear_dynamics(
    latents_path: str,
    save_path: str,
    alpha: float = 1.0,
) -> LinearDynamics:
    """Fit Ridge regression from latents.npz and save to save_path.

    Args:
        latents_path: Path to latents.npz (output of train_hwm_high pre-encode step).
        save_path:    Where to pickle the LinearDynamics object.
        alpha:        Ridge regularization strength.

    Returns:
        Fitted LinearDynamics object.
    """
    cache = Path(save_path)
    if cache.exists():
        print(f"Loading cached Ridge model from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        raise ImportError("scikit-learn is required for plan_linear.py: pip install scikit-learn")

    print("Fitting Ridge linear dynamics model from latents.npz ...")
    d = np.load(latents_path)
    Z           = d["Z"].astype(np.float32)        # (N, D)
    actions     = d["actions"].astype(np.int64)    # (N,)
    boundaries  = d["trajectory_boundaries"]        # (K,)

    # Build (z_t, a_t) -> z_{t+1} pairs, excluding across-episode boundaries
    boundary_set = set(int(b) for b in boundaries)

    X_list, y_list = [], []
    for t in range(len(Z) - 1):
        if (t + 1) in boundary_set:
            continue  # skip episode boundary transitions
        a_oh = np.zeros(ACTION_DIM, dtype=np.float32)
        a_oh[int(actions[t])] = 1.0
        X_list.append(np.concatenate([Z[t], a_oh]))  # (D + A,)
        y_list.append(Z[t + 1])                       # (D,)

    X = np.stack(X_list, axis=0)  # (M, D+A)
    y = np.stack(y_list, axis=0)  # (M, D)
    print(f"  Fitting on {len(X)} transitions  (X: {X.shape}, y: {y.shape})")

    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, y)

    ld = LinearDynamics(W=ridge.coef_.astype(np.float32),
                        b=ridge.intercept_.astype(np.float32))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(ld, f)
    print(f"  Saved Ridge model → {cache}")
    return ld


# ── CEM planner (numpy, no GPU) ───────────────────────────────────────────────

def cem_plan_linear(
    dynamics: LinearDynamics,
    z_curr: np.ndarray,
    z_goal: np.ndarray,
    H: int = 10,
    n_samples: int = 500,
    n_elite: int = 50,
    n_iters: int = 5,
    rng: Optional[np.random.Generator] = None,
    cost_fn=None,
) -> int:
    """CEM over H primitive actions using the linear dynamics model.

    Mirrors cem_plan() from plan_flat.py but operates entirely in numpy.

    Args:
        dynamics:  Fitted LinearDynamics model.
        z_curr:    (D,) current latent state.
        z_goal:    (D,) goal latent state.
        ...CEM hyperparameters...
        rng:       Optional numpy random generator.
        cost_fn:   Optional callable (z_final_np: np.ndarray (n_samples, D))
                   -> np.ndarray (n_samples,) of costs.  When None, falls back
                   to L1 distance between z_final and z_goal (original behaviour).

    Returns:
        best_action (int): First action of the best sequence found.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Per-step action log-probabilities: (H, ACTION_DIM)
    log_probs = np.zeros((H, ACTION_DIM), dtype=np.float32)

    for _ in range(n_iters):
        probs = np.exp(log_probs - log_probs.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)

        # Sample action sequences: (n_samples, H)
        a_samples = np.stack([
            rng.choice(ACTION_DIM, size=n_samples, p=probs[t])
            for t in range(H)
        ], axis=1)  # (n_samples, H)

        # Evaluate each sequence
        z_finals = np.stack([
            dynamics.rollout(z_curr, a_samples[i])[-1]
            for i in range(n_samples)
        ], axis=0)  # (n_samples, D)

        if cost_fn is not None:
            costs = cost_fn(z_finals)  # (n_samples,)
        else:
            costs = np.abs(z_goal[None] - z_finals).sum(axis=-1)  # (n_samples,)

        # Select elite samples
        elite_idx = np.argsort(costs)[:n_elite]
        elite_actions = a_samples[elite_idx]  # (n_elite, H)

        # Update per-step log-probs from elite action frequencies
        for t in range(H):
            counts = np.bincount(elite_actions[:, t], minlength=ACTION_DIM).astype(np.float32)
            new_lp = np.log(counts / counts.sum() + 1e-8)
            log_probs[t] = 0.5 * log_probs[t] + 0.5 * new_lp

    best_action = int(np.argmax(log_probs[0]))
    return best_action


# ── Single-episode runner ──────────────────────────────────────────────────────

def run_episode(
    lewm,
    dynamics: LinearDynamics,
    goal_frame: np.ndarray,
    target_achievement: str,
    seed: int,
    max_steps: int = 1000,
    H: int = 10,
    n_samples: int = 500,
    n_elite: int = 50,
    n_iters: int = 5,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
    record_rollout: bool = False,
    cost_fn=None,
) -> dict:
    """Run one linear-planner episode."""
    import crafter
    import time

    env = crafter.Env(seed=seed)
    obs = env.reset()

    goal_t = torch.from_numpy(goal_frame).float() / 255.0
    goal_t = goal_t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z_goal_t = lewm.encode(goal_t)
    z_goal = z_goal_t.squeeze(0).cpu().numpy()  # (D,)

    rng = np.random.default_rng(seed)
    planning_times = []
    success = False
    prev_ach = {}
    side_achievements: dict[str, int] = {}  # name -> step of first unlock
    rollout_frames: list | None = [] if record_rollout else None
    if rollout_frames is not None:
        rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

    for step in range(max_steps):
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            z_curr_t = lewm.encode(obs_t)
        z_curr = z_curr_t.squeeze(0).cpu().numpy()  # (D,)

        t0 = time.perf_counter()
        action = cem_plan_linear(
            dynamics, z_curr, z_goal,
            H=H, n_samples=n_samples, n_elite=n_elite, n_iters=n_iters, rng=rng,
            cost_fn=cost_fn,
        )
        planning_times.append((time.perf_counter() - t0) * 1000)

        obs, reward, done, info = env.step(action)
        if rollout_frames is not None:
            rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

        curr_ach = info.get("achievements", {})
        for k, v in curr_ach.items():
            if v > prev_ach.get(k, 0):
                if k == target_achievement:
                    success = True
                    if verbose:
                        print(f"  [linear] '{target_achievement}' achieved at step {step+1}")
                elif k not in side_achievements:
                    side_achievements[k] = step + 1
                    if verbose:
                        print(f"  [linear] side '{k}' at step {step+1}")
        prev_ach = dict(curr_ach)

        if done or success:
            break

    out = {
        "condition": "linear",
        "achievement": target_achievement,
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "side_achievements": side_achievements,
        "planning_ms_per_step": float(np.mean(planning_times)) if planning_times else 0.0,
    }
    if rollout_frames is not None:
        out["rollout_frames"] = np.stack(rollout_frames, axis=0)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Linear dynamics CEM planner (Condition B)")
    parser.add_argument("--checkpoint",   default=CHECKPOINT)
    parser.add_argument("--goal_library", default=GOAL_LIBRARY)
    parser.add_argument("--latents_cache", default=LATENTS_CACHE)
    parser.add_argument("--ridge_model",  default=RIDGE_MODEL)
    parser.add_argument("--achievement",  default="collect_wood")
    parser.add_argument("--n_episodes",   type=int,   default=3)
    parser.add_argument("--seed_start",   type=int,   default=100)
    parser.add_argument("--max_steps",    type=int,   default=1000)
    parser.add_argument("--H",            type=int,   default=10)
    parser.add_argument("--n_samples",    type=int,   default=500)
    parser.add_argument("--n_elite",      type=int,   default=50)
    parser.add_argument("--n_iters",      type=int,   default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lewm, _ = load_lewm(args.checkpoint, device=device)

    dynamics = fit_linear_dynamics(args.latents_cache, args.ridge_model)

    gl = np.load(args.goal_library, allow_pickle=True)
    goal_names = list(gl["goal_names"])
    if args.achievement not in goal_names:
        raise ValueError(f"'{args.achievement}' not found in goal library")
    idx = goal_names.index(args.achievement)
    goal_frame = gl["goal_frames"][idx]

    for ep in range(args.n_episodes):
        result = run_episode(
            lewm, dynamics, goal_frame, args.achievement,
            seed=args.seed_start + ep,
            max_steps=args.max_steps,
            H=args.H, n_samples=args.n_samples,
            n_elite=args.n_elite, n_iters=args.n_iters,
            device=device, verbose=True,
        )
        print(result)


if __name__ == "__main__":
    main()
