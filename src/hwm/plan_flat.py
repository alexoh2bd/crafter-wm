"""Condition A — Flat world-model CEM planner.

Uses the frozen LeWM predictor directly as the dynamics model.
CEM samples H=10 primitive action sequences, rolls out each in latent space,
and picks the sequence that brings the predicted final latent closest to z_goal.
Receding-horizon MPC: replans every environment step.

Standalone usage (for quick testing):
    python src/hwm/plan_flat.py --achievement collect_wood --n_episodes 3
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import ACTION_DIM, CHECKPOINT, GOAL_LIBRARY, LATENT_DIM
from hwm.models import load_lewm, one_hot_actions


# ── CEM planner ───────────────────────────────────────────────────────────────

def cem_plan(
    lewm,
    z_curr: torch.Tensor,
    z_goal: torch.Tensor,
    H: int = 10,
    n_samples: int = 500,
    n_elite: int = 50,
    n_iters: int = 5,
    device: torch.device = torch.device("cpu"),
    cost_fn=None,
) -> int:
    """Run CEM over a horizon of H primitive actions and return the first action.

    Args:
        lewm:      Frozen LeWM model.
        z_curr:    (1, latent_dim) current latent state.
        z_goal:    (1, latent_dim) goal latent state.
        H:         Planning horizon (number of primitive actions).
        n_samples: Number of CEM population samples.
        n_elite:   Number of elite samples kept per iteration.
        n_iters:   Number of CEM refinement iterations.
        device:    Compute device.
        cost_fn:   Optional callable (z_final_np: np.ndarray (n_samples, D))
                   -> np.ndarray (n_samples,) of costs.  When None, falls back
                   to L1 distance between z_final and z_goal (original behaviour).

    Returns:
        best_action (int): The first action in the optimal sequence.
    """
    # Per-step categorical distribution represented as logits (H, ACTION_DIM)
    logits = torch.zeros(H, ACTION_DIM, device=device)

    z_curr_exp = z_curr.expand(n_samples, -1)  # (n_samples, D)
    z_goal_exp = z_goal.expand(n_samples, -1)  # (n_samples, D)

    for _ in range(n_iters):
        # Sample action sequences: (n_samples, H)
        probs = torch.softmax(logits, dim=-1)                    # (H, A)
        a_samples = torch.multinomial(
            probs.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * H, ACTION_DIM),
            num_samples=1,
        ).reshape(n_samples, H)                                  # (n_samples, H)

        # Convert to one-hot (n_samples, H, ACTION_DIM)
        a_oh = one_hot_actions(a_samples, ACTION_DIM)

        # Rollout in latent space: (n_samples, H+1, D)
        with torch.no_grad():
            z_traj = lewm.rollout(z_curr_exp, a_oh)             # (n_samples, H+1, D)

        z_final = z_traj[:, -1]                                   # (n_samples, D)

        # Cost: probe-based or L1 fallback
        if cost_fn is not None:
            costs = torch.from_numpy(
                cost_fn(z_final.cpu().numpy())
            ).to(device)                                          # (n_samples,)
        else:
            costs = F.l1_loss(z_final, z_goal_exp, reduction="none").sum(dim=-1)

        # Select elite
        elite_idx = torch.argsort(costs)[:n_elite]               # (n_elite,)
        elite_actions = a_samples[elite_idx]                     # (n_elite, H)

        # Update per-step logits from elite action frequencies
        for t in range(H):
            counts = torch.bincount(elite_actions[:, t], minlength=ACTION_DIM).float()
            # Smooth update: mix old logits with new frequency signal
            logits[t] = logits[t] * 0.5 + (counts / counts.sum()).log().clamp(min=-10) * 0.5

    # Execute the greedy best action at t=0
    best_action = int(torch.argmax(logits[0]).item())
    return best_action


# ── Single-episode runner ──────────────────────────────────────────────────────

def run_episode(
    lewm,
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
    """Run one flat-planner episode and return result dict.

    Args:
        lewm:               Frozen LeWM model (eval mode).
        goal_frame:         (64, 64, 3) uint8 goal image.
        target_achievement: Achievement name to track success.
        seed:               Environment seed.
        max_steps:          Maximum episode length.
        ...CEM hyperparameters...
        device:             Compute device.
        verbose:            Print per-step info.
        record_rollout:     If True, include rollout_frames (T, H, W, 3) uint8.
        cost_fn:            Optional probe cost callable; see cem_plan for signature.

    Returns:
        dict with keys: condition, achievement, seed, success, steps,
                        planning_ms_per_step; optionally rollout_frames.
    """
    import crafter
    import time

    env = crafter.Env(seed=seed)
    obs = env.reset()  # (64, 64, 3) ndarray — no unpacking

    # Encode goal frame once
    goal_t = torch.from_numpy(goal_frame).float() / 255.0
    goal_t = goal_t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, 64, 64)
    with torch.no_grad():
        z_goal = lewm.encode(goal_t)  # (1, D)

    planning_times = []
    success = False
    prev_ach = {}
    side_achievements: dict[str, int] = {}  # name -> step of first unlock
    rollout_frames: list | None = [] if record_rollout else None
    if rollout_frames is not None:
        rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

    for step in range(max_steps):
        # Encode current observation
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            z_curr = lewm.encode(obs_t)  # (1, D)

        # Plan
        t0 = time.perf_counter()
        action = cem_plan(
            lewm, z_curr, z_goal, H=H,
            n_samples=n_samples, n_elite=n_elite, n_iters=n_iters, device=device,
            cost_fn=cost_fn,
        )
        planning_times.append((time.perf_counter() - t0) * 1000)

        obs, reward, done, info = env.step(action)
        if rollout_frames is not None:
            rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

        # Check achievements
        curr_ach = info.get("achievements", {})
        for k, v in curr_ach.items():
            if v > prev_ach.get(k, 0):
                if k == target_achievement:
                    success = True
                    if verbose:
                        print(f"  [flat] '{target_achievement}' achieved at step {step+1}")
                elif k not in side_achievements:
                    side_achievements[k] = step + 1
                    if verbose:
                        print(f"  [flat] side '{k}' at step {step+1}")
        prev_ach = dict(curr_ach)

        if done or success:
            break

    out = {
        "condition": "flat",
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


# ── CLI for quick testing ─────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Flat CEM planner (Condition A)")
    parser.add_argument("--checkpoint",   default=CHECKPOINT)
    parser.add_argument("--goal_library", default=GOAL_LIBRARY)
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

    gl = np.load(args.goal_library, allow_pickle=True)
    goal_names = list(gl["goal_names"])
    if args.achievement not in goal_names:
        raise ValueError(f"'{args.achievement}' not found in goal library")
    idx = goal_names.index(args.achievement)
    goal_frame = gl["goal_frames"][idx]  # (64, 64, 3)

    for ep in range(args.n_episodes):
        result = run_episode(
            lewm, goal_frame, args.achievement,
            seed=args.seed_start + ep,
            max_steps=args.max_steps,
            H=args.H, n_samples=args.n_samples,
            n_elite=args.n_elite, n_iters=args.n_iters,
            device=device, verbose=True,
        )
        print(result)


if __name__ == "__main__":
    main()
