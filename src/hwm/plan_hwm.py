"""Condition C — Two-level HWM planner.

High-level CEM (Gaussian, continuous) over H_hi=3 macro-actions in R^256,
  using HighLevelPredictor to predict the latent H_hi steps ahead.
Low-level CEM (integer, categorical) over H_lo=10 primitive actions,
  using LeWM.rollout to reach the high-level subgoal.

Replanning schedule:
  - High-level reruns every H_lo steps OR when ‖z_curr - z_subgoal‖₁ < threshold
  - Low-level reruns every step (receding-horizon MPC)

Oracle ablation mode (--oracle):
  Instead of using the HighLevelPredictor to produce z_subgoal, retrieve the
  actual human playthrough midpoint frame (from trajectory_dataset.npz) at the
  timestep halfway between the start of the episode and the goal achievement
  timestep, and encode it as z_subgoal.

Standalone usage:
    python src/hwm/plan_hwm.py --achievement collect_wood --n_episodes 3
"""

from __future__ import annotations

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
    HWM_HIGH_CKPT,
    LATENT_DIM,
    LATENTS_CACHE,
    TRAJ_DATASET,
)
from hwm.models import (
    ActionEncoder,
    HighLevelPredictor,
    load_lewm,
    one_hot_actions,
)


# ── Load HWM high-level checkpoint ────────────────────────────────────────────

def load_hwm_high(
    ckpt_path: str = HWM_HIGH_CKPT,
    device: torch.device = torch.device("cpu"),
) -> tuple[ActionEncoder, HighLevelPredictor]:
    """Load trained ActionEncoder and HighLevelPredictor from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get("args", {})

    action_enc = ActionEncoder(
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        max_len=saved_args.get("max_subseq_len", 32),
    ).to(device)
    action_enc.load_state_dict(ckpt["action_encoder"])
    action_enc.eval()
    for p in action_enc.parameters():
        p.requires_grad_(False)

    high_pred = HighLevelPredictor(
        latent_dim=LATENT_DIM,
        depth=6,
        num_heads=16,
        dropout=0.1,
        context_len=3,
    ).to(device)
    high_pred.load_state_dict(ckpt["high_predictor"])
    high_pred.eval()
    for p in high_pred.parameters():
        p.requires_grad_(False)

    return action_enc, high_pred


# ── High-level CEM (continuous Gaussian over macro-actions) ───────────────────

def cem_high(
    high_pred: HighLevelPredictor,
    z_curr: torch.Tensor,
    z_goal: torch.Tensor,
    H_hi: int = 3,
    n_samples: int = 300,
    n_elite: int = 30,
    n_iters: int = 5,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous Gaussian CEM over H_hi macro-actions.

    Args:
        high_pred: Frozen HighLevelPredictor.
        z_curr:    (1, D) current latent state.
        z_goal:    (1, D) goal latent state.
        H_hi:      High-level planning horizon (number of macro-actions).
        n_samples: CEM population size.
        n_elite:   Number of elite samples.
        n_iters:   CEM refinement iterations.

    Returns:
        best_l_seq: (1, H_hi, D) optimal macro-action sequence.
        z_subgoal:  (1, D) predicted latent after first macro-action.
    """
    D = LATENT_DIM

    # Initialize Gaussian distribution over macro-action sequences
    mu    = torch.zeros(H_hi, D, device=device)
    sigma = torch.ones(H_hi, D, device=device)

    z_curr_exp = z_curr.expand(n_samples, -1)   # (n_samples, D)
    z_goal_exp = z_goal.expand(n_samples, -1)   # (n_samples, D)

    best_l_seq = mu.unsqueeze(0)  # (1, H_hi, D) — fallback

    for iteration in range(n_iters):
        # Sample macro-action sequences: (n_samples, H_hi, D)
        eps = torch.randn(n_samples, H_hi, D, device=device)
        l_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

        # Rollout in high-level latent space: (n_samples, H_hi+1, D)
        with torch.no_grad():
            z_traj = high_pred.rollout(z_curr_exp, l_samples)

        z_final = z_traj[:, -1]  # (n_samples, D)

        # L1 cost to goal
        costs = F.l1_loss(z_final, z_goal_exp, reduction="none").sum(dim=-1)  # (n_samples,)

        # Select elite
        elite_idx = torch.argsort(costs)[:n_elite]
        elite_l   = l_samples[elite_idx]  # (n_elite, H_hi, D)

        # Update distribution from elite mean/std
        mu    = elite_l.mean(dim=0)       # (H_hi, D)
        sigma = elite_l.std(dim=0) + 1e-4  # (H_hi, D)

        if iteration == n_iters - 1:
            best_l_seq = mu.unsqueeze(0)  # (1, H_hi, D)

    # Get subgoal: predict one step with the best first macro-action
    with torch.no_grad():
        z_subgoal = high_pred(
            z_curr.unsqueeze(1),             # (1, 1, D)
            best_l_seq[:, :1, :],            # (1, 1, D)
        ).squeeze(1)                         # (1, D)

    return best_l_seq, z_subgoal


# ── Low-level CEM (integer, same as plan_flat.cem_plan) ───────────────────────

def cem_low(
    lewm,
    z_curr: torch.Tensor,
    z_subgoal: torch.Tensor,
    H: int = 10,
    n_samples: int = 500,
    n_elite: int = 50,
    n_iters: int = 5,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Low-level CEM over H primitive actions targeting z_subgoal."""
    logits = torch.zeros(H, ACTION_DIM, device=device)

    z_curr_exp    = z_curr.expand(n_samples, -1)
    z_subgoal_exp = z_subgoal.expand(n_samples, -1)

    for _ in range(n_iters):
        probs = torch.softmax(logits, dim=-1)
        a_samples = torch.multinomial(
            probs.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * H, ACTION_DIM),
            num_samples=1,
        ).reshape(n_samples, H)

        a_oh = one_hot_actions(a_samples, ACTION_DIM)

        with torch.no_grad():
            z_traj = lewm.rollout(z_curr_exp, a_oh)  # (n_samples, H+1, D)

        z_final = z_traj[:, -1]
        costs = F.l1_loss(z_final, z_subgoal_exp, reduction="none").sum(dim=-1)

        elite_idx    = torch.argsort(costs)[:n_elite]
        elite_actions = a_samples[elite_idx]

        for t in range(H):
            counts = torch.bincount(elite_actions[:, t], minlength=ACTION_DIM).float()
            logits[t] = logits[t] * 0.5 + (counts / counts.sum()).log().clamp(min=-10) * 0.5

    return int(torch.argmax(logits[0]).item())


# ── Oracle subgoal: nearest midpoint frame from training trajectories ─────────

def get_oracle_subgoal(
    lewm,
    goal_achievement_step: int,
    current_step: int,
    traj_dataset_path: str,
    latents_path: str,
    device: torch.device,
) -> torch.Tensor:
    """Return z_subgoal from the midpoint frame of a human playthrough.

    Midpoint = (current_step + goal_achievement_step) // 2 timestep in
    the trajectory that achieved the goal earliest.  Uses pre-encoded
    latents if available, otherwise encodes the raw frame on-the-fly.

    Returns:
        z_subgoal: (1, D) tensor on device.
    """
    from pathlib import Path as _P
    lp = _P(latents_path)
    if lp.exists():
        d = np.load(latents_path)
        Z           = d["Z"]
        boundaries  = d["trajectory_boundaries"]
    else:
        # Fallback: re-encode on-the-fly (slow)
        td = np.load(traj_dataset_path)
        obs = td["obs"]
        boundaries = td["trajectory_boundaries"]
        batch = torch.from_numpy(obs).float() / 255.0
        batch = batch.permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            Z = lewm.encode(batch).cpu().numpy()

    mid_step = (current_step + goal_achievement_step) // 2
    mid_step = max(0, min(mid_step, len(Z) - 1))
    z_mid = torch.from_numpy(Z[mid_step].copy()).float().unsqueeze(0).to(device)
    return z_mid


# ── Single-episode runner ──────────────────────────────────────────────────────

def run_episode(
    lewm,
    high_pred: HighLevelPredictor,
    goal_frame: np.ndarray,
    target_achievement: str,
    goal_achievement_step: int,
    seed: int,
    max_steps: int = 1000,
    H_hi: int = 3,
    H_lo: int = 10,
    n_samples_hi: int = 300,
    n_samples_lo: int = 500,
    n_elite_hi: int = 30,
    n_elite_lo: int = 50,
    n_iters: int = 5,
    subgoal_threshold: float = 5.0,
    device: torch.device = torch.device("cpu"),
    oracle: bool = False,
    traj_dataset_path: str = TRAJ_DATASET,
    latents_path: str = LATENTS_CACHE,
    verbose: bool = False,
    record_rollout: bool = False,
) -> dict:
    """Run one HWM episode."""
    import crafter
    import time

    env = crafter.Env(seed=seed)
    obs = env.reset()

    # Encode goal frame once
    goal_t = torch.from_numpy(goal_frame).float() / 255.0
    goal_t = goal_t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z_goal = lewm.encode(goal_t)  # (1, D)

    z_subgoal: Optional[torch.Tensor] = None
    steps_since_replan = H_lo  # force high-level replan on first step
    planning_times = []
    success = False
    prev_ach = {}
    rollout_frames: list | None = [] if record_rollout else None
    if rollout_frames is not None:
        rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

    for step in range(max_steps):
        # Encode current observation
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            z_curr = lewm.encode(obs_t)  # (1, D)

        # Decide whether to run high-level replan
        need_replan = (z_subgoal is None) or (steps_since_replan >= H_lo)
        if z_subgoal is not None and not need_replan:
            dist = F.l1_loss(z_curr, z_subgoal, reduction="none").sum().item()
            need_replan = dist < subgoal_threshold

        t0 = time.perf_counter()

        if need_replan:
            if oracle:
                z_subgoal = get_oracle_subgoal(
                    lewm, goal_achievement_step, step,
                    traj_dataset_path, latents_path, device,
                )
            else:
                _, z_subgoal = cem_high(
                    high_pred, z_curr, z_goal,
                    H_hi=H_hi, n_samples=n_samples_hi,
                    n_elite=n_elite_hi, n_iters=n_iters, device=device,
                )
            steps_since_replan = 0
            if verbose:
                d_to_goal = F.l1_loss(z_subgoal, z_goal, reduction="none").sum().item()
                print(f"  [hwm] step={step} high-level replan  "
                      f"|z_sub - z_goal|₁={d_to_goal:.2f}")

        # Low-level plan toward subgoal
        action = cem_low(
            lewm, z_curr, z_subgoal,
            H=H_lo, n_samples=n_samples_lo,
            n_elite=n_elite_lo, n_iters=n_iters, device=device,
        )
        planning_times.append((time.perf_counter() - t0) * 1000)
        steps_since_replan += 1

        obs, reward, done, info = env.step(action)
        if rollout_frames is not None:
            rollout_frames.append(np.asarray(obs, dtype=np.uint8).copy())

        curr_ach = info.get("achievements", {})
        for k, v in curr_ach.items():
            if v > prev_ach.get(k, 0) and k == target_achievement:
                success = True
                if verbose:
                    print(f"  [hwm{'_oracle' if oracle else ''}] "
                          f"'{target_achievement}' achieved at step {step+1}")
        prev_ach = dict(curr_ach)

        if done or success:
            break

    out = {
        "condition": "hwm_oracle" if oracle else "hwm",
        "achievement": target_achievement,
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "planning_ms_per_step": float(np.mean(planning_times)) if planning_times else 0.0,
    }
    if rollout_frames is not None:
        out["rollout_frames"] = np.stack(rollout_frames, axis=0)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="HWM two-level CEM planner (Condition C)")
    parser.add_argument("--checkpoint",    default=CHECKPOINT)
    parser.add_argument("--hwm_checkpoint", default=HWM_HIGH_CKPT)
    parser.add_argument("--goal_library",  default=GOAL_LIBRARY)
    parser.add_argument("--latents_cache", default=LATENTS_CACHE)
    parser.add_argument("--traj_dataset",  default=TRAJ_DATASET)
    parser.add_argument("--achievement",   default="collect_wood")
    parser.add_argument("--n_episodes",    type=int,   default=3)
    parser.add_argument("--seed_start",    type=int,   default=100)
    parser.add_argument("--max_steps",     type=int,   default=1000)
    parser.add_argument("--H_hi",          type=int,   default=3)
    parser.add_argument("--H_lo",          type=int,   default=10)
    parser.add_argument("--n_samples_hi",  type=int,   default=300)
    parser.add_argument("--n_samples_lo",  type=int,   default=500)
    parser.add_argument("--n_elite_hi",    type=int,   default=30)
    parser.add_argument("--n_elite_lo",    type=int,   default=50)
    parser.add_argument("--n_iters",       type=int,   default=5)
    parser.add_argument("--threshold",     type=float, default=5.0)
    parser.add_argument("--oracle",        action="store_true",
                        help="Use oracle subgoals from human trajectories")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lewm, _ = load_lewm(args.checkpoint, device=device)
    action_enc, high_pred = load_hwm_high(args.hwm_checkpoint, device=device)

    gl = np.load(args.goal_library, allow_pickle=True)
    goal_names = list(gl["goal_names"])
    if args.achievement not in goal_names:
        raise ValueError(f"'{args.achievement}' not found in goal library")
    idx = goal_names.index(args.achievement)
    goal_frame            = gl["goal_frames"][idx]
    goal_achievement_step = int(gl["goal_achievement_steps"][idx])

    for ep in range(args.n_episodes):
        result = run_episode(
            lewm, high_pred, goal_frame, args.achievement,
            goal_achievement_step=goal_achievement_step,
            seed=args.seed_start + ep,
            max_steps=args.max_steps,
            H_hi=args.H_hi, H_lo=args.H_lo,
            n_samples_hi=args.n_samples_hi, n_samples_lo=args.n_samples_lo,
            n_elite_hi=args.n_elite_hi, n_elite_lo=args.n_elite_lo,
            n_iters=args.n_iters, subgoal_threshold=args.threshold,
            device=device, oracle=args.oracle,
            traj_dataset_path=args.traj_dataset,
            latents_path=args.latents_cache,
            verbose=True,
        )
        print(result)


if __name__ == "__main__":
    main()
