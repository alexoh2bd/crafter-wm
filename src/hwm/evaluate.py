"""Evaluation harness — runs all three planning conditions + oracle ablation.

Conditions
----------
flat    — Condition A: flat CEM using LeWM predictor
linear  — Condition B: Ridge linear dynamics + CEM
hwm     — Condition C: two-level HWM CEM
hwm_oracle — Condition C ablation: real human midpoint subgoals

For each condition, 50 episodes are run (seeds 100–149).  Each episode targets
one achievement in tech-tree order, picked from goal_library.npz.

Reproducibility: Crafter uses ``Env(seed=seed_start + ep)`` where ``ep`` is the
0-based episode index in that condition block.  Same CLI and goal library
implies the same world seeds per episode.  ``results.json`` ``config`` stores
artifact paths, resolved absolute paths, optional ``git_commit``, and
``SLURM_JOB_ID`` when set.

Output: results/results.json  with per-episode records and aggregate stats.

With --wandb: after each condition, logs aggregate metrics plus videos for the
last three episodes of that condition (RGB Crafter frames).

Usage
-----
    # All conditions (sequential):
    python src/hwm/evaluate.py

    # One condition only:
    python src/hwm/evaluate.py --condition flat
    python src/hwm/evaluate.py --condition linear
    python src/hwm/evaluate.py --condition hwm
    python src/hwm/evaluate.py --condition hwm_oracle

    # Append to existing results:
    python src/hwm/evaluate.py --condition hwm --append
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    ACHIEVEMENT_NAMES,
    CHECKPOINT,
    GOAL_LIBRARY,
    HWM_HIGH_CKPT,
    LATENTS_CACHE,
    RESULTS_DIR,
    RESULTS_JSON,
    RIDGE_MODEL,
    TRAJ_DATASET,
)
from hwm.models import load_lewm
from hwm.plan_flat import run_episode as run_flat
from hwm.plan_hwm import load_hwm_high, run_episode as run_hwm
from hwm.plan_linear import fit_linear_dynamics, run_episode as run_linear


# ── Achievement ordering (tech-tree, roughly) ─────────────────────────────────
# Ordered by median first-unlock timestep observed in the human playthroughs.
TECH_TREE_ORDER = [
    "collect_wood",
    "place_table",
    "make_wood_pickaxe",
    "collect_stone",
    "collect_drink",
    "eat_cow",
    "defeat_zombie",
    "place_plant",
    "defeat_skeleton",
    "make_wood_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "collect_coal",
    "collect_iron",
    "place_stone",
    "place_furnace",
    "make_iron_pickaxe",
    "wake_up",
    "make_iron_sword",
    "collect_diamond",
    "collect_sapling",
    "eat_plant",
]
# Verify all 22 achievements are represented
assert set(TECH_TREE_ORDER) == set(ACHIEVEMENT_NAMES), \
    "TECH_TREE_ORDER must contain exactly the 22 ACHIEVEMENT_NAMES"


def _ordered_achievements(goal_library: dict) -> list[tuple[str, np.ndarray, int]]:
    """Return (name, frame, step) tuples in TECH_TREE_ORDER, skipping missing ones."""
    goal_names = list(goal_library["goal_names"])
    results = []
    for name in TECH_TREE_ORDER:
        if name not in goal_names:
            continue
        idx = goal_names.index(name)
        frame = goal_library["goal_frames"][idx]
        step  = int(goal_library["goal_achievement_steps"][idx])
        if step < 0:
            continue  # achievement never seen in human data
        results.append((name, frame, step))
    return results


# ── Per-condition evaluation ──────────────────────────────────────────────────

def evaluate_condition(
    condition: str,
    lewm,
    goal_library: dict,
    n_episodes: int,
    seed_start: int,
    max_steps: int,
    device: torch.device,
    # Condition-specific kwargs
    dynamics=None,
    high_pred=None,
    traj_dataset_path: str = TRAJ_DATASET,
    latents_path: str = LATENTS_CACHE,
    verbose: bool = False,
) -> list[dict]:
    """Run *n_episodes* for *condition* and return list of result dicts."""
    achievements = _ordered_achievements(goal_library)
    records = []

    n_rollout_record = min(3, n_episodes)

    for ep in range(n_episodes):
        seed = seed_start + ep

        # Rotate through tech-tree achievements
        name, frame, ach_step = achievements[ep % len(achievements)]

        record_rollout = ep >= n_episodes - n_rollout_record

        t0 = time.time()
        if condition == "flat":
            result = run_flat(
                lewm, frame, name, seed=seed,
                max_steps=max_steps, device=device, verbose=verbose,
                record_rollout=record_rollout,
            )
        elif condition == "linear":
            assert dynamics is not None, "dynamics required for linear condition"
            result = run_linear(
                lewm, dynamics, frame, name, seed=seed,
                max_steps=max_steps, device=device, verbose=verbose,
                record_rollout=record_rollout,
            )
        elif condition in ("hwm", "hwm_oracle"):
            assert high_pred is not None, "high_pred required for hwm condition"
            result = run_hwm(
                lewm, high_pred, frame, name,
                goal_achievement_step=ach_step,
                seed=seed, max_steps=max_steps,
                oracle=(condition == "hwm_oracle"),
                traj_dataset_path=traj_dataset_path,
                latents_path=latents_path,
                device=device, verbose=verbose,
                record_rollout=record_rollout,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        result["wall_time_s"] = time.time() - t0
        result["episode"] = ep
        records.append(result)

        print(
            f"[{condition}] ep={ep:3d}/{n_episodes}  {name:25s}  "
            f"success={result['success']}  steps={result['steps']:4d}  "
            f"plan_ms={result['planning_ms_per_step']:.1f}"
        )

    return records


# ── Aggregate statistics ──────────────────────────────────────────────────────

def aggregate_one_condition(recs: list[dict]) -> dict:
    """Aggregate stats for a single planning condition (one block of episodes)."""
    from collections import defaultdict

    if not recs:
        return {}
    successes = [r for r in recs if r["success"]]
    by_ach: dict[str, list[bool]] = defaultdict(list)
    for r in recs:
        by_ach[r["achievement"]].append(r["success"])
    return {
        "n_episodes": len(recs),
        "n_success": len(successes),
        "success_rate": len(successes) / max(len(recs), 1),
        "mean_steps": float(np.mean([r["steps"] for r in recs])),
        "mean_steps_success": (
            float(np.mean([r["steps"] for r in successes]))
            if successes else None
        ),
        "mean_planning_ms": float(np.mean([r["planning_ms_per_step"] for r in recs])),
        "per_achievement": {
            k: {"n_episodes": len(v), "success_rate": float(np.mean(v))}
            for k, v in by_ach.items()
        },
    }


def compute_aggregates(records: list[dict]) -> dict:
    """Compute per-condition aggregate statistics."""
    from collections import defaultdict

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_condition[r["condition"]].append(r)

    return {cond: aggregate_one_condition(recs) for cond, recs in by_condition.items()}


def wandb_log_condition_final(run, cond: str, stats: dict, step: int) -> None:
    """Log end-of-condition metrics only (global + per-achievement episode counts)."""
    if run is None or not stats:
        return
    payload: dict = {
        f"{cond}/n_episodes": stats["n_episodes"],
        f"{cond}/n_success": stats["n_success"],
        f"{cond}/success_rate": stats["success_rate"],
        f"{cond}/mean_steps": stats["mean_steps"],
        f"{cond}/mean_planning_ms": stats["mean_planning_ms"],
    }
    mss = stats.get("mean_steps_success")
    if mss is not None:
        payload[f"{cond}/mean_steps_success"] = mss
    for ach, sub in stats.get("per_achievement", {}).items():
        payload[f"{cond}/per_achievement/{ach}/n_episodes"] = sub["n_episodes"]
        payload[f"{cond}/per_achievement/{ach}/success_rate"] = sub["success_rate"]
    run.log(payload, step=step)


def wandb_log_rollouts(run, cond: str, records: list[dict], step: int) -> None:
    """Log frame videos for episodes that included rollout_frames (final 3 per condition)."""
    if run is None:
        return
    payload: dict = {}
    for r in records:
        vid_arr = r.get("rollout_frames")
        if vid_arr is None:
            continue
        vid = np.asarray(vid_arr)
        if vid.ndim != 4 or vid.shape[0] < 1:
            continue
        ep = int(r["episode"])
        fps = min(20, max(4, int(150 / max(vid.shape[0], 1))))
        payload[f"{cond}/rollout/ep_{ep:03d}_video"] = _wandb.Video(
            vid, fps=fps, format="mp4"
        )
        payload[f"{cond}/rollout/ep_{ep:03d}_achievement"] = r["achievement"]
        payload[f"{cond}/rollout/ep_{ep:03d}_success"] = int(r["success"])
        payload[f"{cond}/rollout/ep_{ep:03d}_steps"] = int(r["steps"])
    if payload:
        run.log(payload, step=step)


def _path_resolved(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _git_commit() -> str | None:
    repo = Path(__file__).resolve().parent.parent.parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def eval_repro_metadata(
    args: argparse.Namespace,
    conditions_to_run: list[str],
) -> dict:
    """Resolved paths and provenance for reproducibility (JSON + wandb)."""
    return {
        "checkpoint_resolved": _path_resolved(args.checkpoint),
        "hwm_checkpoint_resolved": _path_resolved(args.hwm_checkpoint),
        "ridge_model_resolved": _path_resolved(args.ridge_model),
        "latents_cache_resolved": _path_resolved(args.latents_cache),
        "goal_library_resolved": _path_resolved(args.goal_library),
        "traj_dataset_resolved": _path_resolved(args.traj_dataset),
        "results_path_resolved": _path_resolved(args.results_path),
        "conditions_expanded": list(conditions_to_run),
        "seed_convention": (
            "Crafter Env(seed) uses seed = seed_start + ep where ep is the "
            "0-based episode index within that condition. Changing n_episodes or "
            "goal_library changes which achievement pairs with which seed."
        ),
        "git_commit": _git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all HWM planning conditions")
    parser.add_argument("--condition",    default="all",
                        choices=["all", "flat", "linear", "hwm", "hwm_oracle"],
                        help="Which condition(s) to run")
    parser.add_argument("--checkpoint",      default=CHECKPOINT)
    parser.add_argument("--hwm_checkpoint",  default=HWM_HIGH_CKPT)
    parser.add_argument("--goal_library",    default=GOAL_LIBRARY)
    parser.add_argument("--ridge_model",     default=RIDGE_MODEL)
    parser.add_argument("--latents_cache",   default=LATENTS_CACHE)
    parser.add_argument("--traj_dataset",    default=TRAJ_DATASET)
    parser.add_argument("--results_path",    default=RESULTS_JSON)
    parser.add_argument("--n_episodes",      type=int,   default=50)
    parser.add_argument("--seed_start",      type=int,   default=100)
    parser.add_argument("--max_steps",       type=int,   default=1000)
    parser.add_argument("--append",          action="store_true",
                        help="Append to existing results.json instead of overwriting")
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--wandb",           action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",   default="lewm-crafter")
    parser.add_argument("--wandb_run_name",  default=None)
    args = parser.parse_args()

    conditions_to_run = (
        ["flat", "linear", "hwm", "hwm_oracle"]
        if args.condition == "all"
        else [args.condition]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── wandb init ────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb:
        if _WANDB_AVAILABLE:
            run_name = args.wandb_run_name or f"hwm-eval-{args.condition}"
            wandb_config = dict(vars(args))
            wandb_config.update(eval_repro_metadata(args, conditions_to_run))
            wandb_run = _wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
            )
        else:
            print("Warning: --wandb set but wandb is not installed; "
                  "install with `pip install wandb`")

    # ── Load shared resources ─────────────────────────────────────────────────
    print(f"Loading LeWM from {args.checkpoint}")
    lewm, _ = load_lewm(args.checkpoint, device=device)

    print(f"Loading goal library from {args.goal_library}")
    goal_library = np.load(args.goal_library, allow_pickle=True)

    dynamics    = None
    high_pred   = None

    if "linear" in conditions_to_run:
        dynamics = fit_linear_dynamics(args.latents_cache, args.ridge_model)

    if "hwm" in conditions_to_run or "hwm_oracle" in conditions_to_run:
        print(f"Loading HWM high-level modules from {args.hwm_checkpoint}")
        _, high_pred = load_hwm_high(args.hwm_checkpoint, device=device)

    # ── Run conditions ────────────────────────────────────────────────────────
    all_records: list[dict] = []

    # Load existing results if appending
    results_path = Path(args.results_path)
    if args.append and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        all_records = existing.get("episodes", [])
        print(f"Appending to existing {len(all_records)} records")

    for ci, cond in enumerate(conditions_to_run):
        print(f"\n{'='*60}")
        print(f"Evaluating condition: {cond}")
        print(f"{'='*60}")
        records = evaluate_condition(
            condition=cond,
            lewm=lewm,
            goal_library=goal_library,
            n_episodes=args.n_episodes,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
            device=device,
            dynamics=dynamics,
            high_pred=high_pred,
            traj_dataset_path=args.traj_dataset,
            latents_path=args.latents_cache,
            verbose=args.verbose,
        )
        if wandb_run is not None:
            wandb_log_condition_final(
                wandb_run, cond, aggregate_one_condition(records), step=ci,
            )
            wandb_log_rollouts(wandb_run, cond, records, step=ci)
        for r in records:
            r.pop("rollout_frames", None)
        all_records.extend(records)

    # ── Save results ──────────────────────────────────────────────────────────
    aggregates = compute_aggregates(all_records)

    results_config = {
        "n_episodes": args.n_episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "checkpoint": args.checkpoint,
        "hwm_checkpoint": args.hwm_checkpoint,
        "ridge_model": args.ridge_model,
        "latents_cache": args.latents_cache,
        "goal_library": args.goal_library,
        "traj_dataset": args.traj_dataset,
        "results_path": args.results_path,
        "condition_arg": args.condition,
    }
    results_config.update(eval_repro_metadata(args, conditions_to_run))

    output = {
        "episodes":   all_records,
        "aggregates": aggregates,
        "config":     results_config,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for cond, stats in aggregates.items():
        print(f"  {cond:15s}  success_rate={stats['success_rate']:.2%}  "
              f"mean_steps={stats['mean_steps']:.0f}  "
              f"plan_ms={stats['mean_planning_ms']:.1f}")

    if wandb_run is not None:
        _wandb.finish()


if __name__ == "__main__":
    main()
