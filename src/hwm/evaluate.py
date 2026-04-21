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

With ``--save_rollout_gifs DIR`` and e.g. ``--n_episodes 8``, records every
episode and writes eight GIFs to ``DIR`` (requires ``imageio``).

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

    # Eight rollout GIFs on disk (HWM policy):
    python src/hwm/evaluate.py --condition hwm --n_episodes 8 \\
        --save_rollout_gifs results/gifs_hwm
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    DATA_OUT,
    GOAL_LIBRARY,
    HWM_HIGH_CKPT,
    LATENTS_CACHE,
    NPZ_DIR,
    RESULTS_DIR,
    RESULTS_JSON,
    RIDGE_MODEL,
    TRAJ_DATASET,
)
from hwm.models import load_lewm
from hwm.plan_flat import run_episode as run_flat
from hwm.plan_hwm import load_hwm_high, run_episode as run_hwm
from hwm.plan_linear import fit_linear_dynamics, run_episode as run_linear


# ── Parallel evaluation (ProcessPoolExecutor, one episode per task) ───────────
# Module-level cache populated once per worker process by _worker_init.
_PROC_CACHE: dict = {}


def _worker_init(
    lewm_ckpt: str,
    hwm_ckpt: str,
    latents_cache: str,
    ridge_model_path: str,
    traj_dataset: str,
    device_str: str,
    probes: dict | None,
    src_path: str,
) -> None:
    """Load heavy artifacts once per spawned worker process."""
    import sys as _sys
    if src_path not in _sys.path:
        _sys.path.insert(0, src_path)

    import torch as _torch
    from hwm.models import load_lewm as _load_lewm

    _d = _torch.device(device_str)
    _lewm, _ = _load_lewm(lewm_ckpt, device=_d)
    _lewm.eval()

    _PROC_CACHE.update({
        "lewm": _lewm,
        "device": _d,
        "hwm_ckpt": hwm_ckpt,
        "latents_cache": latents_cache,
        "ridge_model_path": ridge_model_path,
        "traj_dataset": traj_dataset,
        "probes": probes,
    })


def _worker_run_episode(task: dict) -> dict:
    """Execute a single planning episode inside a worker process."""
    import time as _time
    from functools import partial as _partial

    condition      = task["condition"]
    name           = task["name"]
    goal_frame     = task["goal_frame"]
    seed           = task["seed"]
    max_steps      = task["max_steps"]
    ach_step       = task["ach_step"]
    src_ep         = task["src_ep"]
    verbose        = task["verbose"]
    record_rollout = task["record_rollout"]
    planner        = task.get("planner", "grad")
    grad_n_steps   = task.get("grad_n_steps", 30)
    grad_lr        = task.get("grad_lr", 0.05)
    grad_tau_start = task.get("grad_tau_start", 1.0)
    grad_tau_end   = task.get("grad_tau_end", 0.1)

    lewm   = _PROC_CACHE["lewm"]
    device = _PROC_CACHE["device"]
    probes = _PROC_CACHE.get("probes")

    cost_fn = None
    if probes is not None:
        from hwm.probe import probe_cost_batch as _pcb
        cost_fn = _partial(_pcb, probes, ach_name=name)

    t0 = _time.time()

    if condition == "flat":
        from hwm.plan_flat import run_episode as _run
        result = _run(
            lewm, goal_frame, name, seed=seed,
            max_steps=max_steps, device=device,
            verbose=verbose, record_rollout=record_rollout,
            cost_fn=cost_fn,
        )

    elif condition == "linear":
        from hwm.plan_linear import fit_linear_dynamics as _fit, run_episode as _run
        if "dynamics" not in _PROC_CACHE:
            _PROC_CACHE["dynamics"] = _fit(
                _PROC_CACHE["latents_cache"], _PROC_CACHE["ridge_model_path"]
            )
        result = _run(
            lewm, _PROC_CACHE["dynamics"], goal_frame, name, seed=seed,
            max_steps=max_steps, device=device,
            verbose=verbose, record_rollout=record_rollout,
            cost_fn=cost_fn,
        )

    elif condition in ("hwm", "hwm_oracle"):
        from hwm.plan_hwm import load_hwm_high as _lhh, run_episode as _run
        if "high_pred" not in _PROC_CACHE:
            _, _hp, _mm, _ms = _lhh(_PROC_CACHE["hwm_ckpt"], device=device)
            _PROC_CACHE["high_pred"]   = _hp
            _PROC_CACHE["macro_mean"]  = _mm
            _PROC_CACHE["macro_std"]   = _ms
        oracle = condition == "hwm_oracle"
        result = _run(
            lewm, _PROC_CACHE["high_pred"], goal_frame, name,
            goal_achievement_step=ach_step,
            seed=seed, max_steps=max_steps,
            oracle=oracle,
            traj_dataset_path=_PROC_CACHE["traj_dataset"],
            latents_path=_PROC_CACHE["latents_cache"],
            device=device, verbose=verbose,
            record_rollout=record_rollout,
            cost_fn=cost_fn,
            goal_source_ep_idx=src_ep,
            macro_action_mean=_PROC_CACHE.get("macro_mean"),
            macro_action_std=_PROC_CACHE.get("macro_std"),
            planner=planner,
            grad_n_steps=grad_n_steps, grad_lr=grad_lr,
            grad_tau_start=grad_tau_start, grad_tau_end=grad_tau_end,
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")

    result["wall_time_s"] = _time.time() - t0
    result["episode"]     = task["ep_idx"]
    return result


# ── Achievement ordering (tech-tree, roughly) ─────────────────────────────────
# Ordered by median first-unlock timestep observed in the human playthroughs.
TECH_TREE_ORDER = [
    "collect_wood",
    "collect_drink",
    "eat_cow",
    "place_table",
    "make_wood_pickaxe",
    "make_wood_sword",
    "collect_stone",
    "defeat_zombie",
    "defeat_skeleton",
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
    "place_plant",
    "eat_plant",
]
# Verify all 22 achievements are represented
assert set(TECH_TREE_ORDER) == set(ACHIEVEMENT_NAMES), \
    "TECH_TREE_ORDER must contain exactly the 22 ACHIEVEMENT_NAMES"


def _ordered_achievements(
    goal_library: dict,
    up_to: str | None = None,
) -> list[tuple[str, np.ndarray, int, int]]:
    """Return (name, frame, step, source_ep_idx) tuples in TECH_TREE_ORDER.

    Args:
        up_to: If given (and not 'all'), restrict the order to achievements up to
               and including this entry in TECH_TREE_ORDER.  Pass None or 'all'
               to use the full list.
    """
    order = TECH_TREE_ORDER
    if up_to and up_to != "all" and up_to in TECH_TREE_ORDER:
        order = TECH_TREE_ORDER[:TECH_TREE_ORDER.index(up_to) + 1]

    goal_names = list(goal_library["goal_names"])
    # goal_source_ep_idxs added in updated build_goal_library.py; fall back to 0
    ep_idxs = goal_library.get("goal_source_ep_idxs", None)
    results = []
    for name in order:
        if name not in goal_names:
            continue
        idx = goal_names.index(name)
        frame    = goal_library["goal_frames"][idx]
        step     = int(goal_library["goal_achievement_steps"][idx])
        src_ep   = int(ep_idxs[idx]) if ep_idxs is not None else 0
        if step < 0:
            continue  # achievement never seen in human data
        results.append((name, frame, step, src_ep))
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
    # Cost function
    probes: dict | None = None,
    macro_action_mean=None,
    macro_action_std=None,
    achievements_up_to: str | None = "collect_iron",
    planner: str = "grad",
    grad_n_steps: int = 30,
    grad_lr: float = 0.05,
    grad_tau_start: float = 1.0,
    grad_tau_end: float = 0.1,
<<<<<<< HEAD
    record_all_rollouts: bool = False,
=======
>>>>>>> hwm
) -> list[dict]:
    """Run *n_episodes* for *condition* and return list of result dicts.

    Args:
        probes:             When provided (loaded via probe.load_probes), uses
                            probe-based planning cost instead of L1 image-matching.
                            Pass None to use the original L1 cost (default).
        achievements_up_to: Restrict the achievement rotation to tech-tree entries
                            up to and including this name.  Pass 'all' or None for
                            the full list.
    """
    from functools import partial as _partial
    from hwm.probe import probe_cost_batch as _probe_cost_batch

    achievements = _ordered_achievements(goal_library, up_to=achievements_up_to)
    records = []

    if record_all_rollouts:
        n_rollout_record = n_episodes
    else:
        n_rollout_record = min(3, n_episodes)

    for ep in range(n_episodes):
        seed = seed_start + ep

        # Rotate through tech-tree achievements
        name, frame, ach_step, src_ep = achievements[ep % len(achievements)]

        record_rollout = ep >= n_episodes - n_rollout_record

        # Build per-episode cost function
        if probes is not None:
            cost_fn = _partial(_probe_cost_batch, probes, ach_name=name)
        else:
            cost_fn = None

        t0 = time.time()
        if condition == "flat":
            result = run_flat(
                lewm, frame, name, seed=seed,
                max_steps=max_steps, device=device, verbose=verbose,
                record_rollout=record_rollout,
                cost_fn=cost_fn,
            )
        elif condition == "linear":
            assert dynamics is not None, "dynamics required for linear condition"
            result = run_linear(
                lewm, dynamics, frame, name, seed=seed,
                max_steps=max_steps, device=device, verbose=verbose,
                record_rollout=record_rollout,
                cost_fn=cost_fn,
            )
        elif condition in ("hwm", "hwm_oracle"):
            assert high_pred is not None, "high_pred required for hwm condition"
            oracle = condition == "hwm_oracle"
            result = run_hwm(
                lewm, high_pred, frame, name,
                goal_achievement_step=ach_step,
                seed=seed, max_steps=max_steps,
                oracle=oracle,
                traj_dataset_path=traj_dataset_path,
                latents_path=latents_path,
                device=device, verbose=verbose,
                record_rollout=record_rollout,
                cost_fn=cost_fn,
                goal_source_ep_idx=src_ep,
                macro_action_mean=macro_action_mean,
                macro_action_std=macro_action_std,
                planner=planner,
                grad_n_steps=grad_n_steps,
                grad_lr=grad_lr,
                grad_tau_start=grad_tau_start,
                grad_tau_end=grad_tau_end,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        result["wall_time_s"] = time.time() - t0
        result["episode"] = ep
        records.append(result)

        side = result.get("side_achievements", {})
        side_str = (
            "  side=[" + ", ".join(
                f"{k}@{v}" for k, v in sorted(side.items(), key=lambda x: x[1])
            ) + "]"
            if side else ""
        )
        print(
            f"[{condition}] ep={ep:3d}/{n_episodes}  {name:25s}  "
            f"success={result['success']}  steps={result['steps']:4d}  "
            f"plan_ms={result['planning_ms_per_step']:.1f}"
            f"{side_str}"
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


def save_rollout_gifs_to_dir(
    records: list[dict],
    out_dir: Path,
    condition: str,
    *,
    fps: float = 8.0,
) -> int:
    """Write each episode's ``rollout_frames`` (uint8 H W 3) to a GIF under ``out_dir``.

    Returns the number of GIFs written.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise SystemExit(
            "--save_rollout_gifs requires imageio: pip install imageio"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in records:
        vid_arr = r.get("rollout_frames")
        if vid_arr is None:
            continue
        vid = np.asarray(vid_arr)
        if vid.ndim != 4 or vid.shape[0] < 1:
            continue
        ep = int(r["episode"])
        ach = str(r.get("achievement", "unknown"))
        safe = re.sub(r"[^\w\-.]+", "_", ach).strip("_")[:48] or "ach"
        ok = int(r.get("success", 0))
        path = out_dir / f"{condition}_ep{ep:02d}_{safe}_ok{ok}.gif"
        imageio.mimsave(str(path), list(vid), fps=min(20, max(4.0, float(fps))))
        print(f"  rollout GIF → {path}  ({vid.shape[0]} frames)")
        n += 1
    if n == 0:
        print(
            f"  warning: wrote 0 GIFs for {condition} — no rollout_frames "
            "(need more episodes recorded or record_rollout episodes)."
        )
    return n


def wandb_log_rollouts(run, cond: str, records: list[dict], step: int) -> None:
    """Log frame videos for episodes that included rollout_frames (final 3 per condition).

    Tries to log as a GIF written via imageio (no moviepy dependency).  Falls
    back to logging only the first frame as an image if imageio is unavailable.
    """
    if run is None:
        return
    payload: dict = {}
    tmp_files: list[str] = []

    for r in records:
        vid_arr = r.get("rollout_frames")
        if vid_arr is None:
            continue
        vid = np.asarray(vid_arr)
        if vid.ndim != 4 or vid.shape[0] < 1:
            continue
        ep = int(r["episode"])
        fps = min(20, max(4, int(150 / max(vid.shape[0], 1))))

        video_logged = False
        try:
            import imageio
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
                tmp_path = f.name
            tmp_files.append(tmp_path)
            imageio.mimsave(tmp_path, list(vid), fps=fps)
            payload[f"{cond}/rollout/ep_{ep:03d}_video"] = _wandb.Video(tmp_path)
            video_logged = True
        except Exception:
            pass

        if not video_logged:
            # Last-resort: log first frame only
            payload[f"{cond}/rollout/ep_{ep:03d}_frame0"] = _wandb.Image(vid[0])

        payload[f"{cond}/rollout/ep_{ep:03d}_achievement"] = r["achievement"]
        payload[f"{cond}/rollout/ep_{ep:03d}_success"] = int(r["success"])
        payload[f"{cond}/rollout/ep_{ep:03d}_steps"] = int(r["steps"])

    if payload:
        run.log(payload, step=step)

    for p in tmp_files:
        try:
            import os
            os.unlink(p)
        except OSError:
            pass


def _print_achievement_table(aggregates: dict) -> None:
    """Print a fixed-width table to stdout: rows = conditions, cols = achievements."""
    if not aggregates:
        return

    ach_cols = [a for a in TECH_TREE_ORDER if any(
        a in s.get("per_achievement", {}) for s in aggregates.values()
    )]
    cond_w = max(len(c) for c in aggregates) + 2
    num_w = 6

    header = (
        f"{'condition':<{cond_w}}"
        f"{'overall':>{num_w}}"
        f"{'steps':>{num_w}}"
        f"{'ms/step':>{num_w+1}}"
        + "".join(f"  {a[:10]:>10}" for a in ach_cols)
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("ACHIEVEMENT SUCCESS RATE TABLE")
    print(sep)
    print(header)
    print(sep)
    for cond, stats in aggregates.items():
        per_ach = stats.get("per_achievement", {})
        row = (
            f"{cond:<{cond_w}}"
            f"{stats['success_rate']:>{num_w}.1%}"
            f"{stats['mean_steps']:>{num_w}.0f}"
            f"{stats['mean_planning_ms']:>{num_w+1}.1f}"
            + "".join(
                f"  {per_ach[a]['success_rate']:>10.1%}" if a in per_ach else f"  {'—':>10}"
                for a in ach_cols
            )
        )
        print(row)
    print(sep)


def wandb_log_achievement_table(run, aggregates: dict) -> None:
    """Log wandb Tables summarising the evaluation suite.

    Logs two tables:

    ``summary/achievement_table``
        One row per condition.  Columns: condition, n_episodes, n_success,
        success_rate, mean_steps, mean_planning_ms, then one ``ach/<name>``
        column per achievement in TECH_TREE_ORDER containing the **success
        rate** (float 0–1, 0.0 when not attempted).

    ``summary/achievement_counts_table``
        One row per condition.  Columns: condition, n_episodes, n_success,
        then one ``ach/<name>`` column per achievement containing the **total
        number of times that goal was accomplished** (integer count, 0 when
        not attempted).  Provides an at-a-glance view of raw goal completion
        volume across the whole evaluation suite.
    """
    if run is None or not aggregates:
        return

    ach_cols = list(TECH_TREE_ORDER)

    # ── Success-rate table (one row per condition) ────────────────────────────
    rate_columns = (
        ["condition", "n_episodes", "n_success", "success_rate",
         "mean_steps", "mean_planning_ms"]
        + [f"ach/{a}" for a in ach_cols]
    )
    rate_table = _wandb.Table(columns=rate_columns)
    for cond, stats in aggregates.items():
        per_ach = stats.get("per_achievement", {})
        ach_rates = [
            round(per_ach[a]["success_rate"], 4) if a in per_ach else 0.0
            for a in ach_cols
        ]
        rate_table.add_data(
            cond,
            stats["n_episodes"],
            stats["n_success"],
            round(stats["success_rate"], 4),
            round(stats["mean_steps"], 1),
            round(stats["mean_planning_ms"], 2),
            *ach_rates,
        )

    # ── Counts table (total goals accomplished per achievement column) ─────────
    count_columns = (
        ["condition", "n_episodes", "n_success"]
        + [f"ach/{a}" for a in ach_cols]
    )
    count_table = _wandb.Table(columns=count_columns)
    for cond, stats in aggregates.items():
        per_ach = stats.get("per_achievement", {})
        ach_counts = [
            int(round(per_ach[a]["n_episodes"] * per_ach[a]["success_rate"]))
            if a in per_ach else 0
            for a in ach_cols
        ]
        count_table.add_data(
            cond,
            stats["n_episodes"],
            stats["n_success"],
            *ach_counts,
        )

    run.log({
        "summary/achievement_table": rate_table,
        "summary/achievement_counts_table": count_table,
    })


def wandb_log_goals_accomplished_table(run, records: list[dict]) -> None:
    """Log a wandb.Table of raw goal-accomplishment counts per condition × achievement.

    Rows: one per condition.
    Columns: condition, total_accomplished, then one integer column per achievement
             in TECH_TREE_ORDER counting the number of episodes where that specific
             achievement was the target *and* was successfully accomplished.
    """
    if run is None or not records:
        return

    from collections import defaultdict

    # count[cond][ach] = number of successful episodes targeting that achievement
    count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total: dict[str, int] = defaultdict(int)
    for r in records:
        cond = r.get("condition", "unknown")
        if r.get("success"):
            ach = r["achievement"]
            count[cond][ach] += 1
            total[cond] += 1

    ach_cols = list(TECH_TREE_ORDER)
    columns = ["condition", "total_accomplished"] + list(ach_cols)
    table = _wandb.Table(columns=columns)

    for cond in sorted(count.keys()):
        ach_counts = [count[cond].get(a, 0) for a in ach_cols]
        table.add_data(cond, total[cond], *ach_counts)

    run.log({"summary/goals_accomplished_table": table})


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
    parser.add_argument("--n_episodes",      type=int,   default=5)
    parser.add_argument("--seed_start",      type=int,   default=100)
    parser.add_argument("--max_steps",       type=int,   default=1000)
    parser.add_argument("--append",          action="store_true",
                        help="Append to existing results.json instead of overwriting")
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--n_workers",       type=int, default=1,
                        help="Number of parallel worker processes for episode evaluation. "
                             "Each worker loads its own model copy on GPU. "
                             "Set to 0 to use os.cpu_count() workers.")
    parser.add_argument("--wandb",           action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",   default="lewm-crafter")
    parser.add_argument("--wandb_run_name",  default=None)
    # ── Cost function flags ───────────────────────────────────────────────────
    parser.add_argument("--cost",            default="l1",
                        choices=["l1", "probe"],
                        help="Planning cost: 'l1' = L1 image-matching (original), "
                             "'probe' = linear probe achievement probability")
    parser.add_argument("--probe_path",      default=str(
                            Path(DATA_OUT) / "probes.pkl"),
                        help="Path to fitted probe dict (used when --cost probe)")
    parser.add_argument("--achievements_up_to", default="collect_iron",
                        help="Restrict eval to tech-tree achievements up to and "
                             "including this one.  Pass 'all' to use the full "
                             "TECH_TREE_ORDER (22 achievements).")
    parser.add_argument(
        "--planner",
        default="grad",
        choices=["cem", "grad"],
        help="HWM low-level planner: CEM vs Gumbel-softmax + Adam (grad, default)",
    )
    parser.add_argument("--grad_n_steps", type=int, default=30)
    parser.add_argument("--grad_lr", type=float, default=0.05)
    parser.add_argument("--grad_tau_start", type=float, default=1.0)
    parser.add_argument("--grad_tau_end", type=float, default=0.1)
    parser.add_argument("--fit_probes",      action="store_true",
                        help="Fit probes from latents.npz and human NPZ files, "
                             "print accuracy table, then exit (no eval run)")
    parser.add_argument("--npz_dir",         default=NPZ_DIR,
                        help="Directory of human playthrough .npz files "
                             "(used when --fit_probes)")
    parser.add_argument(
        "--save_rollout_gifs",
        type=str,
        default=None,
        metavar="DIR",
        help="Write RGB rollout GIFs to DIR (records all episodes; pair with "
             "--n_episodes 8 for eight GIFs)",
    )
    parser.add_argument(
        "--rollout_gif_fps",
        type=float,
        default=8.0,
        help="Frames per second for --save_rollout_gifs (default: 8)",
    )
    args = parser.parse_args()

    conditions_to_run = (
        ["flat", "linear", "hwm", "hwm_oracle"]
        if args.condition == "all"
        else [args.condition]
    )

    # ── Probe fitting (diagnostic mode — exits after printing accuracy) ───────
    if args.fit_probes:
        from hwm.probe import build_achievement_labels, fit_probes as _fit_probes
        d = np.load(args.latents_cache)
        Z          = d["Z"].astype(np.float32)
        boundaries = d["trajectory_boundaries"]
        print(f"Loaded latents: Z={Z.shape}  boundaries={boundaries.shape}")
        labels = build_achievement_labels(args.npz_dir, Z, boundaries)
        _fit_probes(Z, labels, save_path=args.probe_path)
        return

    # ── Load probes when probe cost is requested ──────────────────────────────
    probes = None
    if args.cost == "probe":
        from hwm.probe import load_probes
        print(f"Loading probe dict from {args.probe_path}")
        probes = load_probes(args.probe_path)

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

    macro_mean = macro_std = None
    if "hwm" in conditions_to_run or "hwm_oracle" in conditions_to_run:
        print(f"Loading HWM high-level modules from {args.hwm_checkpoint}")
        _, high_pred, macro_mean, macro_std = load_hwm_high(
            args.hwm_checkpoint, device=device
        )

    # ── Run conditions ────────────────────────────────────────────────────────
    all_records: list[dict] = []

    # Load existing results if appending
    results_path = Path(args.results_path)
    if args.append and results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        all_records = existing.get("episodes", [])
        print(f"Appending to existing {len(all_records)} records")

    n_workers = args.n_workers if args.n_workers > 0 else None  # None → os.cpu_count()

    if n_workers == 1:
        # ── Sequential path (original behaviour) ─────────────────────────────
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
                probes=probes,
                macro_action_mean=macro_mean,
                macro_action_std=macro_std,
                achievements_up_to=args.achievements_up_to,
                planner=args.planner,
                grad_n_steps=args.grad_n_steps,
                grad_lr=args.grad_lr,
                grad_tau_start=args.grad_tau_start,
                grad_tau_end=args.grad_tau_end,
<<<<<<< HEAD
                record_all_rollouts=args.save_rollout_gifs is not None,
=======
>>>>>>> hwm
            )
            if wandb_run is not None:
                wandb_log_condition_final(
                    wandb_run, cond, aggregate_one_condition(records), step=ci,
                )
                wandb_log_rollouts(wandb_run, cond, records, step=ci)
            if args.save_rollout_gifs:
                save_rollout_gifs_to_dir(
                    records,
                    Path(args.save_rollout_gifs),
                    cond,
                    fps=args.rollout_gif_fps,
                )
            for r in records:
                r.pop("rollout_frames", None)
            all_records.extend(records)

    else:
        # ── Parallel path ─────────────────────────────────────────────────────
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor, as_completed as _as_completed

        condition_order = {c: i for i, c in enumerate(conditions_to_run)}
        src_path = str(Path(__file__).resolve().parent.parent)
        device_str = str(device)

        # Build flat task list: all (condition, episode) pairs
        all_tasks: list[dict] = []
        for cond in conditions_to_run:
            achievements = _ordered_achievements(
                goal_library, up_to=args.achievements_up_to
            )
            n_rollout_record = (
                args.n_episodes
                if args.save_rollout_gifs
                else min(3, args.n_episodes)
            )
            for ep in range(args.n_episodes):
                seed = args.seed_start + ep
                name, frame, ach_step, src_ep = achievements[ep % len(achievements)]
                all_tasks.append({
                    "condition":      cond,
                    "condition_order": condition_order[cond],
                    "ep_idx":         ep,
                    "name":           name,
                    "goal_frame":     frame,
                    "ach_step":       ach_step,
                    "src_ep":         src_ep,
                    "seed":           seed,
                    "max_steps":      args.max_steps,
                    "record_rollout": ep >= args.n_episodes - n_rollout_record,
                    "verbose":        args.verbose,
                    "planner":        args.planner,
                    "grad_n_steps":   args.grad_n_steps,
                    "grad_lr":        args.grad_lr,
                    "grad_tau_start": args.grad_tau_start,
                    "grad_tau_end":   args.grad_tau_end,
                })

        n_total = len(all_tasks)
        print(f"\nDispatching {n_total} episodes across {n_workers} workers "
              f"(conditions: {conditions_to_run})")

        ctx = _mp.get_context("spawn")
        parallel_results: list[dict] = []
        n_done = 0

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                args.checkpoint, args.hwm_checkpoint,
                args.latents_cache, args.ridge_model,
                args.traj_dataset, device_str, probes, src_path,
            ),
        ) as pool:
            futures = {pool.submit(_worker_run_episode, t): t for t in all_tasks}
            for fut in _as_completed(futures):
                result = fut.result()
                task   = futures[fut]
                n_done += 1
                side = result.get("side_achievements", {})
                side_str = (
                    "  side=[" + ", ".join(
                        f"{k}@{v}" for k, v in sorted(side.items(), key=lambda x: x[1])
                    ) + "]" if side else ""
                )
                print(
                    f"[{task['condition']}] ep={task['ep_idx']:3d}/{args.n_episodes}"
                    f"  {task['name']:25s}  success={result['success']}"
                    f"  steps={result['steps']:4d}"
                    f"  plan_ms={result['planning_ms_per_step']:.1f}"
                    f"{side_str}"
                    f"  ({n_done}/{n_total})"
                )
                parallel_results.append(result)

        # Sort into condition order then episode order for consistent aggregation
        parallel_results.sort(
            key=lambda r: (condition_order[r["condition"]], r["episode"])
        )

        # Per-condition wandb logging
        from collections import defaultdict as _dd
        by_cond: dict[str, list[dict]] = _dd(list)
        for r in parallel_results:
            by_cond[r["condition"]].append(r)

        for ci, cond in enumerate(conditions_to_run):
            records = by_cond.get(cond, [])
            if wandb_run is not None:
                wandb_log_condition_final(
                    wandb_run, cond, aggregate_one_condition(records), step=ci,
                )
                wandb_log_rollouts(wandb_run, cond, records, step=ci)
            if args.save_rollout_gifs:
                save_rollout_gifs_to_dir(
                    records,
                    Path(args.save_rollout_gifs),
                    cond,
                    fps=args.rollout_gif_fps,
                )
            for r in records:
                r.pop("rollout_frames", None)

        all_records.extend(parallel_results)

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
        "achievements_up_to": args.achievements_up_to,
        "n_workers": args.n_workers,
        "cost": args.cost,
        "probe_path": args.probe_path if args.cost == "probe" else None,
        "planner": args.planner,
        "grad_n_steps": args.grad_n_steps,
        "grad_lr": args.grad_lr,
        "grad_tau_start": args.grad_tau_start,
        "grad_tau_end": args.grad_tau_end,
<<<<<<< HEAD
        "save_rollout_gifs": args.save_rollout_gifs,
        "rollout_gif_fps": args.rollout_gif_fps if args.save_rollout_gifs else None,
=======
>>>>>>> hwm
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

    # ── Print achievement table ────────────────────────────────────────────────
    _print_achievement_table(aggregates)

    if wandb_run is not None:
        wandb_log_achievement_table(wandb_run, aggregates)
        wandb_log_goals_accomplished_table(wandb_run, all_records)
        _wandb.finish()


if __name__ == "__main__":
    main()
