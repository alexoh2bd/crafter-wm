#!/bin/bash
# ppo_eval.sh  — evaluate PPO baseline on the same seeds used in hwm_launch.sh
# Usage:
#   bash src/scripts/ppo_eval.sh
#   N_EPISODES=65 bash src/scripts/ppo_eval.sh   # full eval (5×13)

set -e

# ── Resolve PROJECT_ROOT ──────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

SRC_DIR="${PROJECT_ROOT}/src"
cd "${SRC_DIR}"

# ── venv ──────────────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
PPO_CKPT="${PROJECT_ROOT}/data/crafter/ppo/ppo_teacher.pt"
RESULTS_JSON="${PROJECT_ROOT}/results/ppo_results.json"

# ── Eval params — match hwm_launch.sh exactly ─────────────────────────────────
N_EPISODES="${N_EPISODES:-5}"     # default 5 seeds; set to 65 for full 5×13 run
SEED_START=100
MAX_STEPS=1000

mkdir -p "${PROJECT_ROOT}/results"

echo "=== PPO baseline eval: ${N_EPISODES} episodes, seeds ${SEED_START}..$((SEED_START + N_EPISODES - 1)) ==="

python - <<PYEOF
import json, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import crafter

sys.path.insert(0, "${SRC_DIR}")
from teacherPPO import ActorCritic, preprocess

# ── tech-tree order (matches evaluate.py TECH_TREE_ORDER, up to collect_iron) ─
TECH_TREE_UP_TO_IRON = [
    "collect_wood", "collect_drink", "eat_cow", "place_table",
    "make_wood_pickaxe", "make_wood_sword", "collect_stone",
    "defeat_zombie", "defeat_skeleton", "make_stone_pickaxe",
    "make_stone_sword", "collect_coal", "collect_iron",
]
N_ACH = len(TECH_TREE_UP_TO_IRON)   # 13

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load policy ───────────────────────────────────────────────────────────────
ckpt = torch.load("${PPO_CKPT}", map_location=device, weights_only=False)
policy = ActorCritic().to(device)
policy.load_state_dict(ckpt["policy"])
policy.eval()
print(f"Loaded PPO from ${PPO_CKPT}")

# ── Run episodes ──────────────────────────────────────────────────────────────
n_episodes = ${N_EPISODES}
seed_start = ${SEED_START}
max_steps  = ${MAX_STEPS}

records = []

for ep in range(n_episodes):
    seed = seed_start + ep
    # Mirror evaluate.py rotation so the "target" achievement is the same
    target = TECH_TREE_UP_TO_IRON[ep % N_ACH]

    env = crafter.Env(seed=seed)
    obs = env.reset()

    prev_ach = {}
    success  = False
    planning_times = []
    t0_ep = time.time()

    for step in range(max_steps):
        obs_t = preprocess(obs).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            action, _, _, _ = policy.get_action(obs_t, deterministic=True)
        planning_times.append((time.perf_counter() - t0) * 1000)

        obs, _reward, done, info = env.step(action.item())
        curr_ach = info.get("achievements", {})

        for k, v in curr_ach.items():
            if v > prev_ach.get(k, 0) and k == target:
                success = True
        prev_ach = dict(curr_ach)

        if done or success:
            break

    # Record every achievement that was unlocked at least once, and at what step
    unlocked = {k: v for k, v in prev_ach.items() if v > 0}

    rec = {
        "condition":           "ppo",
        "achievement":         target,
        "seed":                seed,
        "success":             success,
        "steps":               step + 1,
        "planning_ms_per_step": float(np.mean(planning_times)) if planning_times else 0.0,
        "all_achievements":    unlocked,
        "episode":             ep,
    }
    records.append(rec)
    print(
        f"[ppo] ep={ep:3d}/{n_episodes}  target={target:25s}  "
        f"success={success}  steps={step+1:4d}  "
        f"unlocked={list(unlocked.keys())}"
    )

# ── Aggregate ─────────────────────────────────────────────────────────────────
by_ach = defaultdict(list)
for r in records:
    by_ach[r["achievement"]].append(r["success"])

print("\n── Per-achievement success rate ──────────────────────────────")
for ach in TECH_TREE_UP_TO_IRON:
    if ach in by_ach:
        sr = sum(by_ach[ach]) / len(by_ach[ach])
        print(f"  {ach:<28}  {sr:.1%}  ({sum(by_ach[ach])}/{len(by_ach[ach])})")

overall_sr = sum(r["success"] for r in records) / max(len(records), 1)
print(f"\nOverall success rate: {overall_sr:.1%}  ({sum(r['success'] for r in records)}/{len(records)} episodes)")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "episodes":   records,
    "aggregates": {
        "ppo": {
            "n_episodes":    len(records),
            "n_success":     sum(r["success"] for r in records),
            "success_rate":  overall_sr,
            "mean_steps":    float(np.mean([r["steps"] for r in records])),
            "per_achievement": {
                ach: {"n_episodes": len(v), "success_rate": float(np.mean(v))}
                for ach, v in by_ach.items()
            },
        }
    },
    "config": {
        "condition": "ppo",
        "checkpoint": "${PPO_CKPT}",
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "max_steps":  max_steps,
        "deterministic": True,
    },
}

out_path = Path("${RESULTS_JSON}")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved → ${RESULTS_JSON}")
PYEOF