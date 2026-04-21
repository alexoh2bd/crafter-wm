"""Generate four evaluation figures from results/results.json.

Figures
-------
1. results/fig1_success_rate.png
   Bar chart of achievement success rate per condition (flat / linear / hwm / hwm_oracle).

2. results/fig2_steps_cdf.png
   CDF of steps-to-completion (successful episodes only) per condition.

3. results/fig3_planning_time.png
   Box plot of planning time per step (ms) per condition, log-scale y-axis.

4. results/fig4_subgoal_panel.png
   Qualitative subgoal panel for one HWM episode: shows z_subgoal visualized
   by nearest-neighbour retrieval from the training trajectories (in latent
   space using pre-encoded latents.npz). No decoder needed.

Usage
-----
    python src/hwm/plot_results.py
    python src/hwm/plot_results.py --results results/results.json --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hwm.constants import (
    CHECKPOINT,
    GOAL_LIBRARY,
    LATENTS_CACHE,
    RESULTS_DIR,
    RESULTS_JSON,
    TRAJ_DATASET,
)


# ── Colour palette (colour-blind friendly) ────────────────────────────────────
COND_COLORS = {
    "flat":       "#1f77b4",
    "linear":     "#ff7f0e",
    "hwm":        "#2ca02c",
    "hwm_oracle": "#9467bd",
}
COND_LABELS = {
    "flat":       "Flat WM",
    "linear":     "Linear",
    "hwm":        "HWM",
    "hwm_oracle": "HWM Oracle",
}
COND_ORDER = ["flat", "linear", "hwm", "hwm_oracle"]


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")


# ── Figure 1: Success rate bar chart ─────────────────────────────────────────

def plot_success_rate(aggregates: dict, out_path: str) -> None:
    plt = _import_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811 (needed for type annotations)

    conds = [c for c in COND_ORDER if c in aggregates]
    rates = [aggregates[c]["success_rate"] * 100 for c in conds]
    colors = [COND_COLORS[c] for c in conds]
    labels = [COND_LABELS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.7)

    # Annotate each bar
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Achievement Success Rate by Planning Condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Figure 2: Steps-to-completion CDF ────────────────────────────────────────

def plot_steps_cdf(episodes: list[dict], out_path: str) -> None:
    plt = _import_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811

    fig, ax = plt.subplots(figsize=(7, 4))

    for cond in COND_ORDER:
        steps = sorted([
            r["steps"] for r in episodes
            if r["condition"] == cond and r["success"]
        ])
        if not steps:
            continue
        n = len(steps)
        cdf = np.arange(1, n + 1) / n
        ax.step(steps, cdf, label=COND_LABELS[cond],
                color=COND_COLORS[cond], linewidth=1.8, where="post")

    ax.set_xlabel("Steps to completion")
    ax.set_ylabel("Cumulative fraction of successes")
    ax.set_title("Steps-to-Completion CDF (successful episodes)")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Figure 3: Planning time box plot ─────────────────────────────────────────

def plot_planning_time(episodes: list[dict], out_path: str) -> None:
    plt = _import_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811

    conds = [c for c in COND_ORDER if any(r["condition"] == c for r in episodes)]
    data  = [
        [r["planning_ms_per_step"] for r in episodes if r["condition"] == c]
        for c in conds
    ]
    labels = [COND_LABELS[c] for c in conds]
    colors = [COND_COLORS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_yscale("log")
    ax.set_xticklabels(labels)
    ax.set_ylabel("Planning time per step (ms, log scale)")
    ax.set_title("Planning Time Distribution by Condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Figure 4: Qualitative subgoal panel ──────────────────────────────────────

def plot_subgoal_panel(
    out_path: str,
    results_path: str,
    latents_path: str,
    traj_dataset_path: str,
    checkpoint_path: str,
    goal_library_path: str,
    n_subgoal_frames: int = 5,
) -> None:
    """Show z_subgoal for successive high-level replanning steps of one HWM episode.

    Visualization: for each replanning step, retrieve the nearest-neighbour
    frame from training trajectories (L2 in latent space) and display it
    alongside the real current frame.
    """
    plt = _import_matplotlib()
    import matplotlib.pyplot as plt  # noqa: F811
    import torch

    from hwm.models import load_lewm

    # We need the latents.npz (pre-encoded training frames) for NN retrieval
    lp = Path(latents_path)
    if not lp.exists():
        print(f"  WARNING: {latents_path} not found — skipping fig4 (run train_hwm_high first)")
        return

    d_lat = np.load(latents_path)
    Z_train = d_lat["Z"].astype(np.float32)       # (N, D)

    d_traj = np.load(traj_dataset_path)
    obs_train = d_traj["obs"]                      # (N, 64, 64, 3) uint8

    # Load goal library
    gl = np.load(goal_library_path, allow_pickle=True)
    goal_names = list(gl["goal_names"])

    # Find a representative HWM episode from results
    with open(results_path) as f:
        results = json.load(f)

    hwm_episodes = [r for r in results["episodes"]
                    if r["condition"] == "hwm" and r["success"]]
    if not hwm_episodes:
        print("  No successful HWM episodes found — skipping fig4")
        return

    # Pick the episode with most steps (so we have multiple subgoal frames)
    ep = max(hwm_episodes, key=lambda r: r["steps"])
    target_ach = ep["achievement"]
    seed = ep["seed"]
    print(f"  Qualitative panel: achievement={target_ach}  seed={seed}  steps={ep['steps']}")

    # Re-run a short extract of the episode, recording subgoal latents
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lewm, _ = load_lewm(checkpoint_path, device=device)

    from hwm.constants import HWM_HIGH_CKPT
    from hwm.plan_hwm import load_hwm_high, cem_high, cem_low
    import crafter

    try:
        action_enc, high_pred = load_hwm_high(HWM_HIGH_CKPT, device=device)
    except FileNotFoundError:
        print(f"  WARNING: {HWM_HIGH_CKPT} not found — skipping fig4")
        return

    if target_ach not in goal_names:
        print(f"  Achievement {target_ach} not in goal library — skipping fig4")
        return
    idx = goal_names.index(target_ach)
    goal_frame = gl["goal_frames"][idx]

    goal_t = torch.from_numpy(goal_frame).float() / 255.0
    goal_t = goal_t.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        z_goal = lewm.encode(goal_t)

    env = crafter.Env(seed=seed)
    obs = env.reset()

    subgoal_frames_real  = []  # NN-retrieved frames from training set
    current_frames       = []  # actual current obs at each replan
    H_lo = 10
    steps_since_replan = H_lo  # force replan on step 0

    for step in range(min(ep["steps"], 200)):
        obs_t = torch.from_numpy(obs).float() / 255.0
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            z_curr = lewm.encode(obs_t)

        if steps_since_replan >= H_lo and len(subgoal_frames_real) < n_subgoal_frames:
            with torch.no_grad():
                _, z_subgoal = cem_high(
                    high_pred, z_curr, z_goal, H_hi=3,
                    n_samples=300, n_elite=30, n_iters=5, device=device,
                )

            # NN retrieval: find closest training frame by L2 in latent space
            z_sub_np = z_subgoal.squeeze(0).cpu().numpy()         # (D,)
            dists = np.linalg.norm(Z_train - z_sub_np[None], axis=1)  # (N,)
            nn_idx = int(np.argmin(dists))
            subgoal_frames_real.append(obs_train[nn_idx])          # (64,64,3)
            current_frames.append(obs.copy())
            steps_since_replan = 0

        action = cem_low(lewm, z_curr, z_subgoal,
                         H=H_lo, n_samples=200, n_elite=20, n_iters=3, device=device)
        obs, _, done, info = env.step(action)
        steps_since_replan += 1
        if done:
            break

    if not subgoal_frames_real:
        print("  No subgoal frames collected — skipping fig4")
        return

    n_cols = len(subgoal_frames_real)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, (curr_f, sub_f) in enumerate(zip(current_frames, subgoal_frames_real)):
        axes[0, i].imshow(curr_f)
        axes[0, i].set_title(f"Step {i * H_lo}\n(current)", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(sub_f)
        axes[1, i].set_title("z_subgoal\n(NN frame)", fontsize=8)
        axes[1, i].axis("off")

    fig.suptitle(
        f"HWM Subgoal Visualization — '{target_ach}' (seed {seed})\n"
        "Top: current obs  |  Bottom: nearest-neighbour training frame to z_subgoal",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HWM evaluation results")
    parser.add_argument("--results",       default=RESULTS_JSON)
    parser.add_argument("--out",           default=RESULTS_DIR)
    parser.add_argument("--checkpoint",    default=CHECKPOINT)
    parser.add_argument("--goal_library",  default=GOAL_LIBRARY)
    parser.add_argument("--latents_cache", default=LATENTS_CACHE)
    parser.add_argument("--traj_dataset",  default=TRAJ_DATASET)
    parser.add_argument("--skip_fig4",     action="store_true",
                        help="Skip the slow qualitative subgoal panel")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    episodes   = data["episodes"]
    aggregates = data["aggregates"]

    # Fig 1 — success rate
    plot_success_rate(aggregates, str(out_dir / "fig1_success_rate.png"))

    # Fig 2 — steps CDF
    plot_steps_cdf(episodes, str(out_dir / "fig2_steps_cdf.png"))

    # Fig 3 — planning time
    plot_planning_time(episodes, str(out_dir / "fig3_planning_time.png"))

    # Fig 4 — qualitative subgoal panel
    if not args.skip_fig4:
        plot_subgoal_panel(
            out_path=str(out_dir / "fig4_subgoal_panel.png"),
            results_path=args.results,
            latents_path=args.latents_cache,
            traj_dataset_path=args.traj_dataset,
            checkpoint_path=args.checkpoint,
            goal_library_path=args.goal_library,
        )
    else:
        print("Skipping fig4 (--skip_fig4)")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
