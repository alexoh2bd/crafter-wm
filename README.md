# Hierarchical World Model Planning in Crafter

This repository explores **goal-conditioned latent-space planning** in the
[Crafter](https://github.com/danijar/crafter) benchmark environment.  The core
hypothesis is that a **hierarchical world model** — a low-level latent world
model (LeWM) plus a high-level macro-action predictor (HWM) — can reach specific
achievements more reliably than either a flat planner or a linear dynamics
baseline, by breaking long-horizon tasks into learned macro-subgoals.

---

## Repository layout

```
plan/
├── src/
│   ├── lemodel.py              # LeWM: ViT encoder + causal transformer predictor + SIGReg
│   ├── ledata.py               # Data loading: pickle buffers → CrafterDataset
│   ├── letrain.py              # LeWM DDP training (legacy path)
│   ├── lewm_checkpoint.py      # Load .pt / YAML-merged checkpoints
│   ├── legoal.py               # Older pickle goal library (superseded)
│   ├── visualize_lewm_crafter.py   # Record Crafter rollout video with losses overlaid
│   ├── hwm/
│   │   ├── constants.py        # Shared paths, dims, action names, eval holdout
│   │   ├── models.py           # ActionEncoder, SegmentedActionEncoder, HighLevelPredictor
│   │   ├── train_lewm_v2.py    # LeWM v2 trainer (mixed/random buffers, bf16, diagnostics)
│   │   ├── train_lewm_balanced.py  # LeWM with inverse-sqrt action weighting
│   │   ├── train_hwm_high.py   # HWM training (chain / waypoint objective)
│   │   ├── build_goal_library.py   # Human NPZs → goal_library.npz + trajectory_dataset.npz
│   │   ├── build_human_pkl.py  # Older NPZ → pickle conversion helper
│   │   ├── evaluate.py         # Eval harness: all conditions, parallel workers, GIF export
│   │   ├── plan_flat.py        # Flat CEM planner (LeWM rollout)
│   │   ├── plan_linear.py      # Ridge-dynamics CEM planner
│   │   ├── plan_hwm.py         # Two-level HWM planner (CEM or grad subgoal search)
│   │   ├── probe.py            # Per-achievement logistic probe cost
│   │   ├── plot_results.py     # Figures: success bars, step CDF, planning time, subgoal panel
│   │   ├── diagnose.py         # Action-magnitude diagnostic (per-action ‖Δz‖₂)
│   │   └── data/
│   │       └── mixed_sampler.py    # MixedTransitionSampler: random + PPO replay
│   └── scripts/                # SLURM / bash launch scripts
├── data/crafter/
│   ├── human/                  # 100 human playthrough .npz files
│   ├── random_rollouts/        # random_500k.pkl — 500k uniform-random transitions
│   ├── ppo_rollouts/           # crafter_teacher_data.pkl — PPO-teacher trajectories
│   ├── ppo/                    # ppo_teacher.pt
│   └── world_model/            # All trained checkpoints (see below)
├── results/                    # Eval JSON files and figures
└── logs/                       # SLURM .out files
```

---

## Architecture overview

### LeWM — low-level latent world model

A **ViT encoder** (64×64 image → patch size 8 → 64 tokens, depth 12, heads 3)
embeds each observation into a **latent vector** `z ∈ ℝ²⁵⁶`.  A **causal
transformer predictor** (depth 6, heads 16, hidden 384, context_len 16) takes a
window of `(z_t, a_t)` pairs and predicts `z_{t+1}` via AdaLN action
conditioning.  Training minimises a **prediction MSE + SIGReg** (Siamese
representation regularisation, λ=0.1, M=1024 pairs) to keep the latent space
geometrically meaningful.

Key constants (`constants.py`): `LATENT_DIM=256`, `ACTION_DIM=17`, `CONTEXT_LEN=16`.

### HWM — high-level macro-action model

| Module | Role |
|--------|------|
| **ActionEncoder** | 2-layer bidirectional Transformer; maps an action subsequence of variable length to a `MACRO_DIM=32` vector via CLS pooling |
| **SegmentedActionEncoder** | Nav/interaction CLS heads + type embeddings (ablation) |
| **HighLevelPredictor** | Same block structure as the LeWM predictor; conditions on macro-action embeddings projected to 256-D; `context_len=3` (3 macro steps) |

Training builds **WaypointChainDataset**: chains of K=4–16 human subgoals drawn
from pre-encoded latents.  Loss = MSE on predicted intermediate latents +
action-sequence reconstruction (SIGReg variant).

### Planning conditions

| Condition | Planner | Description |
|-----------|---------|-------------|
| `flat` | CEM, H=10 primitives | CEM directly over LeWM rollouts toward goal latent |
| `linear` | CEM, Ridge dynamics | Ridge regression fit on `(z_t, a_t) → z_{t+1}`; CEM on that model |
| `hwm` | Two-level | HWM proposes macro-action subgoal; low-level CEM or gradient (Gumbel-softmax + Adam) refines primitive sequence |
| `hwm_oracle` | Two-level, oracle mid-point | Same as `hwm` but subgoal latent is the real human midpoint (ablation ceiling) |
| `ppo` | Scripted PPO | PPO teacher policy (behaviour baseline, no planning) |

**Cost functions**: `l1` (L1 distance between rollout-end and goal latent) or
`probe` (logistic probe score `1 − P(achievement | z)`).

---

## Data pipeline

```
100 human NPZ episodes
        │
        ├── build_goal_library.py
        │       hold-out: episodes 95–99 (eval only)
        │       → goal_library.npz        (22 goal frames + first-unlock steps)
        │       → trajectory_dataset.npz  (60 261 frames, 95 train episodes)
        │
        ├── train_hwm_high.py (pre-encode with frozen LeWM)
        │       → latents.npz   Z=(N, 256)
        │
        ├── plan_linear.fit_linear_dynamics
        │       → ridge_model.pkl
        │
        └── probe.fit_probes
                → probes.pkl   (22 logistic probe classifiers)

random_500k.pkl  (500k uniform-random Crafter transitions)
crafter_teacher_data.pkl  (PPO-teacher rollouts)
        │
        └── MixedTransitionSampler → LeWM training buffers
```

**Eval holdout**: human episodes 95–99 are never in `trajectory_dataset.npz` or
the probe training set, but the goal library scans all 100 episodes to find the
canonical achievement frame.

---

## Experiments / ablations

Every experiment uses the same 65-episode protocol unless noted:
seeds `100 + ep`, goals rotating through 13 achievements up to `collect_iron`,
`max_steps=1000`, `--cost probe`.

### LeWM variants trained

| Tag / logdir | Buffer | Notes |
|---|---|---|
| `lewm_teacher_deep` | PPO-teacher rollouts | First deep LeWM (DDP, `letrain.py`) |
| `lewm_teacher_deep_ft` | + human NPZ fine-tune | Fine-tune pass on human data |
| `lewm_human_ft` | Human NPZ | Direct human-data LeWM (used for most HWM evals) |
| `lewm_balanced_ppo` | PPO-heavy + inverse-sqrt action weights | `train_lewm_balanced.py`, rare-action oversampling |
| `lewm_v2` | 70% random + 30% PPO, mixed sampler | `train_lewm_v2.py`, predictor_hidden_dim 384, bf16, 100k steps |
| `lewm_v2_random_wm` | 100% random rollouts | `--random-only`, warm-started from `lewm_v2/latest.pt`; eval uses `step_25000_ratio_0.8.pt` |

**LeWM v2 training diagnostics** (logged every 5 000 steps): per-action mean
‖Δz‖₂ for all 17 actions.  `diag_ratio = craft_delta / move_delta`.  A ratio
near 1 means all action types shift the latent equally — a healthy sign.  The
random-wm run achieves ratio ≈ 0.99 at step 25 000 (spread = 0.05), indicating
well-separated action representations.

### HWM variants trained

| Tag / logdir | ActionEncoder | Training objective | LeWM backbone |
|---|---|---|---|
| `hwm_high` | Standard | Waypoint triplets | `lewm_human_ft` |
| `hwm_high_chain` | Standard | **Chain** (K=4–16 dense subgoals) | `lewm_human_ft` |
| `hwm_high_chain_strong` | Standard | Chain, full-episode mode | `lewm_human_ft` |
| `hwm_high_segenc` | **SegmentedActionEncoder** (nav/interact split) | Chain | `lewm_human_ft` |
| `hwm_high_v2` | Standard, macro_dim 32 | Chain-strong | `lewm_v2` |
| `hwm_high_random_wm` | Standard, macro_dim 32 | Chain-strong, 200 epochs | `lewm_v2_random_wm` step 25k |

**`hwm_high_random_wm` training** (job 11210238): 37 586 chains from 95 episodes,
batch 512, 200 epochs, cosine-decay LR from 3e-4.
Best val loss **155.82** (epoch ~190). ~16s/epoch on RTX PRO 6000 Blackwell.

### Low-level planner variants

Two low-level planners were compared within the HWM condition:

- **`cem`** — cross-entropy method, samples candidate primitive sequences,
  selects low-cost elite set.
- **`grad`** — Gumbel-softmax + Adam (30 steps, lr=0.05, temperature annealing
  1.0→0.1); generally faster than CEM.

Early smoke-tests used `l1` cost; all full benchmarks use `probe` cost.

---

## Results

### Summary table (65 episodes, probe cost, unless noted)

| Condition | LeWM | HWM | n_success / 65 | Success rate | Mean steps | Mean plan ms/step |
|-----------|------|-----|:-:|:-:|:-:|:-:|
| **PPO baseline** | `ppo_teacher.pt` | — | **11** | **16.9%** | 140 | — |
| `hwm_grad` (smoke) | `lewm_teacher_deep` | `hwm_high` | 0/5 | 0% | 229 | 3 355 |
| `hwm` | `lewm_human_ft` | `hwm_high` | 0/65 | 0% | 166 | 931 |
| `hwm_oracle` | `lewm_human_ft` | `hwm_high` | 0/65 | 0% | — | — |
| `hwm` (chain) | `lewm_human_ft` | `hwm_high_chain` | 1/65 | 1.5% | 160 | 285 |
| `hwm_oracle` (chain) | `lewm_human_ft` | `hwm_high_chain` | 2/65 | 3.1% | 164 | 290 |
| `hwm` (chain-strong) | `lewm_human_ft` | `hwm_high_chain_strong` | 1/65 | 1.5% | 151 | 219 |
| `hwm_oracle` (chain-strong) | `lewm_human_ft` | `hwm_high_chain_strong` | 0/65 | 0% | 163 | 223 |
| `hwm` (segenc, CEM) | `lewm_human_ft` | `hwm_high_segenc` | 1/65 | 1.5% | 160 | 215 |
| `hwm_oracle` (segenc, CEM) | `lewm_human_ft` | `hwm_high_segenc` | 0/65 | 0% | 158 | 223 |
| `flat` | `lewm_teacher_deep_ft` | — | 1/5 | 20% | 136 | 384 |
| **`hwm` + `hwm_oracle` (random_wm)** | `lewm_v2_random_wm` step 25k | `hwm_high_random_wm` | *eval running* | — | — | — |

**PPO per-achievement breakdown** (best per-achievement figures currently):
`collect_wood` 80%, `place_table` 60%, `collect_drink` 40%, `eat_cow` 40%.
All crafting / stone+ achievements: 0%.

**All evaluated HWM variants score near 0% or at most 1–3%**, well below PPO.
The single `flat` success (20% on 5 episodes, collect_wood) and HWM's rare
`collect_drink` success suggest the bottleneck is the probe-cost planner
reaching easy low-tier goals but failing to compose long-horizon action
sequences.

### Probe quality (`random_wm`, step 25k latents)

| Achievement | Pos% | Acc | Status |
|---|---|---|---|
| `collect_wood` | 97.0% | 94.1% | ✓ |
| `place_table` | 88.1% | 96.2% | ✓ |
| `make_wood_pickaxe` | 87.4% | 96.5% | ✓ |
| `collect_stone` | 83.7% | 91.7% | ✓ |
| `collect_drink` | 77.1% | 78.8% | ✓ |
| `make_stone_pickaxe` | 67.9% | 81.7% | ✓ |
| `make_stone_sword` | 67.7% | 83.0% | ✓ |
| `place_stone` | 68.7% | 76.4% | ✓ |
| `collect_coal` | 64.6% | 76.0% | ✓ |
| `collect_sapling` | 63.4% | 65.2% | borderline |
| `defeat_zombie` | 57.3% | 67.1% | borderline |
| `wake_up` | 56.2% | 69.6% | borderline |
| `eat_cow` | 79.4% | 75.5% | ✓ |
| `collect_iron` | 46.3% | 67.6% | borderline |
| `make_iron_pickaxe` | 26.3% | 66.2% | borderline |
| `make_iron_sword` | 21.9% | 68.5% | borderline |
| `eat_plant` | 15.9% | 66.9% | borderline |
| `collect_diamond` | 14.3% | 61.2% | borderline |
| `defeat_skeleton` | 28.9% | 57.1% | **⚠ WARNING** |
| `make_wood_sword` | 39.1% | 62.1% | borderline |
| `place_furnace` | 30.7% | 63.6% | borderline |
| `place_plant` | 28.0% | 63.3% | borderline |

`defeat_skeleton` is below 60% — the latent space may not cleanly encode the
preconditions for combat achievements with random-only training data.

---

## Identified failure modes

1. **Probe cost mismatch**: L1 and probe costs both point at a goal in latent
   space, but the planner can reach a nearby latent that doesn't correspond to
   actually performing the task in the world.

2. **Short-horizon planning vs. long-horizon tasks**: Even `collect_stone`
   requires several precondition steps.  With `max_steps=1000` the planner
   tends to wander rather than commit to a multi-step sequence.

3. **HWM subgoal quality**: All HWM variants including `hwm_oracle` (real human
   midpoints) score ≤ 3%.  The oracle near-matches the pure-HWM, suggesting the
   bottleneck is the **low-level primitive execution** step, not subgoal
   selection.

4. **Random-only training data**: Without exploratory / goal-directed transitions,
   the latent space clusters around frequent low-level states and the predictor
   never sees the transition patterns needed for crafting.  The `random_wm`
   ablation (current eval) tests whether the cleaner / unbiased training signal
   compensates.

5. **Planning speed**: Early `hwm_grad` runs took ~3 355 ms/step; chain-strong
   and segenc variants reduced this to ~215–290 ms/step.  Still far slower than
   PPO (no planning overhead).

6. **Probe convergence**: L-BFGS hits 1 000-iteration limit for all 22 probes on
   the random-wm latents.  Probes for rare achievements (≤30% positive rate)
   are unreliable as planning cost signals.

---

## Current status (as of 2026-04-21)

| Stage | Status |
|-------|--------|
| LeWM v2 (mixed) training, `lewm_v2`, 100k steps | ✅ Complete (job 11209986) — reached step 47 300+ |
| LeWM v2 random-only, `lewm_v2_random_wm`, resumed from step 0 warm-started on `lewm_v2/latest.pt` | ✅ `step_25000_ratio_0.8.pt` used for downstream |
| Goal library + trajectory dataset (`wm_cache_random_wm`) | ✅ 60 261 frames, 22/22 achievements found |
| HWM `hwm_high_random_wm`, 200 epochs | ✅ Best val loss 155.82 |
| Probes (`wm_cache_random_wm/probes.pkl`) | ✅ Fit with convergence warnings |
| Eval `hwm` condition, 65 episodes, `random_wm` | 🔄 **Running** (job 11210238, 4 workers) → `results/results_random_wm.json` |
| Eval `hwm_oracle` condition, 65 episodes, `random_wm` | ⏳ Pending (same job, second condition) |
| Figures (`results/figures_random_wm/`) | ⏳ Pending stage 6 |
| GIF export (`save_policy_rollout_gifs.sh`) | ⏳ Not yet run |

---

## How to run

### Prerequisites

```bash
cd plan
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install imageio imageio-ffmpeg scikit-learn matplotlib wandb
```

### Full pipeline (random-only LeWM → HWM → eval)

```bash
# Default: SKIP_LEWM=1 (uses step_25000_ratio_0.8.pt), picks up at Stage 2
sbatch src/scripts/train_lewm_v2_random_only_hwm.sh

# Override RUN_TAG or start LeWM from scratch
RUN_TAG=my_run SKIP_LEWM=0 sbatch src/scripts/train_lewm_v2_random_only_hwm.sh

# Resume LeWM from a specific checkpoint
SKIP_LEWM=0 LEWM_RESUME=data/crafter/world_model/lewm_v2/latest.pt sbatch ...
```

### Just evaluate (given existing checkpoints)

```bash
PYTHONPATH=src python src/hwm/evaluate.py \
  --condition hwm \
  --n_episodes 65 \
  --checkpoint data/crafter/world_model/lewm_v2_random_wm/step_25000_ratio_0.8.pt \
  --hwm_checkpoint data/crafter/world_model/hwm_high_random_wm/best.pt \
  --goal_library data/crafter/wm_cache_random_wm/goal_library.npz \
  --latents_cache data/crafter/wm_cache_random_wm/latents.npz \
  --traj_dataset data/crafter/wm_cache_random_wm/trajectory_dataset.npz \
  --ridge_model data/crafter/wm_cache_random_wm/ridge_model.pkl \
  --probe_path data/crafter/wm_cache_random_wm/probes.pkl \
  --cost probe --n_workers 4
```

### Generate 8 rollout GIFs

```bash
bash src/scripts/save_policy_rollout_gifs.sh
# Output: results/rollout_gifs_random_wm/*.gif
```

### Run action-magnitude diagnostic

```bash
PYTHONPATH=src python src/hwm/diagnose.py \
  --checkpoint data/crafter/world_model/lewm_v2_random_wm/step_25000_ratio_0.8.pt \
  --latents_cache data/crafter/wm_cache_random_wm/latents.npz \
  --plot logs/diag_step25k.png
```

### Fit or re-fit probes

```bash
PYTHONPATH=src python src/hwm/evaluate.py \
  --fit_probes \
  --latents_cache data/crafter/wm_cache_random_wm/latents.npz \
  --npz_dir data/crafter/human \
  --probe_path data/crafter/wm_cache_random_wm/probes.pkl
```

---

## Key hyperparameters at a glance

| Parameter | LeWM v2 | HWM (chain-strong) |
|-----------|---------|-------------------|
| Latent dim | 256 | — (uses frozen z) |
| Predictor hidden | 384 | 256 (via proj) |
| Context len | 16 | 3 macro steps |
| Macro dim | — | 32 |
| Batch size | 256 | 512 |
| LR | 3e-4 (cosine) | 3e-4 (cosine) |
| Steps / Epochs | 100k | 200 epochs |
| SIGReg λ | 0.1 | 0.2 |
| Warmup steps | 1 000 | — |
| Precision | bf16 | fp32 |
| Chain K range | — | 4–16 subgoals |

---

## Wandb

All runs log to project **`lewm-crafter`** at
[https://wandb.ai/humahuma/lewm-crafter](https://wandb.ai/humahuma/lewm-crafter).
Run names follow `lewm-v2-<logdir_name>` for LeWM and `hwm-high-train` for HWM.
