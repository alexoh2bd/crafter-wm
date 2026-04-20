"""Shared constants for the HWM planning pipeline.

The data typo (`achivement_` with one 'e') lives here exclusively.
All other modules call ach_key() rather than formatting the string directly.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are relative to the project root (parent of src/).
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent  # plan/

CHECKPOINT    = str(PROJECT_ROOT / "data" / "crafter" / "world_model" / "lewm_teacher_deep" / "best.pt")
NPZ_DIR       = str(PROJECT_ROOT / "data" / "crafter" / "human")
DATA_OUT      = str(PROJECT_ROOT / "data" / "crafter" / "wm_cache")
GOAL_LIBRARY  = str(PROJECT_ROOT / "data" / "crafter" / "wm_cache" / "goal_library.npz")
TRAJ_DATASET  = str(PROJECT_ROOT / "data" / "crafter" / "wm_cache" / "trajectory_dataset.npz")
LATENTS_CACHE = str(PROJECT_ROOT / "data" / "crafter" / "wm_cache" / "latents.npz")
RIDGE_MODEL   = str(PROJECT_ROOT / "data" / "crafter" / "wm_cache" / "ridge_model.pkl")
HWM_HIGH_CKPT = str(PROJECT_ROOT / "data" / "crafter" / "world_model" / "hwm_high" / "best.pt")
RESULTS_DIR   = str(PROJECT_ROOT / "results")
RESULTS_JSON  = str(PROJECT_ROOT / "results" / "results.json")

# Indices (0-based, by sort order of NPZ files) of human episodes held out for eval.
# Fixed before any training artifact is produced; all training scripts import from here.
EVAL_EP_INDICES: list[int] = list(range(95, 100))  # episodes 95, 96, 97, 98, 99

# ── LeWM architecture (must match best.pt saved args) ────────────────────────
LATENT_DIM      = 256
ACTION_DIM      = 17
CONTEXT_LEN     = 16

# ── HWM macro-action dimension ───────────────────────────────────────────────
# ActionEncoder projects action sequences to MACRO_DIM vectors.
# CEM searches in R^MACRO_DIM (tractable) while HighLevelPredictor
# operates internally in R^LATENT_DIM.
# Backward-compatible: old checkpoints without "macro_dim" in args used 256.
MACRO_DIM       = 32

# ── Action type boundary ──────────────────────────────────────────────────────
# CRAFTER_ACTIONS[0:5]  = noop, move_left, move_right, move_up, move_down
# CRAFTER_ACTIONS[5:17] = do, sleep, place_*, make_*  (interaction / crafting)
NAV_ACTION_THRESHOLD = 5  # action index < 5 → navigation; ≥ 5 → interaction

# ── Crafter action names (index matches env.action_space) ────────────────────
CRAFTER_ACTIONS = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down',
    'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
    'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe',
    'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword',
    'make_iron_sword',
]

# ── Achievement names in canonical Crafter order ─────────────────────────────
ACHIEVEMENT_NAMES = [
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up',
]
N_ACHIEVEMENTS = len(ACHIEVEMENT_NAMES)  # 22


def ach_key(name: str) -> str:
    """Return the NPZ column key for *name* (encapsulates the data typo)."""
    return f"achivement_{name}"  # one 'e' — matches the actual data files
