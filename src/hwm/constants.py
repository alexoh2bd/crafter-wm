"""Shared constants for the HWM planning pipeline.

The data typo (`achivement_` with one 'e') lives here exclusively.
All other modules call ach_key() rather than formatting the string directly.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# All paths are relative to the project root (parent of src/).
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent  # plan/

CHECKPOINT    = str(PROJECT_ROOT / "logs" / "lewm_teacher_deep" / "best.pt")
NPZ_DIR       = str(PROJECT_ROOT / "data" / "human_crafter")
DATA_OUT      = str(PROJECT_ROOT / "data")
GOAL_LIBRARY  = str(PROJECT_ROOT / "data" / "goal_library.npz")
TRAJ_DATASET  = str(PROJECT_ROOT / "data" / "trajectory_dataset.npz")
LATENTS_CACHE = str(PROJECT_ROOT / "data" / "latents.npz")
RIDGE_MODEL   = str(PROJECT_ROOT / "data" / "ridge_model.pkl")
HWM_HIGH_CKPT = str(PROJECT_ROOT / "logs" / "hwm_high" / "best.pt")
RESULTS_DIR   = str(PROJECT_ROOT / "results")
RESULTS_JSON  = str(PROJECT_ROOT / "results" / "results.json")

# ── LeWM architecture (must match best.pt saved args) ────────────────────────
LATENT_DIM      = 256
ACTION_DIM      = 17
CONTEXT_LEN     = 16

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
