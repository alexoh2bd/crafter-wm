"""
LeWM rollout collector for Crafter.

Loads a trained LeWM checkpoint, runs the Crafter environment with an
epsilon-scripted policy, encodes each observation with the LeWM encoder,
and saves:
  - crafter_rollouts.pkl   : full trajectory buffer [{t, obs, z_t, action}]
  - goal_library.pkl       : {achievement: [{obs, z_t, timestep, episode}]}
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import crafter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))
from lewm_checkpoint import load_lewm as load_model, obs_to_tensor  # noqa: E402


def collect_rollouts(
    checkpoint_path: str,
    n_steps: int = 50_000,
    epsilon: float = 0.2,
    rollout_save: str = "crafter_rollouts.pkl",
    goal_save: str = "goal_library.pkl",
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(checkpoint_path, device)
    print(f"Loaded LeWM from {checkpoint_path}")

    env = crafter.Env()
    obs = env.reset()
    prev_counts: dict = {}
    episode = 0
    step = 0

    rollout_buffer = []
    goal_library = defaultdict(list)

    print(f"Collecting {n_steps} steps...")

    while step < n_steps:
        # Epsilon-scripted policy (mirrors ledata.py)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.random.choice([1, 2, 3, 4, 5, 5, 5]))

        # Encode current observation
        with torch.no_grad():
            z_t = model.encode(obs_to_tensor(obs, device))  # (1, D)
        z_np = z_t.squeeze(0).cpu().numpy()  # (D,)

        rollout_buffer.append(
            {
                "t": step,
                "episode": episode,
                "obs": obs.astype(np.uint8),
                "z_t": z_np,
                "action": action,
            }
        )

        obs_next, _reward, done, info = env.step(action)

        # Achievement detection
        curr_counts = info.get("achievements", {})
        for ach, count in curr_counts.items():
            if count > prev_counts.get(ach, 0):
                goal_library[ach].append(
                    {
                        "obs": obs.copy().astype(np.uint8),
                        "z_t": z_np.copy(),
                        "timestep": step,
                        "episode": episode,
                    }
                )
                print(f"  ep={episode} t={step}: '{ach}'")
        prev_counts = dict(curr_counts)

        obs = obs_next
        step += 1

        if done:
            obs = env.reset()
            prev_counts = {}
            episode += 1

        if step % 5000 == 0:
            counts = {k: len(v) for k, v in goal_library.items()}
            print(f"step={step}/{n_steps} | ep={episode} | goals={counts}")

    # Save
    with open(rollout_save, "wb") as f:
        pickle.dump(rollout_buffer, f)
    print(f"Saved {len(rollout_buffer)} transitions -> {rollout_save}")

    with open(goal_save, "wb") as f:
        pickle.dump(dict(goal_library), f)
    goal_counts = {k: len(v) for k, v in goal_library.items()}
    print(f"Saved goal library -> {goal_save}: {goal_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LeWM checkpoint (.pt)",
    )
    parser.add_argument("--n_steps", type=int, default=50_000)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout_save", type=str, default="crafter_rollouts.pkl")
    parser.add_argument("--goal_save", type=str, default="goal_library.pkl")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    collect_rollouts(
        checkpoint_path=args.checkpoint,
        n_steps=args.n_steps,
        epsilon=args.epsilon,
        rollout_save=args.rollout_save,
        goal_save=args.goal_save,
        device_str=args.device,
    )
