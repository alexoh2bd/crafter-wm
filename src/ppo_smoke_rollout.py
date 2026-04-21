"""Run the PPO teacher for one short episode and print each action (sanity check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import crafter  # noqa: E402
from ledata import CRAFTER_ACTIONS  # noqa: E402
from teacherPPO import ActorCritic, preprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO teacher — one episode, print actions")
    parser.add_argument("--checkpoint", type=str, required=True, help="ppo_teacher.pt (policy weights)")
    parser.add_argument("--max-steps", type=int, default=50, help="Stop after this many env steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use policy mean action")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Random action prob (0 = pure policy)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = ActorCritic().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    action_names = list(CRAFTER_ACTIONS)

    env = crafter.Env()
    obs = env.reset()
    done = False
    step = 0

    print(f"PPO smoke rollout  checkpoint={args.checkpoint}  max_steps={args.max_steps}  device={device}")
    print(f"{'step':>4}  {'a':>3}  {'action':<20}  note")
    print("-" * 52)

    while not done and step < args.max_steps:
        if args.epsilon > 0.0 and np.random.rand() < args.epsilon:
            action = int(np.random.randint(env.action_space.n))
            note = "(random ε)"
        else:
            obs_t = preprocess(obs).to(device)
            with torch.no_grad():
                act_t, _, _, _ = policy.get_action(obs_t, deterministic=args.deterministic)
            action = int(act_t.item())
            note = ""

        name = action_names[action] if 0 <= action < len(action_names) else "?"
        print(f"{step:4d}  {action:3d}  {name:<20}  {note}")

        obs, _reward, done, info = env.step(action)
        step += 1

    print("-" * 52)
    print(f"Stopped after {step} steps (done={done})")


if __name__ == "__main__":
    main()
