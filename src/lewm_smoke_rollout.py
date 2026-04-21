"""Short Crafter rollout to sanity-check a trained LeWM (encode + one-step predict).

Runs random actions for ``max_steps``, prints each step's action name and the
teacher-forcing MSE between the predicted next latent and the encoded next frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import crafter  # noqa: E402
from hwm.models import load_lewm  # noqa: E402
from ledata import ACTION_DIM, CRAFTER_ACTIONS  # noqa: E402


def obs_chw(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W,C) uint8 -> (1,C,H,W) float on device."""
    t = torch.from_numpy(obs).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="LeWM smoke — random rollout + one-step latent MSE")
    parser.add_argument("--checkpoint", type=str, required=True, help="LeWM best.pt (model + args)")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_lewm(args.checkpoint, device=device)
    model.eval()

    names = list(CRAFTER_ACTIONS)
    env = crafter.Env()
    obs = env.reset()
    done = False
    step = 0
    print(
        f"LeWM smoke  checkpoint={args.checkpoint}  max_steps={args.max_steps}  "
        f"device={device}"
    )
    print(f"{'step':>4}  {'a':>3}  {'action':<20}  {'pred_mse':>10}")
    print("-" * 56)

    while not done and step < args.max_steps:
        action = int(rng.integers(0, env.action_space.n))
        obs_t = obs_chw(obs, device)
        z = model.encode(obs_t)
        a_oh = F.one_hot(
            torch.tensor([[action]], device=device, dtype=torch.long),
            num_classes=ACTION_DIM,
        ).float()
        z_hat = model.predict(z.unsqueeze(1), a_oh).squeeze(1)

        obs, _reward, done, _info = env.step(action)
        z_next = model.encode(obs_chw(obs, device))
        mse = F.mse_loss(z_hat, z_next).item()

        aname = names[action] if 0 <= action < len(names) else "?"
        print(f"{step:4d}  {action:3d}  {aname:<20}  {mse:10.5f}")
        step += 1

    print("-" * 56)
    print(f"Stopped after {step} steps (done={done})")


if __name__ == "__main__":
    main()
