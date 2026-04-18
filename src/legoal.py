"""
Goal library construction.
After training, runs the trained encoder over collected trajectories
to build a labeled dict: {achievement_name: [z_t, obs_t, ...]}
Used at inference time for HWM goal specification.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from lemodel import LeWM
from ledata import ACTION_DIM


@torch.no_grad()
def build_goal_library(
    checkpoint_path: str,
    data_path: str,
    save_path: str = 'goal_library.pkl',
    device: str = 'cuda',
    latent_dim: int = 256,
):
    """
    For every timestep where an achievement fires, encode the observation
    and store (obs, z_t) keyed by achievement name.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load model — reconstruct from saved args so arch always matches checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get('args', {})
    model = LeWM(
        img_size=64,
        patch_size=8,
        latent_dim=saved.get('latent_dim', latent_dim),
        action_dim=ACTION_DIM,
        encoder_depth=saved.get('encoder_depth', 12),
        encoder_heads=saved.get('encoder_heads', 3),
        predictor_depth=saved.get('predictor_depth', 6),
        predictor_heads=saved.get('predictor_heads', 16),
        context_len=saved.get('context_len', 16),
        sigreg_M=saved.get('sigreg_M', 1024),
        sigreg_lambda=saved.get('sigreg_lambda', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Load trajectories
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    goal_library = defaultdict(list)

    for ep_idx, traj in enumerate(data['trajectories']):
        obs_list = traj['obs']         # list of (64,64,3) uint8
        achievements = traj['achievements']  # list of [ach_name, ...]

        for t, (obs_np, achs) in enumerate(zip(obs_list, achievements)):
            if not achs:
                continue

            # Encode observation
            obs_t = torch.from_numpy(obs_np).float() / 255.0
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)
            z_t = model.encode(obs_t).squeeze(0).cpu().numpy()       # (D,)

            for ach in achs:
                goal_library[ach].append({
                    'obs': obs_np.copy(),     # (64,64,3) uint8 — for visualization
                    'z':   z_t.copy(),        # (D,) — for planning
                    'episode': ep_idx,
                    'timestep': t,
                })

        if ep_idx % 50 == 0:
            print(f"Processed episode {ep_idx}/{len(data['trajectories'])}")
            print(f"  Library: {dict((k, len(v)) for k,v in goal_library.items())}")

    # Compute mean z per achievement (canonical goal embedding)
    canonical = {}
    for ach, entries in goal_library.items():
        zs = np.stack([e['z'] for e in entries])
        canonical[ach] = {
            'z_mean': zs.mean(axis=0),       # mean embedding
            'z_all': zs,                      # all embeddings
            'obs_examples': [e['obs'] for e in entries[:5]],  # first 5 frames
            'count': len(entries),
        }

    result = {
        'goal_library': dict(goal_library),
        'canonical': canonical,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\nGoal library saved to {save_path}")
    print("Achievement counts:")
    for ach, info in canonical.items():
        print(f"  {ach}: {info['count']} examples")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='crafter_data.pkl')
    parser.add_argument('--save_path', type=str, default='goal_library.pkl')
    parser.add_argument('--latent_dim', type=int, default=256)
    args = parser.parse_args()

    build_goal_library(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        save_path=args.save_path,
        latent_dim=args.latent_dim,
    )