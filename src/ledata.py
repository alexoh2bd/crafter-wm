import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path
from collections import defaultdict
import crafter

ACTION_DIM = 17
CRAFTER_ACTIONS = [
    'noop', 'move_left', 'move_right', 'move_up', 'move_down',
    'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
    'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe',
    'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword',
    'make_iron_sword',
]

def collect_crafter_data(
    n_episodes: int = 1000,
    save_path: str = 'crafter_data.pkl',
    epsilon: float = 0.2,   # mix of scripted + random
    max_steps_per_episode: int = 10_000,
):
    env = crafter.Env()
    all_trajectories = []
    goal_library = defaultdict(list)

    for ep in range(n_episodes):
        obs = env.reset()  # (64, 64, 3) ndarray — no unpacking

        traj = {'obs': [], 'actions': [], 'achievements': []}
        prev_counts = {}
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            # Epsilon-scripted policy:
            # With prob epsilon take random action,
            # otherwise take a heuristic action biased toward
            # early tech-tree (move + do repeatedly)
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                # Bias toward movement + interaction
                action = np.random.choice([
                    1, 2, 3, 4,   # move (indices into action space)
                    5,            # do (interact/collect)
                    5, 5,         # weight 'do' more heavily
                ])

            obs_next, reward, done, info = env.step(action)

            # Achievement detection — compare cumulative counts
            curr_counts = info.get('achievements', {})
            new_achievements = [
                k for k, v in curr_counts.items()
                if v > prev_counts.get(k, 0)
            ]
            prev_counts = {k: v for k, v in curr_counts.items()}

            traj['obs'].append(obs.astype(np.uint8))
            traj['actions'].append(action)
            traj['achievements'].append(new_achievements)

            for ach in new_achievements:
                goal_library[ach].append({
                    'obs': obs.copy(),
                    'episode': ep,
                    'timestep': len(traj['obs']) - 1,
                })
                print(f"  ep={ep} t={len(traj['obs'])}: '{ach}'")

            obs = obs_next
            step += 1

        all_trajectories.append(traj)
        if ep % 50 == 0:
            counts = {k: len(v) for k, v in goal_library.items()}
            print(f"Episode {ep}/{n_episodes} | library: {counts}")

    data = {
        'trajectories': all_trajectories,
        'goal_library': dict(goal_library),
        'action_dim': ACTION_DIM,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {save_path}")
    return data

class CrafterDataset(Dataset):
    """
    Sequence dataset for LeWM training.
    Each item: (obs_seq, action_seq) of length context_len.
    obs_seq: (T, C, H, W) float32 in [0,1]
    action_seq: (T, ACTION_DIM) one-hot
    """
    def __init__(self, data_path: str, context_len: int = 16,
                 img_size: int = 64):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.context_len = context_len
        self.img_size = img_size
        self.sequences = []

        for traj in data['trajectories']:
            obs = traj['obs']       # list of (64,64,3) uint8
            actions = traj['actions']  # list of int
            T = len(obs)
            if T < context_len + 1:
                continue

            # Slide window over trajectory
            for start in range(0, T - context_len, context_len // 2):
                end = start + context_len
                obs_chunk = np.stack(obs[start:end])       # (T, H, W, C)
                act_chunk = np.array(actions[start:end])   # (T,)
                self.sequences.append((obs_chunk, act_chunk))

        print(f"Dataset: {len(self.sequences)} sequences of length {context_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        obs_chunk, act_chunk = self.sequences[idx]

        # obs: (T, H, W, C) uint8 -> (T, C, H, W) float32 [0,1]
        obs_t = torch.from_numpy(obs_chunk).float() / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2)  # (T, C, H, W)

        # actions: int -> one-hot (T, ACTION_DIM)
        act_t = torch.zeros(len(act_chunk), ACTION_DIM)
        for i, a in enumerate(act_chunk):
            act_t[i, a] = 1.0

        return obs_t, act_t


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=10000)
    parser.add_argument('--max_steps_per_episode', type=int, default=100_000)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--save_path', type=str, default='/data/crafter/crafter_data.pkl')
    args = parser.parse_args()
    collect_crafter_data(
        n_episodes=args.n_episodes,
        save_path=args.save_path,
        epsilon=args.epsilon,
        max_steps_per_episode=args.max_steps_per_episode,
    )