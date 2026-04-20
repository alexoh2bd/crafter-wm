# ppo_teacher.py
import torch
import torch.nn as nn
import numpy as np
import pickle
import crafter
from collections import defaultdict
from pathlib import Path
from torch.distributions import Categorical

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

ACTION_DIM = 17

class ActorCritic(nn.Module):
    """
    Small CNN + MLP policy for Crafter.
    Shares a CNN backbone between actor and critic.
    Deliberately compact — we want structured exploration,
    not peak performance.
    """
    def __init__(self, action_dim=ACTION_DIM):
        super().__init__()
        # CNN backbone: 64x64x3 -> flat features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),   # -> 15x15
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),  # -> 6x6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),  # -> 4x4
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = 64 * 4 * 4  # 1024

        self.shared = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        # obs: (B, 3, 64, 64) float [0,1]
        x = self.cnn(obs)
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def preprocess(obs):
    # obs: (H, W, C) uint8 -> (1, C, H, W) float [0,1]
    t = torch.from_numpy(obs).float() / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def train_ppo(
    total_steps: int = 500_000,
    n_envs: int = 8,
    rollout_len: int = 128,
    epochs: int = 4,
    minibatches: int = 4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 2.5e-4,
    ent_coef: float = 0.01,
    save_path: str = 'ppo_teacher.pt',
    device: str = 'cuda',
    wandb_project: str = 'lewm-crafter',
    wandb_run_name: str = 'ppo-teacher',
    use_wandb: bool = True,
):
    use_wandb = use_wandb and _WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=dict(
                total_steps=total_steps, n_envs=n_envs,
                rollout_len=rollout_len, epochs=epochs,
                minibatches=minibatches, gamma=gamma,
                gae_lambda=gae_lambda, clip_eps=clip_eps,
                lr=lr, ent_coef=ent_coef,
            ),
        )

    device = torch.device(device)
    policy = ActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    envs = [crafter.Env() for _ in range(n_envs)]
    obs_list = [env.reset() for env in envs]

    global_step = 0
    update = 0

    # Per-episode return tracking across all envs
    ep_returns   = np.zeros(n_envs)
    ep_lengths   = np.zeros(n_envs, dtype=int)
    finished_returns: list[float] = []
    finished_lengths: list[int]  = []

    # Storage buffers
    obs_buf    = torch.zeros(rollout_len, n_envs, 3, 64, 64)
    act_buf    = torch.zeros(rollout_len, n_envs, dtype=torch.long)
    logp_buf   = torch.zeros(rollout_len, n_envs)
    rew_buf    = torch.zeros(rollout_len, n_envs)
    done_buf   = torch.zeros(rollout_len, n_envs)
    val_buf    = torch.zeros(rollout_len, n_envs)

    achievement_counts: dict[str, int] = {}

    while global_step < total_steps:
        policy.eval()

        # ── Collect rollout ───────────────────────────────────────
        for step in range(rollout_len):
            obs_tensor = torch.stack([
                preprocess(o).squeeze(0) for o in obs_list
            ]).to(device)

            with torch.no_grad():
                actions, logps, _, values = policy.get_action(obs_tensor)

            obs_buf[step]  = obs_tensor.cpu()
            act_buf[step]  = actions.cpu()
            logp_buf[step] = logps.cpu()
            val_buf[step]  = values.squeeze(-1).cpu()

            for i, (env, action) in enumerate(zip(envs, actions.cpu().numpy())):
                next_obs, reward, done, info = env.step(action)
                rew_buf[step, i]  = reward
                done_buf[step, i] = done
                ep_returns[i] += reward
                ep_lengths[i]  += 1

                if done:
                    finished_returns.append(float(ep_returns[i]))
                    finished_lengths.append(int(ep_lengths[i]))
                    ep_returns[i] = 0
                    ep_lengths[i] = 0
                    obs_list[i] = env.reset()
                else:
                    obs_list[i] = next_obs

                for k, v in info.get('achievements', {}).items():
                    if v > 0:
                        achievement_counts[k] = achievement_counts.get(k, 0) + 1

            global_step += n_envs

        # ── GAE returns ───────────────────────────────────────────
        with torch.no_grad():
            last_obs = torch.stack([
                preprocess(o).squeeze(0) for o in obs_list
            ]).to(device)
            _, last_val = policy(last_obs)
            last_val = last_val.squeeze(-1).cpu()

        advantages = torch.zeros_like(rew_buf)
        last_gae = 0
        for t in reversed(range(rollout_len)):
            next_val  = last_val if t == rollout_len - 1 else val_buf[t + 1]
            next_done = done_buf[t]
            delta     = rew_buf[t] + gamma * next_val * (1 - next_done) - val_buf[t]
            last_gae  = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # ── PPO update ────────────────────────────────────────────
        policy.train()
        b_obs   = obs_buf.reshape(-1, 3, 64, 64).to(device)
        b_acts  = act_buf.reshape(-1).to(device)
        b_logps = logp_buf.reshape(-1).to(device)
        b_advs  = advantages.reshape(-1).to(device)
        b_rets  = returns.reshape(-1).to(device)
        b_advs  = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        batch_size = rollout_len * n_envs
        mb_size    = batch_size // minibatches

        pg_losses, v_losses, entropies, approx_kls = [], [], [], []
        for _ in range(epochs):
            idx = torch.randperm(batch_size)
            for start in range(0, batch_size, mb_size):
                mb_idx = idx[start:start + mb_size]
                logits, values = policy(b_obs[mb_idx])
                dist       = Categorical(logits=logits)
                new_logps  = dist.log_prob(b_acts[mb_idx])
                entropy    = dist.entropy().mean()

                ratio = (new_logps - b_logps[mb_idx]).exp()
                adv   = b_advs[mb_idx]

                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - clip_eps, 1 + clip_eps),
                ).mean()

                v_loss = 0.5 * (values.squeeze(-1) - b_rets[mb_idx]).pow(2).mean()
                loss   = pg_loss + 0.5 * v_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())
                with torch.no_grad():
                    approx_kls.append(
                        ((ratio - 1) - ratio.log()).mean().item()
                    )

        update += 1

        # ── Logging ───────────────────────────────────────────────
        mean_return = np.mean(finished_returns[-100:]) if finished_returns else 0.0
        mean_len    = np.mean(finished_lengths[-100:])  if finished_lengths else 0.0
        log_dict = {
            'ppo/policy_loss':      float(np.mean(pg_losses)),
            'ppo/value_loss':       float(np.mean(v_losses)),
            'ppo/entropy':          float(np.mean(entropies)),
            'ppo/approx_kl':        float(np.mean(approx_kls)),
            'ppo/mean_return':      mean_return,
            'ppo/mean_ep_len':      mean_len,
            'ppo/mean_reward':      rew_buf.mean().item(),
            'ppo/global_step':      global_step,
            'ppo/update':           update,
            **{f'achievements/{k}': v for k, v in achievement_counts.items()},
        }

        print(f"step={global_step:>8d} | update={update:>4d} | "
              f"return={mean_return:>7.2f} | "
              f"pg={log_dict['ppo/policy_loss']:>7.4f} | "
              f"vf={log_dict['ppo/value_loss']:>7.4f} | "
              f"ent={log_dict['ppo/entropy']:>6.4f} | "
              f"kl={log_dict['ppo/approx_kl']:>7.5f} | "
              f"ach={dict(sorted(achievement_counts.items()))}")

        if use_wandb:
            wandb.log(log_dict, step=global_step)

    torch.save({'policy': policy.state_dict()}, save_path)
    print(f"PPO teacher saved to {save_path}")

    if use_wandb:
        wandb.finish()

    return policy


def collect_rollouts_with_teacher(
    checkpoint_path: str,
    n_episodes: int = 5000,
    save_path: str = 'crafter_teacher_data.pkl',
    max_steps_per_episode: int = 10_000,
    epsilon: float = 0.05,      # small random noise keeps diversity; pure greedy
                                 # collapses to repetitive near-optimal trajectories
    deterministic: bool = False, # False = stochastic policy (more diverse data)
    device: str = 'cuda',
):
    """
    Roll out the trained PPO teacher and save trajectories in the same
    format as ledata.collect_crafter_data() so CrafterDataset can consume
    them without modification.

    Why teacher rollouts beat random/heuristic:
      - The PPO policy has learned which action sequences unlock achievements.
      - Achievement-bearing timesteps are orders of magnitude denser than in
        random data, giving LeWM rich supervision signal for rare events.
      - epsilon > 0 injects just enough noise to prevent mode collapse and
        keep the trajectory distribution diverse.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    policy = ActorCritic().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()
    print(f"Loaded PPO teacher from {checkpoint_path}")

    env = crafter.Env()
    all_trajectories = []
    goal_library = defaultdict(list)
    total_achievements = defaultdict(int)

    for ep in range(n_episodes):
        obs = env.reset()
        traj = {'obs': [], 'actions': [], 'achievements': []}
        prev_counts = {}
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space.n)
            else:
                obs_t = preprocess(obs).to(device)
                with torch.no_grad():
                    act, _, _, _ = policy.get_action(obs_t, deterministic=deterministic)
                action = act.item()

            obs_next, _reward, done, info = env.step(action)

            curr_counts = info.get('achievements', {})
            new_achievements = [
                k for k, v in curr_counts.items()
                if v > prev_counts.get(k, 0)
            ]
            prev_counts = dict(curr_counts)

            traj['obs'].append(obs.astype(np.uint8))
            traj['actions'].append(action)
            traj['achievements'].append(new_achievements)

            for ach in new_achievements:
                total_achievements[ach] += 1
                goal_library[ach].append({
                    'obs': obs.copy(),
                    'episode': ep,
                    'timestep': step,
                })
                print(f"  ep={ep} t={step}: '{ach}'")

            obs = obs_next
            step += 1

        all_trajectories.append(traj)
        if ep % 100 == 0:
            print(f"Episode {ep}/{n_episodes} | "
                  f"steps={step} | achievements: {dict(total_achievements)}")

    data = {
        'trajectories': all_trajectories,
        'goal_library': dict(goal_library),
        'action_dim': ACTION_DIM,
        'collection_method': 'teacher_ppo',
        'teacher_checkpoint': str(checkpoint_path),
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nTeacher rollouts saved to {save_path}")
    print(f"Total episodes: {n_episodes} | Total achievements: {dict(total_achievements)}")
    return data


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_teacher.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'collect', 'all'],
                        default='all',
                        help='"train" = PPO only | "collect" = rollouts only | '
                             '"all" = train then collect')
    # Allow individual overrides
    parser.add_argument('--teacher_save_path',    type=str)
    parser.add_argument('--teacher_checkpoint',   type=str)
    parser.add_argument('--rollout_save_path',    type=str)
    parser.add_argument('--n_episodes',           type=int)
    parser.add_argument('--total_steps',          type=int)
    parser.add_argument('--device',               type=str, default='cuda')
    parser.add_argument('--wandb_project',        type=str)
    parser.add_argument('--wandb_run_name',       type=str)
    parser.add_argument('--no_wandb',             action='store_true',
                        help='Disable wandb logging')
    args = parser.parse_args()

    # Load YAML
    cfg = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        for section in raw.values():
            if isinstance(section, dict):
                cfg.update(section)

    # CLI overrides
    ppo_save   = args.teacher_save_path  or cfg.get(
        'teacher_save_path',  'data/crafter/ppo/ppo_teacher.pt')
    ckpt_path  = args.teacher_checkpoint or cfg.get('teacher_checkpoint', ppo_save)
    rollout_out = args.rollout_save_path or cfg.get(
        'rollout_save_path', 'data/crafter/ppo_rollouts/crafter_teacher_data.pkl')
    n_ep       = args.n_episodes  or cfg.get('n_episodes',  5000)
    t_steps    = args.total_steps or cfg.get('ppo_total_steps', 500_000)

    if args.mode in ('train', 'all'):
        train_ppo(
            total_steps        = t_steps,
            n_envs             = cfg.get('n_envs',       8),
            rollout_len        = cfg.get('rollout_len',  128),
            epochs             = cfg.get('ppo_epochs',   4),
            minibatches        = cfg.get('minibatches',  4),
            gamma              = cfg.get('gamma',        0.99),
            gae_lambda         = cfg.get('gae_lambda',   0.95),
            clip_eps           = cfg.get('clip_eps',     0.2),
            lr                 = cfg.get('ppo_lr',       2.5e-4),
            ent_coef           = cfg.get('ent_coef',     0.01),
            save_path          = ppo_save,
            device             = args.device,
            wandb_project      = args.wandb_project or cfg.get('wandb_project', 'lewm-crafter'),
            wandb_run_name     = args.wandb_run_name or cfg.get('wandb_run_name', 'ppo-teacher'),
            use_wandb          = not args.no_wandb and cfg.get('use_wandb', True),
        )

    if args.mode in ('collect', 'all'):
        collect_rollouts_with_teacher(
            checkpoint_path        = ckpt_path,
            n_episodes             = n_ep,
            save_path              = rollout_out,
            max_steps_per_episode  = cfg.get('max_steps_per_episode', 10_000),
            epsilon                = cfg.get('collection_epsilon',     0.05),
            deterministic          = cfg.get('deterministic',          False),
            device                 = args.device,
        )