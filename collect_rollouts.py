import dreamerv3
import embodied
import numpy as np
import pickle
from pathlib import Path

# Load a trained checkpoint
logdir = Path("~/logdir/crafter_hwm").expanduser()
config = embodied.Config(dreamerv3.Agent.configs["defaults"])
config = config.update(dreamerv3.Agent.configs["crafter"])
config = config.update({"logdir": str(logdir)})

step = embodied.Counter()
env = dreamerv3.wrap_env(
    __import__("crafter").Env(), config.wrapper
)
env = embodied.BatchEnv([env], parallel=False)
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

# Load checkpoint weights
cp = embodied.Checkpoint()
cp.agent = agent
cp.load(logdir / "checkpoint.ckpt", keys=["agent"])

goal_library = {}   # {achievement_name: [(obs, z_t), ...]}
rollout_buffer = [] # all (t, obs, h_t, z_t, achievements) tuples

obs = env.reset()
state = None
done = np.zeros(1, bool)

for t in range(50_000):
    act, state, _ = agent.policy(obs, state, mode="eval")
    obs = env.step(act)

    # Extract RSSM latent — state is (h, z) post-update
    h_t, z_t = state[0], state[1]  # adjust keys per actual state dict

    # Crafter logs achievements as obs keys prefixed with "log_"
    for key, val in obs.items():
        if key.startswith("log_achievement_") and val.any():
            achievement = key.replace("log_achievement_", "")
            frame = obs["image"][0]  # pixel obs
            z_np = np.array(z_t[0])
            h_np = np.array(h_t[0])
            if achievement not in goal_library:
                goal_library[achievement] = []
            goal_library[achievement].append({
                "obs": frame,
                "z_t": z_np,
                "h_t": h_np,
                "timestep": t,
            })
            print(f"  Logged goal: {achievement} at t={t}")

    rollout_buffer.append({
        "t": t,
        "obs": obs["image"][0],
        "z_t": np.array(z_t[0]),
        "h_t": np.array(h_t[0]),
    })

# Save
with open("goal_library.pkl", "wb") as f:
    pickle.dump(goal_library, f)
with open("rollout_buffer.pkl", "wb") as f:
    pickle.dump(rollout_buffer, f)

print(f"Goals collected: { {k: len(v) for k, v in goal_library.items()} }")