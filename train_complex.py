"""
train_complex.py

Trains 6 agents on the complex 3-subnet pentest environment:
  1. Q-Learning     (tabular, classic RL)
  2. PPO            (on-policy, policy gradient)
  3. DQN            (off-policy, value-based)
  4. DDQN           (Double DQN – larger network, tuned LR; SB3 DQN already uses double Q-update)
  5. A2C            (on-policy, actor-critic)
  6. DDPG           (off-policy, continuous actor-critic via argmax wrapper)
    7. A3C-like       (A2C with async-like hyperparams; SB3 has no native A3C)

Usage:
    source .venv/bin/activate
    python train_complex.py
"""

import os
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from pentest_env_complex import PentestEnvComplex
from simulation_extensions import train_dppo_like, train_gail_like

MODELS_DIR = "models_complex"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Continuous-action wrapper for DDPG ────────────────────────────────────────
class ContinuousActionWrapper(gym.Wrapper):
    """
    Wraps a Discrete-action env with a Box action space so DDPG can train on it.
    The agent outputs a continuous vector; we select the action with the highest
    value (argmax), effectively treating it as a learned soft selection.
    """
    def __init__(self, env):
        super().__init__(env)
        n = env.action_space.n
        self.action_space = spaces.Box(
            low=np.full(n, -1.0, dtype=np.float32),
            high=np.full(n,  1.0, dtype=np.float32),
        )

    def step(self, action):
        return self.env.step(int(np.argmax(action)))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ── 1. Q-Learning ─────────────────────────────────────────────────────────────
def train_q_learning(episodes=80_000, alpha=0.05, gamma=0.99,
                     epsilon=1.0, epsilon_decay=0.9999):
    env     = PentestEnvComplex()
    q_table = {}
    key     = lambda obs: tuple(obs.tolist())

    print("[1/6] Q-Learning training ...")
    for ep in range(episodes):
        obs, _ = env.reset()
        state  = key(obs)
        q_table.setdefault(state, np.zeros(env.action_space.n))
        done = False

        while not done:
            if np.random.rand() < epsilon:
                act = env.action_space.sample()
            else:
                act = int(np.argmax(q_table[state]))

            nobs, rew, term, trunc, _ = env.step(act)
            done = term or trunc
            nst  = key(nobs)
            q_table.setdefault(nst, np.zeros(env.action_space.n))

            best = np.argmax(q_table[nst])
            td   = rew + gamma * q_table[nst][best]
            q_table[state][act] += alpha * (td - q_table[state][act])
            state = nst

        epsilon = max(0.01, epsilon * epsilon_decay)
        if (ep + 1) % 20_000 == 0:
            print(f"   ep={ep+1:>7} | ε={epsilon:.4f} | states={len(q_table)}")

    path = f"{MODELS_DIR}/q_table_complex.pkl"
    with open(path, "wb") as f:
        pickle.dump(q_table, f)
    print(f"   Saved → {path}  ({len(q_table)} states)\n")
    return q_table


# ── Generic SB3 trainer ───────────────────────────────────────────────────────
def train_sb3(algo_cls, name, idx, save_name, make_env_fn, timesteps, **model_kwargs):
    print(f"[{idx}/6] Training {name} ...")
    env   = Monitor(make_env_fn())
    model = algo_cls("MlpPolicy", env, verbose=0, **model_kwargs)
    model.learn(total_timesteps=timesteps)
    path  = f"{MODELS_DIR}/{save_name}"
    model.save(path)
    print(f"   Saved → {path}.zip\n")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    TS   = 80_000    # timesteps for PPO / DQN / DDQN / A2C
    TS_D = 150_000   # DDPG needs more steps due to continuous-action exploration

    # 1. Q-Learning
    train_q_learning()

    # 2. PPO
    train_sb3(
        PPO, "PPO", 2, "ppo_complex",
        PentestEnvComplex, TS,
        tensorboard_log="./complex_tensorboard/",
        n_steps=512, learning_rate=3e-4,
    )

    # 3. DQN  (SB3's DQN already uses Double Q-learning by default)
    train_sb3(
        DQN, "DQN", 3, "dqn_complex",
        PentestEnvComplex, TS,
        tensorboard_log="./complex_tensorboard/",
        learning_starts=2_000, batch_size=64,
    )

    # 4. DDQN  (explicit Double DQN: larger 128×128 network + tuned LR)
    train_sb3(
        DQN, "DDQN", 4, "ddqn_complex",
        PentestEnvComplex, TS,
        tensorboard_log="./complex_tensorboard/",
        learning_starts=2_000, batch_size=64,
        learning_rate=5e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    # 5. A2C
    train_sb3(
        A2C, "A2C", 5, "a2c_complex",
        PentestEnvComplex, TS,
        tensorboard_log="./complex_tensorboard/",
    )

    # 6. DDPG  (continuous wrapper: argmax(action_vector) → discrete action)
    n_act        = PentestEnvComplex.NUM_NODES * PentestEnvComplex.ACTIONS_PER_NODE
    action_noise = NormalActionNoise(
        mean=np.zeros(n_act),
        sigma=0.3 * np.ones(n_act),
    )
    train_sb3(
        DDPG, "DDPG", 6, "ddpg_complex",
        lambda: ContinuousActionWrapper(PentestEnvComplex()), TS_D,
        tensorboard_log="./complex_tensorboard/",
        action_noise=action_noise,
        learning_starts=5_000,
        batch_size=256,
        learning_rate=1e-3,
    )

    # 7. A3C-like surrogate
    # SB3 does not provide native asynchronous A3C. This configuration uses A2C
    # with stronger entropy bonus and shorter rollout to mimic exploration behavior.
    train_sb3(
        A2C, "A3C-like", 7, "a3c_like_complex",
        PentestEnvComplex, TS,
        tensorboard_log="./complex_tensorboard/",
        n_steps=32,
        ent_coef=0.02,
        learning_rate=7e-4,
    )

    # 8. DPPO-like surrogate
    print("[8/9] Training DPPO-like ...")
    train_dppo_like(total_timesteps=TS, n_envs=8, save_name="dppo_complex")
    print("   Saved → models_complex/dppo_complex.zip\n")

    # 9. GAIL-like imitation
    print("[9/9] Training GAIL-like ...")
    train_gail_like(save_name="gail_complex.pt")
    print("   Saved → models_complex/gail_complex.pt\n")

    print("=" * 55)
    print("All 9 models trained and saved to models_complex/")
    print("Run:  python compare_complex.py")


if __name__ == "__main__":
    main()
