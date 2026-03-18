"""
evaluate_transfer_networks.py

Zero-shot transfer evaluation of trained models on three new networks:
  - Network X: vulnerability remapping
  - Network Y: key-node pivot topology
  - Network Z: dual-control core topology

Usage:
    source .venv/bin/activate
    python evaluate_transfer_networks.py
"""

import csv
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO, DQN, A2C, DDPG

from a3c_pytorch import ActorCritic
from pentest_env_complex import PentestEnvComplex
from pentest_env_variants import PentestEnvX, PentestEnvY, PentestEnvZ


MODELS_DIR = "models_complex"
N_EVAL = 30
OUT_CSV = "results_transfer_xyz.csv"


class ContinuousActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.action_space.n
        self.action_space = spaces.Box(
            low=np.full(n, -1.0, dtype=np.float32),
            high=np.full(n, 1.0, dtype=np.float32),
        )

    def step(self, action):
        return self.env.step(int(np.argmax(action)))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def eval_sb3(model, env_factory, n=N_EVAL):
    env = env_factory()
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(action))
            total_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(total_reward)
        successes.append(term)
        if term:
            steps_list.append(steps)
    return {
        "sr": np.mean(successes) * 100,
        "mr": np.mean(rewards),
        "std": np.std(rewards),
        "ms": np.mean(steps_list) if steps_list else float("nan"),
    }


def eval_ddpg(model, env_factory, n=N_EVAL):
    env = ContinuousActionWrapper(env_factory())
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(total_reward)
        successes.append(term)
        if term:
            steps_list.append(steps)
    return {
        "sr": np.mean(successes) * 100,
        "mr": np.mean(rewards),
        "std": np.std(rewards),
        "ms": np.mean(steps_list) if steps_list else float("nan"),
    }


def eval_q_table(q_table, env_factory, n=N_EVAL):
    env = env_factory()
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        state = tuple(obs.tolist())
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action = int(np.argmax(q_table[state])) if state in q_table else env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            state = tuple(obs.tolist())
            total_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(total_reward)
        successes.append(term)
        if term:
            steps_list.append(steps)
    return {
        "sr": np.mean(successes) * 100,
        "mr": np.mean(rewards),
        "std": np.std(rewards),
        "ms": np.mean(steps_list) if steps_list else float("nan"),
    }


def eval_a3c(model_path, env_factory, n=N_EVAL):
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    env = env_factory()
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(total_reward)
        successes.append(term)
        if term:
            steps_list.append(steps)
    return {
        "sr": np.mean(successes) * 100,
        "mr": np.mean(rewards),
        "std": np.std(rewards),
        "ms": np.mean(steps_list) if steps_list else float("nan"),
    }


def print_network_header(name, desc, env_cls):
    print("\n" + "=" * 78)
    print(f"  {name}  |  {desc}")
    print(f"  Exploitable vulns: {env_cls.EXPLOITABLE_VULN}")
    print("=" * 78)


def print_table(rows):
    width_name = 30
    header = f"{'Agent':<{width_name}}  {'SR%':>6}  {'Mean Reward':>12}  {'Avg Steps':>9}"
    print(header)
    print("-" * len(header))
    for row in rows:
        ms_str = f"{row['ms']:.1f}" if not np.isnan(row['ms']) else "N/A"
        print(f"{row['agent']:<{width_name}}  {row['sr']:>6.1f}  {row['mr']:>+12.2f}  {ms_str:>9}")


def main():
    networks = [
        ("Network X", "Same topology, new vulnerability mapping", PentestEnvX),
        ("Network Y", "Key-node pivot topology", PentestEnvY),
        ("Network Z", "Dual-control core topology", PentestEnvZ),
    ]

    q_table = pickle.load(open(f"{MODELS_DIR}/q_table_complex.pkl", "rb"))

    evaluators = [
        ("Q-Learning", lambda env_cls: eval_q_table(q_table, env_cls)),
        ("PPO", lambda env_cls: eval_sb3(PPO.load(f"{MODELS_DIR}/ppo_complex.zip", env=env_cls()), env_cls)),
        ("A2C", lambda env_cls: eval_sb3(A2C.load(f"{MODELS_DIR}/a2c_complex.zip", env=env_cls()), env_cls)),
        ("DQN (orig)", lambda env_cls: eval_sb3(DQN.load(f"{MODELS_DIR}/dqn_complex.zip", env=env_cls()), env_cls)),
        ("DDQN (orig)", lambda env_cls: eval_sb3(DQN.load(f"{MODELS_DIR}/ddqn_complex.zip", env=env_cls()), env_cls)),
        ("DQN (fixed)", lambda env_cls: eval_sb3(DQN.load(f"{MODELS_DIR}/dqn_fixed.zip", env=env_cls()), env_cls)),
        ("DDQN (fixed)", lambda env_cls: eval_sb3(DQN.load(f"{MODELS_DIR}/ddqn_fixed.zip", env=env_cls()), env_cls)),
        ("DDPG", lambda env_cls: eval_ddpg(DDPG.load(f"{MODELS_DIR}/ddpg_complex.zip", env=ContinuousActionWrapper(env_cls())), env_cls)),
        ("A3C-like/SB3", lambda env_cls: eval_sb3(A2C.load(f"{MODELS_DIR}/a3c_like_complex.zip", env=env_cls()), env_cls)),
        ("A3C (PyTorch async)", lambda env_cls: eval_a3c(f"{MODELS_DIR}/a3c_pytorch.pt", env_cls)),
    ]

    all_rows = []
    for network_name, desc, env_cls in networks:
        print_network_header(network_name, desc, env_cls)
        rows = []
        for agent_name, evaluator in evaluators:
            metrics = evaluator(env_cls)
            row = {
                "network": network_name,
                "agent": agent_name,
                **metrics,
            }
            rows.append(row)
            all_rows.append(row)
        print_table(rows)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["network", "agent", "sr", "mr", "std", "ms"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n[✓] Transfer results saved →", OUT_CSV)


if __name__ == "__main__":
    main()