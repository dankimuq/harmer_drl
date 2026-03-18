"""
generalize_and_adapt.py

Implements the generalisation ideas discussed in the paper:
  1) environment randomization
  2) topology-aware observation
  3) capability abstraction-backed action semantics
  4) few-shot adaptation on unseen networks

Usage:
    source .venv/bin/activate
    python generalize_and_adapt.py
"""

import csv
import os
import numpy as np

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor

from pentest_env_generalized import RandomizedPentestEnv
from pentest_env_variants import PentestEnvX, PentestEnvY, PentestEnvZ
from topology_wrapper import TopologyAwareObservationWrapper


MODELS_DIR = "models_generalized"
RESULTS_CSV = os.getenv("GENERALIZATION_RESULTS_CSV", "results_generalization_adaptation.csv")
TRAIN_STEPS = int(os.getenv("GENERALIZATION_TRAIN_STEPS", "40000"))
FEW_SHOT_STEPS = int(os.getenv("GENERALIZATION_FEWSHOT_STEPS", "8000"))
N_EVAL = int(os.getenv("GENERALIZATION_EVAL_EPISODES", "20"))

os.makedirs(MODELS_DIR, exist_ok=True)


def make_randomized_env(seed=None):
    return TopologyAwareObservationWrapper(RandomizedPentestEnv(seed=seed))


def make_transfer_env(env_cls):
    class TransferEnv(env_cls):
        def get_topology_features(self):
            topo_one_hot = [1, 0, 0]
            dmz_gate = [0, 0]
            internal_gate = [0, 0]
            dual_core = [0]
            goal_one_hot = [0] * self.NUM_NODES
            goal_one_hot[self.NUM_NODES - 1] = 1
            accessible = [int(self._subnet_accessible(i)) for i in range(self.NUM_SUBNETS)]
            if env_cls.__name__.endswith("Y"):
                topo_one_hot = [0, 1, 0]
                dmz_gate = [0, 1]
                internal_gate = [0, 1]
            elif env_cls.__name__.endswith("Z"):
                topo_one_hot = [0, 0, 1]
                dmz_gate = [1, 0]
                dual_core = [1]
            return np.array(topo_one_hot + dmz_gate + internal_gate + dual_core + goal_one_hot + accessible,
                            dtype=np.int8)

    return TopologyAwareObservationWrapper(TransferEnv())


def eval_model(model, env_factory, n_eval=N_EVAL):
    env = env_factory()
    rewards, successes, steps_list = [], [], []
    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(int(action))
            ep_reward += reward
            steps += 1
            done = term or trunc
        rewards.append(ep_reward)
        successes.append(term)
        if term:
            steps_list.append(steps)
    return {
        "sr": np.mean(successes) * 100,
        "mr": np.mean(rewards),
        "std": np.std(rewards),
        "ms": np.mean(steps_list) if steps_list else float("nan"),
    }


def train_or_load(algo_name, algo_cls, steps=TRAIN_STEPS):
    path = f"{MODELS_DIR}/{algo_name}_generalized"
    zip_path = path + ".zip"
    if os.path.exists(zip_path):
        return algo_cls.load(path, env=Monitor(make_randomized_env()))

    env = Monitor(make_randomized_env())
    kwargs = {}
    if algo_name == "ppo":
        kwargs = dict(n_steps=512, learning_rate=3e-4)
    elif algo_name == "a2c":
        kwargs = dict(learning_rate=3e-4)
    elif algo_name == "dqn":
        kwargs = dict(learning_starts=2000, batch_size=64, learning_rate=1e-3)

    model = algo_cls("MlpPolicy", env, verbose=0, **kwargs)
    model.learn(total_timesteps=steps)
    model.save(path)
    return model


def few_shot_adapt(model, algo_name, algo_cls, target_name, env_factory, steps=FEW_SHOT_STEPS):
    adapted_path = f"{MODELS_DIR}/{algo_name}_fewshot_{target_name.lower()}"
    model = algo_cls.load(f"{MODELS_DIR}/{algo_name}_generalized", env=Monitor(env_factory()))
    model.set_env(Monitor(env_factory()))
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save(adapted_path)
    return model


def print_rows(title, rows):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"{'Agent':<18} {'Phase':<12} {'SR%':>6} {'Mean Reward':>12} {'Avg Steps':>10}")
    print("-" * 80)
    for row in rows:
        ms_str = f"{row['ms']:.1f}" if not np.isnan(row['ms']) else "N/A"
        print(f"{row['agent']:<18} {row['phase']:<12} {row['sr']:>6.1f} {row['mr']:>+12.2f} {ms_str:>10}")


def main():
    algos = [
        ("ppo", PPO),
        ("a2c", A2C),
        ("dqn", DQN),
    ]
    targets = [
        ("X", PentestEnvX),
        ("Y", PentestEnvY),
        ("Z", PentestEnvZ),
    ]

    trained_models = {name: train_or_load(name, cls) for name, cls in algos}

    all_rows = []
    for target_name, target_cls in targets:
        env_factory = lambda cls=target_cls: make_transfer_env(cls)
        rows = []
        for algo_name, algo_cls in algos:
            base_model = trained_models[algo_name]
            zero = eval_model(base_model, env_factory)
            zero_row = dict(network=target_name, agent=algo_name.upper(), phase="zero-shot", **zero)
            rows.append({"agent": algo_name.upper(), "phase": "zero-shot", **zero})
            all_rows.append(zero_row)

            adapted = few_shot_adapt(base_model, algo_name, algo_cls, target_name, env_factory)
            post = eval_model(adapted, env_factory)
            post_row = dict(network=target_name, agent=algo_name.upper(), phase="few-shot", **post)
            rows.append({"agent": algo_name.upper(), "phase": "few-shot", **post})
            all_rows.append(post_row)

        print_rows(f"Generalized Training -> Network {target_name}", rows)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["network", "agent", "phase", "sr", "mr", "std", "ms"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[✓] Results saved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()