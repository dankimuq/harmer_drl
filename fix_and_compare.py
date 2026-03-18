"""
fix_and_compare.py

PHASE 5 – Experiment: Fix DQN dead-loop + stable A3C, then compare ALL agents.

Steps
─────
 1) Train DQN  on PentestEnvFixed  → models_complex/dqn_fixed.zip
 2) Train DDQN on PentestEnvFixed  → models_complex/ddqn_fixed.zip
 3) Train A3C (PyTorch async)      → models_complex/a3c_pytorch.pt   (via a3c_pytorch.py)
 4) Evaluate 12 agents on PentestEnvFixed (30 ep each)
 5) Print a summary table and save results to results_fixed.csv

Usage:
    source .venv/bin/activate
    python fix_and_compare.py
"""

import os, pickle, csv, time
import numpy as np
import torch

# ── SB3 ──────────────────────────────────────────────────────────────────────
from stable_baselines3 import DQN, PPO, A2C

# ── Environments ──────────────────────────────────────────────────────────────
from pentest_env_complex import PentestEnvComplex   # original env (for loading old models)
from pentest_env_fixed   import PentestEnvFixed     # patched env  (scan-idempotency fix)

# ── A3C ───────────────────────────────────────────────────────────────────────
from a3c_pytorch import ActorCritic, train as train_a3c, evaluate as evaluate_a3c

ENV_CLS    = PentestEnvFixed
MODELS_DIR = "models_complex"
N_EVAL_EP  = 30
RESULTS_CSV = "results_fixed.csv"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_env(fixed=True):
    return ENV_CLS() if fixed else PentestEnvComplex()


def eval_sb3(model, fixed=True, n=N_EVAL_EP):
    """Evaluate a loaded SB3 model; returns (sr%, mean_reward, std_reward, mean_steps)."""
    env = make_env(fixed)
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_r, step, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r; step += 1; done = term or trunc
        rewards.append(total_r); successes.append(term)
        if term: steps_list.append(step)
    sr  = np.mean(successes) * 100
    return sr, np.mean(rewards), np.std(rewards), (np.mean(steps_list) if steps_list else float("nan"))


def eval_q_table(q_table, fixed=True, n=N_EVAL_EP):
    env = make_env(fixed)
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        state = tuple(obs)
        total_r, step, done = 0.0, 0, False
        while not done:
            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            state = tuple(obs); total_r += r; step += 1; done = term or trunc
        rewards.append(total_r); successes.append(term)
        if term: steps_list.append(step)
    sr = np.mean(successes) * 100
    return sr, np.mean(rewards), np.std(rewards), (np.mean(steps_list) if steps_list else float("nan"))


def eval_deterministic(fixed=True, n=N_EVAL_EP):
    """HARMer deterministic agent: scan all → exploit correct vuln on each reachable node."""
    env = make_env(fixed)
    ev  = PentestEnvComplex.EXPLOITABLE_VULN
    nv  = PentestEnvComplex.ACTIONS_PER_NODE   # 5 (scan + 4 exploits)
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_r, step, done = 0.0, 0, False
        while not done:
            # Find first reachable & unexploited node; scan then exploit
            action = 0  # fallback
            for node in range(PentestEnvComplex.NUM_NODES):
                base = node * nv
                obs_node = obs[node * PentestEnvComplex.STATE_DIM:(node + 1) * PentestEnvComplex.STATE_DIM]
                if obs_node[0] == 0:          # not scanned → scan first
                    action = base; break
                if obs_node[1] == 0:          # scanned but not owned → exploit
                    action = base + 1 + ev[node]; break
            obs, r, term, trunc, _ = env.step(action)
            total_r += r; step += 1; done = term or trunc
        rewards.append(total_r); successes.append(term)
        if term: steps_list.append(step)
    sr = np.mean(successes) * 100
    return sr, np.mean(rewards), np.std(rewards), (np.mean(steps_list) if steps_list else float("nan"))


def eval_random(fixed=True, n=N_EVAL_EP):
    env = make_env(fixed)
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_r, step, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total_r += r; step += 1; done = term or trunc
        rewards.append(total_r); successes.append(term)
        if term: steps_list.append(step)
    sr = np.mean(successes) * 100
    return sr, np.mean(rewards), np.std(rewards), (np.mean(steps_list) if steps_list else float("nan"))


def eval_a3c_pytorch(model_path, fixed=True, n=N_EVAL_EP):
    model = ActorCritic()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    env = make_env(fixed)
    rewards, steps_list, successes = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_r, step, done = 0.0, 0, False
        while not done:
            a, _, _ = model.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total_r += r; step += 1; done = term or trunc
        rewards.append(total_r); successes.append(term)
        if term: steps_list.append(step)
    sr = np.mean(successes) * 100
    return sr, np.mean(rewards), np.std(rewards), (np.mean(steps_list) if steps_list else float("nan"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Training helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_dqn_fixed():
    path = f"{MODELS_DIR}/dqn_fixed.zip"
    print("\n[1/3] Training DQN on PentestEnvFixed ...")
    env = ENV_CLS()
    model = DQN(
        "MlpPolicy", env, verbose=0,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=120_000)
    model.save(path)
    print(f"    Saved → {path}")
    return model


def train_ddqn_fixed():
    path = f"{MODELS_DIR}/ddqn_fixed.zip"
    print("\n[2/3] Training DDQN on PentestEnvFixed ...")
    env = ENV_CLS()
    model = DQN(
        "MlpPolicy", env, verbose=0,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        policy_kwargs=dict(net_arch=[128, 128]),
        # Double-DQN flag (default True in SB3 ≥ 1.7)
        optimize_memory_usage=False,
    )
    model.learn(total_timesteps=120_000)
    model.save(path)
    print(f"    Saved → {path}")
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── STEP 1-2: Train DQN / DDQN on fixed env (skip if already done) ────────
    dqn_fixed_path  = f"{MODELS_DIR}/dqn_fixed.zip"
    ddqn_fixed_path = f"{MODELS_DIR}/ddqn_fixed.zip"
    dqn_fixed_model  = (DQN.load(dqn_fixed_path,  env=ENV_CLS())
                        if os.path.exists(dqn_fixed_path)  else train_dqn_fixed())
    ddqn_fixed_model = (DQN.load(ddqn_fixed_path, env=ENV_CLS())
                        if os.path.exists(ddqn_fixed_path) else train_ddqn_fixed())

    # ── STEP 3: Load / train A3C (PyTorch async) ──────────────────────────────
    a3c_pt = f"{MODELS_DIR}/a3c_pytorch.pt"
    if os.path.exists(a3c_pt):
        print(f"\n[3/3] A3C already trained — loading {a3c_pt}")
        a3c_model = ActorCritic()
        a3c_model.load_state_dict(torch.load(a3c_pt, map_location="cpu"))
    else:
        print("\n[3/3] Training A3C (PyTorch async) on PentestEnvFixed ...")
        a3c_model = train_a3c()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 4: Evaluate all 12 agents
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "═" * 70)
    print("  EVALUATING ALL AGENTS  (n={} episodes each)".format(N_EVAL_EP))
    print("═" * 70)

    results = []

    def row(name, sr, mr, std, ms, note=""):
        return dict(name=name, sr=sr, mr=mr, std=std, ms=ms, note=note)

    # ── Baselines ────────────────────────────────────────────────────────────
    results.append(row("Random",
                        *eval_random(), note="baseline"))
    results.append(row("Deterministic HARMer",
                        *eval_deterministic(), note="original"))

    # ── Q-Learning (original env) ─────────────────────────────────────────────
    q_path = f"{MODELS_DIR}/q_table_complex.pkl"
    with open(q_path, "rb") as f:
        q_table = pickle.load(f)
    results.append(row("Q-Learning (orig env)",
                        *eval_q_table(q_table, fixed=False), note="original env"))
    results.append(row("Q-Learning (fixed env)",
                        *eval_q_table(q_table, fixed=True), note="fixed env"))

    # ── SB3 models (original env) ─────────────────────────────────────────────
    for tag, fname in [("PPO", "ppo_complex"), ("A2C", "a2c_complex"),
                       ("DQN (orig)", "dqn_complex"), ("DDQN (orig)", "ddqn_complex")]:
        path = f"{MODELS_DIR}/{fname}.zip"
        cls  = PPO if "PPO" in tag else (A2C if "A2C" in tag else DQN)
        m    = cls.load(path, env=PentestEnvComplex())
        results.append(row(f"{tag} (orig env)", *eval_sb3(m, fixed=False),
                           note="original env"))

    # ── DQN / DDQN fixed env (just trained) ───────────────────────────────────
    results.append(row("DQN (fixed env)",  *eval_sb3(dqn_fixed_model,  fixed=True), note="scan-idempotency fix"))
    results.append(row("DDQN (fixed env)", *eval_sb3(ddqn_fixed_model, fixed=True), note="scan-idempotency fix"))

    # ── A2C / PPO on fixed env (transfer from orig models) ────────────────────
    ppo_model = PPO.load(f"{MODELS_DIR}/ppo_complex.zip", env=ENV_CLS())
    a2c_model = A2C.load(f"{MODELS_DIR}/a2c_complex.zip", env=ENV_CLS())
    results.append(row("PPO (fixed env)", *eval_sb3(ppo_model, fixed=True), note="zero-shot transfer"))
    results.append(row("A2C (fixed env)", *eval_sb3(a2c_model, fixed=True), note="zero-shot transfer"))

    # ── A3C-like (SB3 surrogate, orig env) ────────────────────────────────────
    a3c_like = A2C.load(f"{MODELS_DIR}/a3c_like_complex.zip", env=PentestEnvComplex())
    results.append(row("A3C-like/SB3 (orig env)", *eval_sb3(a3c_like, fixed=False), note="A2C surrogate"))

    # ── A3C PyTorch (fixed env) ───────────────────────────────────────────────
    a3c_path = f"{MODELS_DIR}/a3c_pytorch.pt"
    results.append(row("A3C (PyTorch async, fixed env)",
                        *eval_a3c_pytorch(a3c_path, fixed=True), note="async, grad accum fix"))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # STEP 5: Print table
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    w1, w2, w3, w4, w5 = 34, 8, 12, 8, 22
    header = (f"{'Agent':<{w1}}  {'SR%':>{w2}}  {'Mean Rew':>{w3}}  "
              f"{'Avg Steps':>{w5}}  {'Note':<{w5}}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in results:
        ms_str = f"{r['ms']:.1f}" if not np.isnan(r['ms']) else "N/A"
        mr_str = f"{r['mr']:+.2f}"
        print(f"{r['name']:<{w1}}  {r['sr']:>{w2}.1f}  "
              f"{mr_str:>{w3}}  "
              f"{ms_str:>{w5}}  {r['note']:<{w5}}")
    print("─" * len(header))

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sr", "mr", "std", "ms", "note"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Results saved → {RESULTS_CSV}")
