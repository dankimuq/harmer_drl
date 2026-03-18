"""
reward_tuning.py

Tests 3 reward function variants to understand why DQN fails and whether
changing the reward structure can fix it.

┌─────────────────────────────────────────────────────────────────┐
│  Variant A  ─  step-focused (force efficiency)                  │
│    R_STEP  = -2.0  (heavy penalty → very short episodes)        │
│    R_SCAN  = +0.5  (scan nearly useless; better to try exploit) │
│    R_PIVOT = +5.0  (small intermediate reward)                  │
│    R_WIN   = +200  │
├─────────────────────────────────────────────────────────────────┤
│  Variant B  ─  balanced (current default)                       │
│    R_STEP  = -0.5  │  R_SCAN = +2  │  R_PIVOT = +10  │ R_WIN=200│
├─────────────────────────────────────────────────────────────────┤
│  Variant C  ─  coverage-focused (reward exploration)            │
│    R_STEP  = -0.1  (very light → long episodes OK)              │
│    R_SCAN  = +2.0                                               │
│    R_PIVOT = +30.0 (big intermediate reward → explore all nodes) │
│    R_WIN   = +200                                               │
│    R_WRONG = -1.0  (softer wrong-exploit penalty)               │
└─────────────────────────────────────────────────────────────────┘

Algorithms trained per variant:  PPO · DQN · A2C
Total:  3 variants × 3 algorithms = 9 models (≈ 2 min total)

Hypothesis: DQN fails in Variant B because Scan (+2) dominates the
greedy policy after partial successes. Variant A makes Scan nearly
worthless, which should force DQN to try exploits directly.

Usage:
    source .venv/bin/activate
    python reward_tuning.py
"""

import os
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor

from pentest_env_complex import PentestEnvComplex

MODELS_DIR = "models_reward_tuning"
os.makedirs(MODELS_DIR, exist_ok=True)
N_EVAL = 30
TS     = 80_000


# ── Reward variants ───────────────────────────────────────────────────────────
class PentestEnvStepFocused(PentestEnvComplex):
    """Variant A: heavy step penalty, tiny scan reward."""
    R_STEP  = -2.0
    R_SCAN  = +0.5
    R_PIVOT = +5.0
    R_WIN   = +200.0
    R_WRONG = -5.0


class PentestEnvCoverageFocused(PentestEnvComplex):
    """Variant C: lightweight step penalty, big pivot reward, softer wrong-exploit."""
    R_STEP  = -0.1
    R_SCAN  = +2.0
    R_PIVOT = +30.0
    R_WIN   = +200.0
    R_WRONG = -1.0    # softer: encourages the agent to try more exploits


VARIANTS = [
    ("A: Step-Focused  (step=-2, scan=+0.5, pivot=+5)",    PentestEnvStepFocused),
    ("B: Balanced      (step=-0.5, scan=+2, pivot=+10)",   PentestEnvComplex),
    ("C: Coverage      (step=-0.1, scan=+2, pivot=+30)",   PentestEnvCoverageFocused),
]

ALGORITHMS = [
    ("PPO", PPO, dict(n_steps=512, learning_rate=3e-4)),
    ("DQN", DQN, dict(learning_starts=2_000, batch_size=64)),
    ("A2C", A2C, {}),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def train_model(algo_cls, algo_kwargs, env_cls, name):
    env   = Monitor(env_cls())
    model = algo_cls("MlpPolicy", env, verbose=0, **algo_kwargs)
    model.learn(total_timesteps=TS)
    path  = f"{MODELS_DIR}/{name}"
    model.save(path)
    return model


def evaluate_model(model, env_cls, n=N_EVAL):
    env = env_cls()
    rewards, successes, steps = [], [], []
    for _ in range(n):
        obs, _ = env.reset()
        total_r, s, ok = 0.0, 0, False
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total_r += r; s += 1
            if term: ok = True; break
            if trunc: break
        rewards.append(total_r)
        successes.append(ok)
        if ok: steps.append(s)
    sr  = np.mean(successes) * 100
    mr  = np.mean(rewards)
    ms  = np.mean(steps) if steps else float("nan")
    return sr, mr, ms


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Reward Function Tuning Experiment")
    print(f"  3 variants × 3 algorithms = 9 models  ({TS:,} steps each)")
    print("=" * 72)

    # Grid: results[variant_label][algo_name] = (sr, mr, ms)
    results = {v[0]: {} for v in VARIANTS}
    models  = {}

    total  = len(VARIANTS) * len(ALGORITHMS)
    done   = 0
    for v_label, env_cls in VARIANTS:
        for a_name, algo_cls, algo_kwargs in ALGORITHMS:
            done += 1
            tag  = f"{a_name}_{v_label[:1]}"  # e.g. PPO_A
            print(f"  [{done}/{total}] Training {a_name} on variant {v_label[:1]} ...",
                  end=" ", flush=True)
            model = train_model(algo_cls, algo_kwargs, env_cls, tag)
            sr, mr, ms = evaluate_model(model, env_cls)
            results[v_label][a_name] = (sr, mr, ms)
            models[tag] = model
            print(f"success={sr:.0f}%  reward={mr:+.1f}  steps={ms:.1f}")

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  Results summary  (Success% | MeanReward | AvgSteps)")
    print("=" * 72)
    # Header
    print(f"  {'Variant':<42}", end="")
    for a_name, _, __ in ALGORITHMS:
        print(f"  {a_name:^20}", end="")
    print()
    print("-" * 72)

    for v_label, _ in VARIANTS:
        print(f"  {v_label:<42}", end="")
        for a_name, _, __ in ALGORITHMS:
            sr, mr, ms = results[v_label][a_name]
            ms_s = f"{ms:.1f}" if not np.isnan(ms) else " N/A"
            print(f"  {sr:>4.0f}% {mr:>+7.1f} {ms_s:>5}  ", end="")
        print()

    print("=" * 72)

    # ── DQN-specific focus ────────────────────────────────────────────────────
    print("\n── DQN Success Rate Across Variants ─────────────────────────────────────────")
    for v_label, _ in VARIANTS:
        sr, mr, ms = results[v_label]["DQN"]
        ms_s = f"{ms:.1f}" if not np.isnan(ms) else "N/A"
        print(f"  {v_label[:1]}  {v_label[3:45]:<42}  "
              f"DQN: {sr:>5.1f}%  reward={mr:>+7.1f}  steps={ms_s}")

    print()
    print("  Interpretation:")
    print("  • If Variant A fixes DQN → scan reward was the culprit (scan dominated)")
    print("    because DQN overvalued the +2 scan signal and looped on it.")
    print("  • If Variant C also fixes DQN → softer wrong-exploit penalty helped,")
    print("    giving DQN more gradient signal to learn the correct exploit order.")
    print("  • If DQN fails in ALL variants → the core issue is architectural:")
    print("    off-policy replay with +reward signals creates loops that DQN cannot")
    print("    break without much more training steps or experience replay tuning.")

    # ── Cross-comparison: best algorithm per variant ──────────────────────────
    print("\n── Best Algorithm Per Variant ───────────────────────────────────────────────")
    for v_label, _ in VARIANTS:
        best_name, best_sr = max(
            ((a, results[v_label][a][0]) for a, _, __ in ALGORITHMS),
            key=lambda x: x[1]
        )
        _, best_mr, best_ms = results[v_label][best_name]
        print(f"  {v_label[:1]}  Winner: {best_name:<5}  "
              f"success={best_sr:.0f}%  reward={best_mr:+.1f}")


if __name__ == "__main__":
    main()
