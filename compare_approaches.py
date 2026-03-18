"""
compare_approaches.py

Compares the original HARMer deterministic (sequential) approach against
trained RL agents (PPO, DQN, A2C, Q-Learning) on the PentestEnv.

Metrics collected per episode:
  - total_reward   : sum of all rewards
  - steps_to_goal  : number of steps taken (None if goal not reached)
  - success        : whether the exploit succeeded within max_steps
  - actions_taken  : sequence of action indices

Run:
    source .venv/bin/activate
    python compare_approaches.py
"""

import pickle
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from pentest_env import PentestEnv

N_EVAL_EPISODES = 30   # episodes per approach

# ──────────────────────────────────────────────────────────────
# 1.  DETERMINISTIC BASELINE  (mimics original HARMer behaviour)
#     Fixed order: Scan first, then try each exploit in order 1→2→3
# ──────────────────────────────────────────────────────────────

class DeterministicHARMer:
    """
    Replicates the original HARMer sequential, rule-based attacker.
    Strategy:
      Step 1 – always Scan (action 0) to enumerate vulnerabilities.
      Step 2 – try exploits in fixed order: 1, 2, 3.
    Once a step is done it is not repeated (no re-scan, no retry).
    """
    def __init__(self, env: PentestEnv):
        self.env = env
        self._plan = [0, 1, 2, 3]   # Scan then Exploit-1, Exploit-2, Exploit-3
        self._idx  = 0

    def reset(self):
        self._idx = 0

    def predict(self, obs):
        if self._idx < len(self._plan):
            action = self._plan[self._idx]
            self._idx += 1
        else:
            # Plan exhausted – repeat last exploit attempt (stuck behaviour)
            action = self._plan[-1]
        return action


def run_episode(agent, env, is_deterministic_harmer=False):
    obs, _ = env.reset()
    if is_deterministic_harmer:
        agent.reset()

    total_reward = 0.0
    steps        = 0
    success      = False
    actions      = []

    while True:
        if is_deterministic_harmer:
            action = agent.predict(obs)
        else:
            action, _ = agent.predict(obs, deterministic=True)
            action = int(action)

        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps        += 1

        if terminated:
            success = True
            break
        if truncated:
            break

    steps_to_goal = steps if success else None
    return {
        "total_reward" : total_reward,
        "steps_to_goal": steps_to_goal,
        "success"      : success,
        "actions"      : actions,
    }


def run_q_learning_episode(q_table, env):
    obs, _ = env.reset()
    state  = tuple(obs.tolist())
    total_reward = 0.0
    steps        = 0
    success      = False
    actions      = []

    while True:
        if state in q_table:
            action = int(np.argmax(q_table[state]))
        else:
            action = env.action_space.sample()

        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        state = tuple(obs.tolist())
        total_reward += reward
        steps        += 1

        if terminated:
            success = True
            break
        if truncated:
            break

    return {
        "total_reward" : total_reward,
        "steps_to_goal": steps if success else None,
        "success"      : success,
        "actions"      : actions,
    }


def summarise(results: list, name: str):
    rewards     = [r["total_reward"]  for r in results]
    successes   = [r["success"]       for r in results]
    steps_list  = [r["steps_to_goal"] for r in results if r["steps_to_goal"] is not None]

    success_rate = np.mean(successes) * 100
    mean_reward  = np.mean(rewards)
    std_reward   = np.std(rewards)
    mean_steps   = np.mean(steps_list) if steps_list else float("nan")

    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    print(f"  Episodes       : {len(results)}")
    print(f"  Success rate   : {success_rate:.1f}%  ({sum(successes)}/{len(results)})")
    print(f"  Mean reward    : {mean_reward:+.2f}  ±{std_reward:.2f}")
    print(f"  Avg steps/goal : {mean_steps:.1f}" if steps_list else "  Avg steps/goal : N/A (no success)")

    # Show a representative action trace (first successful episode, or last episode)
    for ep in results:
        if ep["success"]:
            action_names = {0: "Scan", 1: "Exploit-1(FTP)", 2: "Exploit-2(distcc)✓", 3: "Exploit-3(Samba)"}
            trace = " → ".join(action_names.get(a, f"A{a}") for a in ep["actions"])
            print(f"  Example trace  : {trace}")
            break

    return {
        "name"        : name,
        "success_rate": success_rate,
        "mean_reward" : mean_reward,
        "std_reward"  : std_reward,
        "mean_steps"  : mean_steps,
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 52)
    print("  HARMer  –  Deterministic vs RL Comparison")
    print(f"  Evaluation episodes per approach: {N_EVAL_EPISODES}")
    print("=" * 52)

    env = PentestEnv()
    all_summaries = []

    # ── 1. Deterministic HARMer baseline ───────────────────────
    print("\n[1/5] Running Deterministic HARMer baseline...")
    det_agent   = DeterministicHARMer(env)
    det_results = [run_episode(det_agent, env, is_deterministic_harmer=True)
                   for _ in range(N_EVAL_EPISODES)]
    all_summaries.append(summarise(det_results, "Deterministic HARMer (baseline)"))

    # ── 2. Q-Learning ──────────────────────────────────────────
    print("\n[2/5] Loading Q-Learning agent...")
    with open("models/q_table_agent.pkl", "rb") as f:
        q_table = pickle.load(f)
    ql_results = [run_q_learning_episode(q_table, env)
                  for _ in range(N_EVAL_EPISODES)]
    all_summaries.append(summarise(ql_results, "Q-Learning"))

    # ── 3. PPO ─────────────────────────────────────────────────
    print("\n[3/5] Loading PPO agent...")
    ppo_model   = PPO.load("models/ppo_pentest_agent")
    ppo_results = [run_episode(ppo_model, env) for _ in range(N_EVAL_EPISODES)]
    all_summaries.append(summarise(ppo_results, "PPO (Proximal Policy Optimization)"))

    # ── 4. DQN ─────────────────────────────────────────────────
    print("\n[4/5] Loading DQN agent...")
    dqn_model   = DQN.load("models/dqn_pentest_agent")
    dqn_results = [run_episode(dqn_model, env) for _ in range(N_EVAL_EPISODES)]
    all_summaries.append(summarise(dqn_results, "DQN (Deep Q-Network)"))

    # ── 5. A2C ─────────────────────────────────────────────────
    print("\n[5/5] Loading A2C agent...")
    a2c_model   = A2C.load("models/a2c_pentest_agent")
    a2c_results = [run_episode(a2c_model, env) for _ in range(N_EVAL_EPISODES)]
    all_summaries.append(summarise(a2c_results, "A2C (Advantage Actor-Critic)"))

    # ── Final summary table ─────────────────────────────────────
    print("\n\n" + "=" * 65)
    print(f"  {'Approach':<38} {'Success%':>8} {'MeanRew':>9} {'AvgSteps':>9}")
    print("=" * 65)
    for s in all_summaries:
        steps_str = f"{s['mean_steps']:.1f}" if not np.isnan(s['mean_steps']) else "  N/A "
        print(f"  {s['name']:<38} {s['success_rate']:>7.1f}% {s['mean_reward']:>+9.2f} {steps_str:>9}")
    print("=" * 65)
    print("\nNote: lower avg_steps = faster exploit path found.")


if __name__ == "__main__":
    main()
