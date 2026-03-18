"""
compare_complex.py

Compares Deterministic HARMer (sequential rule-based) vs 6 RL agents on the
complex network: 3 subnets × 2 nodes = 6 nodes, 4 vulns/node, lateral movement.

Agents evaluated:
  0. Random
  1. Deterministic HARMer (sequential, original approach)
  2. Q-Learning
  3. PPO
  4. DQN
  5. DDQN (Double DQN)
  6. A2C
  7. DDPG (continuous action wrapper)
    8. A3C-like (A2C surrogate; SB3 has no native A3C)

Usage:
    source .venv/bin/activate
    python compare_complex.py
"""

import csv
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, DDPG

from pentest_env_complex import PentestEnvComplex
from simulation_extensions import load_gail_policy

N_EVAL     = 30
MODELS_DIR = "models_complex"
RESULTS_CSV = "results_compare_complex_simulation.csv"


# ── Continuous-action wrapper (identical to train_complex.py) ─────────────────
class ContinuousActionWrapper(gym.Wrapper):
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


# ── Deterministic HARMer baseline ─────────────────────────────────────────────
class DeterministicHARMerComplex:
    """
    Replicates the original HARMer sequential rule-based strategy on the complex
    network.  It knows the subnet topology but has NO knowledge of which exploit
    works on which node.

    Strategy per node (in order 0 → 5):
      1. Scan
      2. Try Exploit-0, Exploit-1, Exploit-2, Exploit-3 in fixed order
      3. Once a node is exploited, advance to the next node
      Only operates on nodes whose subnet is accessible (discovered=1).
    """

    def __init__(self, env: PentestEnvComplex):
        self.env = env
        # Pre-build action plan per node: [Scan, E0, E1, E2, E3]
        self._plans = [
            [n * env.ACTIONS_PER_NODE + i for i in range(env.ACTIONS_PER_NODE)]
            for n in range(env.NUM_NODES)
        ]
        self.reset()

    def reset(self):
        self._ptrs = [0] * self.env.NUM_NODES
        self._curr = 0

    def predict(self, obs) -> int:
        state = obs.reshape(self.env.NUM_NODES, self.env.STATE_DIM)

        # Advance past exploited nodes
        while self._curr < self.env.NUM_NODES and state[self._curr][1] == 1:
            self._curr += 1

        if self._curr >= self.env.NUM_NODES:
            return 0  # shouldn't happen before episode terminates

        node = self._curr

        # If this node's subnet is not yet accessible, find the first one that is
        if state[node][0] == 0:
            for n in range(self.env.NUM_NODES):
                if state[n][0] == 1 and state[n][1] == 0:
                    node = n
                    break
            else:
                return 0  # all accessible nodes already exploited (edge case)

        ptr = self._ptrs[node]
        if ptr >= len(self._plans[node]):
            # Exhausted plan for this node – move on
            self._curr = node + 1
            return self.predict(obs)

        action = self._plans[node][ptr]
        self._ptrs[node] += 1
        return action


# ── Episode runners ────────────────────────────────────────────────────────────

def _result(total_r, steps, success, actions):
    return dict(total_reward=total_r, steps_to_goal=steps if success else None,
                success=success, actions=actions)


def run_random(env):
    obs, _  = env.reset()
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a = env.action_space.sample()
        acts.append(a)
        obs, rew, term, trunc, _ = env.step(a)
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


def run_deterministic(agent, env):
    obs, _ = env.reset()
    agent.reset()
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a = agent.predict(obs)
        acts.append(a)
        obs, rew, term, trunc, _ = env.step(a)
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


def run_rl(model, env):
    obs, _ = env.reset()
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        a    = int(a)
        acts.append(a)
        obs, rew, term, trunc, _ = env.step(a)
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


def run_rl_ddpg(model, env: ContinuousActionWrapper):
    obs, _ = env.reset()
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        acts.append(int(np.argmax(a)))   # record discrete equivalent for trace
        obs, rew, term, trunc, _ = env.step(a)
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


def run_gail(policy, env):
    obs, _ = env.reset()
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a, _ = policy.act(obs, deterministic=True)
        acts.append(a)
        obs, rew, term, trunc, _ = env.step(a)
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


def run_q_learning(q_table, env):
    obs, _ = env.reset()
    state  = tuple(obs.tolist())
    r, s, ok, acts = 0.0, 0, False, []
    while True:
        a = int(np.argmax(q_table[state])) if state in q_table else env.action_space.sample()
        acts.append(a)
        obs, rew, term, trunc, _ = env.step(a)
        state = tuple(obs.tolist())
        r += rew; s += 1
        if term: ok = True; break
        if trunc: break
    return _result(r, s, ok, acts)


# ── Action trace formatter ─────────────────────────────────────────────────────
def _fmt(a):
    n_per = PentestEnvComplex.ACTIONS_PER_NODE
    ev    = PentestEnvComplex.EXPLOITABLE_VULN
    node  = a // n_per
    atype = a % n_per
    if atype == 0:
        return f"Scan[N{node}]"
    v = atype - 1
    mark = "✓" if v == ev[node] else "✗"
    return f"E{v}[N{node}]{mark}"


# ── Summary printer ────────────────────────────────────────────────────────────
def summarise(results, name):
    rews   = [r["total_reward"]  for r in results]
    sukcs  = [r["success"]       for r in results]
    steps  = [r["steps_to_goal"] for r in results if r["steps_to_goal"] is not None]

    sr   = np.mean(sukcs) * 100
    mr   = np.mean(rews)
    std  = np.std(rews)
    ms   = np.mean(steps) if steps else float("nan")

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Episodes       : {len(results)}")
    print(f"  Success rate   : {sr:.1f}%  ({sum(sukcs)}/{len(results)})")
    print(f"  Mean reward    : {mr:+.2f}  ±{std:.2f}")
    if steps:
        print(f"  Avg steps/goal : {ms:.1f}")
    else:
        print(f"  Avg steps/goal : N/A (never reached crown jewel)")

    # Print one successful episode trace (or last episode if no success)
    sample = next((ep for ep in results if ep["success"]), results[-1])
    trace  = " → ".join(_fmt(a) for a in sample["actions"])
    label  = "Successful trace" if sample["success"] else "Best attempt   "
    # Truncate long traces for readability
    if len(trace) > 120:
        trace = trace[:117] + "..."
    print(f"  {label}: {trace}")

    return dict(name=name, sr=sr, mr=mr, std=std, ms=ms)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env = PentestEnvComplex()

    print("=" * 60)
    print("  HARMer Complex Network  –  Comparison Experiment")
    print("  3 Subnets × 2 Nodes = 6 Nodes | 4 Vulns/Node | 30 Actions")
    print(f"  Evaluation episodes per agent: {N_EVAL}")
    print("=" * 60)
    print(f"\n  Network topology:")
    print(f"    Subnet 0 (DMZ)     : Node 0, Node 1   [always accessible]")
    print(f"    Subnet 1 (Internal): Node 2, Node 3   [pivot from Subnet 0]")
    print(f"    Subnet 2 (Core)    : Node 4, Node 5   [pivot from Subnet 1]")
    print(f"  Crown Jewel: Node 5  |  Exploitable vulns: {PentestEnvComplex.EXPLOITABLE_VULN}")

    summaries = []

    # 0. Random
    print("\n[0/10] Random agent ...")
    summaries.append(summarise(
        [run_random(env) for _ in range(N_EVAL)], "Random (baseline)"))

    # 1. Deterministic HARMer
    print("\n[1/10] Deterministic HARMer (sequential) ...")
    det = DeterministicHARMerComplex(env)
    summaries.append(summarise(
        [run_deterministic(det, env) for _ in range(N_EVAL)],
        "Deterministic HARMer (sequential)"))

    # 2. Q-Learning
    print("\n[2/10] Loading Q-Learning ...")
    with open(f"{MODELS_DIR}/q_table_complex.pkl", "rb") as f:
        qt = pickle.load(f)
    summaries.append(summarise(
        [run_q_learning(qt, env) for _ in range(N_EVAL)], "Q-Learning"))

    # 3. PPO
    print("\n[3/10] Loading PPO ...")
    ppo = PPO.load(f"{MODELS_DIR}/ppo_complex")
    summaries.append(summarise(
        [run_rl(ppo, env) for _ in range(N_EVAL)], "PPO"))

    # 4. DQN
    print("\n[4/10] Loading DQN ...")
    dqn = DQN.load(f"{MODELS_DIR}/dqn_complex")
    summaries.append(summarise(
        [run_rl(dqn, env) for _ in range(N_EVAL)], "DQN"))

    # 5. DDQN
    print("\n[5/10] Loading DDQN ...")
    ddqn = DQN.load(f"{MODELS_DIR}/ddqn_complex")
    summaries.append(summarise(
        [run_rl(ddqn, env) for _ in range(N_EVAL)], "DDQN (Double DQN, 128×128)"))

    # 6. A2C
    print("\n[6/10] Loading A2C ...")
    a2c = A2C.load(f"{MODELS_DIR}/a2c_complex")
    summaries.append(summarise(
        [run_rl(a2c, env) for _ in range(N_EVAL)], "A2C"))

    # 7. DDPG
    print("\n[7/10] Loading DDPG ...")
    ddpg_env = ContinuousActionWrapper(PentestEnvComplex())
    ddpg     = DDPG.load(f"{MODELS_DIR}/ddpg_complex")
    summaries.append(summarise(
        [run_rl_ddpg(ddpg, ddpg_env) for _ in range(N_EVAL)],
        "DDPG (continuous argmax wrapper)"))

    # 8. A3C-like surrogate
    print("\n[8/10] Loading A3C-like ...")
    a3c_like = A2C.load(f"{MODELS_DIR}/a3c_like_complex")
    summaries.append(summarise(
        [run_rl(a3c_like, env) for _ in range(N_EVAL)],
        "A3C-like (A2C surrogate)"))

    # 9. DPPO-like surrogate
    print("\n[9/10] Loading DPPO-like ...")
    dppo = PPO.load(f"{MODELS_DIR}/dppo_complex")
    summaries.append(summarise(
        [run_rl(dppo, env) for _ in range(N_EVAL)],
        "DPPO-like (vectorized PPO surrogate)"))

    # 10. GAIL-like imitation
    print("\n[10/10] Loading GAIL-like ...")
    gail = load_gail_policy(f"{MODELS_DIR}/gail_complex.pt")
    summaries.append(summarise(
        [run_gail(gail, env) for _ in range(N_EVAL)],
        "GAIL-like (adversarial imitation)"))

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print(f"  {'Approach':<44} {'Success%':>8} {'MeanRew':>9} {'AvgSteps':>9}")
    print("=" * 75)
    for s in summaries:
        ms_str = f"{s['ms']:.1f}" if not np.isnan(s['ms']) else "  N/A"
        print(f"  {s['name']:<44} {s['sr']:>7.1f}% {s['mr']:>+9.2f} {ms_str:>9}")
    print("=" * 75)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "sr", "mr", "std", "ms"])
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nSaved simulation comparison results -> {RESULTS_CSV}")

    print("\nNotes:")
    print("  • AvgSteps = mean steps taken in successful episodes (lower = more efficient)")
    print("  • DDPG uses continuous argmax mapping; not natively designed for discrete actions")
    print("  • A3C-like uses A2C surrogate because SB3 has no native A3C implementation")
    print("  • DPPO-like uses vectorized PPO surrogate because SB3 has no native DPPO implementation")
    print("  • GAIL-like is simulation-only adversarial imitation from oracle demonstrations")
    print("  • Exploitable vulns per node: ", PentestEnvComplex.EXPLOITABLE_VULN)
    print("  • Deterministic HARMer scans every node and tries exploits in fixed order (0→3)")


if __name__ == "__main__":
    main()
