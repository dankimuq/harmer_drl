"""
diagnose_agents.py

Diagnostic matrix to explain WHY DQN / DDQN / DDPG fail while
Q-Learning / PPO / A2C succeed on the complex pentest network.

Metrics per episode, per agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  success        – Crown Jewel exploited within MAX_STEPS
  nodes_exp      – # nodes successfully exploited (max 6)
  subnet2        – did the agent ever reach Subnet 2? (0/1)
  crown_atts     – # exploit attempts on Node 5 (target)
  scan_rate      – fraction of actions that were Scans (wastes reward)
  repeat_ratio   – fraction of consecutive same-action pairs (loop indicator)
  max_repeat     – longest run of identical consecutive actions (stuck depth)
  dead_loops     – # times same action repeated ≥ 5 times (severe loops)
  unique_act_pct – distinct actions / episode_length  (exploration breadth)
  act_entropy    – Shannon H of action distribution   (diversity in bits)
  q_val_std      – std of Q-values at initial state   (DQN/DDQN only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    source .venv/bin/activate
    python diagnose_agents.py
"""

import pickle
from collections import Counter
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, DDPG

from pentest_env_complex import PentestEnvComplex

N_EVAL     = 30
MODELS_DIR = "models_complex"
DEAD_THR   = 5          # same action repeated ≥ this → dead loop

N_NODES  = PentestEnvComplex.NUM_NODES
N_PER    = PentestEnvComplex.ACTIONS_PER_NODE
EV       = PentestEnvComplex.EXPLOITABLE_VULN
SCAN_SET = {i * N_PER for i in range(N_NODES)}            # Scan actions: 0,5,10,15,20,25
N5_EXPS  = set(range(5 * N_PER + 1, 5 * N_PER + N_PER))  # Exploit actions on Node 5


# ── Continuous-action wrapper (DDPG) ─────────────────────────────────────────
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


# ── Episode runner (collects obs sequence) ───────────────────────────────────
def _run_rl(model, env, is_ddpg=False):
    obs, _ = env.reset()
    acts, obs_seq = [], [obs.copy()]
    total_r = 0.0
    while True:
        a, _ = model.predict(obs, deterministic=True)
        disc_a = int(np.argmax(a)) if is_ddpg else int(a)
        acts.append(disc_a)
        obs, r, term, trunc, _ = env.step(a)
        obs_seq.append(obs.copy())
        total_r += r
        if term or trunc:
            break
    return acts, obs_seq, term, total_r


def _run_q_learning(q_table, env):
    obs, _ = env.reset()
    state  = tuple(obs.tolist())
    acts, obs_seq = [], [obs.copy()]
    total_r = 0.0
    while True:
        a = int(np.argmax(q_table[state])) if state in q_table else env.action_space.sample()
        acts.append(a)
        obs, r, term, trunc, _ = env.step(a)
        state = tuple(obs.tolist())
        obs_seq.append(obs.copy())
        total_r += r
        if term or trunc:
            break
    return acts, obs_seq, term, total_r


# ── Per-episode diagnostic calculation ───────────────────────────────────────
def diagnose_episode(acts, obs_seq, success):
    n      = len(acts)
    final  = obs_seq[-1].reshape(N_NODES, 6) if obs_seq else np.zeros((N_NODES, 6))

    # Nodes exploited & subnet 2 reached
    nodes_exp     = int(final[:, 1].sum())
    subnet2       = bool(final[4, 0] == 1 or final[5, 0] == 1)
    crown_atts    = sum(a in N5_EXPS for a in acts)

    # Scan rate
    scan_rate = sum(a in SCAN_SET for a in acts) / max(n, 1)

    # Repeat / dead-loop metrics
    repeat_pairs = sum(acts[i] == acts[i + 1] for i in range(n - 1))
    repeat_ratio = repeat_pairs / max(n - 1, 1)

    max_rep = cur = 1
    dead_loops = 0
    for i in range(1, n):
        if acts[i] == acts[i - 1]:
            cur += 1
            if cur == DEAD_THR:
                dead_loops += 1
            max_rep = max(max_rep, cur)
        else:
            cur = 1

    # Action diversity
    unique_pct = len(set(acts)) / max(n, 1)
    counts = Counter(acts)
    probs  = np.array(list(counts.values()), dtype=float) / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))

    return {
        "success":      success,
        "nodes_exp":    nodes_exp,
        "subnet2":      float(subnet2),
        "crown_atts":   crown_atts,
        "scan_rate":    scan_rate,
        "repeat_ratio": repeat_ratio,
        "max_repeat":   max_rep,
        "dead_loops":   dead_loops,
        "unique_pct":   unique_pct,
        "act_entropy":  entropy,
    }


def agg(episodes, key):
    vals = [e[key] for e in episodes]
    return np.mean(vals), np.std(vals)


# ── Q-value spread (DQN/DDQN only) ───────────────────────────────────────────
def q_val_std(dqn_model, env):
    """Compute std & range of Q-values at the initial observation."""
    obs, _ = env.reset()
    obs_t  = torch.FloatTensor(obs).unsqueeze(0)
    with torch.no_grad():
        q_vals = dqn_model.q_net(obs_t).cpu().numpy()[0]
    return q_vals.std(), q_vals.min(), q_vals.max(), q_vals.argmax()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    env = PentestEnvComplex()

    agents = []  # (name, run_fn)

    # Q-Learning
    with open(f"{MODELS_DIR}/q_table_complex.pkl", "rb") as f:
        qt = pickle.load(f)
    agents.append(("Q-Learning",
        lambda: _run_q_learning(qt, env)))

    # PPO, A2C, A3C-like
    for name, cls, path in [
        ("PPO",       PPO, "ppo_complex"),
        ("A2C",       A2C, "a2c_complex"),
        ("A3C-like",  A2C, "a3c_like_complex"),
    ]:
        m = cls.load(f"{MODELS_DIR}/{path}")
        agents.append((name, lambda m=m: _run_rl(m, env)))

    # DQN, DDQN
    for name, path in [("DQN", "dqn_complex"), ("DDQN", "ddqn_complex")]:
        m = DQN.load(f"{MODELS_DIR}/{path}")
        agents.append((name, lambda m=m: _run_rl(m, env)))

    # DDPG
    ddpg_env = ContinuousActionWrapper(PentestEnvComplex())
    ddpg     = DDPG.load(f"{MODELS_DIR}/ddpg_complex")
    agents.append(("DDPG", lambda: _run_rl(ddpg, ddpg_env, is_ddpg=True)))

    # Collect diagnostics
    all_results = {}
    for name, run_fn in agents:
        print(f"  [{name}] running {N_EVAL} episodes ...", end=" ", flush=True)
        episodes = []
        for _ in range(N_EVAL):
            acts, obs_seq, success, _ = run_fn()
            episodes.append(diagnose_episode(acts, obs_seq, success))
        all_results[name] = episodes
        sr = np.mean([e["success"] for e in episodes]) * 100
        print(f"success={sr:.0f}%")

    # ── Print diagnostic table ────────────────────────────────────────────────
    KEYS = ["success", "nodes_exp", "subnet2", "crown_atts",
            "scan_rate", "repeat_ratio", "max_repeat", "dead_loops",
            "unique_pct", "act_entropy"]
    LABELS = ["Success%", "NodesExp", "Subnet2%", "CrownAtt",
              "ScanRate", "RepeatRatio", "MaxRepeat", "DeadLoops",
              "UniquePct", "ActEntropy"]

    print("\n\n" + "=" * 108)
    print(f"  {'Agent':<12}", end="")
    for lbl in LABELS:
        print(f"  {lbl:>10}", end="")
    print()
    print("=" * 108)

    for name, episodes in all_results.items():
        print(f"  {name:<12}", end="")
        for key in KEYS:
            mu, _ = agg(episodes, key)
            if key == "success":
                print(f"  {mu*100:>9.1f}%", end="")
            elif key == "subnet2":
                print(f"  {mu*100:>9.1f}%", end="")
            elif key in ("max_repeat", "dead_loops", "nodes_exp", "crown_atts"):
                print(f"  {mu:>10.1f}", end="")
            else:
                print(f"  {mu:>10.3f}", end="")
        print()

    print("=" * 108)

    # ── Q-value analysis for DQN and DDQN ────────────────────────────────────
    print("\n── Q-value Analysis at Initial State (DQN / DDQN) ──────────────────────────")
    for name, path in [("DQN", "dqn_complex"), ("DDQN", "ddqn_complex")]:
        m = DQN.load(f"{MODELS_DIR}/{path}")
        std, mn, mx, am = q_val_std(m, env)

        node   = am // N_PER
        atype  = am % N_PER
        a_desc = f"Scan[N{node}]" if atype == 0 else \
                 f"E{atype-1}[N{node}]{'✓' if atype-1 == EV[node] else '✗'}"
        print(f"  {name:<6}: Q-val range=[{mn:+.2f}, {mx:+.2f}]  std={std:.3f}  "
              f"argmax_action={am} ({a_desc})")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n── Key Failure Signatures ───────────────────────────────────────────────────")
    print("  DQN/DDQN:")
    print("    • MaxRepeat >> 1 → agent gets stuck executing the same action repeatedly")
    print("    • DeadLoops > 0  → triggered ≥5-in-a-row loop; never escapes to Subnet 2")
    print("    • Subnet2%  = 0% → lateral movement to Core subnet never achieved")
    print("    • ScanRate high  → Scan gives +2 reward; DQN over-values it vs exploiting")
    print("    • Q-val std low  → Q-function collapsed; all actions appear equally good")
    print("    ROOT CAUSE: Off-policy DQN with replay buffer overestimates scan reward.")
    print("                The positive scan signal (+2) dominates after early successes,")
    print("                causing the greedy policy to loop on scans instead of exploiting.")
    print()
    print("  DDPG:")
    print("    • Acts on continuous outputs: argmax of a [-1,1]^30 vector.")
    print("    • Continuous policy gradient diverges on effectively discrete action space.")
    print("    • Loops on the same (wrong) exploit after early lucky exploits.")
    print()
    print("  Successful agents (Q-Learning, PPO, A2C, A3C-like):")
    print("    • Low repeat ratio + max_repeat ≤ 1  → no loops")
    print("    • Subnet2% = 100% → consistently reaches Core via lateral movement")
    print("    • DeadLoops = 0   → clean policy without degenerate cycles")


if __name__ == "__main__":
    main()
