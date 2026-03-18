"""
a3c_pytorch.py

True Asynchronous Advantage Actor-Critic (A3C) using PyTorch with
multiple parallel worker threads (each with its own env copy).

Architecture
─────────────────────────────────────────────────────────────────
  ActorCritic    – shared MLP backbone, separate policy & value heads
  A3CWorker      – thread subclass; rollout N steps → compute grads
                   → apply to shared global model (async)
  SharedAdam     – Adam with state tensors kept in shared memory

Training
─────────────────────────────────────────────────────────────────
  N_WORKERS = 4 threads, each with its own PentestEnvComplex copy
  TOTAL_STEPS = 300_000 env steps across all workers
  Saves model to: models_complex/a3c_pytorch.pt

Usage:
    source .venv/bin/activate
    python a3c_pytorch.py
"""

import os
# Prevent PyTorch internal OMP threads from fighting with our worker threads
# (main cause of SIGTRAP on macOS with threaded PyTorch)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import threading
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F

from pentest_env_complex import PentestEnvComplex
from pentest_env_fixed   import PentestEnvFixed

# ── Hyper-parameters ──────────────────────────────────────────────────────────
# FIX: run on PentestEnvFixed so that DQN-style dead-loop is also patched here;
#      A3C uses the same env for fair comparison in fix_and_compare.py
ENV_CLS   = PentestEnvFixed

N_WORKERS    = 4
N_STEPS      = 80       # longer rollout → reduces stale-gradient window
GAMMA        = 0.99
LR           = 3e-4     # restored; stability now comes from per-worker grad accum
ENT_COEF     = 0.05     # strong entropy bonus maintained
VALUE_COEF   = 0.5
GRAD_CLIP    = 5.0
TOTAL_STEPS  = 400_000
MODELS_DIR   = "models_complex"

OBS_DIM = PentestEnvComplex.NUM_NODES * PentestEnvComplex.STATE_DIM   # 36
N_ACT   = PentestEnvComplex.NUM_NODES * PentestEnvComplex.ACTIONS_PER_NODE  # 30
EV      = PentestEnvComplex.EXPLOITABLE_VULN


# ── Network ───────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_act=N_ACT, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_act)   # policy logits
        self.v  = nn.Linear(hidden, 1)       # state value

    def forward(self, x: torch.Tensor):
        h = self.shared(x.float())
        return self.pi(h), self.v(h)

    def act(self, obs: np.ndarray, deterministic=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(obs_t)
        dist   = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1).item() if deterministic else dist.sample().item()
        return action, dist.log_prob(torch.tensor(action)), value.squeeze()


# ── Thread-safe step counter ──────────────────────────────────────────────────
class StepCounter:
    def __init__(self):
        self._v    = 0
        self._lock = threading.Lock()

    @property
    def value(self):
        return self._v

    def add(self, n: int):
        with self._lock:
            self._v += n


# ── Worker thread ─────────────────────────────────────────────────────────────
class A3CWorker(threading.Thread):
    def __init__(self, wid, global_model, optimizer, lock, counter, total_steps):
        super().__init__(daemon=True)
        self.wid          = wid
        self.global_model = global_model
        self.optimizer    = optimizer
        self.lock         = lock          # guards global model updates
        self.counter      = counter
        self.total_steps  = total_steps
        self.local_model  = ActorCritic()
        self.env          = ENV_CLS()
        self.ep_rewards   = []
        self.ep_lengths   = []

    def run(self):
        obs, _ = self.env.reset()

        while self.counter.value < self.total_steps:
            # ── Sync local ← global ──────────────────────────────────────────
            self.local_model.load_state_dict(self.global_model.state_dict())

            # ── Rollout ──────────────────────────────────────────────────────
            states, actions, rewards, dones = [], [], [], []
            ep_r, ep_len = 0.0, 0

            for _ in range(N_STEPS):
                act, _, _ = self.local_model.act(obs)
                nobs, r, term, trunc, _ = self.env.step(act)
                done = term or trunc

                states.append(obs.copy())
                actions.append(act)
                rewards.append(r)
                dones.append(done)
                ep_r   += r
                ep_len += 1

                obs = nobs
                if done:
                    self.ep_rewards.append(ep_r)
                    self.ep_lengths.append(ep_len)
                    ep_r, ep_len = 0.0, 0
                    obs, _ = self.env.reset()

            # ── Bootstrap return ─────────────────────────────────────────────
            if dones[-1]:
                R = 0.0
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    _, v = self.local_model(obs_t)
                R = v.item()

            # ── Compute returns & advantages ──────────────────────────────────
            returns = []
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + GAMMA * R * (1 - float(d))
                returns.insert(0, R)

            states_t  = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)
            returns_t = torch.FloatTensor(returns)

            logits, values = self.local_model(states_t)
            dist           = torch.distributions.Categorical(logits=logits)
            log_probs      = dist.log_prob(actions_t)
            advantages     = returns_t - values.squeeze().detach()

            policy_loss = -(log_probs * advantages).mean()
            value_loss  = F.mse_loss(values.squeeze(), returns_t)
            entropy     = dist.entropy().mean()
            loss        = policy_loss + VALUE_COEF * value_loss - ENT_COEF * entropy

            # ── Async update ──────────────────────────────────────────────────
            # FIX: The optimizer tracks GLOBAL params, so optimizer.zero_grad()
            #      would zero GLOBAL grads (a race condition).
            #      Instead, manually zero LOCAL model grads before backward.
            for p in self.local_model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            loss.backward()   # grads accumulate on LOCAL model only (thread-safe)

            with self.lock:
                # Overwrite global grads with local grads then step
                for lp, gp in zip(self.local_model.parameters(),
                                   self.global_model.parameters()):
                    if lp.grad is not None:
                        gp.grad = lp.grad.clone()
                nn.utils.clip_grad_norm_(self.global_model.parameters(), GRAD_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()   # safe: we hold the lock

            self.counter.add(N_STEPS)


# ── Training entry point ──────────────────────────────────────────────────────
def train(total_steps=TOTAL_STEPS, n_workers=N_WORKERS):
    os.makedirs(MODELS_DIR, exist_ok=True)

    global_model = ActorCritic()
    optimizer    = torch.optim.Adam(global_model.parameters(), lr=LR)
    lock         = threading.Lock()
    counter      = StepCounter()

    workers = [A3CWorker(i, global_model, optimizer, lock, counter, total_steps)
               for i in range(n_workers)]

    print(f"[A3C] Starting {n_workers} workers — target {total_steps:,} steps ...")
    t0 = time.time()
    for w in workers:
        w.start()

    # Periodic evaluation to track and save the BEST checkpoint
    best_sr        = -1.0
    best_state     = None
    eval_env       = ENV_CLS()
    eval_interval  = total_steps // 10
    last_eval      = 0

    while counter.value < total_steps:
        time.sleep(1)
        if counter.value - last_eval >= eval_interval:
            pct     = counter.value / total_steps * 100
            elapsed = time.time() - t0

            # Quick eval (10 ep) of current global model
            successes = []
            for _ in range(10):
                obs, _ = eval_env.reset()
                ep_r   = 0.0
                done   = False
                while not done:
                    act, _, _ = global_model.act(obs, deterministic=True)
                    obs, r, term, trunc, _ = eval_env.step(act)
                    ep_r += r
                    done  = term or trunc
                successes.append(term)
            sr = np.mean(successes) * 100

            # Track best
            if sr > best_sr:
                best_sr    = sr
                best_state = {k: v.clone() for k, v in global_model.state_dict().items()}

            recent_rewards = []
            for wk in workers:
                recent_rewards.extend(wk.ep_rewards[-5:])
            avg_r = np.mean(recent_rewards) if recent_rewards else float("nan")
            print(f"  {pct:5.1f}%  steps={counter.value:>7,}  "
                  f"eval_sr={sr:.0f}%  best_sr={best_sr:.0f}%  "
                  f"avg_ep_reward={avg_r:+.1f}  elapsed={elapsed:.1f}s")
            last_eval = counter.value

    for w in workers:
        w.join(timeout=3)

    # Restore best checkpoint found during training
    if best_state is not None and best_sr > 0:
        global_model.load_state_dict(best_state)
        print(f"\n[A3C] Restored best checkpoint (eval success={best_sr:.0f}%)")

    elapsed = time.time() - t0
    path    = f"{MODELS_DIR}/a3c_pytorch.pt"
    torch.save(global_model.state_dict(), path)
    print(f"[A3C] Training complete in {elapsed:.1f}s")
    print(f"[A3C] Model saved → {path}")
    return global_model


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model=None, model_path=None, n_episodes=30):
    if model is None:
        path  = model_path or f"{MODELS_DIR}/a3c_pytorch.pt"
        model = ActorCritic()
        model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    env      = ENV_CLS()

    rewards, steps_list, successes = [], [], []
    action_traces = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r, step, done = 0.0, 0, False
        trace = []

        while not done:
            a, _, _ = model.act(obs, deterministic=True)
            trace.append(a)
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            step    += 1
            done     = term or trunc

        rewards.append(total_r)
        successes.append(term)
        if term:
            steps_list.append(step)
        action_traces.append(trace)

    sr  = np.mean(successes) * 100
    mr  = np.mean(rewards)
    std = np.std(rewards)
    ms  = np.mean(steps_list) if steps_list else float("nan")

    n_per = PentestEnvComplex.ACTIONS_PER_NODE
    ev    = PentestEnvComplex.EXPLOITABLE_VULN

    def fmt(a):
        node  = a // n_per
        atype = a % n_per
        if atype == 0:
            return f"Scan[N{node}]"
        v = atype - 1
        return f"E{v}[N{node}]{'✓' if v == ev[node] else '✗'}"

    print("\n" + "─" * 60)
    print("  A3C (PyTorch, async threads)")
    print("─" * 60)
    print(f"  Episodes       : {n_episodes}")
    print(f"  Success rate   : {sr:.1f}%  ({sum(successes)}/{n_episodes})")
    print(f"  Mean reward    : {mr:+.2f}  ±{std:.2f}")
    print(f"  Avg steps/goal : {ms:.1f}" if steps_list else "  Avg steps/goal : N/A")

    sample = next((t for t, s in zip(action_traces, successes) if s), action_traces[-1])
    trace  = " → ".join(fmt(a) for a in sample)
    if len(trace) > 110:
        trace = trace[:107] + "..."
    label = "Successful trace" if any(successes) else "Best attempt"
    print(f"  {label}: {trace}")
    print("─" * 60)

    return dict(name="A3C (PyTorch async)", sr=sr, mr=mr, std=std, ms=ms)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trained_model = train()
    evaluate(model=trained_model)
