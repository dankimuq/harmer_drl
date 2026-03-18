"""
simulation_extensions.py

Simulation-only extensions for additional agents:
  - DPPO-like: vectorized PPO surrogate for distributed PPO
  - GAIL-like: lightweight adversarial imitation learning on expert rollouts

These implementations are explicitly simulation-focused and do not perform any
real exploitation or tool execution.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pentest_env_complex import PentestEnvComplex


MODELS_DIR = "models_complex"


class OracleExpertComplex:
    """Shortest-path oracle for the fixed simulation environment."""

    def __init__(self):
        self.targets = [0, 2, PentestEnvComplex.NUM_NODES - 1]

    def predict(self, obs: np.ndarray) -> int:
        state = obs.reshape(PentestEnvComplex.NUM_NODES, PentestEnvComplex.STATE_DIM)
        for node in self.targets:
            if state[node][1] == 0:
                vuln = PentestEnvComplex.EXPLOITABLE_VULN[node]
                return node * PentestEnvComplex.ACTIONS_PER_NODE + 1 + vuln
        return 0


def train_dppo_like(total_timesteps=120_000, n_envs=8, save_name="dppo_complex"):
    env = make_vec_env(PentestEnvComplex, n_envs=n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        tensorboard_log="./complex_tensorboard/",
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=total_timesteps)
    path = f"{MODELS_DIR}/{save_name}"
    model.save(path)
    return model


def collect_expert_dataset(expert, n_episodes=200):
    env = PentestEnvComplex()
    states, actions = [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = expert.predict(obs)
            states.append(obs.copy())
            actions.append(action)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64)


class GAILPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs_t):
        hidden = self.net(obs_t.float())
        return self.pi(hidden), self.v(hidden).squeeze(-1)

    def act(self, obs: np.ndarray, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = self(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(value.item())


class Discriminator(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_t, act_t, n_actions):
        one_hot = F.one_hot(act_t.long(), n_actions).float()
        return self.net(torch.cat([obs_t.float(), one_hot], dim=-1)).squeeze(-1)


@dataclass
class GAILArtifacts:
    policy: GAILPolicy
    discriminator: Discriminator


def _rollout_policy(policy, n_episodes=24):
    env = PentestEnvComplex()
    trajectories = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode = []
        done = False
        while not done:
            action, _ = policy.act(obs, deterministic=False)
            next_obs, _, term, trunc, _ = env.step(action)
            episode.append((obs.copy(), action, term or trunc))
            obs = next_obs
            done = term or trunc
        trajectories.append(episode)
    return trajectories


def train_gail_like(
    total_rounds=120,
    bc_epochs=20,
    disc_updates=6,
    policy_updates=4,
    save_name="gail_complex.pt",
):
    env = PentestEnvComplex()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    expert = OracleExpertComplex()
    expert_states, expert_actions = collect_expert_dataset(expert)

    policy = GAILPolicy(obs_dim, n_actions)
    discriminator = Discriminator(obs_dim, n_actions)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=3e-4)

    expert_states_t = torch.as_tensor(expert_states, dtype=torch.float32)
    expert_actions_t = torch.as_tensor(expert_actions, dtype=torch.int64)

    for _ in range(bc_epochs):
        logits, _ = policy(expert_states_t)
        loss = F.cross_entropy(logits, expert_actions_t)
        policy_opt.zero_grad()
        loss.backward()
        policy_opt.step()

    for _ in range(total_rounds):
        trajectories = _rollout_policy(policy, n_episodes=16)
        policy_states = np.asarray([step[0] for ep in trajectories for step in ep], dtype=np.float32)
        policy_actions = np.asarray([step[1] for ep in trajectories for step in ep], dtype=np.int64)

        policy_states_t = torch.as_tensor(policy_states, dtype=torch.float32)
        policy_actions_t = torch.as_tensor(policy_actions, dtype=torch.int64)

        for _ in range(disc_updates):
            expert_logits = discriminator(expert_states_t, expert_actions_t, n_actions)
            policy_logits = discriminator(policy_states_t, policy_actions_t, n_actions)
            disc_loss = (
                F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits)) +
                F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits))
            )
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()

        for _ in range(policy_updates):
            states = []
            actions = []
            returns = []
            for episode in trajectories:
                discounted = []
                running = 0.0
                for obs, action, done in reversed(episode):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act_t = torch.as_tensor([action], dtype=torch.int64)
                    with torch.no_grad():
                        disc_logit = discriminator(obs_t, act_t, n_actions)
                        reward = -F.logsigmoid(-disc_logit).item()
                    running = reward + 0.99 * running
                    discounted.insert(0, running)
                for (obs, action, _), ret in zip(episode, discounted):
                    states.append(obs)
                    actions.append(action)
                    returns.append(ret)

            states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32)
            actions_t = torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.int64)
            returns_t = torch.as_tensor(np.asarray(returns, dtype=np.float32), dtype=torch.float32)
            logits, values = policy(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            advantages = returns_t - values.detach()
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns_t)
            entropy_bonus = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
            policy_opt.zero_grad()
            loss.backward()
            policy_opt.step()

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = f"{MODELS_DIR}/{save_name}"
    torch.save(
        {
            "policy": policy.state_dict(),
            "discriminator": discriminator.state_dict(),
            "obs_dim": int(obs_dim),
            "n_actions": int(n_actions),
        },
        path,
    )
    return GAILArtifacts(policy=policy, discriminator=discriminator)


def load_gail_policy(path=f"{MODELS_DIR}/gail_complex.pt"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = GAILPolicy(checkpoint["obs_dim"], checkpoint["n_actions"])
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    return policy