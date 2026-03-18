"""
topology_wrapper.py

Observation wrapper that appends static/dynamic topology descriptors to the
base host state so the policy can condition on network structure.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TopologyAwareObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        base_dim = int(np.prod(env.observation_space.shape))
        topo_dim = len(self.env.get_topology_features())
        self.observation_space = spaces.MultiBinary(base_dim + topo_dim)

    def observation(self, observation):
        topo = self.env.get_topology_features()
        return np.concatenate([np.asarray(observation, dtype=np.int8), topo]).astype(np.int8)