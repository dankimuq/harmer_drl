"""
network_task_generator.py

Task-family generator for generalisation experiments.

Instead of evaluating on a few hand-crafted networks only, this module samples
network tasks from a broader distribution over topology, goal placement, and
vulnerability assignments.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class NetworkTask:
    topology_id: int
    topology_name: str
    subnet_sizes: List[int]
    host_subnet: List[int]
    exploitable_vuln: List[int]
    goal_node: int
    dmz_gate_node: Optional[int]
    internal_gate_node: Optional[int]
    requires_dual_internal: bool
    initial_visible_subnets: List[int]
    node_permutation: List[int]
    scan_noise: float
    decoy_nodes: List[int]


class NetworkTaskGenerator:
    TOPOLOGY_LINEAR = 0
    TOPOLOGY_KEYNODE = 1
    TOPOLOGY_DUALCORE = 2

    TOPOLOGY_NAMES = {
        TOPOLOGY_LINEAR: "linear",
        TOPOLOGY_KEYNODE: "keynode",
        TOPOLOGY_DUALCORE: "dualcore",
    }

    def __init__(self, num_nodes=6, num_vulns=4, subnet_size=2, seed=None):
        self.num_nodes = num_nodes
        self.num_vulns = num_vulns
        self.subnet_size = subnet_size
        self.num_subnets = num_nodes // subnet_size
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def _sample_topology_id(self, split: str) -> int:
        if split == "test-ood-topology":
            return self.TOPOLOGY_DUALCORE
        if split == "val":
            return int(self._rng.choice([self.TOPOLOGY_LINEAR, self.TOPOLOGY_KEYNODE]))
        return int(self._rng.integers(0, 3))

    def _sample_goal_node(self, topology_id: int) -> int:
        if topology_id == self.TOPOLOGY_DUALCORE:
            return int(self._rng.choice([4, 5]))
        return int(self._rng.choice([4, 5]))

    def _sample_vuln_mapping(self, split: str) -> List[int]:
        if split == "test-ood-security":
            return [int((i + 2) % self.num_vulns) for i in range(self.num_nodes)]
        return [int(self._rng.integers(0, self.num_vulns)) for _ in range(self.num_nodes)]

    def _sample_decoys(self, split: str) -> List[int]:
        if split.startswith("test") and self._rng.random() < 0.35:
            return [int(self._rng.choice([1, 4]))]
        return []

    def sample(self, split: str = "train") -> NetworkTask:
        topology_id = self._sample_topology_id(split)
        topology_name = self.TOPOLOGY_NAMES[topology_id]
        subnet_sizes = [self.subnet_size] * self.num_subnets
        host_subnet = [node // self.subnet_size for node in range(self.num_nodes)]

        dmz_gate_node = None
        internal_gate_node = None
        requires_dual_internal = False

        if topology_id in (self.TOPOLOGY_KEYNODE, self.TOPOLOGY_DUALCORE):
            dmz_gate_node = int(self._rng.choice([0, 1]))
        if topology_id == self.TOPOLOGY_KEYNODE:
            internal_gate_node = int(self._rng.choice([2, 3]))
        if topology_id == self.TOPOLOGY_DUALCORE:
            requires_dual_internal = True

        exploitable_vuln = self._sample_vuln_mapping(split)
        goal_node = self._sample_goal_node(topology_id)
        initial_visible_subnets = [0]
        node_permutation = self._rng.permutation(self.num_nodes).tolist()
        scan_noise = 0.0 if split == "train" else 0.05
        decoy_nodes = self._sample_decoys(split)

        return NetworkTask(
            topology_id=topology_id,
            topology_name=topology_name,
            subnet_sizes=subnet_sizes,
            host_subnet=host_subnet,
            exploitable_vuln=exploitable_vuln,
            goal_node=goal_node,
            dmz_gate_node=dmz_gate_node,
            internal_gate_node=internal_gate_node,
            requires_dual_internal=requires_dual_internal,
            initial_visible_subnets=initial_visible_subnets,
            node_permutation=node_permutation,
            scan_noise=scan_noise,
            decoy_nodes=decoy_nodes,
        )