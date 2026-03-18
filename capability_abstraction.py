"""
capability_abstraction.py

Abstraction layer for actions/exploits so the agent is not tied to only
Metasploit modules. A capability may be backed by:
  - a Metasploit module
  - a custom Python exploit script
  - an external tool / PoC
  - a manual workflow placeholder
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Capability:
    capability_id: str
    action_type: int
    kind: str
    backend: str
    reference: str
    description: str


DEFAULT_CAPABILITIES = {
    "scan": Capability(
        capability_id="scan",
        action_type=0,
        kind="recon",
        backend="builtin",
        reference="scan",
        description="Enumerate a host and reveal its vulnerability families.",
    ),
    "exploit_family_0": Capability(
        capability_id="exploit_family_0",
        action_type=1,
        kind="exploit",
        backend="metasploit",
        reference="family/0",
        description="Exploit family 0. Can be mapped to a Metasploit or custom module.",
    ),
    "exploit_family_1": Capability(
        capability_id="exploit_family_1",
        action_type=2,
        kind="exploit",
        backend="metasploit",
        reference="family/1",
        description="Exploit family 1. Can be mapped to a Metasploit or custom module.",
    ),
    "exploit_family_2": Capability(
        capability_id="exploit_family_2",
        action_type=3,
        kind="exploit",
        backend="custom",
        reference="family/2",
        description="Exploit family 2. Intended for non-Metasploit custom exploit chains.",
    ),
    "exploit_family_3": Capability(
        capability_id="exploit_family_3",
        action_type=4,
        kind="exploit",
        backend="external",
        reference="family/3",
        description="Exploit family 3. Intended for PoC scripts or external tools.",
    ),
}


class CapabilityRegistry:
    def __init__(self, base: Optional[Dict[str, Capability]] = None):
        self._caps = dict(base or DEFAULT_CAPABILITIES)

    def register(self, capability: Capability):
        self._caps[capability.capability_id] = capability

    def get(self, capability_id: str) -> Capability:
        return self._caps[capability_id]

    def action_to_capability(self, action_type: int) -> Capability:
        for cap in self._caps.values():
            if cap.action_type == action_type:
                return cap
        raise KeyError(f"No capability registered for action_type={action_type}")

    def capability_to_action(self, capability_id: str) -> int:
        return self._caps[capability_id].action_type

    def exploit_backend_for_family(self, vuln_family: int) -> str:
        return self.action_to_capability(vuln_family + 1).backend


def encode_node_capability_action(node: int, capability_id: str, registry: CapabilityRegistry, actions_per_node: int) -> int:
    return node * actions_per_node + registry.capability_to_action(capability_id)


def decode_node_capability_action(action: int, registry: CapabilityRegistry, actions_per_node: int):
    node = int(action) // actions_per_node
    action_type = int(action) % actions_per_node
    return node, registry.action_to_capability(action_type)