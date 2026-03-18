"""
prepare_real_execution_configs.py

Convert real_experiments network JSON files into execution-oriented configs and
runbooks for later live testing.

This script does NOT access Docker, Metasploit, or any target. It only creates
structured preparation files.
"""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
NETWORKS_DIR = BASE / "networks"
CONFIGS_DIR = BASE / "configs"
RUNBOOKS_DIR = BASE / "runbooks"


def build_target_config(network_spec: dict) -> dict:
    targets = [host["ip"] for host in network_spec["hosts"]]
    return {
        "approach": f"real_experiment::{network_spec['experiment_name']}",
        "targets": targets,
        "path-file": f"real_experiments/runbooks/{network_spec['experiment_name']}_attack_paths.txt",
        "metadata": {
            "goal_host": network_spec["goal"]["target_host"],
            "goal_objective": network_spec["goal"]["objective"],
            "description": network_spec["description"],
        },
    }


def build_runbook(network_spec: dict) -> dict:
    hosts = {host["id"]: host for host in network_spec["hosts"]}
    return {
        "experiment_name": network_spec["experiment_name"],
        "scope": {
            "attacker_entry_subnet": network_spec["attacker"]["entry_subnet"],
            "goal": network_spec["goal"],
        },
        "ordered_preparation_steps": [
            "Confirm Docker lab topology exists and hosts are reachable from the attacker container.",
            "Validate Metasploit RPC connectivity and workspace creation.",
            "Run safe service fingerprinting only within approved scope.",
            "Map discovered services to capability families or custom exploit backends.",
            "Execute exploits only after verifying the target and preconditions.",
            "Record every attempt into real_experiments/results/real_experiments_execution_log.json.",
        ],
        "hosts": [
            {
                "id": host["id"],
                "ip": host["ip"],
                "role": host["role"],
                "services": host["services"],
                "vulnerabilities": host["vulnerabilities"],
                "expected_backend": [
                    "metasploit" if vuln in {"family_0", "family_1"} else "custom_or_external"
                    for vuln in host["vulnerabilities"]
                ],
            }
            for host in network_spec["hosts"]
        ],
        "reachability_rules": network_spec["reachability_rules"],
        "notes": "Preparation artifact only. No live exploitation performed by this script.",
    }


def write_attack_path_placeholder(network_name: str, network_spec: dict):
    lines = [
        "# Placeholder attack paths for later live execution",
        f"# goal={network_spec['goal']['target_host']}",
        "[]",
    ]
    (RUNBOOKS_DIR / f"{network_name}_attack_paths.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    for network_path in sorted(NETWORKS_DIR.glob("real_experiments_network_*.json")):
        spec = json.loads(network_path.read_text(encoding="utf-8"))
        name = spec["experiment_name"]

        target_config = build_target_config(spec)
        runbook = build_runbook(spec)

        (CONFIGS_DIR / f"{name}_target_config.json").write_text(
            json.dumps(target_config, indent=2), encoding="utf-8"
        )
        (RUNBOOKS_DIR / f"{name}_runbook.json").write_text(
            json.dumps(runbook, indent=2), encoding="utf-8"
        )
        write_attack_path_placeholder(name, spec)


if __name__ == "__main__":
    main()