"""
Generate JSON and Mermaid assets for real execution experiments.

Usage:
    source ../.venv/bin/activate
    python generate_real_experiment_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BASE = Path(__file__).resolve().parent
NETWORKS_DIR = BASE / "networks"
DIAGRAMS_DIR = BASE / "diagrams"
RESULTS_DIR = BASE / "results"
FIGURES_DIR = BASE / "figures"


NETWORK_SPECS = {
    "real_experiments_network_alpha": {
        "experiment_name": "real_experiments_network_alpha",
        "description": "DMZ to internal to crown-jewel path with one Linux and one Windows branch.",
        "attacker": {"entry_subnet": "dmz", "starting_visibility": ["dmz"]},
        "subnets": [
            {"id": "dmz", "cidr": "10.20.0.0/24"},
            {"id": "internal", "cidr": "10.20.1.0/24"},
            {"id": "core", "cidr": "10.20.2.0/24"},
        ],
        "hosts": [
            {"id": "web01", "ip": "10.20.0.10", "role": "dmz-web", "os": "ubuntu-20.04", "services": ["ssh", "http"], "vulnerabilities": ["family_1", "family_3"]},
            {"id": "jump01", "ip": "10.20.0.20", "role": "dmz-jump", "os": "windows-2019", "services": ["rdp", "winrm"], "vulnerabilities": ["family_0"]},
            {"id": "app01", "ip": "10.20.1.10", "role": "internal-app", "os": "ubuntu-22.04", "services": ["ssh", "http"], "vulnerabilities": ["family_2"]},
            {"id": "db01", "ip": "10.20.1.20", "role": "internal-db", "os": "windows-2016", "services": ["mssql"], "vulnerabilities": ["family_1"]},
            {"id": "files01", "ip": "10.20.2.10", "role": "core-files", "os": "windows-2019", "services": ["smb"], "vulnerabilities": ["family_0"]},
            {"id": "crown01", "ip": "10.20.2.20", "role": "crown-jewel", "os": "ubuntu-22.04", "services": ["ssh"], "vulnerabilities": ["family_3"]},
        ],
        "reachability_rules": [
            {"from_subnet": "dmz", "to_subnet": "internal", "condition": "compromise_any(dmz)"},
            {"from_subnet": "internal", "to_subnet": "core", "condition": "compromise_any(internal)"},
        ],
        "goal": {"target_host": "crown01", "objective": "obtain_session"},
    },
    "real_experiments_network_beta": {
        "experiment_name": "real_experiments_network_beta",
        "description": "Key-node pivot network where only one DMZ jump host opens the internal subnet.",
        "attacker": {"entry_subnet": "dmz", "starting_visibility": ["dmz"]},
        "subnets": [
            {"id": "dmz", "cidr": "10.30.0.0/24"},
            {"id": "internal", "cidr": "10.30.1.0/24"},
            {"id": "core", "cidr": "10.30.2.0/24"},
        ],
        "hosts": [
            {"id": "edge01", "ip": "10.30.0.10", "role": "edge-web", "os": "ubuntu-20.04", "services": ["http"], "vulnerabilities": ["family_2"]},
            {"id": "gate01", "ip": "10.30.0.20", "role": "pivot-gateway", "os": "windows-2019", "services": ["rdp", "winrm"], "vulnerabilities": ["family_1"]},
            {"id": "erp01", "ip": "10.30.1.10", "role": "erp", "os": "windows-2016", "services": ["http", "mssql"], "vulnerabilities": ["family_0"]},
            {"id": "ops01", "ip": "10.30.1.20", "role": "ops", "os": "ubuntu-22.04", "services": ["ssh"], "vulnerabilities": ["family_3"]},
            {"id": "backup01", "ip": "10.30.2.10", "role": "backup", "os": "ubuntu-22.04", "services": ["ssh", "nfs"], "vulnerabilities": ["family_1"]},
            {"id": "crown02", "ip": "10.30.2.20", "role": "crown-jewel", "os": "windows-2019", "services": ["smb"], "vulnerabilities": ["family_2"]},
        ],
        "reachability_rules": [
            {"from_subnet": "dmz", "to_subnet": "internal", "condition": "compromise_host(gate01)"},
            {"from_subnet": "internal", "to_subnet": "core", "condition": "compromise_host(ops01)"},
        ],
        "goal": {"target_host": "crown02", "objective": "obtain_session"},
    },
    "real_experiments_network_gamma": {
        "experiment_name": "real_experiments_network_gamma",
        "description": "Dual-control internal zone where two hosts must be compromised before core access.",
        "attacker": {"entry_subnet": "dmz", "starting_visibility": ["dmz"]},
        "subnets": [
            {"id": "dmz", "cidr": "10.40.0.0/24"},
            {"id": "internal", "cidr": "10.40.1.0/24"},
            {"id": "core", "cidr": "10.40.2.0/24"},
        ],
        "hosts": [
            {"id": "proxy01", "ip": "10.40.0.10", "role": "proxy", "os": "ubuntu-20.04", "services": ["http", "ssh"], "vulnerabilities": ["family_0"]},
            {"id": "mail01", "ip": "10.40.0.20", "role": "mail", "os": "windows-2019", "services": ["smtp", "winrm"], "vulnerabilities": ["family_3"]},
            {"id": "auth01", "ip": "10.40.1.10", "role": "auth", "os": "windows-2016", "services": ["ldap", "kerberos"], "vulnerabilities": ["family_1"]},
            {"id": "ci01", "ip": "10.40.1.20", "role": "ci-server", "os": "ubuntu-22.04", "services": ["ssh", "http"], "vulnerabilities": ["family_2"]},
            {"id": "finance01", "ip": "10.40.2.10", "role": "finance", "os": "windows-2019", "services": ["smb", "mssql"], "vulnerabilities": ["family_0"]},
            {"id": "crown03", "ip": "10.40.2.20", "role": "crown-jewel", "os": "ubuntu-22.04", "services": ["ssh"], "vulnerabilities": ["family_2"]},
        ],
        "reachability_rules": [
            {"from_subnet": "dmz", "to_subnet": "internal", "condition": "compromise_host(proxy01)"},
            {"from_subnet": "internal", "to_subnet": "core", "condition": "compromise_all([auth01, ci01])"},
        ],
        "goal": {"target_host": "crown03", "objective": "obtain_session"},
    },
    "real_experiments_network_delta": {
        "experiment_name": "real_experiments_network_delta",
        "description": "Hub-and-spoke style network with a decoy service in the DMZ and a database crown jewel.",
        "attacker": {"entry_subnet": "dmz", "starting_visibility": ["dmz"]},
        "subnets": [
            {"id": "dmz", "cidr": "10.50.0.0/24"},
            {"id": "branch-a", "cidr": "10.50.1.0/24"},
            {"id": "branch-b", "cidr": "10.50.2.0/24"},
            {"id": "core", "cidr": "10.50.3.0/24"},
        ],
        "hosts": [
            {"id": "portal01", "ip": "10.50.0.10", "role": "public-portal", "os": "ubuntu-20.04", "services": ["http"], "vulnerabilities": ["family_1"]},
            {"id": "decoy01", "ip": "10.50.0.30", "role": "decoy", "os": "ubuntu-20.04", "services": ["http"], "vulnerabilities": []},
            {"id": "brancha01", "ip": "10.50.1.10", "role": "branch-a-app", "os": "windows-2019", "services": ["winrm", "http"], "vulnerabilities": ["family_0"]},
            {"id": "branchb01", "ip": "10.50.2.10", "role": "branch-b-app", "os": "ubuntu-22.04", "services": ["ssh"], "vulnerabilities": ["family_3"]},
            {"id": "hub01", "ip": "10.50.3.10", "role": "hub-control", "os": "windows-2016", "services": ["rdp", "smb"], "vulnerabilities": ["family_2"]},
            {"id": "crown04", "ip": "10.50.3.20", "role": "crown-jewel-db", "os": "windows-2019", "services": ["mssql"], "vulnerabilities": ["family_1"]},
        ],
        "reachability_rules": [
            {"from_subnet": "dmz", "to_subnet": "branch-a", "condition": "compromise_host(portal01)"},
            {"from_subnet": "dmz", "to_subnet": "branch-b", "condition": "compromise_host(portal01)"},
            {"from_subnet": "branch-a", "to_subnet": "core", "condition": "compromise_any(branch-a)"},
            {"from_subnet": "branch-b", "to_subnet": "core", "condition": "compromise_any(branch-b)"},
        ],
        "goal": {"target_host": "crown04", "objective": "obtain_session"},
    },
}


MERMAID = {
    "real_experiments_network_alpha": """flowchart LR\n    A[Attacker] --> D1[DMZ: web01]\n    A --> D2[DMZ: jump01]\n    D1 --> I1[Internal: app01]\n    D2 --> I2[Internal: db01]\n    I1 --> C1[Core: files01]\n    I2 --> C2[Core: crown01]\n""",
    "real_experiments_network_beta": """flowchart LR\n    A[Attacker] --> E1[DMZ: edge01]\n    A --> G1[DMZ Gate: gate01]\n    G1 --> I1[Internal: erp01]\n    G1 --> I2[Internal: ops01]\n    I2 --> C1[Core: backup01]\n    I2 --> C2[Core: crown02]\n""",
    "real_experiments_network_gamma": """flowchart LR\n    A[Attacker] --> P1[DMZ: proxy01]\n    A --> M1[DMZ: mail01]\n    P1 --> I1[Internal: auth01]\n    P1 --> I2[Internal: ci01]\n    I1 --> G{Need auth01 and ci01}\n    I2 --> G\n    G --> C1[Core: finance01]\n    G --> C2[Core: crown03]\n""",
    "real_experiments_network_delta": """flowchart LR\n    A[Attacker] --> P1[DMZ: portal01]\n    A --> D1[DMZ: decoy01]\n    P1 --> B1[Branch-A: brancha01]\n    P1 --> B2[Branch-B: branchb01]\n    B1 --> H1[Core: hub01]\n    B2 --> H1\n    H1 --> C1[Core: crown04]\n""",
}


def main():
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, spec in NETWORK_SPECS.items():
        with (NETWORKS_DIR / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)
        with (DIAGRAMS_DIR / f"{name}.mmd").open("w", encoding="utf-8") as f:
            f.write(MERMAID[name])
        render_network_png(name, spec, FIGURES_DIR / f"{name}.png")

    results_csv = RESULTS_DIR / "real_experiments_results.csv"
    if not results_csv.exists():
        results_csv.write_text(
            "experiment_name,target_host,execution_stage,status,session_opened,notes,timestamp\n",
            encoding="utf-8",
        )

    execution_log = RESULTS_DIR / "real_experiments_execution_log.json"
    if not execution_log.exists():
        execution_log.write_text(json.dumps({"executions": []}, indent=2), encoding="utf-8")


def render_network_png(name, spec, output_path):
    subnet_order = [subnet["id"] for subnet in spec["subnets"]]
    hosts_by_subnet = {subnet_id: [] for subnet_id in subnet_order}
    host_to_subnet = {}
    for host in spec["hosts"]:
        subnet_id = next(
            subnet["id"] for subnet in spec["subnets"]
            if host["ip"].startswith(".".join(subnet["cidr"].split(".")[:3]))
            or subnet["id"] in host["role"]
            or subnet["id"] in host["id"]
        ) if False else None
        host_to_subnet[host["id"]] = None

    # Assign hosts by CIDR third octet heuristic.
    subnet_octet = {subnet["id"]: subnet["cidr"].split(".")[2] for subnet in spec["subnets"]}
    for host in spec["hosts"]:
        octet = host["ip"].split(".")[2]
        subnet_id = next(subnet_id for subnet_id, subnet_oct in subnet_octet.items() if subnet_oct == octet)
        hosts_by_subnet[subnet_id].append(host)
        host_to_subnet[host["id"]] = subnet_id

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_title(name.replace("_", " "))
    ax.axis("off")

    x_step = 3.1
    y_step = 1.2
    box_w = 2.0
    box_h = 0.65
    colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f2dede"]
    node_positions = {}

    for subnet_index, subnet_id in enumerate(subnet_order):
        x = subnet_index * x_step
        subnet_hosts = hosts_by_subnet[subnet_id]
        subnet_color = colors[subnet_index % len(colors)]
        ax.text(x + 0.9, 2.65, subnet_id, fontsize=11, fontweight="bold", ha="center")
        for host_index, host in enumerate(subnet_hosts):
            y = 2.0 - host_index * y_step
            node_positions[host["id"]] = (x, y)
            patch = FancyBboxPatch(
                (x, y), box_w, box_h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.1,
                edgecolor="#333333",
                facecolor=subnet_color,
            )
            ax.add_patch(patch)
            vuln_text = ", ".join(host["vulnerabilities"]) if host["vulnerabilities"] else "none"
            ax.text(x + 0.08, y + 0.44, f"{host['id']} ({host['ip']})", fontsize=8, ha="left")
            ax.text(x + 0.08, y + 0.18, f"{host['role']} | vulns: {vuln_text}", fontsize=7, ha="left")

    ax.text(-1.4, 2.2, "Attacker", fontsize=10, fontweight="bold")
    first_subnet_hosts = hosts_by_subnet[subnet_order[0]]
    for host in first_subnet_hosts:
        hx, hy = node_positions[host["id"]]
        ax.annotate("", xy=(hx, hy + 0.3), xytext=(-0.2, 2.2),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#444444"))

    for rule in spec["reachability_rules"]:
        from_hosts = hosts_by_subnet[rule["from_subnet"]]
        to_hosts = hosts_by_subnet[rule["to_subnet"]]
        if not from_hosts or not to_hosts:
            continue
        fx, fy = node_positions[from_hosts[-1]["id"]]
        tx, ty = node_positions[to_hosts[0]["id"]]
        ax.annotate(
            rule["condition"],
            xy=(tx, ty + 0.3),
            xytext=(fx + box_w, fy + 0.95),
            fontsize=7,
            ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#777777"),
        )

    goal = spec["goal"]["target_host"]
    gx, gy = node_positions[goal]
    ax.text(gx + 0.9, gy - 0.18, "GOAL", fontsize=9, fontweight="bold", color="#b30000", ha="center")
    ax.set_xlim(-1.8, len(subnet_order) * x_step + 1.0)
    ax.set_ylim(-0.8, 3.1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
