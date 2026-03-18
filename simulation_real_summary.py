"""
simulation_real_summary.py

Create publication-ready summary tables and figures for:
  1) simulation results
  2) real_experiments preparation assets

Outputs are saved under:
  - simulation_figures/
  - real_experiments/results/
"""

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SIM_OUT = ROOT / "simulation_figures"
REAL_OUT = ROOT / "real_experiments" / "results"
SIM_OUT.mkdir(exist_ok=True)
REAL_OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value):
    return float(value) if value not in ("nan", "", None) else float("nan")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_simulation_main():
    rows = load_csv(ROOT / "results_compare_complex_simulation.csv")
    parsed = [
        {
            "name": r["name"],
            "sr": to_float(r["sr"]),
            "mr": to_float(r["mr"]),
            "std": to_float(r["std"]),
            "ms": to_float(r["ms"]),
        }
        for r in rows
    ]
    out_csv = SIM_OUT / "simulation_table_main.csv"
    write_csv(out_csv, ["name", "sr", "mr", "std", "ms"], parsed)

    names = [r["name"] for r in parsed]
    short = {
        "Random (baseline)": "Random",
        "Deterministic HARMer (sequential)": "Det.\nHARMer",
        "Q-Learning": "Q-Learn",
        "PPO": "PPO",
        "DQN": "DQN",
        "DDQN (Double DQN, 128×128)": "DDQN",
        "A2C": "A2C",
        "DDPG (continuous argmax wrapper)": "DDPG",
        "A3C-like (A2C surrogate)": "A3C-like",
        "DPPO-like (vectorized PPO surrogate)": "DPPO-like",
        "GAIL-like (adversarial imitation)": "GAIL-like",
    }
    labels = [short.get(n, n) for n in names]
    x = np.arange(len(parsed))

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes[0].bar(x, [r["sr"] for r in parsed], color="#4daf4a")
    axes[0].set_title("Simulation Fig. 1 — Success Rate on Complex Network")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0)
    axes[0].axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.4)

    axes[1].bar(x, [r["mr"] for r in parsed], color="#377eb8")
    axes[1].set_title("Simulation Fig. 2 — Mean Reward")
    axes[1].set_ylabel("Mean Reward")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=0)
    axes[1].axhline(0, color="black", linewidth=0.7, alpha=0.4)

    succ_idx = [i for i, r in enumerate(parsed) if not np.isnan(r["ms"])]
    axes[2].bar(np.arange(len(succ_idx)), [parsed[i]["ms"] for i in succ_idx], color="#ff7f00")
    axes[2].set_title("Simulation Fig. 3 — Average Steps to Goal (successful only)")
    axes[2].set_ylabel("Avg Steps")
    axes[2].set_xticks(np.arange(len(succ_idx)))
    axes[2].set_xticklabels([labels[i] for i in succ_idx], rotation=0)
    axes[2].axhline(3, color="#984ea3", linestyle="--", linewidth=1, alpha=0.6)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(SIM_OUT / "simulation_fig_main.png", dpi=300)
    fig.savefig(SIM_OUT / "simulation_fig_main.pdf")
    return parsed


def summarize_transfer_and_adaptation():
    transfer = load_csv(ROOT / "results_transfer_xyz.csv")
    adaptation = load_csv(ROOT / "results_generalization_adaptation.csv")

    transfer_rows = [
        {
            "network": r["network"],
            "agent": r["agent"],
            "sr": to_float(r["sr"]),
            "mr": to_float(r["mr"]),
            "ms": to_float(r["ms"]),
        }
        for r in transfer
    ]
    adaptation_rows = [
        {
            "network": r["network"],
            "agent": r["agent"],
            "phase": r["phase"],
            "sr": to_float(r["sr"]),
            "mr": to_float(r["mr"]),
            "ms": to_float(r["ms"]),
        }
        for r in adaptation
    ]

    write_csv(SIM_OUT / "simulation_table_transfer.csv", ["network", "agent", "sr", "mr", "ms"], transfer_rows)
    write_csv(SIM_OUT / "simulation_table_adaptation.csv", ["network", "agent", "phase", "sr", "mr", "ms"], adaptation_rows)

    networks = ["X", "Y", "Z"]
    agents = ["PPO", "A2C", "DQN"]
    zero = np.zeros((len(agents), len(networks)))
    few = np.zeros((len(agents), len(networks)))
    for i, agent in enumerate(agents):
        for j, network in enumerate(networks):
            for row in adaptation_rows:
                if row["agent"] == agent and row["network"] == network and row["phase"] == "zero-shot":
                    zero[i, j] = row["sr"]
                if row["agent"] == agent and row["network"] == network and row["phase"] == "few-shot":
                    few[i, j] = row["sr"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, mat, title in zip(axes, [zero, few], ["Zero-shot", "Few-shot"]):
        im = ax.imshow(mat, vmin=0, vmax=100, cmap="YlGn")
        ax.set_xticks(np.arange(len(networks)))
        ax.set_xticklabels(networks)
        ax.set_yticks(np.arange(len(agents)))
        ax.set_yticklabels(agents)
        ax.set_title(f"Simulation Fig. 4 — {title} Success Rate")
        for i in range(len(agents)):
            for j in range(len(networks)):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Success Rate (%)")
    fig.tight_layout()
    fig.savefig(SIM_OUT / "simulation_fig_transfer_adaptation.png", dpi=300)
    fig.savefig(SIM_OUT / "simulation_fig_transfer_adaptation.pdf")


def summarize_real_preparation():
    networks_dir = ROOT / "real_experiments" / "networks"
    rows = []
    for path in sorted(networks_dir.glob("real_experiments_network_*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": spec["experiment_name"],
                "subnets": len(spec["subnets"]),
                "hosts": len(spec["hosts"]),
                "goal_host": spec["goal"]["target_host"],
                "reachability_rules": len(spec["reachability_rules"]),
                "has_decoy": int(any(host.get("role") == "decoy" for host in spec["hosts"])),
            }
        )

    out_csv = REAL_OUT / "real_experiments_preparation_summary.csv"
    write_csv(out_csv, ["experiment_name", "subnets", "hosts", "goal_host", "reachability_rules", "has_decoy"], rows)

    x = np.arange(len(rows))
    labels = [r["experiment_name"].replace("real_experiments_network_", "") for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.35
    ax.bar(x - width / 2, [r["hosts"] for r in rows], width, label="Hosts", color="#377eb8")
    ax.bar(x + width / 2, [r["reachability_rules"] for r in rows], width, label="Reachability Rules", color="#ff7f00")
    ax.set_title("Real Experiments Fig. 1 — Prepared Network Complexity")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(REAL_OUT / "real_experiments_preparation_summary.png", dpi=300)
    fig.savefig(REAL_OUT / "real_experiments_preparation_summary.pdf")


def main():
    summarize_simulation_main()
    summarize_transfer_and_adaptation()
    summarize_real_preparation()
    print(f"Saved simulation figures -> {SIM_OUT}")
    print(f"Saved real experiment preparation summary -> {REAL_OUT}")


if __name__ == "__main__":
    main()