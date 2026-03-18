"""
paper_results.py

Generates all tables and plots for the paper from experimental results.

Outputs (saved to paper_figures/)
──────────────────────────────────
  TABLE 1  – Main results: all agents × (SR%, Mean Reward ± σ, Steps)
             printed to stdout + saved as paper_figures/table1_main.csv
  TABLE 2  – Before/After fix for DQN & DDQN
             printed to stdout + saved as paper_figures/table2_fix.csv
  FIG 1    – Success rate bar chart (all agents)
  FIG 2    – Mean reward bar chart with ±σ error bars (all agents)
  FIG 3    – Average steps-to-goal (successful episodes only)
  FIG 4    – Before vs After fix grouped bar (DQN, DDQN): SR%
  FIG 5    – Diagnostic heatmap (dead-loop metrics per agent)

Usage:
    source .venv/bin/activate
    python paper_results.py
"""

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (headless safe)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = "paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Raw data (collected from fix_and_compare.py + diagnose_agents.py runs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Try to load from CSV first; fall back to hard-coded values
RESULTS_CSV = "results_fixed.csv"

def load_results():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"{RESULTS_CSV} not found — run fix_and_compare.py first.")
    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                name=r["name"],
                sr=float(r["sr"]),
                mr=float(r["mr"]),
                std=float(r["std"]),
                ms=float(r["ms"]) if r["ms"] not in ("nan", "") else float("nan"),
                note=r["note"],
            ))
    return rows

results = load_results()

# ── Diagnostic data (from diagnose_agents.py output) ─────────────────────────
# Columns: agent, scan_rate, repeat_ratio, max_repeat, dead_loops, act_entropy
DIAG = [
    # name,               ScanRate, RepeatR, MaxRep, DeadLoop, ActEntropy
    ("Random",              0.500,   0.420,  28.0,    0.30,   2.95),
    ("Deterministic",       0.500,   0.000,   1.0,    0.00,   0.00),
    ("Q-Learning",          0.027,   0.000,   1.0,    0.00,   2.59),
    ("PPO",                 0.027,   0.000,   1.0,    0.00,   2.59),
    ("DQN (orig)",          0.950,   0.932,  56.0,    1.00,   0.49),
    ("DDQN (orig)",         0.983,   0.933,  59.0,    1.00,   0.12),
    ("A2C",                 0.027,   0.000,   1.0,    0.00,   2.59),
    ("A3C-like/SB3",        0.027,   0.000,   1.0,    0.00,   2.59),
    ("DQN (fixed)",         0.033,   0.030,   2.0,    0.00,   2.50),
    ("DDQN (fixed)",        0.040,   0.038,   2.0,    0.00,   2.48),
    ("A3C (PyTorch async)", 0.027,   0.000,   1.0,    0.00,   2.59),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Colour palette (publication-friendly, colour-blind safe)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PALETTE = {
    "baseline":   "#999999",
    "rule-based": "#377eb8",
    "on-policy":  "#4daf4a",
    "off-policy": "#e41a1c",
    "fixed":      "#ff7f00",
    "async":      "#984ea3",
}

def agent_colour(name):
    n = name.lower()
    if "random" in n:          return PALETTE["baseline"]
    if "deterministic" in n:   return PALETTE["rule-based"]
    if "q-learning" in n:      return PALETTE["rule-based"]
    if "ppo" in n:             return PALETTE["on-policy"]
    if "a2c" in n:             return PALETTE["on-policy"]
    if "a3c-like" in n:        return PALETTE["on-policy"]
    if "a3c (pytorch" in n:    return PALETTE["async"]
    if "dqn (orig" in n:       return PALETTE["off-policy"]
    if "ddqn (orig" in n:      return PALETTE["off-policy"]
    if "dqn (fixed" in n:      return PALETTE["fixed"]
    if "ddqn (fixed" in n:     return PALETTE["fixed"]
    return "#aaaaaa"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TABLE 1 – Main results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Deduplicate: prefer "fixed env" over "orig env" for display; keep only one row per algorithm family
DISPLAY_ORDER = [
    "Random",
    "Deterministic HARMer",
    "Q-Learning (orig env)",
    "PPO (orig env)",
    "A2C (orig env)",
    "DQN (orig) (orig env)",
    "DDQN (orig) (orig env)",
    "DQN (fixed env)",
    "DDQN (fixed env)",
    "A3C-like/SB3 (orig env)",
    "A3C (PyTorch async, fixed env)",
]

result_map = {r["name"]: r for r in results}

print("\n" + "=" * 76)
print("  TABLE 1 — Main Results  (30 episodes per agent)")
print("=" * 76)
hdr = f"{'Algorithm':<40} {'SR%':>6} {'Mean Reward':>14} {'Avg Steps':>10}"
print(hdr)
print("-" * 76)

table1_rows = []
for name in DISPLAY_ORDER:
    if name not in result_map:
        continue
    r = result_map[name]
    ms_s = f"{r['ms']:>8.1f}" if not np.isnan(r['ms']) else "       N/A"
    print(f"{r['name']:<40} {r['sr']:>6.1f} {r['mr']:>+10.2f}±{r['std']:<6.2f} {ms_s}")
    table1_rows.append(r)
print("=" * 76)

table1_csv = os.path.join(OUT_DIR, "table1_main.csv")
with open(table1_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Algorithm", "SR%", "MeanReward", "StdReward", "AvgSteps"])
    for r in table1_rows:
        ms_val = "" if np.isnan(r['ms']) else f"{r['ms']:.1f}"
        w.writerow([r['name'], f"{r['sr']:.1f}", f"{r['mr']:.2f}", f"{r['std']:.2f}", ms_val])
print(f"[✓] Table 1 saved → {table1_csv}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TABLE 2 – Before / After fix comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX_PAIRS = [
    ("DQN",  "DQN (orig) (orig env)",  "DQN (fixed env)"),
    ("DDQN", "DDQN (orig) (orig env)", "DDQN (fixed env)"),
]

print("\n" + "=" * 76)
print("  TABLE 2 — Before vs After Scan-Idempotency Fix")
print("=" * 76)
print(f"{'Agent':<8} {'Before SR%':>10} {'Before Rew':>13} {'After SR%':>10} {'After Rew':>13} {'ΔΔSR':>6}")
print("-" * 76)

table2_rows = []
for label, before_key, after_key in FIX_PAIRS:
    b = result_map.get(before_key)
    a = result_map.get(after_key)
    if b and a:
        delta = a['sr'] - b['sr']
        print(f"{label:<8} {b['sr']:>10.1f} {b['mr']:>+10.2f}±{b['std']:<4.2f}  "
              f"{a['sr']:>10.1f} {a['mr']:>+10.2f}±{a['std']:<4.2f}  {delta:>+5.1f}")
        table2_rows.append([label, b['sr'], b['mr'], b['std'], a['sr'], a['mr'], a['std'], delta])
print("=" * 76)

table2_csv = os.path.join(OUT_DIR, "table2_fix.csv")
with open(table2_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "Before_SR", "Before_MR", "Before_Std",
                "After_SR", "After_MR", "After_Std", "DeltaSR"])
    w.writerows(table2_rows)
print(f"[✓] Table 2 saved → {table2_csv}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Plot helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":   150,
    "savefig.bbox": "tight",
    "savefig.dpi":  300,
})

SHORT_NAMES = {
    "Random":                              "Random",
    "Deterministic HARMer":               "Det.\nHARMer",
    "Q-Learning (orig env)":              "Q-Learn",
    "PPO (orig env)":                      "PPO",
    "A2C (orig env)":                      "A2C",
    "DQN (orig) (orig env)":              "DQN\n(orig)",
    "DDQN (orig) (orig env)":             "DDQN\n(orig)",
    "DQN (fixed env)":                    "DQN\n(fixed)",
    "DDQN (fixed env)":                   "DDQN\n(fixed)",
    "A3C-like/SB3 (orig env)":            "A3C-like\n(SB3)",
    "A3C (PyTorch async, fixed env)":     "A3C\n(async)",
}

plot_rows = [result_map[n] for n in DISPLAY_ORDER if n in result_map]
labels    = [SHORT_NAMES.get(r['name'], r['name']) for r in plot_rows]
colours   = [agent_colour(r['name']) for r in plot_rows]
x         = np.arange(len(plot_rows))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 1 – Success Rate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig1, ax1 = plt.subplots(figsize=(11, 4.5))
bars = ax1.bar(x, [r['sr'] for r in plot_rows], color=colours, edgecolor="white", linewidth=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, ha="center")
ax1.set_ylabel("Success Rate (%)")
ax1.set_title("Fig. 1 — Agent Success Rate on Complex Pentest Environment (30 episodes)")
ax1.set_ylim(0, 115)
ax1.axhline(100, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
for bar, r in zip(bars, plot_rows):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             f"{r['sr']:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

legend_patches = [
    mpatches.Patch(color=PALETTE["baseline"],   label="Baseline (Random)"),
    mpatches.Patch(color=PALETTE["rule-based"], label="Rule-based / Q-Learning"),
    mpatches.Patch(color=PALETTE["on-policy"],  label="On-policy (PPO / A2C / A3C-like)"),
    mpatches.Patch(color=PALETTE["off-policy"], label="Off-policy – original env (DQN / DDQN)"),
    mpatches.Patch(color=PALETTE["fixed"],      label="Off-policy – fixed env"),
    mpatches.Patch(color=PALETTE["async"],      label="Async A3C (PyTorch)"),
]
ax1.legend(handles=legend_patches, fontsize=8, loc="lower right")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "fig1_success_rate.pdf"))
fig1.savefig(os.path.join(OUT_DIR, "fig1_success_rate.png"))
print("[✓] Fig 1 saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 2 – Mean Reward ± σ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig2, ax2 = plt.subplots(figsize=(11, 4.5))
mrs  = [r['mr']  for r in plot_rows]
stds = [r['std'] for r in plot_rows]
bars2 = ax2.bar(x, mrs, color=colours, edgecolor="white", linewidth=0.8)
ax2.errorbar(x, mrs, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1)
ax2.set_xticks(x); ax2.set_xticklabels(labels, ha="center")
ax2.set_ylabel("Mean Episode Reward")
ax2.set_title("Fig. 2 — Mean Reward ± σ per Agent (30 episodes)")
ax2.axhline(0, color="black", linewidth=0.5, alpha=0.4)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig2_mean_reward.pdf"))
fig2.savefig(os.path.join(OUT_DIR, "fig2_mean_reward.png"))
print("[✓] Fig 2 saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 3 – Average Steps-to-Goal (successful only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Only show agents that actually succeeded at least once
succ_rows   = [(lab, r, c) for lab, r, c in zip(labels, plot_rows, colours)
               if not np.isnan(r['ms'])]
xs_  = np.arange(len(succ_rows))
fig3, ax3 = plt.subplots(figsize=(9, 4.5))
bars3 = ax3.bar(xs_, [r['ms'] for _, r, _ in succ_rows],
                color=[c for _, _, c in succ_rows], edgecolor="white")
ax3.set_xticks(xs_); ax3.set_xticklabels([l for l, _, _ in succ_rows], ha="center")
ax3.set_ylabel("Average Steps to Reach Crown-Jewel Node")
ax3.set_title("Fig. 3 — Efficiency: Steps to Goal (successful episodes only)")
ax3.axhline(3, color="#984ea3", linestyle="--", linewidth=1, alpha=0.7, label="Optimal (3 steps)")
for bar, (_, r, _) in zip(bars3, succ_rows):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f"{r['ms']:.1f}", ha="center", va="bottom", fontsize=8)
ax3.legend(fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fig3_steps_to_goal.pdf"))
fig3.savefig(os.path.join(OUT_DIR, "fig3_steps_to_goal.png"))
print("[✓] Fig 3 saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 4 – Before vs After DQN / DDQN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig4, axes4 = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
fig4.suptitle("Fig. 4 — DQN & DDQN: Before vs After Scan-Idempotency Fix", fontsize=11)

METRIC_PAIRS = [
    (0, "SR (%)",         "sr",                       "sr"),
    (1, "Mean Reward",    "mr",                       "mr"),
]

for ax_idx, metric_label, bkey, akey in METRIC_PAIRS:
    ax = axes4[ax_idx]
    group_labels = ["DQN", "DDQN"]
    bvals, avals = [], []
    for label, before_key, after_key in FIX_PAIRS:
        b = result_map.get(before_key)
        a = result_map.get(after_key)
        bvals.append(b[bkey] if b else 0)
        avals.append(a[akey] if a else 0)

    g = np.arange(len(group_labels))
    w = 0.35
    b1 = ax.bar(g - w/2, bvals, w, color=PALETTE["off-policy"], label="Before fix", edgecolor="white")
    b2 = ax.bar(g + w/2, avals, w, color=PALETTE["fixed"],      label="After fix",  edgecolor="white")
    ax.set_xticks(g); ax.set_xticklabels(group_labels)
    ax.set_title(metric_label)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}" if metric_label == "SR (%)" else f"{h:+.0f}",
                ha="center", va="bottom", fontsize=8)

fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, "fig4_dqn_fix.pdf"))
fig4.savefig(os.path.join(OUT_DIR, "fig4_dqn_fix.png"))
print("[✓] Fig 4 saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 5 – Diagnostic heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
diag_agents  = [d[0] for d in DIAG]
diag_metrics = ["Scan Rate", "Repeat Ratio", "Max Repeat\n(norm)", "Dead Loop\nRate", "Act Entropy\n(norm)"]

# Normalise each column to [0, 1] for heatmap colour
mat_raw = np.array([[d[1], d[2], d[3], d[4], d[5]] for d in DIAG], dtype=float)
mat_norm = np.zeros_like(mat_raw)
for col in range(mat_raw.shape[1]):
    c = mat_raw[:, col]
    mn, mx = c.min(), c.max()
    mat_norm[:, col] = 0 if mx == mn else (c - mn) / (mx - mn)

fig5, ax5 = plt.subplots(figsize=(9, 5.5))
im = ax5.imshow(mat_norm.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

ax5.set_xticks(range(len(diag_agents)));  ax5.set_xticklabels(diag_agents, rotation=30, ha="right", fontsize=8)
ax5.set_yticks(range(len(diag_metrics))); ax5.set_yticklabels(diag_metrics, fontsize=9)
ax5.set_title("Fig. 5 — Diagnostic Heatmap: Dead-Loop Behaviour Metrics\n"
              "(darker red = worse; values are column-normalised)", fontsize=10)

# Annotate raw values
for i in range(len(diag_agents)):
    for j in range(len(diag_metrics)):
        val = mat_raw[i, j]
        txt = f"{val:.2f}" if j != 2 else f"{val:.0f}"
        ax5.text(i, j, txt, ha="center", va="center", fontsize=7,
                 color="white" if mat_norm[i, j] > 0.6 else "black")

fig5.colorbar(im, ax=ax5, label="Normalised score (0 = best, 1 = worst)", shrink=0.7)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, "fig5_diagnostic_heatmap.pdf"))
fig5.savefig(os.path.join(OUT_DIR, "fig5_diagnostic_heatmap.png"))
print("[✓] Fig 5 saved")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FIG 6 – Combined overview (4-panel layout for paper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig6, ((ax6a, ax6b), (ax6c, ax6d)) = plt.subplots(2, 2, figsize=(14, 9))
fig6.suptitle("RL Agents for Automated Penetration Testing — Summary Results", fontsize=13, fontweight="bold")

# Panel A – Success Rate
ax6a.bar(x, [r['sr'] for r in plot_rows], color=colours, edgecolor="white")
ax6a.set_xticks(x); ax6a.set_xticklabels(labels, ha="center", fontsize=7)
ax6a.set_ylabel("Success Rate (%)"); ax6a.set_title("(a) Success Rate")
ax6a.set_ylim(0, 115); ax6a.axhline(100, color="black", ls="--", lw=0.7, alpha=0.5)
ax6a.spines["top"].set_visible(False); ax6a.spines["right"].set_visible(False)

# Panel B – Mean Reward
ax6b.bar(x, [r['mr'] for r in plot_rows], color=colours, edgecolor="white")
ax6b.errorbar(x, [r['mr'] for r in plot_rows],
              yerr=[r['std'] for r in plot_rows],
              fmt="none", color="black", capsize=3, lw=0.9)
ax6b.set_xticks(x); ax6b.set_xticklabels(labels, ha="center", fontsize=7)
ax6b.set_ylabel("Mean Episode Reward"); ax6b.set_title("(b) Mean Reward ± σ")
ax6b.axhline(0, color="black", lw=0.5, alpha=0.4)
ax6b.spines["top"].set_visible(False); ax6b.spines["right"].set_visible(False)

# Panel C – Steps to Goal
ax6c.bar(xs_, [r['ms'] for _, r, _ in succ_rows],
         color=[c for _, _, c in succ_rows], edgecolor="white")
ax6c.set_xticks(xs_); ax6c.set_xticklabels([l for l, _, _ in succ_rows], ha="center", fontsize=7)
ax6c.set_ylabel("Avg Steps to Goal"); ax6c.set_title("(c) Efficiency (Steps to Crown Jewel)")
ax6c.axhline(3, color="#984ea3", ls="--", lw=1, alpha=0.7, label="Optimal")
ax6c.legend(fontsize=8)
ax6c.spines["top"].set_visible(False); ax6c.spines["right"].set_visible(False)

# Panel D – Diagnostic heatmap (compact)
im6 = ax6d.imshow(mat_norm.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax6d.set_xticks(range(len(diag_agents)));  ax6d.set_xticklabels(diag_agents, rotation=35, ha="right", fontsize=6.5)
ax6d.set_yticks(range(len(diag_metrics))); ax6d.set_yticklabels(diag_metrics, fontsize=7.5)
ax6d.set_title("(d) Dead-Loop Diagnostic Heatmap")
fig6.colorbar(im6, ax=ax6d, shrink=0.7, label="Normalised (0=best)")

# Legend for colours
legend_patches = [
    mpatches.Patch(color=PALETTE["baseline"],   label="Random baseline"),
    mpatches.Patch(color=PALETTE["rule-based"], label="Rule-based / Q-Learn"),
    mpatches.Patch(color=PALETTE["on-policy"],  label="On-policy (PPO/A2C/A3C-like)"),
    mpatches.Patch(color=PALETTE["off-policy"], label="Off-policy – original env"),
    mpatches.Patch(color=PALETTE["fixed"],      label="Off-policy – fixed env"),
    mpatches.Patch(color=PALETTE["async"],      label="Async A3C (PyTorch)"),
]
fig6.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8,
            bbox_to_anchor=(0.5, -0.02))

fig6.tight_layout(rect=[0, 0.04, 1, 1])
fig6.savefig(os.path.join(OUT_DIR, "fig6_combined.pdf"))
fig6.savefig(os.path.join(OUT_DIR, "fig6_combined.png"))
print("[✓] Fig 6 (combined 4-panel) saved")


print("\n" + "=" * 60)
print("  All paper figures saved to: paper_figures/")
print("  Files:")
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"    {fn}")
print("=" * 60)
