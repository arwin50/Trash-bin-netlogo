import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load and clean ─────────────────────────────────────────────
df = pd.read_csv("fuente_dataset.csv")
df = df[df["arrival-rate"] > 0]
df_final = df[df["ticks"] == 7500].copy()
df_time  = df.copy()
df_high  = df_time[df_time["arrival-rate"] == 5].copy()

bp_palette = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868"}
factors = ["laziness-factor", "litter-sensitivity-factor", "group-litter-factor"]
factor_labels = ["Laziness Factor", "Litter Sensitivity Factor", "Group Litter Factor"]

# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Main descriptive overview
# ══════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle(
    "Fuente Osmeña ABM — Littering Behavior Overview",
    fontsize=14, fontweight="bold"
)

# 1. Litter events by bin-placement
sns.boxplot(data=df_final, x="bin-placement", y="total-litter-events",
            hue="bin-placement", palette=bp_palette, legend=False,
            ax=axes[0, 0])
axes[0, 0].set_title("Litter Events by Bin Placement")
axes[0, 0].set_xlabel("Bin Placement Level")
axes[0, 0].set_ylabel("Total Litter Events")
axes[0, 0].set_xticklabels(["Sparse (1)", "Moderate (2)", "Dense (3)"])

# 2. Litter events by laziness-factor
sns.boxplot(data=df_final, x="laziness-factor", y="total-litter-events",
            hue="laziness-factor", palette="Blues_d", legend=False,
            ax=axes[0, 1])
axes[0, 1].set_title("Litter Events by Laziness Factor")
axes[0, 1].set_xlabel("Laziness Factor")
axes[0, 1].set_ylabel("Total Litter Events")

# 3. Litter events by arrival-rate
sns.boxplot(data=df_final, x="arrival-rate", y="total-litter-events",
            hue="arrival-rate", palette="Oranges_d", legend=False,
            ax=axes[0, 2])
axes[0, 2].set_title("Litter Events by Arrival Rate")
axes[0, 2].set_xlabel("Arrival Rate")
axes[0, 2].set_ylabel("Total Litter Events")

# 4. Bin utilization by bin-placement
sns.boxplot(data=df_final, x="bin-placement", y="bin-utilization-rate",
            hue="bin-placement", palette=bp_palette, legend=False,
            ax=axes[1, 0])
axes[1, 0].set_title("Bin Utilization Rate by Bin Placement")
axes[1, 0].set_xlabel("Bin Placement Level")
axes[1, 0].set_ylabel("Bin Utilization Rate")
axes[1, 0].set_xticklabels(["Sparse (1)", "Moderate (2)", "Dense (3)"])

# 5. Litter events by litter-sensitivity-factor
sns.boxplot(data=df_final, x="litter-sensitivity-factor", y="total-litter-events",
            hue="litter-sensitivity-factor", palette="Greens_d", legend=False,
            ax=axes[1, 1])
axes[1, 1].set_title("Litter Events by Litter Sensitivity")
axes[1, 1].set_xlabel("Litter Sensitivity Factor")
axes[1, 1].set_ylabel("Total Litter Events")

# 6. Litter events by group-litter-factor
sns.boxplot(data=df_final, x="group-litter-factor", y="total-litter-events",
            hue="group-litter-factor", palette="Purples_d", legend=False,
            ax=axes[1, 2])
axes[1, 2].set_title("Litter Events by Group Litter Factor")
axes[1, 2].set_xlabel("Group Litter Factor")
axes[1, 2].set_ylabel("Total Litter Events")

plt.tight_layout()
plt.savefig("fig1_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Interactions: bin-placement x each behavioral factor
# ══════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle(
    "Interaction: Bin Placement × Behavioral Factors",
    fontsize=14, fontweight="bold"
)

for ax, factor, label in zip(axes, factors, factor_labels):
    means = df_final.groupby(["bin-placement", factor])[
        "total-litter-events"].mean().reset_index()
    stds = df_final.groupby(["bin-placement", factor])[
        "total-litter-events"].std().reset_index()
    for bp in sorted(df_final["bin-placement"].unique()):
        sub_m = means[means["bin-placement"] == bp]
        sub_s = stds[stds["bin-placement"] == bp]
        ax.plot(sub_m[factor], sub_m["total-litter-events"],
                marker="o", linewidth=2.5,
                color=bp_palette[bp], label=f"Bin Placement {bp}")
        ax.fill_between(sub_m[factor],
                        sub_m["total-litter-events"] - sub_s["total-litter-events"],
                        sub_m["total-litter-events"] + sub_s["total-litter-events"],
                        alpha=0.12, color=bp_palette[bp])
    ax.set_title(f"Bin Placement × {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Mean Litter Events")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("fig2_interactions.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Bin placement gap across laziness levels
# Shows how much MORE littering sparse bins cause vs dense bins
# as laziness increases
# ══════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    "Bin Placement Effect Gap Across Laziness Levels",
    fontsize=14, fontweight="bold"
)

# Left: absolute means per bin level per laziness value
means_laz = df_final.groupby(["bin-placement", "laziness-factor"])[
    "total-litter-events"].mean().reset_index()
for bp in sorted(df_final["bin-placement"].unique()):
    sub = means_laz[means_laz["bin-placement"] == bp]
    axes[0].plot(sub["laziness-factor"], sub["total-litter-events"],
                 marker="o", linewidth=2.5,
                 color=bp_palette[bp], label=f"Bin Placement {bp}")
axes[0].set_title("Mean Litter Events by\nLaziness × Bin Placement")
axes[0].set_xlabel("Laziness Factor")
axes[0].set_ylabel("Mean Litter Events")
axes[0].legend()

# Right: gap between bin levels at each laziness value
pivot = means_laz.pivot(index="laziness-factor",
                        columns="bin-placement",
                        values="total-litter-events")
laz_vals = [str(v) for v in sorted(df_final["laziness-factor"].unique())]

# Gap 1 vs 2, 1 vs 3, 2 vs 3
gap_combos = []
bp_levels = sorted(df_final["bin-placement"].unique())
for i in range(len(bp_levels)):
    for j in range(i+1, len(bp_levels)):
        bp_a, bp_b = bp_levels[i], bp_levels[j]
        if bp_a in pivot.columns and bp_b in pivot.columns:
            gap = pivot[bp_a] - pivot[bp_b]
            gap_combos.append((f"BP{bp_a} − BP{bp_b}", gap.values))

x = np.arange(len(laz_vals))
width = 0.25
gap_colors = ["#e74c3c", "#9b59b6", "#3498db"]
for idx, (label, gap_vals) in enumerate(gap_combos):
    axes[1].bar(x + idx * width, gap_vals, width,
                label=label, color=gap_colors[idx])

axes[1].set_title("Litter Difference Between\nBin Placement Levels")
axes[1].set_xlabel("Laziness Factor")
axes[1].set_ylabel("Mean Litter Difference")
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(laz_vals)
axes[1].legend(fontsize=8)
axes[1].axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig("fig3_gap_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Time series: litter accumulation over time
# ══════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle(
    "Litter Accumulation Over Time (arrival-rate=5)",
    fontsize=14, fontweight="bold"
)

# By bin-placement
ts_bp = df_high.groupby(["ticks", "bin-placement"])[
    "total-litter-events"].mean().reset_index()
for bp in sorted(df_high["bin-placement"].unique()):
    sub = ts_bp[ts_bp["bin-placement"] == bp]
    axes[0].plot(sub["ticks"], sub["total-litter-events"],
                 marker="o", markersize=3,
                 color=bp_palette[bp], label=f"Bin Placement {bp}")
axes[0].set_title("By Bin Placement")
axes[0].set_xlabel("Ticks")
axes[0].set_ylabel("Mean Cumulative Litter Events")
axes[0].legend(fontsize=8)

# By laziness-factor
ts_laz = df_high.groupby(["ticks", "laziness-factor"])[
    "total-litter-events"].mean().reset_index()
palette_laz = {0.0: "#2166ac", 0.5: "#f4a582", 1.0: "#d6604d"}
for val in [0.0, 0.5, 1.0]:
    sub = ts_laz[ts_laz["laziness-factor"] == val]
    axes[1].plot(sub["ticks"], sub["total-litter-events"],
                 marker="o", markersize=3,
                 color=palette_laz[val], label=f"Laziness={val}")
axes[1].set_title("By Laziness Factor")
axes[1].set_xlabel("Ticks")
axes[1].set_ylabel("Mean Cumulative Litter Events")
axes[1].legend(fontsize=8)

# Heatmap: sensitivity × group factor (both ns, shows flatness)
means_ns = df_final.groupby(
    ["litter-sensitivity-factor", "group-litter-factor"]
)["total-litter-events"].mean().reset_index()
pivot_ns = means_ns.pivot(
    index="litter-sensitivity-factor",
    columns="group-litter-factor",
    values="total-litter-events"
)
sns.heatmap(pivot_ns, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=axes[2], cbar_kws={"label": "Mean Litter Events"})
axes[2].set_title("Sensitivity × Group Factor\n(both ns — confirms no effect)")
axes[2].set_xlabel("Group Litter Factor")
axes[2].set_ylabel("Litter Sensitivity Factor")

plt.tight_layout()
plt.savefig("fig4_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# Export clean CSV for R
# ══════════════════════════════════════════════════════════════
df_final.to_csv("fuente_clean_for_R.csv", index=False)
print("All figures saved. Clean CSV exported as fuente_clean_for_R.csv")
print(f"\nFinal dataset: {len(df_final)} rows")
print(f"Bin placements: {sorted(df_final['bin-placement'].unique())}")
print(f"Arrival rates: {sorted(df_final['arrival-rate'].unique())}")