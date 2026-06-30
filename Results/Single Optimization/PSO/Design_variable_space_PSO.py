import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
from pandas.plotting import parallel_coordinates

# ── Config ───────────────────────────────────────────────────────────────────
SEEDS    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
POP_SIZE = 20
METRIC   = "NPV_over_CAPEX"
TOP_N    = 10   # for box plots: top-N individuals per seed

DESIGN_VARS = [
    "clearance", "sp", "p_rated", "Nwt",
    "wind_MW_per_km2", "solar_MW", "surface_tilt", "surface_azimuth"
]

# ── Load all data ─────────────────────────────────────────────────────────────
frames = []
for s in SEEDS:
    df = pd.read_csv(f"Results/history_ALPSO_seed{s}_c12.0_c21.0.csv")
    df = df[df["success"] == 1].copy()
    df["generation"] = df["iteration"] // POP_SIZE
    df["seed"] = str(s)   # string for categorical coloring
    frames.append(df)

all_data = pd.concat(frames, ignore_index=True)
colors   = cm.tab10(np.linspace(0, 1, len(SEEDS)))
seed_strs = [str(s) for s in SEEDS]

# ── A: Parallel coordinates — best individual per seed ────────────────────────
best_per_seed = (
    all_data.loc[all_data.groupby("seed")[METRIC].idxmin(), DESIGN_VARS + ["seed"]]
    .reset_index(drop=True)
)
# Normalise each variable to [0,1] for readability
norm = best_per_seed.copy()
for v in DESIGN_VARS:
    vmin, vmax = all_data[v].min(), all_data[v].max()
    norm[v] = (best_per_seed[v] - vmin) / (vmax - vmin + 1e-12)

'''
fig, ax = plt.subplots(figsize=(13, 5))
parallel_coordinates(norm, class_column="seed", ax=ax,
                     color=colors, linewidth=2.5, alpha=0.85)

# Annotate raw values on top
for _, row in best_per_seed.iterrows():
    for j, v in enumerate(DESIGN_VARS):
        ax.text(j, norm.loc[norm["seed"] == row["seed"], v].values[0] + 0.03,
                f"{row[v]:.2g}", ha="center", fontsize=6.5, alpha=0.8)

ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value", fontsize=10)
ax.set_title("A — Parallel Coordinates: best individual per seed",
             fontsize=13, fontweight="bold")
ax.legend(title="Seed", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot/plotA_parallel_coords_pso.png", dpi=150, bbox_inches="tight")
# plt.show()
'''

# ── A2: Box plots — distribution of best-per-seed values across design vars ──
fig, ax = plt.subplots(figsize=(13, 5))
 
bp = ax.boxplot(
    [norm[v].values for v in DESIGN_VARS],
    patch_artist=True,
    notch=False,
    medianprops=dict(color="crimson", linewidth=2),
    whiskerprops=dict(linewidth=1.4),
    capprops=dict(linewidth=1.4),
    flierprops=dict(marker="o", markersize=5, alpha=0.6),
)
 
cmap = plt.get_cmap("tab10", len(DESIGN_VARS))
for patch, i in zip(bp["boxes"], range(len(DESIGN_VARS))):
    patch.set_facecolor(cmap(i))
    patch.set_alpha(0.7)
 
# Overlay individual seed points with jitter
for i, v in enumerate(DESIGN_VARS):
    x = np.random.normal(i + 1, 0.04, size=len(norm))
    ax.scatter(x, norm[v].values, color=cmap(i), edgecolors="black",
               s=40, zorder=3, linewidths=0.6)
 
ax.set_xticks(range(1, len(DESIGN_VARS) + 1))
ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value", fontsize=10)
# ax.set_title("A2 — Box Plots: best individual per seed, across design variables (PSO)",
#              fontsize=13, fontweight="bold")
ax.set_ylim(-0.1, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plot/plotA2_boxplots_pso.png", dpi=150, bbox_inches="tight")

top_n_per_seed = all_data.sort_values(by=METRIC, ascending=False).groupby("seed").head(TOP_N)

# 2. Compute the mean and standard deviation for each design variable per seed
summary_stats = top_n_per_seed.groupby("seed")[DESIGN_VARS].agg(["mean", "std"])

# 3. Optional: Flatten the MultiIndex columns to make the final DataFrame flatter and easier to read
summary_stats.columns = [f"{var}_{stat}" for var, stat in summary_stats.columns]
summary_stats = summary_stats.reset_index()

# ── Compute the "Total" row (Across Everything) ──────────────────────────────
# This calculates the mean and std of all top-N individuals combined together
total_row = {"seed": "Total"}
for var in DESIGN_VARS:
    total_row[f"{var}_mean"] = top_n_per_seed[var].mean()
    total_row[f"{var}_std"] = top_n_per_seed[var].std()

# Append the Total row to the summary dataframe
summary_stats = pd.concat([summary_stats, pd.DataFrame([total_row])], ignore_index=True)

# ── View and save the results ────────────────────────────────────────────────
print(summary_stats)
summary_stats.to_csv("results_design_variables_summary.csv", index=False)

'''
# ── B: Scatter matrix (pairplot) — all individuals ───────────────────────────
n = len(DESIGN_VARS)
fig, axes = plt.subplots(n, n, figsize=(18, 18))

for i, vi in enumerate(DESIGN_VARS):
    for j, vj in enumerate(DESIGN_VARS):
        ax = axes[i][j]
        if i == j:
            # Diagonal: KDE / histogram per seed
            for seed, color in zip(seed_strs, colors):
                sub = all_data[all_data["seed"] == seed][vi]
                ax.hist(sub, bins=15, color=color, alpha=0.45, density=True)
            ax.set_ylabel("")
        else:
            for seed, color in zip(seed_strs, colors):
                sub = all_data[all_data["seed"] == seed]
                ax.scatter(sub[vj], sub[vi], color=color,
                           s=6, alpha=0.35, linewidths=0)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if i == n - 1:
            ax.set_xlabel(vj, fontsize=8, rotation=15, ha="right")
        if j == 0:
            ax.set_ylabel(vi, fontsize=8)

# Shared legend
handles = [plt.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=8, label=f"Seed {s}")
           for s, c in zip(SEEDS, colors)]
fig.legend(handles=handles, title="Seed", loc="upper right",
           fontsize=9, title_fontsize=10, framealpha=0.9)
fig.suptitle("B — Scatter Matrix: all individuals, coloured by seed",
             fontsize=14, fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig("plot/plotB_scatter_matrix_pso.png", dpi=150, bbox_inches="tight")
# plt.show()

# ── C: Variable evolution over generations — best individual per generation ───
fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
axes = axes.flatten()

for idx, var in enumerate(DESIGN_VARS):
    ax = axes[idx]
    for seed, color in zip(seed_strs, colors):
        sub = all_data[all_data["seed"] == seed]
        best_gen = (
            sub.loc[sub.groupby("generation")[METRIC].idxmin()]
            .sort_values("generation")
        )
        ax.plot(best_gen["generation"], best_gen[var],
                marker="o", markersize=3, linewidth=1.6,
                color=color, label=f"Seed {seed}", alpha=0.85)
    ax.set_ylabel(var, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Seed", loc="center right",
           fontsize=9, title_fontsize=10, bbox_to_anchor=(1.0, 0.5))
fig.supxlabel("Generation", fontsize=11)
fig.suptitle("C — Variable Evolution: best individual per generation, per seed",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig("plot/plotC_var_evolution_pso.png", dpi=150, bbox_inches="tight")
# plt.show()

# ── D: Box plots per variable — top-N individuals per seed ────────────────────
fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharey=False)
axes = axes.flatten()

for idx, var in enumerate(DESIGN_VARS):
    ax = axes[idx]
    box_data = []
    for seed in seed_strs:
        sub = all_data[all_data["seed"] == seed]
        topn = sub.nsmallest(TOP_N, METRIC)[var].values  # min = best for -NPV
        box_data.append(topn)

    bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(SEEDS) + 1))
    ax.set_xticklabels([f"S{s}" for s in SEEDS], fontsize=8)
    ax.set_ylabel(var, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"D — Box Plots: top-{TOP_N} individuals per seed, per variable",
             fontsize=13, fontweight="bold")
fig.supxlabel("Seed", fontsize=11)
plt.tight_layout()
plt.savefig("plot/plotD_boxplots_pso.png", dpi=150, bbox_inches="tight")
plt.show()
'''

print("Done — saved 4 plots to plot/")