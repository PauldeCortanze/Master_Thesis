import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pandas.plotting import parallel_coordinates

# ── Config ────────────────────────────────────────────────────────────────────
METRIC = "NPV_over_CAPEX"
TOP_N  = 1

# CSV column name (with units) → short display name used in plots
DESIGN_VAR_MAP = {
    "clearance [m]":            "clearance",
    "sp [W/m2]":                "sp",
    "p_rated [MW]":             "p_rated",
    "Nwt":                      "Nwt",
    "wind_MW_per_km2 [MW/km2]": "wind_MW_per_km2",
    "solar_MW [MW]":            "solar_MW",
    "surface_tilt [deg]":       "surface_tilt",
    "surface_azimuth [deg]":    "surface_azimuth",
}
DESIGN_COLS = list(DESIGN_VAR_MAP.keys())    # actual CSV column names
DESIGN_VARS = list(DESIGN_VAR_MAP.values())  # short display names used throughout

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("Results_EGO.csv")

# Strip surrounding whitespace from column names
df.columns = df.columns.str.strip()

# Keep only rows with a valid objective
df = df.dropna(subset=[METRIC]).copy()

# Rename CSV columns → short display names
df = df.rename(columns=DESIGN_VAR_MAP)

# Rename Iteration → iteration
df = df.rename(columns={"Iteration": "iteration"})

# Clean integer index
df = df.reset_index(drop=True)

# Sanity check
missing = [v for v in DESIGN_VARS if v not in df.columns]
if missing:
    raise KeyError(f"These design variables are missing after rename: {missing}\n"
                   f"Available columns: {list(df.columns)}")

top_20_df = df.sort_values(by=METRIC, ascending=True).head(100)

# ── Helper ────────────────────────────────────────────────────────────────────
def best_per_group(data, group_col, metric, cols):
    """One row per group: the row with the lowest metric value."""
    all_cols = list(dict.fromkeys([group_col, metric] + cols))
    return (
        data[all_cols]
        .sort_values(metric)
        .groupby(group_col, sort=False)
        .first()
        .reset_index()
        [[group_col] + cols]
    )

best_per_iter = best_per_group(df, "iteration", METRIC, DESIGN_VARS)

norm = best_per_iter.copy()
for v in DESIGN_VARS:
    vmin, vmax = df[v].min(), df[v].max()
    norm[v] = (best_per_iter[v] - vmin) / (vmax - vmin + 1e-12)

norm_1 = top_20_df.copy()
for v in DESIGN_VARS:
    # We still use the global min/max of the entire df to keep the 0-1 scale context
    vmin, vmax = df[v].min(), df[v].max()
    norm_1[v] = (top_20_df[v] - vmin) / (vmax - vmin + 1e-12)

norm["iteration"] = norm["iteration"].astype(str)

n_iters  = norm["iteration"].nunique()
colors_A = cm.viridis(np.linspace(0, 1, n_iters))

# ── A: Parallel Coordinates — best individual per iteration ───────────────────
'''
fig, ax = plt.subplots(figsize=(13, 5))
parallel_coordinates(norm, class_column="iteration", ax=ax,
                     color=colors_A, linewidth=2.0, alpha=0.75)
ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value", fontsize=10)
ax.set_title("A — Parallel Coordinates: best individual per iteration (EGO)",
             fontsize=13, fontweight="bold")
ax.legend(title="Iteration", bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=7, ncol=2)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plotA_parallel_coords_ego.png", dpi=150, bbox_inches="tight")
print("Saved plotA")
'''

# ── A: Box Plots — best-per-iteration distribution across design variables ───
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

for i, v in enumerate(DESIGN_VARS):
    x = np.random.normal(i + 1, 0.04, size=len(norm))
    ax.scatter(x, norm[v].values, color=cmap(i), edgecolors="black",
               s=40, zorder=3, linewidths=0.6)

ax.set_xticks(range(1, len(DESIGN_VARS) + 1))
ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value  [0 = min limit, 1 = max limit]", fontsize=10)
ax.set_title("A2 — Box Plots: best individual per iteration, across design variables (EGO)",
             fontsize=13, fontweight="bold")
ax.set_ylim(-0.1, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("plotA_boxplots_ego.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved plotA")



'''
# ── B: Scatter Matrix — all individuals, coloured by NPV/CAPEX ───────────────
n = len(DESIGN_VARS)
fig, axes = plt.subplots(n, n, figsize=(18, 18))

metric_vals    = df[METRIC].values
vmin_m, vmax_m = metric_vals.min(), metric_vals.max()
norm_metric    = (metric_vals - vmin_m) / (vmax_m - vmin_m + 1e-12)
scatter_colors = cm.plasma(norm_metric)

for i, vi in enumerate(DESIGN_VARS):
    for j, vj in enumerate(DESIGN_VARS):
        ax = axes[i][j]
        if i == j:
            ax.hist(df[vi], bins=20, color="#4C72B0", alpha=0.6, density=True)
        else:
            ax.scatter(df[vj], df[vi], c=scatter_colors, s=6, alpha=0.5, linewidths=0)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        if i == n - 1:
            ax.set_xlabel(vj, fontsize=8, rotation=15, ha="right")
        if j == 0:
            ax.set_ylabel(vi, fontsize=8)

sm = plt.cm.ScalarMappable(cmap="plasma",
                            norm=plt.Normalize(vmin=vmin_m, vmax=vmax_m))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label(METRIC, fontsize=11)
fig.suptitle("B — Scatter Matrix: all individuals, coloured by NPV/CAPEX (EGO)",
             fontsize=14, fontweight="bold", y=1.002)
plt.savefig("plotB_scatter_matrix_ego.png", dpi=150, bbox_inches="tight")
print("Saved plotB")

# ── C: Variable evolution — best individual per iteration ─────────────────────
fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
axes = axes.flatten()

best_per_gen = (
    best_per_group(df, "iteration", METRIC, DESIGN_VARS + [METRIC])
    .sort_values("iteration")
)

for idx, var in enumerate(DESIGN_VARS):
    ax = axes[idx]
    ax.plot(best_per_gen["iteration"], best_per_gen[var],
            marker="o", markersize=4, linewidth=1.8,
            color="#2196F3", alpha=0.9)

    running       = best_per_gen[METRIC].cummin()
    new_best_mask = best_per_gen[METRIC] == running
    ax.scatter(best_per_gen.loc[new_best_mask, "iteration"],
               best_per_gen.loc[new_best_mask, var],
               color="crimson", zorder=5, s=35, label="New best")

    ax.set_ylabel(var, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(fontsize=8)
fig.supxlabel("Iteration", fontsize=11)
fig.suptitle("C — Variable Evolution: best individual per iteration (EGO)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plotC_var_evolution_ego.png", dpi=150, bbox_inches="tight")
print("Saved plotC")

# ── D: Box plots by iteration quartile ───────────────────────────────────────
n_iters_total = df["iteration"].nunique()
quartile_size = max(1, n_iters_total // 4)
iter_sorted   = sorted(df["iteration"].unique())

quartile_labels = []
quartile_data   = {v: [] for v in DESIGN_VARS}

for q in range(4):
    start   = q * quartile_size
    end     = (q + 1) * quartile_size if q < 3 else n_iters_total
    iters_q = iter_sorted[start:end]
    sub     = df[df["iteration"].isin(iters_q)]
    quartile_labels.append(f"Q{q+1}\n(it {iters_q[0]}-{iters_q[-1]})")
    for v in DESIGN_VARS:
        quartile_data[v].append(sub[v].values)

fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharey=False)
axes = axes.flatten()
q_colors = cm.Set2(np.linspace(0, 1, 4))

for idx, var in enumerate(DESIGN_VARS):
    ax = axes[idx]
    bp = ax.boxplot(quartile_data[var], patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], q_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(quartile_labels, fontsize=8)
    ax.set_ylabel(var, fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("D — Box Plots: design variable distributions by iteration quartile (EGO)",
             fontsize=13, fontweight="bold")
fig.supxlabel("Iteration quartile", fontsize=11)
plt.tight_layout()
plt.savefig("plotD_boxplots_ego.png", dpi=150, bbox_inches="tight")
print("Saved plotD")

plt.show()
'''

print("\nDone — saved plots")