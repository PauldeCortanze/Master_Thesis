import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
from pandas.plotting import parallel_coordinates

# ── Config ───────────────────────────────────────────────────────────────────
Pc_list = [0.75, 0.78, 0.8, 0.82, 0.85]
Pm_list = [0.14, 0.16]
POP_SIZE = 20
METRIC   = "NPV_over_CAPEX"
TOP_N    = 10   # for box plots: top-N individuals per combo

DESIGN_VARS = [
    "clearance", "sp", "p_rated", "Nwt",
    "wind_MW_per_km2", "solar_MW", "surface_tilt", "surface_azimuth"
]

variables = {
        "clearance": {"var_type": "design", "limits": [10, 120], "types": "int"},
        "sp": {"var_type": "design", "limits": [200, 360], "types": "int"},
        "p_rated": {"var_type": "design", "limits": [5, 20], "types": "int"},
        "Nwt": {"var_type": "design", "limits": [5, 50], "types": "int"},
        "wind_MW_per_km2": {"var_type": "design", "limits": [1, 10], "types": "float"},
        "solar_MW": {"var_type": "design", "limits": [30, 200], "types": "float"},
        "surface_tilt": {"var_type": "design", "limits": [0, 90], "types": "float"},
        "surface_azimuth": {
            "var_type": "design",
            "limits": [150, 210],
            "types": "float",
        },
        "DC_AC_ratio": {"var_type": "fixed", "value": 1.479},
        "b_P": {"var_type": "fixed", "value": 20},
        "b_E_h": {"var_type": "fixed", "value": 4},
        "cost_of_battery_P_fluct_in_peak_price_ratio": {"var_type": "fixed", "value": 8.75},
    }

var_limits = {
    k: v["limits"]
    for k, v in variables.items()
    if v["var_type"] == "design"
}

# ── Load all data ─────────────────────────────────────────────────────────────
frames = []
combos = []  # list of (Pc, Pm) tuples actually loaded

for Pc in Pc_list:
    for Pm in Pm_list:
        if (Pc == 0.65 and Pm == 0.2) or (Pc == 0.65 and Pm == 0.25):
            continue
        df = pd.read_csv(f"results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv")
        df = df[df["success"] == 1].copy()
        df["generation"] = df["iteration"] // POP_SIZE
        df["combo"] = f"Pc{Pc}_Pm{Pm}"   # ← label replaces "seed"
        frames.append(df)
        combos.append((Pc, Pm))

all_data   = pd.concat(frames, ignore_index=True)
combo_strs = [f"Pc{Pc}_Pm{Pm}" for Pc, Pm in combos]   #
n_combos   = len(combo_strs)
colors     = cm.tab10(np.linspace(0, 1, n_combos))


# ── A1: Parallel coordinates — best individual per combo ─────────────────────
best_per_combo = (
    all_data.loc[all_data.groupby("combo")[METRIC].idxmin(),
                 DESIGN_VARS + ["combo"]]
    .reset_index(drop=True)
)

# Normalise each variable to [0,1] using defined limits
norm = best_per_combo.copy()
for v in DESIGN_VARS:
    vmin, vmax = var_limits[v]
    norm[v] = (best_per_combo[v] - vmin) / (vmax - vmin + 1e-12)


'''
fig, ax = plt.subplots(figsize=(13, 5))
parallel_coordinates(norm, class_column="combo", ax=ax,
                     color=colors, linewidth=2.5, alpha=0.85)

# Annotate raw values on top
for _, row in best_per_combo.iterrows():
    for j, v in enumerate(DESIGN_VARS):
        y_pos = norm.loc[norm["combo"] == row["combo"], v].values[0]
        ax.text(j, y_pos + 0.03, f"{row[v]:.2g}",
                ha="center", fontsize=6.5, alpha=0.8)

ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value", fontsize=10)
ax.set_title("A — Parallel Coordinates: best individual per (Pc, Pm) combo",
             fontsize=13, fontweight="bold")
ax.legend(title="Combo", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Plot/plotA_parallel_coords_NSGA2.png", dpi=150, bbox_inches="tight")
'''

# ── A2: Box plots — distribution of best-per-combo values across design vars ─
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

# Overlay individual combo points
for i, v in enumerate(DESIGN_VARS):
    x = np.random.normal(i + 1, 0.04, size=len(norm))
    ax.scatter(x, norm[v].values, color=cmap(i), edgecolors="black",
               s=40, zorder=3, linewidths=0.6)

ax.set_xticks(range(1, len(DESIGN_VARS) + 1))
ax.set_xticklabels(DESIGN_VARS, rotation=20, ha="right", fontsize=10)
ax.set_ylabel("Normalised value", fontsize=10)
# ax.set_title("A — Box Plots: best individual per (Pc, Pm) combo, across design variables (GA)",
#              fontsize=13, fontweight="bold")
ax.set_ylim(-0.1, 1.15)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("Plot/plotA_boxplots_NSGA2.png", dpi=150, bbox_inches="tight")

top_n_per_combo = all_data.sort_values(by=METRIC, ascending=False).groupby("combo").head(TOP_N)

# 2. Compute the mean and standard deviation for each design variable per combo
summary_stats = top_n_per_combo.groupby("combo")[DESIGN_VARS].agg(["mean", "std"])

# 3. Optional: Flatten the MultiIndex columns to make the final DataFrame flatter and easier to read
summary_stats.columns = [f"{var}_{stat}" for var, stat in summary_stats.columns]
summary_stats = summary_stats.reset_index()

# ── Compute the "Total" row (Across Everything) ──────────────────────────────
# This calculates the mean and std of all top-N individuals combined together
total_row = {"combo": "Total"}
for var in DESIGN_VARS:
    total_row[f"{var}_mean"] = top_n_per_combo[var].mean()
    total_row[f"{var}_std"] = top_n_per_combo[var].std()

# Append the Total row to the summary dataframe
summary_stats = pd.concat([summary_stats, pd.DataFrame([total_row])], ignore_index=True)

# ── View and save the results ────────────────────────────────────────────────
print(summary_stats)
summary_stats.to_csv("results_design_variables_summary.csv", index=False)



print("Done — saved 4 plots to Plot/")