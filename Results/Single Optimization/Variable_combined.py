import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── Config ───────────────────────────────────────────────────────────────────
POP_SIZE = 20
METRIC   = "NPV_over_CAPEX"

DESIGN_VARS = [
    "clearance", "sp", "p_rated", "Nwt",
    "wind_MW_per_km2", "solar_MW", "surface_tilt", "surface_azimuth"
]

# Slide-friendly formatted LaTeX labels for the X-axis
DISPLAY_LABELS = [
    r"Rotor tip clearance ($h$)",
    r"Turbine specific power ($sp$)",
    r"Turbine rated power ($P_{\mathrm{rated}}$)",
    r"Number of wind turbines ($N_{\mathrm{WT}}$)",
    r"Wind installation density ($\rho_W$)",
    r"PV installed capacity ($C_{\mathrm{PV}}$)",
    r"PV tilt angle ($\theta_{\mathrm{tilt}}$)",
    r"PV azimuth angle ($\theta_{\mathrm{azimuth}}$)"
]

# Uniform limits to bind both datasets between [0, 1] identically
var_limits = {
    "clearance": [10, 120],
    "sp": [200, 360],
    "p_rated": [5, 20],
    "Nwt": [5, 50],
    "wind_MW_per_km2": [1, 10],
    "solar_MW": [30, 200],
    "surface_tilt": [0, 90],
    "surface_azimuth": [150, 210]
}

# ── 1. Load and Clean GA Data ────────────────────────────────────────────────
Pc_list = [0.75, 0.78, 0.8, 0.82, 0.85]
Pm_list = [0.14, 0.16]

ga_frames = []
for Pc in Pc_list:
    for Pm in Pm_list:
        try:
            df = pd.read_csv(f"GA/Results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv")
            df = df[df["success"] == 1].copy()
            df["generation"] = df["iteration"] // POP_SIZE
            df["combo"] = f"Pc{Pc}_Pm{Pm}"
            ga_frames.append(df)
        except FileNotFoundError:
            continue

if ga_frames:
    all_data_ga = pd.concat(ga_frames, ignore_index=True)
    best_per_combo = (
        all_data_ga.loc[all_data_ga.groupby("combo")[METRIC].idxmin(), DESIGN_VARS + ["combo"]]
        .reset_index(drop=True)
    )
    norm_ga = best_per_combo.copy()
    for v in DESIGN_VARS:
        vmin, vmax = var_limits[v]
        norm_ga[v] = (best_per_combo[v] - vmin) / (vmax - vmin + 1e-12)
else:
    norm_ga = pd.DataFrame(columns=DESIGN_VARS)

# ── 2. Load and Clean PSO Data ───────────────────────────────────────────────
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

pso_frames = []
for s in SEEDS:
    try:
        df = pd.read_csv(f"PSO/Results/history_ALPSO_seed{s}_c12.0_c21.0.csv")
        df = df[df["success"] == 1].copy()
        df["generation"] = df["iteration"] // POP_SIZE
        df["seed"] = str(s)
        pso_frames.append(df)
    except FileNotFoundError:
        continue

if pso_frames:
    all_data_pso = pd.concat(pso_frames, ignore_index=True)
    best_per_seed = (
        all_data_pso.loc[all_data_pso.groupby("seed")[METRIC].idxmin(), DESIGN_VARS + ["seed"]]
        .reset_index(drop=True)
    )
    norm_pso = best_per_seed.copy()
    for v in DESIGN_VARS:
        vmin, vmax = var_limits[v]
        norm_pso[v] = (best_per_seed[v] - vmin) / (vmax - vmin + 1e-12)
else:
    norm_pso = pd.DataFrame(columns=DESIGN_VARS)

# ── 3. Build the Grouped Plot ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

# Establish side-by-side x positions
positions_ga  = np.arange(len(DESIGN_VARS)) * 2.0 - 0.35
positions_pso = np.arange(len(DESIGN_VARS)) * 2.0 + 0.35

# Professional presentation palette (High contrast & slide friendly)
color_ga  = "#2b5c8f"  # Elegant Slate Blue
color_pso = "#e69f00"  # Warm Amber Orange

# Plot GA Boxes & Jitter Points
if not norm_ga.empty:
    bp_ga = ax.boxplot(
        [norm_ga[v].values for v in DESIGN_VARS],
        positions=positions_ga, widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color="crimson", linewidth=2),
        whiskerprops=dict(linewidth=1.2, color="#444444"),
        capprops=dict(linewidth=1.2, color="#444444")
    )
    for patch in bp_ga["boxes"]:
        patch.set_facecolor(color_ga)
        patch.set_alpha(0.7)
    
    for i, v in enumerate(DESIGN_VARS):
        x = np.random.normal(positions_ga[i], 0.05, size=len(norm_ga))
        ax.scatter(x, norm_ga[v].values, color=color_ga, edgecolors="black",
                   s=40, zorder=3, linewidths=0.5, alpha=0.8)

# Plot PSO Boxes & Jitter Points
if not norm_pso.empty:
    bp_pso = ax.boxplot(
        [norm_pso[v].values for v in DESIGN_VARS],
        positions=positions_pso, widths=0.5, patch_artist=True, showfliers=False,
        medianprops=dict(color="crimson", linewidth=2),
        whiskerprops=dict(linewidth=1.2, color="#444444"),
        capprops=dict(linewidth=1.2, color="#444444")
    )
    for patch in bp_pso["boxes"]:
        patch.set_facecolor(color_pso)
        patch.set_alpha(0.7)
    
    for i, v in enumerate(DESIGN_VARS):
        x = np.random.normal(positions_pso[i], 0.05, size=len(norm_pso))
        ax.scatter(x, norm_pso[v].values, color=color_pso, edgecolors="black",
                   s=40, zorder=3, linewidths=0.5, alpha=0.8)

# Formatting adjustments
ax.set_xticks(np.arange(len(DESIGN_VARS)) * 2.0)
ax.set_xticklabels(DISPLAY_LABELS, rotation=15, ha="right", fontsize=11)
ax.set_ylabel("Normalised Value", fontsize=12, fontweight="bold")
ax.set_title("Variability of Optimal Design Variables: GA vs PSO", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(-0.05, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

# Legend Creation
legend_elements = [
    Patch(facecolor=color_ga, edgecolor='#333333', alpha=0.7, label='GA Best Solutions'),
    Patch(facecolor=color_pso, edgecolor='#333333', alpha=0.7, label='PSO Best Solutions')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)

plt.tight_layout()
plt.savefig("Plot_Combined_Variability.png", bbox_inches="tight")
print("Successfully generated 'Plot_Combined_Variability.png'")