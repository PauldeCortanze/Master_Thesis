import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────────
METRIC   = "NPV_over_CAPEX"
POP_SIZE = 20  # used to compute generation for GA / PSO

DESIGN_VARS = [
    "clearance", "sp", "p_rated", "Nwt",
    "wind_MW_per_km2", "solar_MW", "surface_tilt", "surface_azimuth"
]

# Defined design variable bounds (used for [0,1] normalisation)
VAR_LIMITS = {
    "clearance":        [10,  120],
    "sp":               [200, 360],
    "p_rated":          [5,   20],
    "Nwt":              [5,   50],
    "wind_MW_per_km2":  [1,   10],
    "solar_MW":         [30,  200],
    "surface_tilt":     [0,   90],
    "surface_azimuth":  [150, 210],
}

# ── EGO column rename map ─────────────────────────────────────────────────────
EGO_COL_MAP = {
    "clearance [m]":            "clearance",
    "sp [W/m2]":                "sp",
    "p_rated [MW]":             "p_rated",
    "Nwt":                      "Nwt",
    "wind_MW_per_km2 [MW/km2]": "wind_MW_per_km2",
    "solar_MW [MW]":            "solar_MW",
    "surface_tilt [deg]":       "surface_tilt",
    "surface_azimuth [deg]":    "surface_azimuth",
}

# ── GA combos ─────────────────────────────────────────────────────────────────
Pc_list = [0.75, 0.78, 0.80, 0.82, 0.85]
Pm_list = [0.14, 0.16]

# ── PSO seeds ─────────────────────────────────────────────────────────────────
PSO_SEEDS = list(range(10))


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

frames = []

# ── EGO ───────────────────────────────────────────────────────────────────────
ego_df = pd.read_csv("EGO/Results_EGO.csv")
ego_df.columns = ego_df.columns.str.strip()
ego_df = ego_df.rename(columns=EGO_COL_MAP)
ego_df = ego_df.rename(columns={"Iteration": "iteration"})
ego_df = ego_df.dropna(subset=[METRIC]).copy()
ego_df["algorithm"] = "EGO"
ego_df["run_id"]    = "EGO_run0"
ego_df["generation"] = ego_df["iteration"]   # EGO: 1 eval per iteration
frames.append(ego_df)

print(f"EGO     : {len(ego_df):>5} rows loaded")

# ── GA ────────────────────────────────────────────────────────────────────────
ga_count = 0
for Pc in Pc_list:
    for Pm in Pm_list:
        path = f"GA/Results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv"
        try:
            df = pd.read_csv(path)
            df = df[df["success"] == 1].copy()
            df["generation"] = df["iteration"] // POP_SIZE
            df["algorithm"]  = "GA"
            df["run_id"]     = f"GA_Pc{Pc}_Pm{Pm}"
            frames.append(df)
            ga_count += len(df)
        except FileNotFoundError:
            print(f"  WARNING: not found — {path}")

print(f"GA      : {ga_count:>5} rows loaded")

# ── PSO ───────────────────────────────────────────────────────────────────────
pso_count = 0
for s in PSO_SEEDS:
    path = f"PSO/Results/history_ALPSO_seed{s}_c12.0_c21.0.csv"
    try:
        df = pd.read_csv(path)
        df = df[df["success"] == 1].copy()
        df["generation"] = df["iteration"] // POP_SIZE
        df["algorithm"]  = "PSO"
        df["run_id"]     = f"PSO_seed{s}"
        frames.append(df)
        pso_count += len(df)
    except FileNotFoundError:
        print(f"  WARNING: not found — {path}")

print(f"PSO     : {pso_count:>5} rows loaded")

# ── Combine ───────────────────────────────────────────────────────────────────
all_data = pd.concat(frames, ignore_index=True)
all_data = all_data.dropna(subset=DESIGN_VARS + [METRIC]).copy()

print(f"\nCombined: {len(all_data):>5} rows  |  "
      f"algorithms: {all_data['algorithm'].unique().tolist()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. NORMALISE TO [0, 1] USING DEFINED VARIABLE LIMITS
# ══════════════════════════════════════════════════════════════════════════════

norm_data = all_data[DESIGN_VARS].copy()
for v in DESIGN_VARS:
    lo, hi = VAR_LIMITS[v]
    norm_data[v] = (all_data[v] - lo) / (hi - lo + 1e-12)

# Clip to [-0.05, 1.05] to catch minor out-of-bounds without discarding rows
norm_data = norm_data.clip(-0.05, 1.05)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PCA  (StandardScaler applied on top of the normalised values)
# ══════════════════════════════════════════════════════════════════════════════

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(norm_data)

pca        = PCA(n_components=len(DESIGN_VARS))
X_pca      = pca.fit_transform(X_scaled)

explained  = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\nExplained variance per PC:")
for i, (ev, cv) in enumerate(zip(explained, cumulative), 1):
    print(f"  PC{i}: {ev*100:5.1f}%   cumulative: {cv*100:5.1f}%")

print("\nPC1 loadings:")
for var, loading in zip(DESIGN_VARS, pca.components_[0]):
    print(f"  {var:<20s}  {loading:+.4f}")
 
print("\nPC2 loadings:")
for var, loading in zip(DESIGN_VARS, pca.components_[1]):
    print(f"  {var:<20s}  {loading:+.4f}")

# Attach PC scores back to metadata
pca_df = all_data[["algorithm", "run_id", "generation", METRIC]].copy().reset_index(drop=True)
for i in range(len(DESIGN_VARS)):
    pca_df[f"PC{i+1}"] = X_pca[:, i]


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

algo_colors = {"EGO": "#e6194b", "GA": "#3cb44b", "PSO": "#4363d8"}
algos       = ["EGO", "GA", "PSO"]

# ── Shared PC axis limits (same scale across all three figures) ───────────────
pc1_all = pca_df["PC1"].values
pc2_all = pca_df["PC2"].values
pad = 0.5
pc1_lim = (pc1_all.min() - pad, pc1_all.max() + pad)
pc2_lim = (pc2_all.min() - pad, pc2_all.max() + pad)

# ── Plot 1: Scree plot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, len(DESIGN_VARS) + 1), explained * 100,
       color="#4C72B0", alpha=0.75, label="Individual")
ax.plot(range(1, len(DESIGN_VARS) + 1), cumulative * 100,
        marker="o", color="crimson", linewidth=2, label="Cumulative")
ax.axhline(90, color="grey", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Explained Variance [%]", fontsize=11)
ax.set_title("Scree Plot — Combined EGO + GA + PSO", fontsize=13, fontweight="bold")
ax.set_xticks(range(1, len(DESIGN_VARS) + 1))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pca_scree.png", dpi=150, bbox_inches="tight")
print("\nSaved pca_scree.png")

# ── Plots 2–4: One figure per algorithm, coloured by generation ───────────────
#
# Colour scale: early iterations → dark (viridis low end)
#               late iterations  → bright/yellow (viridis high end)
#
# For GA (10 combos) and PSO (10 seeds) each run has its own generation counter
# starting at 0, so we normalise within each run_id and then scale to the
# global [0, max_generation] range of that algorithm so the colourbar is
# comparable across runs.

algo_labels = {
    "EGO": "Iteration",   # EGO has no population, so label as iteration
    "GA":  "Generation",
    "PSO": "Generation",
}

for algo in algos:
    sub = pca_df[pca_df["algorithm"] == algo].copy()

    # Use raw generation values; normalise per-run so every run spans [0, 1]
    # then re-scale to the maximum generation count of this algorithm so that
    # the colourbar reflects real iteration depth.
    max_gen = sub["generation"].max()

    # Normalise within each run to [0, max_gen] for a fair colour mapping
    def norm_gen(grp):
        g_min, g_max = grp["generation"].min(), grp["generation"].max()
        span = g_max - g_min if g_max > g_min else 1
        grp = grp.copy()
        grp["gen_norm"] = (grp["generation"] - g_min) / span * max_gen
        return grp

    sub = sub.groupby("run_id", group_keys=False).apply(norm_gen)

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        sub["PC1"], sub["PC2"],
        c=sub["gen_norm"],
        cmap="plasma",
        vmin=0, vmax=max_gen,
        s=10, alpha=0.55, linewidths=0,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(algo_labels[algo], fontsize=10)

    ax.set_xlim(pc1_lim)
    ax.set_ylim(pc2_lim)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=11)
    ax.set_title(
        f"PCA Projection — {algo}  (colored by {algo_labels[algo].lower()})",
        fontsize=13, fontweight="bold"
    )
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    fname = f"pca_pc1_pc2_{algo.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")

# ── Plot 5: Loadings / biplot arrows ─────────────────────────────────────────
loadings = pca.components_.T   # shape (n_vars, n_components)

fig, ax = plt.subplots(figsize=(7, 6))
for i, var in enumerate(DESIGN_VARS):
    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
             head_width=0.03, head_length=0.02,
             fc="#333333", ec="#333333", linewidth=1.4)
    ax.text(loadings[i, 0] * 1.12, loadings[i, 1] * 1.12,
            var, fontsize=9, ha="center", va="center")

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
circle = plt.Circle((0, 0), 1, color="grey", fill=False, linestyle="--", linewidth=1)
ax.add_patch(circle)
ax.axhline(0, color="grey", linewidth=0.8, alpha=0.5)
ax.axvline(0, color="grey", linewidth=0.8, alpha=0.5)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=11)
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=11)
ax.set_title("PCA Loadings — PC1 vs PC2", fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pca_loadings.png", dpi=150, bbox_inches="tight")
print("Saved pca_loadings.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. INDIVIDUAL SEED / RUN FIGURES
# ══════════════════════════════════════════════════════════════════════════════
# ── SPECIFIC RUNS TO PLOT INDIVIDUALLY ────────────────────────────────────────
SPECIFIC_GA_PC = 0.8     # Choose from Pc_list
SPECIFIC_GA_PM = 0.14     # Choose from Pm_list
SPECIFIC_PSO_SEED = 0     # Choose from PSO_SEEDS

# ── Target IDs ────────────────────────────────────────────────────────────────
target_ga_id  = f"GA_Pc{SPECIFIC_GA_PC}_Pm{SPECIFIC_GA_PM}"
target_pso_id = f"PSO_seed{SPECIFIC_PSO_SEED}"

# ── Helper to plot a single run ───────────────────────────────────────────────
def plot_single_run(run_id, algo_name, filename_suffix):
    single_df = pca_df[pca_df["run_id"] == run_id].copy()
    
    if single_df.empty:
        print(f"WARNING: No data found for run_id: {run_id}")
        return

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(
        single_df["PC1"], single_df["PC2"],
        c=single_df["generation"],
        cmap="plasma",  # Unique colormap to differentiate from the summary plots
        s=15, alpha=0.8, linewidths=0
    )
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    if algo_name == "GA Single Run":
        cbar.set_label("Generation", fontsize=10)
    else:
        cbar.set_label("Iteration", fontsize=10)
    
    ax.set_xlim(pc1_lim)
    ax.set_ylim(pc2_lim)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=11)
    # ax.set_title(f"PCA Space — {algo_name} ({run_id})", fontsize=12, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    
    plt.tight_layout()
    fname = f"pca_single_{filename_suffix}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")

# Generate the two requested single run figures
print("\nGenerating individual seed plots...")
plot_single_run(target_ga_id, "GA Single Run", f"ga_Pc{SPECIFIC_GA_PC}_Pm{SPECIFIC_GA_PM}")
plot_single_run(target_pso_id, "PSO Single Seed", f"pso_seed{SPECIFIC_PSO_SEED}")


# ── Helper to generate a 3x3 grid for the remaining 9 runs ────────────────────
def plot_3x3_grid(run_ids, title_prefix, filename, custom_titles=None):
    # A4 Page proportions: 8.27 x 11.69 inches
    fig, axes = plt.subplots(3, 3, figsize=(8.27, 10.5), sharex=True, sharey=True)
    axes = axes.flatten()
    
    sc = None
    for idx, run_id in enumerate(run_ids[:9]): # Strict limit to 9 subplots
        ax = axes[idx]
        sub_df = pca_df[pca_df["run_id"] == run_id].copy()
        
        if sub_df.empty:
            ax.text(0.5, 0.5, f"No Data\n{run_id}", ha='center', va='center', fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            continue
            
        sc = ax.scatter(
            sub_df["PC1"], sub_df["PC2"],
            c=sub_df["generation"],
            cmap="plasma",
            s=6, alpha=0.7, linewidths=0
        )
        
        # Determine subplot title: use custom title if provided, else fallback to cleaned run_id
        if custom_titles and idx < len(custom_titles):
            subplot_title = custom_titles[idx]
        else:
            subplot_title = run_id.replace("GA_", "").replace("PSO_", "").replace("_", " ")
            
        ax.set_title(subplot_title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlim(pc1_lim)
        ax.set_ylim(pc2_lim)
        ax.grid(linestyle="--", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        
    # Add common outer axis labels to save spatial overhead
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)", fontsize=11, labelpad=10)
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)", fontsize=11, labelpad=15)
    
    # Add a unified vertical colorbar mapping to the right edge of the sheet
    if sc is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='vertical')
        label_type = "Generation" if "GA" in title_prefix else "Iteration"
        cbar.set_label(f"{label_type}", fontsize=10, labelpad=10)
        cbar.ax.tick_params(labelsize=8)

    # plt.suptitle(f"{title_prefix} — Remaining 9 Runs", fontsize=13, fontweight="bold", y=0.96)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved grid figure: {filename}")


# ── Generate the two 3x3 grids ────────────────────────────────────────────────
print("\nGenerating 3x3 subfigure grids...")

# 1. Filter out the already isolated PSO Seed 0, and get the other 9
remaining_pso = [f"PSO_seed{s}" for s in PSO_SEEDS if f"PSO_seed{s}" != target_pso_id]
plot_3x3_grid(remaining_pso, "PSO Optimization Paths", "pca_grid_pso_remaining.png")

# 2. Filter out the already isolated GA combo, get the other 9, and pass custom "Seed X" titles
all_ga_combos = [f"GA_Pc{Pc}_Pm{Pm}" for Pc in Pc_list for Pm in Pm_list]
remaining_ga = [combo for combo in all_ga_combos if combo != target_ga_id]
ga_seed_titles = [f"Seed {i}" for i in range(1, 10)]

plot_3x3_grid(remaining_ga, "GA Parameter Combinations", "pca_grid_ga_remaining.png", custom_titles=ga_seed_titles)

print("\nDone.")