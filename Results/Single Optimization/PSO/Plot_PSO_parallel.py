import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
POP_SIZE     = 20         # number of particles in the swarm
METRIC       = "NPV_over_CAPEX"
OUT_DIR      = "plots"     # output folder for convergence plot and GIFs

DESIGN_VARS = [
    "clearance", "sp", "p_rated", "Nwt",
    "wind_MW_per_km2", "solar_MW", "surface_tilt", "surface_azimuth"
]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all data ─────────────────────────────────────────────────────────────
frames = []
for s in SEEDS:
    df = pd.read_csv(f"Results/history_ALPSO_seed{s}_c12.0_c21.0.csv")
    df = df.iloc[1:].copy().reset_index(drop=True)
    df = df[df["success"] == 1].copy()
    df["particle_id"] = df["iteration"] % POP_SIZE
    df["generation"]  = df["iteration"] // POP_SIZE
    df["seed"] = s
    frames.append(df)

# Palette: one colour per seed
SEED_COLORS = plt.cm.tab10(np.linspace(0, 0.8, len(SEEDS)))

# ── 1. Combine all seeds ──────────────────────────────────────────────────────
all_data = pd.concat(frames, ignore_index=True)

# ── 2. Global PCA (fitted once on all seeds) ──────────────────────────────────
print("Fitting PCA …")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(all_data[DESIGN_VARS])
pca      = PCA(n_components=2)
coords   = pca.fit_transform(X_scaled)
all_data["PCA1"] = coords[:, 0]
all_data["PCA2"] = coords[:, 1]
var1, var2 = pca.explained_variance_ratio_ * 100

# PCA velocity arrows: where will each particle be NEXT generation?
all_data = all_data.sort_values(["seed", "particle_id", "generation"])
all_data["u_pca"] = all_data.groupby(["seed", "particle_id"])["PCA1"].shift(-1) - all_data["PCA1"]
all_data["v_pca"] = all_data.groupby(["seed", "particle_id"])["PCA2"].shift(-1) - all_data["PCA2"]

pca1_lim = (all_data["PCA1"].min() - 0.8, all_data["PCA1"].max() + 0.8)
pca2_lim = (all_data["PCA2"].min() - 0.8, all_data["PCA2"].max() + 0.8)

# ── 3. Convergence plot ───────────────────────────────────────────────────────
print("Building convergence plot …")

# Best (minimum) metric per generation per seed
gen_best_all = all_data.groupby(["seed", "generation"])[METRIC].min().reset_index()
gen_best_all.columns = ["seed", "generation", "best"]

# Running cumulative best so the curve is monotone decreasing
def running_best(grp):
    result = grp.copy()
    result["best"] = result["best"].cummin()
    result["seed"] = grp.name
    return result

gen_best_all = (
    gen_best_all.groupby("seed")[["generation", "best"]]
    .apply(running_best)
    .reset_index(drop=True)
)

# For mean ± std band we need a common generation index
all_gens   = sorted(gen_best_all["generation"].unique())
pivot      = - gen_best_all.pivot(index="generation", columns="seed", values="best")
mean_curve = pivot.mean(axis=1)
std_curve  = pivot.std(axis=1)

fig, ax = plt.subplots(figsize=(11, 6))

# Shaded band (mean ± std)
ax.fill_between(all_gens,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.18, color="steelblue", label="Mean ± std")

# One line per seed
for idx, seed in enumerate(SEEDS):
    sub = gen_best_all[gen_best_all["seed"] == seed]
    ax.plot(sub["generation"], - sub["best"],
            color=SEED_COLORS[idx], linewidth=1.6,
            alpha=0.85, label=f"Seed {seed}")

# Bold mean line on top
ax.plot(all_gens, mean_curve,
        color="steelblue", linewidth=2.8, linestyle="--",
        alpha=0.9, label="Mean", zorder=5)

ax.set_xlabel("Generation", fontsize=13)
ax.set_ylabel("NPV / CAPEX (best so far)", fontsize=13)
ax.set_title("PSO Convergence — Best NPV/CAPEX per Generation\n"
             "(cumulative best, all 8 seeds)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.85)
ax.grid(True, linestyle="--", alpha=0.45)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

conv_path = os.path.join(OUT_DIR, "pso_convergence.png")
plt.savefig(conv_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {conv_path}")

# ── 3B. Convergence plot — Best NPV/CAPEX over TIME (EGO-style, per generation) ──
print("Building convergence plot …")

fig, ax = plt.subplots(figsize=(11, 6))

for idx, seed in enumerate(SEEDS):
    sub = all_data[all_data["seed"] == seed].copy()

    # Per generation: best objective + time when generation finished
    gen_stats = sub.groupby("generation").agg(
        best_obj = (METRIC, "min"),
        time_min = ("time_sec", "max"),   # ← replace "time_sec" with your actual time column name
    ).reset_index()

    gen_stats["time_min"]    = gen_stats["time_min"] / 60.0
    gen_stats["best_so_far"] = - gen_stats["best_obj"].cummin()

    ax.plot(gen_stats["time_min"], gen_stats["best_so_far"],
            marker="o", markersize=5, linewidth=2, color=SEED_COLORS[idx], alpha=0.85,
            label=f"Seed {seed}")

ax.set_xlabel("Time (in min)", fontsize=13)
ax.set_ylabel("NPV / CAPEX (best value)", fontsize=13)
# ax.set_title("PSO Convergence — Best NPV/CAPEX over Time\n"
#              "(cumulative best per generation, all 8 seeds)",
#              fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.85)
ax.grid(True, linestyle="--", alpha=0.45)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

conv_path = os.path.join(OUT_DIR, "pso_convergence_time.png")
plt.savefig(conv_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"  Saved → {conv_path}")

# ── 4. Animated GIF per seed ──────────────────────────────────────────────────
print("Building animated GIFs (one per seed) …")

# Shared colormap range for metric colouring
metric_min = - all_data[METRIC].quantile(0.95)
metric_max = - all_data[METRIC].quantile(0.05)

for seed in SEEDS:
    print(f"  Seed {seed} …", end=" ", flush=True)
    sub      = all_data[all_data["seed"] == seed].copy()
    num_gens = int(sub["generation"].max())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spines[["top", "right"]].set_visible(False)

    # Persistent colourbar
    sm = cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=metric_min, vmax=metric_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("NPV / CAPEX", fontsize=10)

    def update(gen, ax=ax, sub=sub, num_gens=num_gens, seed=seed):
        ax.clear()
        curr = sub[sub["generation"] == gen]

        ax.scatter(
            curr["PCA1"], curr["PCA2"],
            c= - curr[METRIC], cmap="plasma",
            vmin=metric_min, vmax=metric_max,
            edgecolors="k", linewidths=0.5,
            s=90, zorder=3
        )

        # Velocity arrows → next generation
        valid = curr.dropna(subset=["u_pca", "v_pca"])
        if gen < num_gens and not valid.empty:
            ax.quiver(
                valid["PCA1"], valid["PCA2"],
                valid["u_pca"], valid["v_pca"],
                angles="xy", scale_units="xy", scale=1,
                color="crimson", alpha=0.55, width=0.004, zorder=2
            )

        ax.set_xlim(*pca1_lim)
        ax.set_ylim(*pca2_lim)
        ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=10)
        ax.set_title(
            f"PSO Swarm — Seed {seed}   |   Generation {gen} → {gen+1}\n"
            f"(8D design space projected to 2D via PCA)",
            fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.3)

    ani = FuncAnimation(fig, update,
                        frames=range(num_gens + 1),
                        interval=1400, repeat=True)

    gif_path = os.path.join(OUT_DIR, f"pso_swarm_seed{seed}.gif")
    ani.save(gif_path, writer="pillow", fps=0.72)
    plt.close()
    print(f"saved → {gif_path}")

# ── 4b. Combined GIF — all seeds overlaid in one single plot ──────────────────
print("Building combined GIF (all seeds overlaid) …")

num_gens = int(all_data["generation"].max())

fig, ax = plt.subplots(figsize=(9, 7))
ax.spines[["top", "right"]].set_visible(False)

sm = cm.ScalarMappable(cmap="plasma",
                       norm=plt.Normalize(vmin=metric_min, vmax=metric_max))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("NPV / CAPEX", fontsize=11)

def update_all(gen):
    ax.clear()
    # All particles from all seeds at this generation (POP_SIZE × n_seeds points)
    curr = all_data[all_data["generation"] == gen]

    ax.scatter(
        curr["PCA1"], curr["PCA2"],
        c=-curr[METRIC], cmap="plasma",
        vmin=metric_min, vmax=metric_max,
        edgecolors="k", linewidths=0.4,
        s=70, zorder=3
    )

    valid = curr.dropna(subset=["u_pca", "v_pca"])
    if gen < num_gens and not valid.empty:
        ax.quiver(
            valid["PCA1"], valid["PCA2"],
            valid["u_pca"], valid["v_pca"],
            angles="xy", scale_units="xy", scale=1,
            color="crimson", alpha=0.5, width=0.003, zorder=2
        )

    ax.set_xlim(*pca1_lim)
    ax.set_ylim(*pca2_lim)
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(
        f"PSO Swarm — All {len(SEEDS)} seeds overlaid   |   Generation {gen} → {gen+1}\n"
        f"({POP_SIZE * len(SEEDS)} particles per frame — 8D design space projected to 2D via PCA)",
        fontsize=11, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.3)

ani_all = FuncAnimation(fig, update_all,
                        frames=range(num_gens + 1),
                        interval=1400, repeat=True)

gif_all_path = os.path.join(OUT_DIR, "pso_swarm_all_seeds.gif")
ani_all.save(gif_all_path, writer="pillow", fps=0.72)
plt.close()
print(f"  saved → {gif_all_path}")

print("\n✓ All done.")
print(f"  Convergence plot : {conv_path}")
print(f"  GIFs per seed    : {OUT_DIR}/pso_swarm_seed{{1..8}}.gif")
print(f"  Combined GIF     : {gif_all_path}")

# ── 5. Print PCA loadings ─────────────────────────────────────────────────────
loadings = pd.DataFrame(
    pca.components_.T * np.sqrt(pca.explained_variance_),
    index=DESIGN_VARS, columns=["PC1", "PC2"]
)
print("\nPCA loadings (scaled):")
print(loadings.round(3))
