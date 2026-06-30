from matplotlib.pylab import seed
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
Pc_list = [0.75, 0.78, 0.8, 0.82, 0.85]
Pm_list = [0.14, 0.16]
Seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
POP_SIZE = 20
METRIC   = "NPV_over_CAPEX"

colors = cm.tab10(np.linspace(0, 1, len(Pc_list) * len(Pm_list)))

# ═════════════════════════════════════════════════════════════════════════════
# Plot 1 — Best per GENERATION  (original, unchanged)
# ═════════════════════════════════════════════════════════════════════════════
results_gen = {}
for Pc in Pc_list:
    for Pm in Pm_list:
        df = pd.read_csv(f"results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv")
        df["generation"] = df["iteration"] // POP_SIZE
        best = (
            df[df["success"] == 1]
            .groupby("generation")[METRIC]
            .min()
            .reset_index()
        )
        best["best_so_far"] = -best[METRIC].cummin()
        results_gen[(Pc, Pm)] = best

all_gens = sorted(set(g for b in results_gen.values() for g in b["generation"]))
aligned_gen = {}
for (Pc, Pm), best in results_gen.items():
    s = best.set_index("generation")["best_so_far"].reindex(all_gens).ffill()
    aligned_gen[(Pc, Pm)] = s

matrix_gen = np.vstack([s.values for s in aligned_gen.values()])
mean_gen   = matrix_gen.mean(axis=0)
std_gen    = matrix_gen.std(axis=0)

fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(all_gens, mean_gen - std_gen, mean_gen + std_gen,
                alpha=0.18, color="steelblue", label="Mean ± std")
for ((Pc, Pm), s), color in zip(aligned_gen.items(), colors):
    ax.plot(all_gens, s.values, linewidth=1.6, color=color, alpha=0.85,
            label=f"Pc={Pc}, Pm={Pm}")
ax.plot(all_gens, mean_gen, linestyle="--", linewidth=2.2,
        color="steelblue", label="Mean", zorder=5)

ax.set_xlabel("Generation", fontsize=13)
ax.set_ylabel("NPV / CAPEX (best value)", fontsize=13)
ax.set_title("NSGA2 Convergence — Best NPV/CAPEX per Generation\n"
             "(cumulative best, all parameter combinations)",
             fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=2, fontsize=9, framealpha=0.85, loc="lower right")
plt.tight_layout()
plt.savefig("Plot/best_across_seeds_nsga2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → Plot/best_across_seeds_nsga2.png")

# ═════════════════════════════════════════════════════════════════════════════
# Plot 2 — Cumulative best over TIME  (EGO-style: per generation)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

for i, ((Pc, Pm), color) in enumerate(zip(
    [(Pc, Pm) for Pc in Pc_list for Pm in Pm_list], colors
)):
    df = pd.read_csv(f"results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv")
    df["generation"] = df["iteration"] // POP_SIZE

    ok = df[df["success"] == 1].copy()

    # Per generation: best objective value + time to complete the generation
    gen_stats = ok.groupby("generation").agg(
        best_obj  = (METRIC, "min"),        # best (most negative) raw value
        time_min  = ("time_sec", "max"),    # time when generation finished
    ).reset_index()

    gen_stats["time_min"]    = gen_stats["time_min"] / 60.0
    gen_stats["best_so_far"] = -gen_stats["best_obj"].cummin()

    ax.plot(gen_stats["time_min"], gen_stats["best_so_far"],
            marker="o", markersize=5, linewidth=2, color=color, alpha=0.85,
            label=f"seed {i}")

ax.set_xlabel("Time (in min)", fontsize=13)
ax.set_ylabel("NPV / CAPEX (best value)", fontsize=13)
# ax.set_title("NSGA2 Convergence — Best NPV/CAPEX over Time\n"
#              "(cumulative best per generation, all parameter combinations)",
#              fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=2, fontsize=9, framealpha=0.85, loc="lower right")
plt.tight_layout()
plt.savefig("Plot/best_over_time_nsga2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → Plot/best_over_time_nsga2.png")