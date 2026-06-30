import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# ── 1. Load data ──────────────────────────────────────────────────────────────
csv_path = "Results/history_NSGA_seed0_ew0.0.csv"
df = pd.read_csv(csv_path)

# Keep only successful runs
df = df[df["success"] == 1].copy()

obj1 = "NPV_over_CAPEX"          # maximise
obj2 = "annual_carbon_offset"    # maximise (more offset = better)

df = df[[obj1, obj2]].dropna().reset_index(drop=True)

# Stored as negated values — flip sign so both objectives are maximised
df[obj1] = -df[obj1]
df[obj2] = -df[obj2]

# ── 2. Pareto-dominance (maximise both objectives) ────────────────────────────
def is_pareto_efficient(costs):
    """
    Returns a boolean mask: True where the point is Pareto-efficient.
    Assumes MAXIMISATION of all columns.
    """
    n = len(costs)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        # A point j dominates i if it is >= in all objectives and > in at least one
        dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
        dominated[i] = False          # a point doesn't dominate itself
        if dominated.any():
            efficient[i] = False
    return efficient

values = df[[obj1, obj2]].to_numpy(dtype=float)
pareto_mask = is_pareto_efficient(values)

df["pareto"] = pareto_mask

pareto_pts = df[pareto_mask].sort_values(obj1)
dominated_pts = df[~pareto_mask]

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# Dominated points
ax.scatter(
    dominated_pts[obj1], dominated_pts[obj2],
    c="#a8c8e8", edgecolors="#5a8ab0", s=40, linewidths=0.4,
    alpha=0.85, zorder=3, label="Dominated solutions"
)

# Pareto-front points
ax.scatter(
    pareto_pts[obj1], pareto_pts[obj2],
    c="#e84545", edgecolors="#8b0000", s=50, linewidths=0.6,
    alpha=0.95, zorder=5, label="Pareto-front solutions"
)

# Staircase line connecting the Pareto front
# ax.plot(
#     pareto_pts[obj1], pareto_pts[obj2], color="#c0392b", linewidth=1.5,
#     alpha=0.6, zorder=4
# )

# Labels & formatting
ax.set_xlabel("NPV / CAPEX", fontsize=12)
ax.set_ylabel("Annual Carbon Offset  [tonne CO₂ eq.]", fontsize=12)
# Values were stored negated in the CSV; signs have been flipped for plotting
ax.set_title("Pareto Front — NPV/CAPEX vs Annual Carbon Offset", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

out_path = "Plots/pareto_front_NSGA.png"
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()