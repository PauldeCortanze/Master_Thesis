import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── 1. Load data ──────────────────────────────────────────────────────────────
df_pareto_pso  = pd.read_csv("PSO/pareto_front_VEPSO_seed0_ew0.0.csv").sort_values("NPV_over_CAPEX").reset_index(drop=True)

NPV           = -df_pareto_pso["NPV_over_CAPEX"].values
carbon_offset = -df_pareto_pso["annual_carbon_offset"].values

df_history = pd.read_csv("GA/Results/history_NSGA_seed0_ew0.0.csv")

# Compute generation from the raw solution iteration index
PopSize = 20
df_history["generation"] = df_history["iteration"] // PopSize

# Flip signs as in your original script to get positive objective values
df_history["NPV"] = -df_history["NPV_over_CAPEX"]
df_history["carbon_offset"] = -df_history["annual_carbon_offset"]

# Filter for successful (feasible) solutions
df_feasible = df_history[df_history["success"] == 1].copy()


# ── Helper: Extract the True Pareto Front ─────────────────────────────────────
def get_pareto_front(df, x_col, y_col):
    """Extracts non-dominated solutions (assuming maximization for both objectives)."""
    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[False, False])
    pareto_points = []
    max_y = float("-inf")

    for _, row in sorted_df.iterrows():
        if row[y_col] >= max_y:
            pareto_points.append(row)
            max_y = row[y_col]

    pareto_df = pd.DataFrame(pareto_points)
    return pareto_df.sort_values(by=x_col)

# Extract Pareto front points for plotting lines
pareto_front = get_pareto_front(df_feasible, "NPV", "carbon_offset")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Pareto front (Standalone)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6))

# NSGA Pareto front (blue)
ax1.plot(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="#1f77b4",      # blue
    linewidth=1.5,
    alpha=0.9,
    zorder=1,
    label="GA Pareto front"
)

ax1.scatter(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="#4C9ED9",      # lighter blue
    s=55,
    zorder=2,
    edgecolors="#0B4F8A", # dark blue border
    linewidths=0.5,
)

# VEPSO Pareto front (red)
ax1.plot(
    NPV,
    carbon_offset,
    color="#d62728",      # red
    linewidth=1.5,
    alpha=0.9,
    zorder=1,
    label="VEPSO Pareto front"
)

ax1.scatter(
    NPV,
    carbon_offset,
    color="#F06A6A",      # lighter red
    s=55,
    zorder=2,
    edgecolors="#A11D1D", # dark red border
    linewidths=0.5,
)

ax1.set_xlabel("NPV / CAPEX", fontsize=12)
ax1.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq./yr)", fontsize=12)
# ax1.set_title("VEPSO Multi-Objective PSO — Pareto Front", fontsize=13)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.tick_params(labelsize=10)
ax1.spines[["top", "right"]].set_visible(False)

plt.legend(fontsize=10, loc="upper right", frameon=True, framealpha=0.9)
plt.tight_layout()
out1 = "pareto_fronts_GA_VEPSO.png"
plt.savefig(out1, dpi=150)
print(f"Saved → {out1}")