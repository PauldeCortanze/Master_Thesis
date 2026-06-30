import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── Load and Preprocess Data ──────────────────────────────────────────────────
# Load the actual CSV file
df_history = pd.read_csv("Results/history_NSGA_seed0_ew0.0.csv")

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

# Plot the computed Pareto front line and points
ax1.plot(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="#1D9E75",
    linewidth=1.5,
    alpha=0.8,
    zorder=1,
)
ax1.scatter(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="#1D9E75",
    s=55,
    zorder=2,
    edgecolors="#0F6E56",
    linewidths=0.5,
)

ax1.set_xlabel("NPV / CAPEX", fontsize=12)
ax1.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq./yr)", fontsize=12)
ax1.set_title("Pareto Front — NPV/CAPEX vs Annual Carbon Offset",
              fontsize=14, fontweight="bold")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.tick_params(labelsize=10)
ax1.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = "Plots/pareto_front_NSGA_ew0.0_only.png"
plt.savefig(out1, dpi=150)
print(f"Saved → {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: All solutions from the single population (history)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

# Highlight infeasible/unsuccessful solutions if any exist
df_inf = df_history[df_history["success"] == 0]
if not df_inf.empty:
    ax2.scatter(
        df_inf["NPV"],
        df_inf["carbon_offset"],
        color="lightgrey",
        s=8,
        alpha=0.35,
        zorder=1,
        label=f"Infeasible ({len(df_inf)})",
    )

# All successful iterations plotted together as a single blue population
ax2.scatter(
    df_feasible["NPV"],
    df_feasible["carbon_offset"],
    color="#2A6DB5",
    s=12,
    alpha=0.55,
    zorder=2,
    edgecolors="none",
    label=f"Evaluated solutions",
)

# Overlay the final non-dominated Pareto front curve
ax2.plot(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="white",
    linewidth=2.5,
    zorder=3,
)
ax2.plot(
    pareto_front["NPV"],
    pareto_front["carbon_offset"],
    color="#1D9E75",
    linewidth=1.5,
    zorder=4,
    label="Pareto front",
)

ax2.set_xlabel("NPV / CAPEX", fontsize=12)
ax2.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq./yr)", fontsize=12)
ax2.set_title("Pareto Front NSGA2 - NPV/CAPEX vs Annual Carbon Offset",
              fontsize=14, fontweight="bold")
ax2.legend(fontsize=9, loc="best", framealpha=0.9)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(labelsize=10)

plt.tight_layout()
out2 = "Plots/history_NSGA_ew0.0.png"
plt.savefig(out2, dpi=150)
print(f"Saved → {out2}")

plt.show()
