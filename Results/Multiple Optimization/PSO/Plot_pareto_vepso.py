import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── load ──────────────────────────────────────────────────────────────────────
df_pareto  = pd.read_csv("pareto_front_VEPSO_seed0_ew0.0.csv").sort_values("NPV_over_CAPEX").reset_index(drop=True)
df_history = pd.read_csv("swarm_history_VEPSO_seed0_ew0.0.csv")

NPV           = -df_pareto["NPV_over_CAPEX"].values
carbon_offset = -df_pareto["annual_carbon_offset"].values

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Pareto front (Standalone)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(NPV, carbon_offset, color="#1D9E75", linewidth=1.2, alpha=0.4, zorder=1)
ax1.scatter(NPV, carbon_offset, color="#1D9E75", s=55, zorder=2,
            edgecolors="#0F6E56", linewidths=0.5)

ax1.set_xlabel("NPV / CAPEX", fontsize=12)
ax1.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq./yr)", fontsize=12)
# ax1.set_title("VEPSO Multi-Objective PSO — Pareto Front", fontsize=13)
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.tick_params(labelsize=10)
ax1.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out1 = "plot/pareto_front_VEPSO_ew0.0_only.png"
plt.savefig(out1, dpi=150)
print(f"Saved → {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: All particles coloured by swarm (Standalone)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

df_h = df_history.copy()
df_h["NPV_over_CAPEX"]       = -df_h["NPV_over_CAPEX"]
df_h["annual_carbon_offset"] = -df_h["annual_carbon_offset"]

n_iter = df_h["iteration"].max()
n_part = df_h["particle"].nunique()

df_inf = df_h[df_h["feasible"] == 0]
df_s0  = df_h[(df_h["feasible"] == 1) & (df_h["swarm"] == 0)]
df_s1  = df_h[(df_h["feasible"] == 1) & (df_h["swarm"] == 1)]

# ax2.scatter(df_inf["NPV_over_CAPEX"], df_inf["annual_carbon_offset"],
#             c="lightgrey", s=8, alpha=0.35, zorder=1,
#             label=f"Infeasible ({len(df_inf)})")

ax2.scatter(df_s0["NPV_over_CAPEX"], df_s0["annual_carbon_offset"],
            color="#2A6DB5", s=12, alpha=0.55, zorder=2, edgecolors="none",
            label=f"Swarm 0 — NPV")

ax2.scatter(df_s1["NPV_over_CAPEX"], df_s1["annual_carbon_offset"],
            color="#C0392B", s=12, alpha=0.55, zorder=2, edgecolors="none",
            label=f"Swarm 1 — Carbon offset")

ax2.plot(NPV, carbon_offset, color="white",   linewidth=2.5, zorder=3)
ax2.plot(NPV, carbon_offset, color="#1D9E75", linewidth=1.5, zorder=4,
         label="Pareto front")

ax2.set_xlabel("NPV / CAPEX", fontsize=12)
ax2.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq./yr)", fontsize=12)
# ax2.set_title(f"Particle Trajectories — {n_iter} iter × {n_part} particles × 2 swarms",
#               fontsize=13, fontweight="bold")
ax2.set_title("Pareto Front VEPSO - NPV/CAPEX vs Annual Carbon Offset", fontsize=14, fontweight="bold")
ax2.legend(fontsize=9, loc="best", framealpha=0.9)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
# ax2.set_xlim(-0.05, 0.9)
# ax2.set_ylim(150000, 470000)
ax2.tick_params(labelsize=10)

plt.tight_layout()
out2 = "plot/swarm_history_VEPSO_ew0.0.png"
plt.savefig(out2, dpi=150)
print(f"Saved → {out2}")

plt.show()