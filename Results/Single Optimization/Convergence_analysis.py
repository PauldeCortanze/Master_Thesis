import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
# GA Config
Pc_list  = [0.75, 0.78, 0.8, 0.82, 0.85]
Pm_list  = [0.14, 0.16]
GA_POP   = 20

# PSO Config
PSO_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
PSO_POP   = 20

METRIC    = "NPV_over_CAPEX"

# ═════════════════════════════════════════════════════════════════════════════
# 1. Process GA Data (Best per Gen + true timestamp)
# ═════════════════════════════════════════════════════════════════════════════
ga_runs = {}
for Pc in Pc_list:
    for Pm in Pm_list:
        try:
            df_ga = pd.read_csv(f"GA/Results/history_NSGA2_seed0_Pc{Pc}_Pm{Pm}.csv")
            df_ga["generation"] = df_ga["iteration"] // GA_POP
            ok_ga = df_ga[df_ga["success"] == 1].copy()
            
            gen_stats = ok_ga.groupby("generation").agg(
                best_obj = (METRIC, "min"),
                time_min = ("time_sec", "max")
            ).reset_index()
            
            gen_stats["time_min"] = gen_stats["time_min"] / 60.0
            ga_runs[(Pc, Pm)] = gen_stats
        except FileNotFoundError:
            continue

all_ga_gens = sorted(set(g for df in ga_runs.values() for g in df["generation"]))
ga_timeline = []

for gen in all_ga_gens:
    best_obj_this_gen = float('inf')
    corresponding_time = None
    
    for (Pc, Pm), df in ga_runs.items():
        row = df[df["generation"] == gen]
        if not row.empty:
            current_obj = row["best_obj"].values[0]
            if current_obj < best_obj_this_gen:
                best_obj_this_gen = current_obj
                corresponding_time = row["time_min"].values[0]
                
    if corresponding_time is not None:
        ga_timeline.append({
            "generation": gen,
            "time_min": corresponding_time,
            "best_obj": best_obj_this_gen
        })

df_ga_final = pd.DataFrame(ga_timeline, columns=["generation", "time_min", "best_obj"])
df_ga_final = df_ga_final.sort_values("generation")
df_ga_final["best_so_far"] = -df_ga_final["best_obj"].cummin()

# ═════════════════════════════════════════════════════════════════════════════
# 2. Process PSO Data (Best per Gen + true timestamp)
# ═════════════════════════════════════════════════════════════════════════════
pso_runs = {}
for s in PSO_SEEDS:
    try:
        df_pso = pd.read_csv(f"PSO/Results/history_ALPSO_seed{s}_c12.0_c21.0.csv")
        df_pso = df_pso.iloc[1:].copy().reset_index(drop=True)
        df_pso = df_pso[df_pso["success"] == 1].copy()
        df_pso["generation"] = df_pso["iteration"] // PSO_POP
        
        gen_stats = df_pso.groupby("generation").agg(
            best_obj = (METRIC, "min"),
            time_min = ("time_sec", "max")
        ).reset_index()
        
        gen_stats["time_min"] = gen_stats["time_min"] / 60.0
        pso_runs[s] = gen_stats
    except FileNotFoundError:
        continue

all_pso_gens = sorted(set(g for df in pso_runs.values() for g in df["generation"]))
pso_timeline = []

for gen in all_pso_gens:
    best_obj_this_gen = float('inf')
    corresponding_time = None
    
    for s, df in pso_runs.items():
        row = df[df["generation"] == gen]
        if not row.empty:
            current_obj = row["best_obj"].values[0]
            if current_obj < best_obj_this_gen:
                best_obj_this_gen = current_obj
                corresponding_time = row["time_min"].values[0]
                
    if corresponding_time is not None:
        pso_timeline.append({
            "generation": gen,
            "time_min": corresponding_time,
            "best_obj": best_obj_this_gen
        })

df_pso_final = pd.DataFrame(pso_timeline, columns=["generation", "time_min", "best_obj"])
df_pso_final = df_pso_final.sort_values("generation")
df_pso_final["best_so_far"] = -df_pso_final["best_obj"].cummin()

# ═════════════════════════════════════════════════════════════════════════════
# 3. Process EGO Data
# ═════════════════════════════════════════════════════════════════════════════
df_ego = pd.read_csv("EGO/Results_EGO.csv")

meta_cols = ["Iteration", "Eval", "New simulations", "Time [min]"]
all_cols = df_ego.columns.tolist()
obj_col  = [c for c in all_cols if c not in meta_cols][0]

iter_time = df_ego.groupby("Iteration")["Time [min]"].first()
cum_time  = iter_time.cumsum() 
iter_best    = df_ego.groupby("Iteration")[obj_col].min()
running_best = iter_best.cummin()

shared_idx   = iter_best.index.intersection(cum_time.index)
ego_time     = cum_time.loc[shared_idx].values
ego_best     = -running_best.loc[shared_idx].values

time_mask = ego_time <= 210
ego_time  = ego_time[time_mask]
ego_best  = ego_best[time_mask]

# ═════════════════════════════════════════════════════════════════════════════
# 4. Plot Comparison Figure
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

# Plot GA
if not df_ga_final.empty:
    ax.plot(df_ga_final["time_min"], df_ga_final["best_so_far"],
            marker="o", markersize=4, linewidth=2, color="#1f77b4", 
            label="Genetic Algorithm (Best per Generation)")

# Plot PSO
if not df_pso_final.empty:
    ax.plot(df_pso_final["time_min"], df_pso_final["best_so_far"],
            marker="s", markersize=4, linewidth=2, color="#2ca02c", 
            label="Particle Swarm (Best per Iteration)")

# Plot EGO
ax.plot(ego_time, ego_best,
        marker="x", markersize=8, linewidth=2, color="#d33215", label="EGO")

# Styling
ax.set_xlabel("Time (in min)", fontsize=12)
ax.set_ylabel("NPV / CAPEX (best value)", fontsize=12)
ax.set_title("Convergence Comparison", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, loc="lower right", framealpha=0.9)

plt.tight_layout()
os.makedirs("Plot", exist_ok=True)
plt.savefig("comprehensive_convergence_comparison.png", dpi=180, bbox_inches="tight")
plt.show()
print("Saved → comprehensive_convergence_comparison.png")