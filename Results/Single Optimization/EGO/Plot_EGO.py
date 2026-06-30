# -*- coding: utf-8 -*-
"""
Plot EGO optimization results from Results_EGO.csv:
  1. Best objective value vs cumulative time
  2. PCA of all evaluated designs, colored by iteration
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------
# Load results
# -----------------
df = pd.read_csv("Results_EGO.csv")

# Columns that are not design variables
meta_cols = ["Iteration", "Eval", "New simulations", "Time [min]"]

# Detect objective column: first numeric column after meta columns
all_cols = df.columns.tolist()
obj_col  = [c for c in all_cols if c not in meta_cols][0]

# Design variable columns: everything after the objective
design_cols = [c for c in all_cols if c not in meta_cols + [obj_col]]

# -----------------
# Cumulative time per iteration
# -----------------
# Use .first() — time is the same for every row in an iteration,
# so we want ONE value per iteration, not the sum of duplicates.
iter_time = df.groupby("Iteration")["Time [min]"].first()
cum_time  = iter_time.cumsum()                          # minutes

# Best (max) objective per iteration, then running cumulative best
iter_best    = df.groupby("Iteration")[obj_col].min()
running_best = iter_best.cummin()                       # running best NPV/CAPEX

# Align on shared iteration index
shared_idx   = iter_best.index.intersection(cum_time.index)
cum_time_arr = cum_time.loc[shared_idx].values
best_arr     = running_best.loc[shared_idx].values

time_mask    = cum_time_arr <= 210
cum_time_arr = cum_time_arr[time_mask]
best_arr     = best_arr[time_mask]

# -----------------
# Plot 1 — Convergence curve vs cumulative time
# -----------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cum_time_arr, - best_arr,
        marker="o", markersize=5, linewidth=2, color="#d33215") 

ax.set_xlabel("Time (in min)", fontsize=12)
ax.set_ylabel("NPV/CAPEX (best value)", fontsize=12)
ax.set_title("Best objective value", fontsize=13, fontweight='bold')
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
# ax.set_xlim(0, 250)
plt.tight_layout()
plt.savefig("ego_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved ego_convergence.png")

# -----------------
# Plot 2 — PCA of design variables, colored by iteration
# -----------------
X          = df[design_cols].values
iterations = df["Iteration"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(7, 6))
norm = plt.Normalize(vmin=iterations.min(), vmax=iterations.max())
sc   = ax.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=iterations, cmap=cm.viridis, norm=norm,
    s=30, alpha=0.8, linewidths=0
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Iteration", fontsize=11)
ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=12)
ax.set_title("PCA of evaluated designs", fontsize=13)
ax.grid(True, linewidth=0.5, alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("ego_pca.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved ego_pca.png")

# -----------------
# Print PCA loadings
# -----------------
loadings = pd.DataFrame(
    pca.components_.T,
    index=design_cols,
    columns=["PC1", "PC2"]
).round(3)
print("\nPCA loadings:")
print(loadings.to_string())