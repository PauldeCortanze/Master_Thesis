import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ── Config ────────────────────────────────────────────────────────────────────
EW_VALUES = [0.0, 100.0, 200.0, 500.0, 1000.0]
NPV_TPL    = "Results/swarm_history_VEPSO_seed0_ew{}.csv"

obj1 = "NPV_over_CAPEX"         # maximise (stored negated → flip)
obj2 = "annual_carbon_offset"   # maximise

# ── Hard thresholds to remove degenerate lower-left cluster ──────────────────
# NPV_MIN    = -1       # discard solutions with NPV/CAPEX below this
# CARBON_MIN = -100_000    # discard solutions with carbon offset below this (t CO2)

COLORS = ["#e84545", "#f4a261", "#2a9d8f", "#457b9d", "#6a4c93"]

# ── Pareto helper (maximise both objectives) ──────────────────────────────────
def pareto_mask(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not efficient[i]:
            continue
        dominated = np.all(arr >= arr[i], axis=1) & np.any(arr > arr[i], axis=1)
        dominated[i] = False
        if dominated.any():
            efficient[i] = False
    return efficient

# ── Load, merge, filter ───────────────────────────────────────────────────────
def load_and_combine(ew):
    path = NPV_TPL.format(ew)
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None

    df = pd.read_csv(path)
    if "success" in df.columns:
        df = df[df["success"] == 1]
    df = df[[obj1, obj2]].dropna().copy()
    df["source"] = "NPV"   # ← add this

    df[obj1] = -df[obj1]
    df[obj2] = -df[obj2]

    before = len(df)
    # df = df[
    #     (df[obj1] >= NPV_MIN) &
    #     (df[obj2] >= CARBON_MIN)
    # ].reset_index(drop=True)
    print(f"  ew={ew}: removed {before - len(df)} bad solutions ({len(df)} kept)")

    return df


# ── Build dataset dict ────────────────────────────────────────────────────────
datasets = {}
for ew in EW_VALUES:
    df = load_and_combine(ew)
    if df is None:
        continue
    vals = df[[obj1, obj2]].to_numpy(dtype=float)
    df["pareto"] = pareto_mask(vals)
    datasets[ew] = df
    n_pareto = df["pareto"].sum()
    print(f"ew={ew:>4}  total={len(df):>5}  pareto={n_pareto:>4}")

if not datasets:
    raise FileNotFoundError("No CSV files found. Check NPV_TPL / CARBON_TPL paths.")

# ── Plot 1: individual subplots (3 Rows x 2 Columns) ──────────────────────────
n     = len(datasets)
ncols = 2
nrows = 3

# Dimensions configured to beautifully fill out a standard report page (8.5 x 11)
fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 11), sharex=True, sharey=True)
axes = np.array(axes).flatten()

for ax_idx, ((ew, df), color) in enumerate(zip(datasets.items(), COLORS)):
    ax        = axes[ax_idx]
    dominated = df[~df["pareto"]]
    pareto    = df[df["pareto"]].sort_values(obj1)

    # Colour dominated points by source file
    ax.scatter(dominated[obj1], dominated[obj2],
               c="#c6d9ed", edgecolors="#7aaac8",
               s=25, linewidths=0.4, alpha=0.75, zorder=3, label="Dominated")

    ax.scatter(pareto[obj1], pareto[obj2],
               c=color, edgecolors="black", s=60,
               linewidths=0.5, alpha=0.95, zorder=5, label="Pareto front")

    ax.set_title(f"$ew = {ew}$", fontsize=12, fontweight="bold")
    ax.set_xlabel("NPV / CAPEX", fontsize=10)
    ax.set_ylabel("Annual Carbon Offset (t $\mathrm{CO}_2$ eq.)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, framealpha=0.85)

# Since axes[5] (row 3, col 2) is hidden, axes[3] (row 2, col 2) is the bottom-most 
# visible plot in its column. We manually re-enable its x-axis labels here.
axes[3].xaxis.set_tick_params(labelbottom=True)

# Hide the unused 6th subplot slot
for ax in axes[len(datasets):]:
    ax.set_visible(False)

# fig.suptitle("Pareto Fronts — NSGA-II (NPV + Carbon pooled per $ew$)",
#              fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
os.makedirs("Plot", exist_ok=True)
out1 = "Plot/pareto_nsga2_combined_subplots.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.show()

# ── Plot 2: overlay all Pareto fronts ─────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))

for (ew, df), color in zip(datasets.items(), COLORS):
    pareto = df[df["pareto"]].sort_values(obj1)
    ax2.scatter(pareto[obj1], pareto[obj2],
                c=color, edgecolors="black", s=20,
                linewidths=0.2, alpha=0.9, zorder=5)
    ax2.plot(pareto[obj1], pareto[obj2], c=color, alpha=1, zorder=4, label=f"ew = {ew}")

ax2.set_xlabel("NPV / CAPEX", fontsize=12)
ax2.set_ylabel("Annual Carbon Offset (t CO₂ eq.)", fontsize=12)
ax2.set_title("Pareto Fronts — all Ew values",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, framealpha=0.9, title="Carbon weight (ew)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

out2 = "Plot/pareto_vepso_combined_overlay.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.show()


'''
# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Solution density in the NPV axis (KDE / histogram overlay)
# ══════════════════════════════════════════════════════════════════════════════
fig2, (ax_hist, ax_kde) = plt.subplots(1, 2, figsize=(14, 5))
 
bins = np.linspace(
    min(df[obj1].min() for df in datasets.values()),
    max(df[obj1].max() for df in datasets.values()),
    40
)
 
for (ew, df), color in zip(datasets.items(), COLORS):
    # Histogram (counts)
    ax_hist.hist(df[obj1], bins=bins, color=color, alpha=0.35,
                 label=f"ew = {ew}", edgecolor="none")
 
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df[obj1], bw_method=0.15)
    x_grid = np.linspace(bins[0], bins[-1], 400)
    ax_kde.plot(x_grid, kde(x_grid), color=color, linewidth=2,
                label=f"ew = {ew}")
    ax_kde.fill_between(x_grid, kde(x_grid), alpha=0.08, color=color)
 
for ax, title in [(ax_hist, "Histogram — solution density along NPV/CAPEX"),
                  (ax_kde,  "KDE — solution density along NPV/CAPEX")]:
    ax.set_xlabel("NPV / CAPEX", fontsize=11)
    ax.set_ylabel("Count / Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, title="Carbon weight (ew)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
 
plt.tight_layout()
out2 = "Plot/diagnostic_2_density.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.show()

fig2b, (ax_hist2, ax_kde2) = plt.subplots(1, 2, figsize=(14, 5))
 
bins_c = np.linspace(
    min(df[obj2].min() for df in datasets.values()),
    max(df[obj2].max() for df in datasets.values()),
    40
)
 
for (ew, df), color in zip(datasets.items(), COLORS):
    ax_hist2.hist(df[obj2], bins=bins_c, color=color, alpha=0.35,
                  label=f"ew = {ew}", edgecolor="none")
 
    kde_c = gaussian_kde(df[obj2], bw_method=0.15)
    y_grid = np.linspace(bins_c[0], bins_c[-1], 400)
    ax_kde2.plot(y_grid, kde_c(y_grid), color=color, linewidth=2,
                 label=f"ew = {ew}")
    ax_kde2.fill_between(y_grid, kde_c(y_grid), alpha=0.08, color=color)
 
for ax, title in [(ax_hist2, "Histogram — solution density along Carbon Offset"),
                  (ax_kde2,  "KDE — solution density along Carbon Offset")]:
    ax.set_xlabel("Annual Carbon Offset [t CO\u2082 eq.]", fontsize=11)
    ax.set_ylabel("Count / Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, title="Carbon weight (ew)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
 
plt.tight_layout()
out2b = "plot/diagnostic_2b_density_Carbon.png"
plt.savefig(out2b, dpi=150, bbox_inches="tight")
print(f"Saved \u2192 {out2b}")
plt.show()
'''

'''
# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Centroid trajectory (average NPV vs average Carbon per ew)
# ══════════════════════════════════════════════════════════════════════════════
ew_list, cx_list, cy_list = [], [], []
for ew, df in datasets.items():
    ew_list.append(ew)
    cx_list.append(df[obj1].mean())
    cy_list.append(df[obj2].mean())
 
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
 
# 3a — scatter of centroids coloured by ew
norm  = plt.Normalize(vmin=min(ew_list), vmax=max(ew_list))
cmap  = cm.plasma
smap  = cm.ScalarMappable(cmap=cmap, norm=norm)
 
for ew, cx, cy, color in zip(ew_list, cx_list, cy_list, COLORS):
    axes3[0].scatter(cx, cy, c=[color], s=160, edgecolors="black",
                     linewidths=0.8, zorder=5)
    axes3[0].annotate(f"ew={ew}", (cx, cy),
                      textcoords="offset points", xytext=(6, 4), fontsize=9)
 
# connect centroids with an arrow path ordered by ew
for i in range(len(ew_list) - 1):
    axes3[0].annotate("",
        xy=(cx_list[i+1], cy_list[i+1]),
        xytext=(cx_list[i], cy_list[i]),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
 
axes3[0].set_xlabel("Mean NPV / CAPEX", fontsize=11)
axes3[0].set_ylabel("Mean Annual Carbon Offset [t CO₂ eq.]", fontsize=11)
axes3[0].set_title("Centroid trajectory\n(ordered by ew)", fontsize=12, fontweight="bold")
axes3[0].grid(True, linestyle="--", alpha=0.35)
axes3[0].spines[["top", "right"]].set_visible(False)
 
# 3b — mean NPV vs ew
axes3[1].plot(ew_list, cx_list, "o-", color="#457b9d", linewidth=2, markersize=8)
for ew, cx, color in zip(ew_list, cx_list, COLORS):
    axes3[1].scatter(ew, cx, c=[color], s=80, edgecolors="black", zorder=5)
axes3[1].set_xlabel("Carbon Weight (ew)", fontsize=11)
axes3[1].set_ylabel("Mean NPV / CAPEX", fontsize=11)
axes3[1].set_title("Mean NPV vs ew\n(should decrease ↓)", fontsize=12, fontweight="bold")
axes3[1].grid(True, linestyle="--", alpha=0.35)
axes3[1].spines[["top", "right"]].set_visible(False)
 
# 3c — mean Carbon vs ew
axes3[2].plot(ew_list, cy_list, "o-", color="#2a9d8f", linewidth=2, markersize=8)
for ew, cy, color in zip(ew_list, cy_list, COLORS):
    axes3[2].scatter(ew, cy, c=[color], s=80, edgecolors="black", zorder=5)
axes3[2].set_xlabel("Carbon Weight (ew)", fontsize=11)
axes3[2].set_ylabel("Mean Annual Carbon Offset [t CO₂ eq.]", fontsize=11)
axes3[2].set_title("Mean Carbon Offset vs ew\n(should increase ↑)", fontsize=12, fontweight="bold")
axes3[2].grid(True, linestyle="--", alpha=0.35)
axes3[2].spines[["top", "right"]].set_visible(False)
 
fig3.suptitle("Plot 3 — Centroid Analysis per ew Value",
              fontsize=14, fontweight="bold")
plt.tight_layout()
out3 = "plot/diagnostic_3_centroids.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")
plt.show()
'''