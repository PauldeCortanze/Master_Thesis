import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
Pc_vals = [0.5, 0.6, 0.7, 0.8, 0.9]
Pm_vals = [0.01, 0.05, 0.1, 0.15, 0.2]
POP_SIZE = 20
METRIC = "NPV_over_CAPEX"

# ── Load & process all files ─────────────────────────────────────────────────
def load_best(pc, pm):
    fname = f"Results/history_NSGA2_seed0_Pc{pc}_Pm{pm}.csv"

    if not os.path.exists(fname):
        print(f"Skipping: {fname} (File not found)")
        return None
    
    df = pd.read_csv(fname)
    df["generation"] = df["iteration"] // POP_SIZE
    best = (
        df[df["success"] == 1]
        .groupby("generation")[METRIC]
        .min()
        .reset_index()
    )
    best["best_so_far"] = best[METRIC].cummin()
    best["best_so_far"] = - best["best_so_far"]
    return best

data = {}
for pc in Pc_vals:
    for pm in Pm_vals:
        res = load_best(pc, pm)
        if res is not None:  # <--- CRUCIAL FIX: Only store if it's a valid DataFrame
            data[(pc, pm)] = res

# ── Shared style ─────────────────────────────────────────────────────────────
YLABEL = "NPV / CAPEX"
TITLE_FS = 13
LABEL_FS = 11
cmap_lines = cm.plasma

# ── Plot 1: Heatmap ──────────────────────────────────────────────────────────
matrix = np.full((len(Pm_vals), len(Pc_vals)), np.nan)

for i, pm in enumerate(Pm_vals):
    for j, pc in enumerate(Pc_vals):
        if (pc, pm) in data:  # <--- CRUCIAL FIX: Safely check if the key exists
            matrix[i, j] = data[(pc, pm)]["best_so_far"].iloc[-1]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap="viridis_r", aspect="auto", origin="lower")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(YLABEL, fontsize=LABEL_FS)
ax.set_xticks(range(len(Pc_vals)))
ax.set_xticklabels(Pc_vals)
ax.set_yticks(range(len(Pm_vals)))
ax.set_yticklabels(Pm_vals)
ax.set_xlabel("Pc (crossover rate)", fontsize=LABEL_FS)
ax.set_ylabel("Pm (mutation rate)", fontsize=LABEL_FS)
ax.set_title("Final Best Value per (Pc, Pm)", fontsize=TITLE_FS, fontweight="bold")
# Annotate cells
for i in range(len(Pm_vals)):
    for j in range(len(Pc_vals)):
        if i == 0 and j == 0:
            ax.text(0, 0, f"{matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=8, color="black")
        else:
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i, j] < matrix.mean() else "black")
        
plt.tight_layout()
plt.savefig("Plots/plot1_heatmap.png", dpi=150)


# ── Plot 2: Lines grouped by Pc ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(Pm_vals)))

for idx, pc in enumerate(Pc_vals):
    ax = axes[idx]
    for pm, color in zip(Pm_vals, colors):
        
        # ── CRUCIAL FIX: Check if the combination exists before accessing it ──
        if (pc, pm) in data:
            d = data[(pc, pm)]
            ax.plot(d["generation"], d["best_so_far"],
                    marker="o", markersize=2, linewidth=1.6,
                    label=f"Pm={pm}", color=color)
            
    ax.set_title(f"Pc = {pc}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Pm", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
fig.suptitle("Convergence grouped by Pc", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot2_lines_by_Pc.png", dpi=150)


# ── Plot 3: Lines grouped by Pm ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(Pc_vals)))

for idx, pm in enumerate(Pm_vals):
    ax = axes[idx]
    for pc, color in zip(Pc_vals, colors):
        if (pc, pm) not in data:
            continue  # Skip if this combination is missing
        d = data[(pc, pm)]
        ax.plot(d["generation"], d["best_so_far"],
                marker="o", markersize=2, linewidth=1.6,
                label=f"Pc={pc}", color=color)
    ax.set_title(f"Pm = {pm}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Pc", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
fig.suptitle("Convergence grouped by Pm", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot3_lines_by_Pm.png", dpi=150)



# ── Plot 2b: Lines grouped by Pc ──────────────────────────────────────────────
Pc_vals_plot = [0.6, 0.7, 0.8]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(Pm_vals)))

for idx, pc in enumerate(Pc_vals_plot):
    ax = axes[idx]
    for pm, color in zip(Pm_vals, colors):
        
        # ── CRUCIAL FIX: Check if the combination exists before accessing it ──
        if (pc, pm) in data:
            d = data[(pc, pm)]
            ax.plot(d["generation"], d["best_so_far"],
                    marker="o", markersize=2, linewidth=1.6,
                    label=f"Pm={pm}", color=color)
            
    ax.set_title(f"Pc = {pc}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Pm", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
# fig.suptitle("Convergence grouped by Pc", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot2b_lines_by_Pc.png", dpi=150)


# ── Plot 3: Lines grouped by Pm ──────────────────────────────────────────────
Pm_vals_plot = [0.05, 0.1, 0.15]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(Pc_vals)))

for idx, pm in enumerate(Pm_vals_plot):
    ax = axes[idx]
    for pc, color in zip(Pc_vals, colors):
        if (pc, pm) not in data:
            continue  # Skip if this combination is missing
        d = data[(pc, pm)]
        ax.plot(d["generation"], d["best_so_far"],
                marker="o", markersize=2, linewidth=1.6,
                label=f"Pc={pc}", color=color)
    ax.set_title(f"Pm = {pm}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Pc", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
# fig.suptitle("Convergence grouped by Pm", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot3b_lines_by_Pm.png", dpi=150)

# plt.show()


# ── Plot 4: All curves in one figure ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Use the exact same color setup you used for the Pm subplots
colors_pm = cmap_lines(np.linspace(0.1, 0.9, len(Pm_vals)))

for idx, pm in enumerate(Pm_vals):
    color = colors_pm[idx]
    
    # We only want to label the very first line of this Pm group to keep the legend clean
    is_first = True 
    
    for pc in Pc_vals:
        if (pc, pm) in data:
            d = data[(pc, pm)]
            
            ax.plot(
                d["generation"], 
                d["best_so_far"],
                marker="o", 
                markersize=1.5, 
                linewidth=1.2,
                color=color,
                # If it's the first curve for this Pm, give it a label; otherwise, ignore
                label=f"Pm={pm}" if is_first else "_" 
            )
            is_first = False

ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Generation", fontsize=LABEL_FS)
ax.set_ylabel(YLABEL, fontsize=LABEL_FS)
ax.set_title("All Convergence Curves (Colored by Pm)", fontsize=TITLE_FS, fontweight="bold")

# Neat, single-column legend with just the 5 Pm variants
ax.legend(
    title="Mutation Rate (Pm)", 
    loc="best",
    fontsize=10, 
    title_fontsize=11, 
    framealpha=0.85
)

plt.tight_layout()
plt.savefig("Plots/plot4_all_curves_by_Pm.png", dpi=150)

# ── Plot 4: All curves in one figure ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Use the exact same color setup you used for the Pm subplots
colors_pc = cmap_lines(np.linspace(0.1, 0.9, len(Pc_vals)))

for idx, pc in enumerate(Pc_vals):
    color = colors_pc[idx]
    
    # We only want to label the very first line of this Pc group to keep the legend clean
    is_first = True 
    
    for pm in Pm_vals:
        if (pc, pm) in data:
            d = data[(pc, pm)]
            
            ax.plot(
                d["generation"], 
                d["best_so_far"],
                marker="o", 
                markersize=1.5, 
                linewidth=1.2,
                color=color,
                # If it's the first curve for this Pc, give it a label; otherwise, ignore
                label=f"Pc={pc}" if is_first else "_" 
            )
            is_first = False

ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Generation", fontsize=LABEL_FS)
ax.set_ylabel(YLABEL, fontsize=LABEL_FS)
ax.set_title("All Convergence Curves (Colored by Pc)", fontsize=TITLE_FS, fontweight="bold")

# Neat, single-column legend with just the 5 Pm variants
ax.legend(
    title="Crossover Rate (Pc)", 
    loc="best",
    fontsize=10, 
    title_fontsize=11, 
    framealpha=0.85
)

plt.tight_layout()
plt.savefig("Plots/plot4_all_curves_by_Pc.png", dpi=150)