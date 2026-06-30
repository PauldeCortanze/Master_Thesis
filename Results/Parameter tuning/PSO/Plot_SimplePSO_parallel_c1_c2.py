import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
c1_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
c2_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
POP_SIZE = 20
METRIC = "NPV_over_CAPEX"

# ── Load & process all files ─────────────────────────────────────────────────
def load_best(c1, c2):
    fname = f"Results/history_ALPSO_seed0_c1{c1}_c2{c2}.csv"

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
for c1 in c1_vals:
    for c2 in c2_vals:
        res = load_best(c1, c2)
        if res is not None:  # <--- CRUCIAL FIX: Only store if it's a valid DataFrame
            data[(c1, c2)] = res

# ── Shared style ─────────────────────────────────────────────────────────────
YLABEL = "NPV / CAPEX"
TITLE_FS = 13
LABEL_FS = 11
cmap_lines = cm.plasma

# ── Plot 1: Heatmap ──────────────────────────────────────────────────────────
matrix = np.full((len(c1_vals), len(c2_vals)), np.nan)

for i, c1 in enumerate(c1_vals):
    for j, c2 in enumerate(c2_vals):
        if (c1, c2) in data:  # <--- CRUCIAL FIX: Safely check if the key exists
            matrix[j, i] = data[(c1, c2)]["best_so_far"].iloc[-1]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(matrix, cmap="viridis_r", aspect="auto", origin="lower")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(YLABEL, fontsize=LABEL_FS)
ax.set_xticks(range(len(c1_vals)))
ax.set_xticklabels(c1_vals)
ax.set_yticks(range(len(c2_vals)))
ax.set_yticklabels(c2_vals)
ax.set_xlabel("c1 (cognitive coefficient)", fontsize=LABEL_FS)
ax.set_ylabel("c2 (social coefficient)", fontsize=LABEL_FS)
ax.set_title("Final Best Value per (c1, c2)", fontsize=TITLE_FS, fontweight="bold")
# Annotate cells
for i in range(len(c2_vals)):
    for j in range(len(c1_vals)):
        if i == 3 and j !=2:
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=8, color="black")
        else:
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                fontsize=8, color="white" if matrix[i, j] < matrix.mean() else "black")
plt.tight_layout()
plt.savefig("Plots/plot1_heatmap.png", dpi=150)


# ── Plot 2: Lines grouped by c1 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(c2_vals)))

for idx, c1 in enumerate(c1_vals):
    ax = axes[idx]
    for c2, color in zip(c2_vals, colors):
        # ── CRUCIAL FIX: Check if the combination exists before accessing it ──
        if (c1, c2) in data:
            d = data[(c1, c2)]
            ax.plot(d["generation"], d["best_so_far"],
                    marker="o", markersize=2, linewidth=1.6,
                    label=f"c2={c2}", color=color)
            
    ax.set_title(f"c1 = {c1}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="c2", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
fig.suptitle("Convergence grouped by c1", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot2_lines_by_c1.png", dpi=150)


# ── Plot 3: Lines grouped by c2 ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(c2_vals)))

for idx, c2 in enumerate(c2_vals):
    ax = axes[idx]
    for c1, color in zip(c1_vals, colors):
        d = data[(c1, c2)]
        ax.plot(d["generation"], d["best_so_far"],
                marker="o", markersize=2, linewidth=1.6,
                label=f"c1={c1}", color=color)
    ax.set_title(f"c2 = {c2}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="c2", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
fig.suptitle("Convergence grouped by c2", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot3_lines_by_c2.png", dpi=150)

# ── Plot 2b: Lines grouped by c1 ──────────────────────────────────────────────
c1_vals_plot = [1.0, 1.5, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(c2_vals)))

for idx, c1 in enumerate(c1_vals_plot):
    ax = axes[idx]
    for c2, color in zip(c2_vals, colors):
        # ── CRUCIAL FIX: Check if the combination exists before accessing it ──
        if (c1, c2) in data:
            d = data[(c1, c2)]
            ax.plot(d["generation"], d["best_so_far"],
                    marker="o", markersize=2, linewidth=1.6,
                    label=f"c2={c2}", color=color)
            
    ax.set_title(f"c1 = {c1}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="c2", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
# fig.suptitle("Convergence grouped by c1", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot2b_lines_by_c1.png", dpi=150)


# ── Plot 3b: Lines grouped by c2 ──────────────────────────────────────────────
c2_vals_plot = [1.0, 1.5, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
axes = axes.flatten()
colors = cmap_lines(np.linspace(0.1, 0.9, len(c2_vals)))

for idx, c2 in enumerate(c2_vals_plot):
    ax = axes[idx]
    for c1, color in zip(c1_vals, colors):
        d = data[(c1, c2)]
        ax.plot(d["generation"], d["best_so_far"],
                marker="o", markersize=2, linewidth=1.6,
                label=f"c1={c1}", color=color)
    ax.set_title(f"c2 = {c2}", fontsize=TITLE_FS)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="c2", loc="center right",
           fontsize=10, title_fontsize=11, framealpha=0.85)
fig.supxlabel("Generation", fontsize=LABEL_FS)
fig.supylabel(YLABEL, fontsize=LABEL_FS)
# fig.suptitle("Convergence grouped by c2", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 0.88, 0.97])
plt.savefig("Plots/plot3b_lines_by_c2.png", dpi=150)


print("Done — saved 5 plots to Plots/")


# ── Plot 4: All curves colored only by Pm ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Use the exact same color setup you used for the Pm subplots
colors_c1 = cmap_lines(np.linspace(0.1, 0.9, len(c1_vals)))

for idx, c1 in enumerate(c1_vals):
    color = colors_c1[idx]
    
    # We only want to label the very first line of this Pm group to keep the legend clean
    is_first = True 
    
    for c2 in c2_vals:
        if (c1, c2) in data:
            d = data[(c1, c2)]
            
            ax.plot(
                d["generation"], 
                d["best_so_far"],
                marker="o", 
                markersize=1.5, 
                linewidth=1.2,
                color=color,
                # If it's the first curve for this Pm, give it a label; otherwise, ignore
                label=f"C1={c1}" if is_first else "_" 
            )
            is_first = False

ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Generation", fontsize=LABEL_FS)
ax.set_ylabel(YLABEL, fontsize=LABEL_FS)
ax.set_title("All Convergence Curves (Colored by C1)", fontsize=TITLE_FS, fontweight="bold")

# Neat, single-column legend with just the 5 Pm variants
ax.legend(
    title="Inertia Weight (C1)", 
    loc="center left", 
    bbox_to_anchor=(1.02, 0.5), 
    fontsize=10, 
    title_fontsize=11, 
    framealpha=0.85
)

plt.tight_layout()
plt.savefig("Plots/plot4_all_curves_by_C1.png", dpi=150)


# ── Plot 4: All curves colored only by Pm ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Use the exact same color setup you used for the Pm subplots
colors_c2 = cmap_lines(np.linspace(0.1, 0.9, len(c2_vals)))

for idx, c2 in enumerate(c2_vals):
    color = colors_c2[idx]
    
    # We only want to label the very first line of this Pm group to keep the legend clean
    is_first = True 
    
    for c1 in c1_vals:
        if (c1, c2) in data:
            d = data[(c1, c2)]
            
            ax.plot(
                d["generation"], 
                d["best_so_far"],
                marker="o", 
                markersize=1.5, 
                linewidth=1.2,
                color=color,
                # If it's the first curve for this Pm, give it a label; otherwise, ignore
                label=f"C2={c2}" if is_first else "_" 
            )
            is_first = False

ax.grid(True, linestyle="--", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Generation", fontsize=LABEL_FS)
ax.set_ylabel(YLABEL, fontsize=LABEL_FS)
ax.set_title("All Convergence Curves (Colored by C2)", fontsize=TITLE_FS, fontweight="bold")

# Neat, single-column legend with just the 5 Pm variants
ax.legend(
    title="Inertia Weight (C2)", 
    loc="center left", 
    bbox_to_anchor=(1.02, 0.5), 
    fontsize=10, 
    title_fontsize=11, 
    framealpha=0.85
)

plt.tight_layout()
plt.savefig("Plots/plot4_all_curves_by_C2.png", dpi=150)