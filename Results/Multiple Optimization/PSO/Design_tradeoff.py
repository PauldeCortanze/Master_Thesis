import os
import matplotlib.pyplot as plt
import pandas as pd


def generate_full_evaluation_plots(csv_path):
    # 1. Load the full swarm history data
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return
    df = pd.read_csv(csv_path)

    # 2. Convert objectives to positive values for presentation
    df["Carbon_Offset_Positive"] = -df["annual_carbon_offset"]
    df["NPV_over_CAPEX_Positive"] = -df["NPV_over_CAPEX"]

    # List of all design variables to analyze
    design_vars = [
        "clearance",
        "sp",
        "p_rated",
        "Nwt",
        "wind_MW_per_km2",
        "solar_MW",
        "surface_tilt",
        "surface_azimuth",
        "b_P",
        "b_E_h",
    ]

    # 3. Create individual plots for all run evaluations
    print("Generating individual scatter plots for all run evaluations...")
    for var in design_vars:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            df["NPV_over_CAPEX_Positive"],
            df["Carbon_Offset_Positive"],
            c=df[var],
            cmap="plasma",
            edgecolors="none",  # Removed borders to prevent overlapping clutter
            alpha=0.6,  # Added transparency to see point density
            s=10,  # Adjusted size for 1,200 points
            linewidths=0,
        )

        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(var.replace("_", " ").title(), fontsize=11)

        # Formatting
        ax.set_ylabel("Annual Carbon Offset", fontsize=12)
        ax.set_xlabel("NPV / CAPEX", fontsize=12)
        # ax.set_title(
        #     f"All Evaluations colored by {var.replace('_', ' ').title()}",
        #     fontsize=13,
        #     fontweight="bold",
        # )

        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        ax.set_xlim(0.8, df["NPV_over_CAPEX_Positive"].max() * 1.05)
        ax.set_ylim(200000, df["Carbon_Offset_Positive"].max() * 1.05)
        plt.tight_layout()

        # Save plot
        fig.savefig(f"Design/all_evals_scatter_{var}.png", dpi=300)
        plt.close(fig)

    # 4. Create a unified grid plot for the Appendix
    print("Generating the consolidated appendix grid plot for all evaluations...")
    fig, axes = plt.subplots(5, 2, figsize=(14, 25))
    axes = axes.flatten()

    for i, var in enumerate(design_vars):
        ax = axes[i]
        sc = ax.scatter(
            df["NPV_over_CAPEX_Positive"],
            df["Carbon_Offset_Positive"],
            c=df[var],
            cmap="plasma",
            edgecolors="none",
            alpha=0.6,
            s=10,
            linewidths=0,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(var.replace("_", " ").title(), fontsize=10)

        ax.set_xlabel("NPV / CAPEX", fontsize=10)
        ax.set_ylabel("Annual Carbon Offset", fontsize=10)
        ax.set_title(
            f"All Simulations - Colored by: {var.replace('_', ' ').title()}",
            fontsize=11,
            fontweight="bold",
        )

    for ax in axes:
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0.8, df["NPV_over_CAPEX_Positive"].max() * 1.05)
        ax.set_ylim(200000, df["Carbon_Offset_Positive"].max() * 1.05)

    plt.tight_layout()

    fig.savefig("Design/all_evals_variables_grid.png", dpi=300)
    plt.close(fig)

    print("All evaluation plots saved successfully to your directory!")


if __name__ == "__main__":
    # Pointing to the swarm history file instead of the Pareto front file
    csv_file_path = "swarm_history_VEPSO_seed0_ew0.0.csv"
    generate_full_evaluation_plots(csv_file_path)
