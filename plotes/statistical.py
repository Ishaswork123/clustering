import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import ListedColormap


# ============================================================
# GLOBAL FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    # Main font
    "font.size": 12,

    # Axes
    "axes.labelsize": 13,
    "axes.titlesize": 14,

    # Ticks
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Legend
    "legend.fontsize": 11,

    # Lines
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    # Embed fonts correctly in PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 1. STATISTICAL ANALYSIS
# ============================================================

def perform_statistical_analysis(
    df,
    output_dir="statistical_analysis"
):

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # CLEAN AND STANDARDIZE DATA
    # --------------------------------------------------------

    df.columns = [c.strip() for c in df.columns]

    cols_lower = {
        c.lower(): c for c in df.columns
    }

    m_col = (
        cols_lower.get("method")
        or cols_lower.get("algorithm")
        or df.columns[0]
    )

    d_col = (
        cols_lower.get("dataset")
        or cols_lower.get("data")
        or df.columns[1]
    )

    a_col = (
        cols_lower.get("ari")
        or cols_lower.get("score")
        or "ARI"
    )

    # --------------------------------------------------------
    # METHOD NAME STANDARDIZATION
    # --------------------------------------------------------
    # IMPORTANT:
    # Novel methods are renamed BEFORE generating the plots.
    # --------------------------------------------------------

    name_mapping = {

        # -------------------------------
        # Novel methods
        # -------------------------------

        "Novel Divisive Clustering (proximity based)":
            "Proximity-Based",

        "Novel Divisive Clustering (proximity based":
            "Proximity-Based",

        "Novel divisive clustering algorithm(order based)":
            "Order-Based",

        "Novel Divisive Clustering Algorithm (order based)":
            "Order-Based",

        "Novel Divisive (Order)":
            "Order-Based",

        "Novel Divisive (Prox)":
            "Proximity-Based",

        # -------------------------------
        # Density
        # -------------------------------

        "density-based":
            "Density-Based",

        # -------------------------------
        # Agglomerative
        # -------------------------------

        "AgglomerativeClustering (ward)":
            "Agglo (Ward)",

        "AgglomerativeClustering (complete)":
            "Agglo (Complete)",

        "AgglomerativeClustering (average)":
            "Agglo (Average)",

        "AgglomerativeClustering (single)":
            "Agglo (Single)"
    }

    # Clean method names
    df[m_col] = (
        df[m_col]
        .astype(str)
        .str.strip()
    )

    # Apply name mapping
    df[m_col] = df[m_col].replace(name_mapping)

    # --------------------------------------------------------
    # Additional case-insensitive normalization
    # --------------------------------------------------------

    def standardize_method(name):

        name_lower = name.lower().strip()

        # Novel methods
        if "proximity" in name_lower:
            return "Proximity-Based"

        if "order" in name_lower:
            return "Order-Based"

        # Density
        if "density" in name_lower:
            return "Density-Based"

        # Agglomerative
        if "ward" in name_lower:
            return "Agglo (Ward)"

        if "complete" in name_lower:
            return "Agglo (Complete)"

        if "average" in name_lower:
            return "Agglo (Average)"

        if "single" in name_lower:
            return "Agglo (Single)"

        return name.strip()


    df[m_col] = df[m_col].apply(
        standardize_method
    )

    # --------------------------------------------------------
    # Convert ARI to numeric
    # --------------------------------------------------------

    df[a_col] = pd.to_numeric(
        df[a_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[a_col]
    )

    # --------------------------------------------------------
    # Average repeated observations
    # --------------------------------------------------------

    df_clean = (
        df.groupby(
            [d_col, m_col]
        )[a_col]
        .mean()
        .reset_index()
    )

    methods = df_clean[m_col].unique()
    datasets = df_clean[d_col].unique()

    # --------------------------------------------------------
    # FRIEDMAN TEST DATA
    # --------------------------------------------------------

    friedman_data = []

    for dataset in datasets:

        dataset_scores = []

        for method in methods:

            val = df_clean[
                (df_clean[d_col] == dataset)
                &
                (df_clean[m_col] == method)
            ][a_col].values

            dataset_scores.append(
                val[0] if len(val) > 0 else 0.0
            )

        friedman_data.append(
            dataset_scores
        )

    # --------------------------------------------------------
    # STATISTICAL RESULTS
    # --------------------------------------------------------

    heatmap_data = {
        "rank_map": {},
        "cd_value": 0.0,
        "friedman_p": 1.0
    }

    if (
        len(friedman_data) > 1
        and len(friedman_data[0]) > 1
    ):

        _, p_value = stats.friedmanchisquare(
            *friedman_data
        )

        heatmap_data["friedman_p"] = p_value

        # ----------------------------------------------------
        # Critical Difference
        # ----------------------------------------------------

        if p_value < 0.05:

            k = len(methods)
            N = len(datasets)

            q_alpha = 2.948

            CD = (
                q_alpha
                *
                np.sqrt(
                    k * (k + 1)
                    /
                    (6 * N)
                )
            )

            heatmap_data["cd_value"] = CD

            # Rank: higher ARI = better = lower rank
            ranks_list = [
                stats.rankdata(
                    [-s for s in ds]
                )
                for ds in friedman_data
            ]

            avg_ranks = np.mean(
                ranks_list,
                axis=0
            )

            heatmap_data["rank_map"] = dict(
                zip(
                    methods,
                    avg_ranks
                )
            )

    print(
        f"Cleaned Analysis complete. "
        f"CD: {heatmap_data['cd_value']:.4f}"
    )

    print(
        f"Friedman p-value: "
        f"{heatmap_data['friedman_p']:.4e}"
    )

    print("\nMethods used in analysis:")

    for method in methods:
        print(f"  - {method}")

    return heatmap_data


# ============================================================
# 2. GENERATE FIGURES
# ============================================================

def generate_combined_plots(data):

    rank_map = data["rank_map"]
    cd = data["cd_value"]
    p_val = data["friedman_p"]

    if not rank_map:

        print(
            "No significant differences found."
        )

        return

    # --------------------------------------------------------
    # SORT METHODS
    # Lower rank = better
    # --------------------------------------------------------

    sorted_methods = sorted(
        rank_map.keys(),
        key=lambda x: rank_map[x]
    )

    sorted_ranks = [
        rank_map[m]
        for m in sorted_methods
    ]

    # ========================================================
    # PLOT 1: AVERAGE RANK BAR CHART
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.5, 6.5)
    )

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    colors = plt.cm.RdYlGn_r(
        np.linspace(
            0.2,
            0.8,
            len(sorted_methods)
        )
    )

    bars = ax.barh(
        sorted_methods,
        sorted_ranks,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        height=0.65
    )

    # Best method at top
    ax.invert_yaxis()

    # --------------------------------------------------------
    # Bold Novel Method Labels
    # --------------------------------------------------------

    for label, method in zip(
        ax.get_yticklabels(),
        sorted_methods
    ):

        if method in [
            "Order-Based",
            "Proximity-Based"
        ]:

            label.set_fontweight(
                "bold"
            )

    # --------------------------------------------------------
    # Rank Labels
    # --------------------------------------------------------

    for i, bar in enumerate(bars):

        ax.text(
            bar.get_width() + 0.08,
            bar.get_y()
            + bar.get_height() / 2,
            f"{sorted_ranks[i]:.2f}",
            va="center",
            fontsize=11
        )

    # --------------------------------------------------------
    # Critical Difference Boundary
    # --------------------------------------------------------

    if cd > 0:

        ax.axvline(
            x=sorted_ranks[0] + cd,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Critical Difference"
        )

    # --------------------------------------------------------
    # Axis Label
    # --------------------------------------------------------

    ax.set_xlabel(
        "Average Rank (lower = better)",
        labelpad=8
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.5
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Remove unnecessary spines
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    if cd > 0:

        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=1,
            edgecolor="black"
        )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save PDF for Springer/LaTeX
    # --------------------------------------------------------

    plt.savefig(
        "rank_bar_chart.pdf",
        bbox_inches="tight"
    )

    # High-resolution PNG
    plt.savefig(
        "rank_bar_chart.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # ========================================================
    # PLOT 2: STATISTICAL SIGNIFICANCE HEATMAP
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    n = len(sorted_methods)

    sig_matrix = np.zeros(
        (n, n)
    )

    annot_matrix = np.empty(
        (n, n),
        dtype=object
    )

    annot_matrix[:] = ""

    # --------------------------------------------------------
    # Build significance matrix
    # --------------------------------------------------------

    for i, m1 in enumerate(
        sorted_methods
    ):

        for j, m2 in enumerate(
            sorted_methods
        ):

            if i == j:
                continue

            diff = (
                rank_map[m2]
                -
                rank_map[m1]
            )

            if diff > cd:

                sig_matrix[i, j] = 1
                annot_matrix[i, j] = ">"

            elif diff < -cd:

                sig_matrix[i, j] = -1
                annot_matrix[i, j] = "<"

    # --------------------------------------------------------
    # Color Map
    # --------------------------------------------------------

    cmap = ListedColormap([
        "#a50026",
        "#ffffbf",
        "#006837"
    ])

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    labels = [
        f"{m}\n(rank: {rank_map[m]:.2f})"
        for m in sorted_methods
    ]

    sns.heatmap(
        sig_matrix,
        annot=annot_matrix,
        fmt="",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=False,
        linewidths=1,
        linecolor="gray",
        annot_kws={
            "size": 14,
            "weight": "bold"
        },
        ax=ax
    )

    # --------------------------------------------------------
    # Bold Order-Based / Proximity-Based labels
    # --------------------------------------------------------

    for label, method in zip(
        ax.get_xticklabels(),
        sorted_methods
    ):

        if method in [
            "Order-Based",
            "Proximity-Based"
        ]:

            label.set_fontweight(
                "bold"
            )

    for label, method in zip(
        ax.get_yticklabels(),
        sorted_methods
    ):

        if method in [
            "Order-Based",
            "Proximity-Based"
        ]:

            label.set_fontweight(
                "bold"
            )

    # --------------------------------------------------------
    # Axis Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Methods",
        fontsize=13,
        labelpad=10
    )

    ax.set_ylabel(
        "Methods",
        fontsize=13,
        labelpad=10
    )

    # --------------------------------------------------------
    # No title
    # --------------------------------------------------------
    # Springer LaTeX caption should be used instead.
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save PDF
    # --------------------------------------------------------

    plt.savefig(
        "statistical_heatmap.pdf",
        bbox_inches="tight"
    )

    # High-resolution PNG
    plt.savefig(
        "statistical_heatmap.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    file_path = (
        "cleaned_clustering_results.csv"
    )

    if os.path.exists(file_path):

        raw_df = pd.read_csv(
            file_path
        )

        stats_results = (
            perform_statistical_analysis(
                raw_df
            )
        )

        generate_combined_plots(
            stats_results
        )

    else:

        print(
            f"File not found: {file_path}"
        )