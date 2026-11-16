import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# GLOBAL FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 9,

    "axes.labelsize": 9,
    "axes.titlesize": 10,

    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "domain_algorithm_average_scores.csv"
)

df["Algorithm"] = (
    df["Algorithm"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 2. STANDARDIZE METHOD NAMES
# ============================================================

mapping = {

    # --------------------------------------------------------
    # Novel methods
    # --------------------------------------------------------

    "Novel Divisive (Prox)": "Proximity-Based",

    "Novel Divisive (Proximity)": "Proximity-Based",

    "Novel Divisive Clustering (proximity based":
        "Proximity-Based",

    "Novel Divisive Clustering (proximity based)":
        "Proximity-Based",

    "Novel divisive clustering algorithm(proximity based)":
        "Proximity-Based",

    "Novel Divisive (Order)": "Order-Based",

    "Novel divisive clustering algorithm(order based)":
        "Order-Based",

    "Novel Divisive Clustering (Order Based)":
        "Order-Based",

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    "density-based": "Density-Based",

    "Density based": "Density-Based",

    # --------------------------------------------------------
    # Agglomerative
    # --------------------------------------------------------

    "AgglomerativeClustering (average)":
        "Agglo (Average)",

    "AgglomerativeClustering (complete)":
        "Agglo (Complete)",

    "AgglomerativeClustering (single)":
        "Agglo (Single)",

    "AgglomerativeClustering (ward)":
        "Agglo (Ward)"
}


df["Algorithm"] = (
    df["Algorithm"]
    .replace(mapping)
)


# ============================================================
# 3. FAIL-SAFE NORMALIZATION
# ============================================================

df.loc[
    df["Algorithm"].str.contains(
        "prox",
        case=False,
        na=False
    ),
    "Algorithm"
] = "Proximity-Based"


df.loc[
    df["Algorithm"].str.contains(
        "order",
        case=False,
        na=False
    ),
    "Algorithm"
] = "Order-Based"


# ============================================================
# 4. CLEAN ARI
# ============================================================

df["ARI"] = pd.to_numeric(
    df["ARI"],
    errors="coerce"
)

df = df.dropna(
    subset=["ARI"]
)


# ============================================================
# 5. MERGE DUPLICATES
# ============================================================

df = (
    df.groupby(
        ["Algorithm", "Domain"],
        as_index=False
    )["ARI"]
    .mean()
)


# ============================================================
# 6. DOMAIN ORDER
# ============================================================

# Keep domains in their existing CSV order
domains = df["Domain"].drop_duplicates().tolist()

print("\nDomains:")
for domain in domains:
    print(" -", domain)


# ============================================================
# 7. GLOBAL X-AXIS LIMIT
# ============================================================

# SAME x-axis for every subplot.
# This is important because it allows visual comparison
# between domains.

max_ari = df["ARI"].max()

x_max = max(
    0.10,
    np.ceil(max_ari * 1.15 * 100) / 100
)


# ============================================================
# 8. CREATE 2-COLUMN GRID
# ============================================================

n_domains = len(domains)

n_cols = 2
n_rows = int(
    np.ceil(n_domains / n_cols)
)


fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        10,
        2.9 * n_rows
    ),
    sharex=True
)


# Make axes iterable
axes = np.array(
    axes
).reshape(-1)


# ============================================================
# 9. COLORS
# ============================================================

# Keep a restrained color palette.
# Blue = traditional/divisive
# Orange = agglomerative
# Green = proposed methods

category_colors = {
    "traditional": "#4C78A8",
    "agglomerative": "#F28E2B",
    "novel": "#59A14F"
}


def get_color(method):

    method_lower = method.lower()

    if method in [
        "Order-Based",
        "Proximity-Based"
    ]:
        return category_colors["novel"]

    if "agglo" in method_lower:
        return category_colors["agglomerative"]

    return category_colors["traditional"]


# ============================================================
# 10. DRAW EACH DOMAIN
# ============================================================

for ax, domain in zip(
    axes,
    domains
):

    domain_df = df[
        df["Domain"] == domain
    ].copy()


    # --------------------------------------------------------
    # Sort: best method at top
    # --------------------------------------------------------

    domain_df = domain_df.sort_values(
        "ARI",
        ascending=True
    )


    # --------------------------------------------------------
    # Best method
    # --------------------------------------------------------

    best_idx = domain_df["ARI"].idxmax()

    best_method = domain_df.loc[
        best_idx,
        "Algorithm"
    ]

    best_score = domain_df.loc[
        best_idx,
        "ARI"
    ]


    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    colors = [
        get_color(method)
        for method in domain_df["Algorithm"]
    ]


    # --------------------------------------------------------
    # Horizontal bars
    # --------------------------------------------------------

    bars = ax.barh(
        domain_df["Algorithm"],
        domain_df["ARI"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        height=0.65
    )


    # --------------------------------------------------------
    # Highlight winner
    # --------------------------------------------------------

    for bar, method, score in zip(
        bars,
        domain_df["Algorithm"],
        domain_df["ARI"]
    ):

        if method == best_method:

            bar.set_edgecolor(
                "black"
            )

            bar.set_linewidth(
                1.8
            )


    # --------------------------------------------------------
    # Bold proposed methods
    # --------------------------------------------------------

    for label in ax.get_yticklabels():

        if label.get_text() in [
            "Order-Based",
            "Proximity-Based"
        ]:

            label.set_fontweight(
                "bold"
            )


    # --------------------------------------------------------
    # ARI values
    # --------------------------------------------------------

    for bar, score, method in zip(
        bars,
        domain_df["ARI"],
        domain_df["Algorithm"]
    ):

        label = f"{score:.2f}"

        if method == best_method:
            label += " ★"

        ax.text(
            score + 0.005,
            bar.get_y()
            + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight=(
                "bold"
                if method == best_method
                else "normal"
            )
        )


    # --------------------------------------------------------
    # Domain title
    # --------------------------------------------------------

    ax.set_title(
        f"Domain: {domain}",
        fontweight="bold",
        pad=5
    )


    # --------------------------------------------------------
    # Winner annotation
    # --------------------------------------------------------

    ax.text(
        0.98,
        0.93,
        f"Best: {best_method} ({best_score:.2f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color="darkred",
        fontweight="bold"
    )


    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        x_max
    )

    ax.set_xlabel(
        "Average ARI"
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.5,
        alpha=0.45
    )

    ax.grid(
        axis="y",
        visible=False
    )

    ax.set_axisbelow(True)


    # --------------------------------------------------------
    # Spines
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 11. REMOVE EMPTY SUBPLOTS
# ============================================================

for ax in axes[n_domains:]:
    ax.remove()


# ============================================================
# 12. FINAL LAYOUT
# ============================================================

plt.tight_layout(
    h_pad=2.0,
    w_pad=1.5
)


# ============================================================
# 13. SAVE FOR SPRINGER
# ============================================================

plt.savefig(
    "domain_algorithm_comparison.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "domain_algorithm_comparison.png",
    dpi=600,
    bbox_inches="tight"
)


plt.show()

plt.close()