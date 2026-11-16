import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# GLOBAL STYLE — CONSISTENT WITH PREVIOUS FIGURES
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 11,

    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 9,
    "legend.title_fontsize": 10,

    "axes.linewidth": 0.8,

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

    # Novel methods
    "Novel Divisive (Prox)": "Proximity-Based",
    "Novel Divisive Clustering (proximity based":
        "Proximity-Based",
    "Novel Divisive Clustering (proximity based)":
        "Proximity-Based",
    "Novel divisive clustering algorithm(proximity based)":
        "Proximity-Based",
    "Novel Divisive (Proximity)":
        "Proximity-Based",

    "Novel Divisive (Order)": "Order-Based",
    "Novel divisive clustering algorithm(order based)":
        "Order-Based",
    "Novel Divisive Clustering (Order Based)":
        "Order-Based",

    # Density
    "density-based": "Density-Based",
    "Density based": "Density-Based",

    # Agglomerative
    "AgglomerativeClustering (average)":
        "Agglo (Average)",
    "AgglomerativeClustering (complete)":
        "Agglo (Complete)",
    "AgglomerativeClustering (single)":
        "Agglo (Single)",
    "AgglomerativeClustering (ward)":
        "Agglo (Ward)"
}

df["Algorithm"] = df["Algorithm"].replace(mapping)


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
# 6. FIND BEST METHOD PER DOMAIN
# ============================================================

best_by_domain = (
    df.loc[
        df.groupby("Domain")["ARI"].idxmax()
    ]
    .reset_index(drop=True)
)


# ============================================================
# 7. ORDER ALGORITHMS
# ============================================================

# Overall average ARI is used only for ordering.
# This keeps the plot easy to read.

algorithm_order = (
    df.groupby("Algorithm")["ARI"]
    .mean()
    .sort_values()
    .index
    .tolist()
)


# ============================================================
# 8. CREATE SMALL, CLEAN FIGURE
# ============================================================

sns.set_style("whitegrid")

fig, ax = plt.subplots(
    figsize=(8.2, 6.8)
)


# ============================================================
# 9. GROUPED HORIZONTAL BARPLOT
# ============================================================

sns.barplot(
    data=df,
    y="Algorithm",
    x="ARI",
    hue="Domain",
    order=algorithm_order,
    palette="viridis",
    edgecolor="black",
    linewidth=0.35,
    dodge=True,
    ax=ax
)


# ============================================================
# 10. BOLD NOVEL METHODS
# ============================================================

for label in ax.get_yticklabels():

    if label.get_text() in [
        "Order-Based",
        "Proximity-Based"
    ]:

        label.set_fontweight("bold")


# ============================================================
# 11. MARK BEST METHOD IN EACH DOMAIN
# ============================================================

# Seaborn creates bars grouped by domain.
# We compare every bar with the best ARI
# of its corresponding domain.

for patch, (_, row) in zip(
    ax.patches,
    df.iterrows()
):

    domain = row["Domain"]
    ari = row["ARI"]

    best_ari = best_by_domain.loc[
        best_by_domain["Domain"] == domain,
        "ARI"
    ].iloc[0]

    if np.isclose(
        ari,
        best_ari
    ):

        # Highlight best result
        patch.set_edgecolor("black")
        patch.set_linewidth(1.8)


# ============================================================
# 12. VALUE LABELS
# ============================================================

for patch, (_, row) in zip(
    ax.patches,
    df.iterrows()
):

    width = patch.get_width()

    if np.isnan(width):
        continue

    domain = row["Domain"]
    ari = row["ARI"]

    best_ari = best_by_domain.loc[
        best_by_domain["Domain"] == domain,
        "ARI"
    ].iloc[0]

    is_best = np.isclose(
        ari,
        best_ari
    )

    # Keep labels short
    text = f"{ari:.2f}"

    if is_best:
        text += " ★"

    ax.annotate(
        text,
        (
            width,
            patch.get_y()
            + patch.get_height() / 2
        ),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight=(
            "bold" if is_best else "normal"
        )
    )


# ============================================================
# 13. AXES
# ============================================================

ax.set_xlabel(
    "Average ARI",
    labelpad=6
)

ax.set_ylabel(
    "Algorithm",
    labelpad=6
)


# ============================================================
# 14. REMOVE TITLE
# ============================================================

# No title here.
# Springer LaTeX caption will provide the title.


# ============================================================
# 15. GRID
# ============================================================

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.6,
    alpha=0.4
)

ax.grid(
    axis="y",
    visible=False
)

ax.set_axisbelow(True)


# ============================================================
# 16. CLEAN SPINES
# ============================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# 17. LEGEND AT BOTTOM
# ============================================================

ax.legend(
    title="Domain",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    frameon=False
)


# ============================================================
# 18. X-AXIS LIMIT
# ============================================================

max_ari = df["ARI"].max()

ax.set_xlim(
    0,
    max_ari * 1.18
)


# ============================================================
# 19. FINAL LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)


# ============================================================
# 20. SAVE
# ============================================================

plt.savefig(
    "cleaned_domain_performance.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "cleaned_domain_performance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

plt.close()