import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --------------------------------------------------
# Global Figure Font Settings
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12,

    "axes.labelsize": 13,
    "axes.titlesize": 14,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    "legend.fontsize": 11,

    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --------------------------------------------------
# Load File
# --------------------------------------------------

df = pd.read_csv("algorithm_baseline_summary.csv")

# --------------------------------------------------
# Clean Data
# --------------------------------------------------

df["Algorithm"] = df["Algorithm"].astype(str).str.strip()

# Remove duplicates
df = df.drop_duplicates(subset="Algorithm")

# ARI
df["ARI"] = pd.to_numeric(df["ARI"], errors="coerce")

# Remove missing values
df = df.dropna(subset=["ARI"])

# --------------------------------------------------
# Standardize Algorithm Names
# --------------------------------------------------

def standardize_name(name):

    name_lower = name.lower().strip()

    # Novel methods
    if "order" in name_lower:
        return "Order-Based"

    elif "proximity"  in name_lower:
        return "Proximity-Based"

    # Agglomerative methods
    elif "agglo" in name_lower:
        return name.strip()

    # Divisive methods
    elif "divisive" in name_lower:
        return name.strip()

    elif "diana" in name_lower:
        return name.strip()

    elif "density" in name_lower:
        return name.strip()

    else:
        return name.strip()


df["DisplayName"] = df["Algorithm"].apply(standardize_name)

# --------------------------------------------------
# Categorization
# --------------------------------------------------

def categorize(name):

    name_lower = name.lower()

    if "order" in name_lower or "proximity" in name_lower:
        return "Novel"

    elif "agglo" in name_lower or "agglomerative" in name_lower:
        return "Agglomerative"

    elif (
        "divisive" in name_lower
        or "diana" in name_lower
        or "density" in name_lower
    ):
        return "Divisive"

    else:
        return "Divisive"


df["Category"] = df["Algorithm"].apply(categorize)

# --------------------------------------------------
# Colors
# --------------------------------------------------

color_map = {
    "Agglomerative": "#E69F00",
    "Divisive": "#0072B2",
    "Novel": "#009E73"
}

df["Color"] = df["Category"].map(color_map)

# --------------------------------------------------
# Sort for Visualization
# --------------------------------------------------

df = df.sort_values(by="ARI")

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8.5, 6.5))

y_pos = np.arange(len(df))

bars = ax.barh(
    y_pos,
    df["ARI"],
    color=df["Color"],
    edgecolor="black",
    linewidth=0.6,
    height=0.65
)

# --------------------------------------------------
# Y-axis Labels
# --------------------------------------------------

ax.set_yticks(y_pos)
ax.set_yticklabels(df["DisplayName"])

# Make novel method labels bold
for label, category in zip(ax.get_yticklabels(), df["Category"]):

    if category == "Novel":
        label.set_fontweight("bold")

# --------------------------------------------------
# Axis Labels
# --------------------------------------------------

ax.set_xlabel("Average ARI", labelpad=8)

# No title here.
# Use the LaTeX figure caption instead.

# --------------------------------------------------
# Value Labels
# --------------------------------------------------

for i, v in enumerate(df["ARI"]):

    ax.text(
        v + 0.005,
        i,
        f"{v:.3f}",
        va="center",
        fontsize=11
    )

# --------------------------------------------------
# Legend
# --------------------------------------------------

legend_elements = [

    Patch(
        facecolor=color_map["Divisive"],
        edgecolor="black",
        label="Divisive"
    ),

    Patch(
        facecolor=color_map["Agglomerative"],
        edgecolor="black",
        label="Agglomerative"
    ),

    Patch(
        facecolor=color_map["Novel"],
        edgecolor="black",
        label="Novel Methods"
    )
]

ax.legend(
    handles=legend_elements,
    loc="lower right",
    frameon=True,
    framealpha=1,
    edgecolor="black"
)

# --------------------------------------------------
# Grid
# --------------------------------------------------

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.5
)

ax.set_axisbelow(True)

# --------------------------------------------------
# Clean Spines
# --------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --------------------------------------------------
# Layout
# --------------------------------------------------

plt.tight_layout()

# --------------------------------------------------
# Save for LaTeX / Springer
# --------------------------------------------------

plt.savefig(
    "bar_plot_avg_ARI.pdf",
    bbox_inches="tight"
)

# Optional high-resolution PNG
plt.savefig(
    "bar_plot_avg_ARI.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()