import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import math

# =========================
# 1. Load & Clean Data
# =========================
df = pd.read_csv('domain_algorithm_average_scores.csv')
df['Algorithm'] = df['Algorithm'].str.strip()

# Mapping to fix duplications found in your CSV
mapping = {
    'Novel Divisive (Prox)': 'Novel Divisive (Proximity)',
    'Novel Divisive Clustering (proximity based': 'Novel Divisive (Proximity)',
    'Novel Divisive Clustering (proximity based)': 'Novel Divisive (Proximity)',
    'Novel divisive clustering algorithm(proximity based)': 'Novel Divisive (Proximity)',
    'Novel Divisive (Order)': 'Novel Divisive (Order)',
    'Novel divisive clustering algorithm(order based)': 'Novel Divisive (Order)',
    'Novel Divisive Clustering (Order Based)': 'Novel Divisive (Order)',
    'density-based': 'Density-Based',
    'Density based': 'Density-Based',
    'AgglomerativeClustering (average)': 'Agglo (average)',
    'AgglomerativeClustering (complete)': 'Agglo (complete)',
    'AgglomerativeClustering (single)': 'Agglo (single)',
    'AgglomerativeClustering (ward)': 'Agglo (ward)'
}

df['Algorithm'] = df['Algorithm'].replace(mapping)

# Fail-safe: Merge any remaining 'prox' strings
df.loc[df['Algorithm'].str.contains('prox', case=False, na=False), 'Algorithm'] = 'Novel Divisive (Proximity)'

# CRITICAL: Group by to merge the duplicate ARI values into one mean score
df = df.groupby(['Algorithm', 'Domain'], as_index=False)['ARI'].mean()

# =========================
# 2. Algorithm Categories
# =========================
def categorize_algo(name):
    name_low = name.lower()
    if 'agglo' in name_low:
        return 'Agglomerative Methods'
    elif any(x in name_low for x in ['div', 'diana', 'novel']):
        return 'Divisive Methods'
    else:
        return 'Other Methods'

df['Method_Type'] = df['Algorithm'].apply(categorize_algo)

# =========================
# Plotting Setup
# =========================
palette = {
    'Agglomerative Methods': '#ff7f0e',
    'Divisive Methods': '#1f77b4',
    'Other Methods': '#2ca02c'
}

domains = sorted(df['Domain'].unique())
cols = 2
rows = math.ceil(len(domains) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
axes = axes.flatten()

# =========================
# 3. Draw Plots
# =========================
for i, domain in enumerate(domains):
    ax = axes[i]
    
    # Filter and sort so the strongest algorithm is at the top
    subset = df[df['Domain'] == domain].sort_values(by='ARI', ascending=False)

    sns.barplot(
        data=subset,
        x='ARI',
        y='Algorithm',
        hue='Method_Type',
        palette=palette,
        dodge=False,
        ax=ax,
        legend=False # Use global legend instead
    )

    ax.set_title(f"Domain: {domain}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Average ARI', fontsize=10)
    ax.set_ylabel('')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Set X-limit based on data range to show detail, or keep at 1.0 for consistency
    max_ari = df['ARI'].max()
# Change Line 83 to:
    ax.set_xlim(0, max_ari * 1.4)  # Increased from 1.1 to 1.4
    # Annotation for the winner
   # Annotation for the winner
    if not subset.empty:
        best_row = subset.iloc[0]
        
        # Position the text at the top right of the subplot area
        # 0.98 (x) is nearly all the way to the right
        # 0.95 (y) is nearly all the way to the top
        ax.text(
            0.98, 0.95, 
            f"Winner: {best_row['Algorithm']} ({best_row['ARI']:.3f})",
            transform=ax.transAxes, 
            ha='right', 
            va='top',
            fontsize=9, 
            color='darkred', 
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver', pad=3)
        )
# Remove unused subplots
for j in range(len(domains), len(axes)):
    fig.delaxes(axes[j])

# =========================
# 4. Global Legend & Final Touch
# =========================
legend_elements = [
    Line2D([0], [0], color='#1f77b4', lw=8, label='Divisive Methods'),
    Line2D([0], [0], color='#ff7f0e', lw=8, label='Agglomerative Methods'),
    Line2D([0], [0], color='#2ca02c', lw=8, label='Other Methods')
]

fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=3,
    fontsize=12,
    frameon=True,
    bbox_to_anchor=(0.5, 0.02)
)

# Line 116: The main title at the very top
plt.suptitle('...', fontsize=22, fontweight='bold', y=0.98)

# Line 119: rect=[left, bottom, right, top] 
# Increase 0.06 to give more room for the legend at the bottom
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('domain_performance_faceted.png', dpi=300, bbox_inches='tight')
plt.show()