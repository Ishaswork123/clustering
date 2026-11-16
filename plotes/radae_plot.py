import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('domain_algorithm_average_scores.csv')
df['Algorithm'] = df['Algorithm'].str.strip()

# ============================================================
# UPDATED MAPPING BASED ON YOUR CSV
# ============================================================
algorithm_mapping = {
    # Proximity Group - Catching the specific typos in your file
    'Novel Divisive (Prox)': 'Novel Divisive (Proximity)',
    'Novel Divisive Clustering (proximity based': 'Novel Divisive (Proximity)',
    'Novel Divisive Clustering (proximity based)': 'Novel Divisive (Proximity)',
    'Novel divisive clustering algorithm(proximity based)': 'Novel Divisive (Proximity)',

    # Order Group
    'Novel Divisive (Order)': 'Novel Divisive (Order)',
    'Novel divisive clustering algorithm(order based)': 'Novel Divisive (Order)',
    'Novel Divisive Clustering (Order Based)': 'Novel Divisive (Order)',

    # Density Group
    'density-based': 'Density-Based',
    'Density based': 'Density-Based',

    # Agglomerative variants
    'AgglomerativeClustering (average)': 'Agglo (average)',
    'AgglomerativeClustering (complete)': 'Agglo (complete)',
    'AgglomerativeClustering (single)': 'Agglo (single)',
    'AgglomerativeClustering (ward)': 'Agglo (ward)'
}

# Apply the manual mapping
df['Algorithm'] = df['Algorithm'].replace(algorithm_mapping)

# --- THE FAIL-SAFE FIX ---
# This looks for 'prox' (case insensitive) and merges them regardless of typos
df.loc[df['Algorithm'].str.contains('prox', case=False, na=False), 'Algorithm'] = 'Novel Divisive (Proximity)'
# -------------------------

# ============================================================
# NORMALIZATION & PIVOT
# ============================================================
group_min = df.groupby('Domain')['ARI'].transform('min')
group_max = df.groupby('Domain')['ARI'].transform('max')

df['Score'] = np.where(
    group_max > group_min,
    (df['ARI'] - group_min) / (group_max - group_min),
    1.0
)

# PIVOT: This will now merge all 'Novel Divisive (Proximity)' rows into ONE
pivot_df = df.pivot_table(
    index='Algorithm',
    columns='Domain',
    values='Score',
    aggfunc='mean'
).fillna(0)

# ============================================================
# RADAR SETUP & PLOTTING (Remains same as your original)
# ============================================================
categories = list(pivot_df.columns)
num_vars = len(categories)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

plt.xticks(angles[:-1], categories, size=11)


ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)
plt.ylim(0, 1.0)

colors = plt.cm.tab20(np.linspace(0, 1, len(pivot_df)))

for i, (algo, row) in enumerate(pivot_df.iterrows()):
    values = row.values.flatten().tolist()
    values += values[:1]
    
    lw = 3 if 'Novel' in algo else 1.5
    alpha_val = 0.15 if 'Novel' in algo else 0.05

    ax.plot(angles, values, linewidth=lw, label=algo, color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=alpha_val)

plt.title('Algorithm Performance Across Domains\n(Normalized ARI)', y=1.08, fontsize=15, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1.15, 1.05), fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig("radar.png", dpi=300, bbox_inches='tight')

plt.show()