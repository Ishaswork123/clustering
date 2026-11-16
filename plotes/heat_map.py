import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load and Fix Missing Data
df = pd.read_csv('algorithm_baseline_summary.csv')

# Create a full grid to ensure every Algorithm/Dataset combo exists
algorithms = df['Algorithm'].unique()
types = df['Dataset_Type'].unique()
full_index = pd.MultiIndex.from_product([algorithms, types], names=['Algorithm', 'Dataset_Type'])
full_df = pd.DataFrame(index=full_index).reset_index()

# Merge and CRITICAL: Use merged_df for all steps below
merged_df = pd.merge(full_df, df, on=['Algorithm', 'Dataset_Type'], how='left')
merged_df['ARI'] = merged_df['ARI'].fillna(0.0)

# 2. Generate the Summary Data (using merged_df)
summary_data = []
for dt in types:
    subset = merged_df[merged_df['Dataset_Type'] == dt]
    best = subset.loc[subset['ARI'].idxmax()]
    worst = subset.loc[subset['ARI'].idxmin()]
    summary_data.append([
        dt, 
        best['Algorithm'], 
        round(best['ARI'], 3), 
        worst['Algorithm'], 
        round(worst['ARI'], 3)
    ])

summary_cols = ['Dataset Type', 'Best Method', 'Best ARI', 'Worst Method', 'Worst ARI']

# 3. Visualization: Heatmap + Embedded Table
# Increase figure height to make room for the table at the bottom
fig, (ax_heat, ax_table) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

# --- Heatmap ---
heatmap_data = merged_df.pivot(index='Algorithm', columns='Dataset_Type', values='ARI')
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
            linewidths=.5, cbar_kws={'label': 'ARI Score'}, ax=ax_heat)

ax_heat.set_title('Clustering Performance: Heatmap & Summary', fontsize=16, pad=20)
ax_heat.set_xlabel('Dataset Category', fontsize=12)
ax_heat.set_ylabel('Algorithm', fontsize=12)

# --- Table ---
ax_table.axis('off') # Hide axes for the table area
tbl = ax_table.table(cellText=summary_data, colLabels=summary_cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5) # Adjust scaling to make it look like a clean report

# 4. Save and Show
plt.tight_layout()
plt.savefig('performance_report_with_table.png', dpi=300)
plt.show()