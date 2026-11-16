
import pandas as pd

df=pd.read_csv('cleaned_clustering_results.csv')
baseline = df.groupby(['Algorithm', 'Dataset_Type'], as_index=False).agg({
    'ARI': 'mean',
    'Silhouette': 'mean'
})

# 4. Final Formatting
# Rounding for a cleaner CSV (optional)
baseline['ARI'] = baseline['ARI'].round(5)
baseline['Silhouette'] = baseline['Silhouette'].round(5)

# Sort by best performing ARI
baseline = baseline.sort_values(by='ARI', ascending=False)

# 5. Save to CSV
baseline.to_csv('algorithm_baseline_summary.csv', index=False)

print("Baseline CSV 'algorithm_baseline_summary.csv' created successfully.")
print("\nTop 5 Algorithms by Average ARI:")
print(baseline.head())