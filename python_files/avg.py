import pandas as pd

# Load your results file
input_file = "categorized_benchmarks.csv"
df = pd.read_csv(input_file)

# Check columns (optional)
print("Columns in file:", df.columns)

# Group by Algorithm and calculate average of all metrics
avg_df = df.groupby("Algorithm")[["ARI", "Silhouette"]].mean().reset_index()

# Round values for clean output
avg_df = avg_df.round(4)

# Sort by ARI (best algorithm on top)
avg_df = avg_df.sort_values(by="ARI", ascending=False)

# Save new file
output_file = "average_metrics_per_algorithm.csv"
avg_df.to_csv(output_file, index=False)

print("\n✅ Average metrics saved to:", output_file)
print("\n📊 Preview:")
print(avg_df)