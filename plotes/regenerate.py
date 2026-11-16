import pandas as pd

# =========================
# Load files
# =========================
df_meta = pd.read_csv("clean_dataset_details.csv")
df_results = pd.read_csv("categorized_benchmarks-selected_1.csv")

# =========================
# Rename column for matching
# =========================
df_meta.rename(columns={"Name": "Dataset"}, inplace=True)

# =========================
# Remove old Dataset_Type column
# =========================
if "Dataset_Type" in df_results.columns:
    df_results.drop(columns=["Dataset_Type"], inplace=True)

# =========================
# Merge domain information
# =========================
df = df_results.merge(
    df_meta[["Dataset", "Domain"]],
    on="Dataset",
    how="left"
)

# =========================
# Fill missing domains
# =========================
df["Domain"] = df["Domain"].fillna("Unknown")

# =========================
# Calculate average scores
# Group by Domain + Algorithm
# =========================
final_df = df.groupby(
    ["Domain", "Algorithm"],
    as_index=False
).agg({
    "ARI": "mean",
    "Silhouette": "mean"
})

# =========================
# Round values (optional)
# =========================
final_df["ARI"] = final_df["ARI"].round(4)
final_df["Silhouette"] = final_df["Silhouette"].round(4)

# =========================
# Save final CSV
# =========================
final_df.to_csv(
    "domain_algorithm_average_scores.csv",
    index=False
)

print("File saved successfully!")
print(final_df.head())