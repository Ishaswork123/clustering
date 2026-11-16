
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Read CSV file
df = pd.read_csv("categorized_benchmarks-selected_1.csv" )


# Remove empty rows
df = df.dropna(subset=["Dataset"])

# Remove extra spaces
df["Dataset"] = df["Dataset"].str.strip()

# =========================
# STEP 2: REMOVE DUPLICATES
# =========================
# Keep only unique dataset names
unique_datasets = df["Dataset"].drop_duplicates()

print(f"Total Unique Datasets: {len(unique_datasets)}")

# =========================
# STEP 3: FETCH DETAILS
# =========================
results = []

# To avoid processing same dataset again
processed = set()

for dataset_name in unique_datasets:

    # Skip if already processed
    if dataset_name.lower() in processed:
        continue

    print(f"\nSearching: {dataset_name}")

    try:
        # Fetch dataset
        dataset = fetch_ucirepo(name=dataset_name)

        meta = dataset.metadata

        # Save results
        results.append({
            "Name": meta.name,
            "ID": meta.uci_id,
            "Instances": meta.num_instances,
            "Attributes": meta.num_features,
            "Domain": meta.area
        })

        # Mark as processed
        processed.add(dataset_name.lower())

        print(f"Found: {meta.name}")

    except Exception as e:

        print(f"Not Found: {dataset_name}")

        results.append({
            "Name": dataset_name,
            "ID": "Not Found",
            "Instances": "",
            "Attributes": "",
            "Domain": ""
        })

# =========================
# STEP 4: SAVE CLEAN FILE
# =========================
output_df = pd.DataFrame(results)

# Save clean dataset details
output_df.to_csv("clean_dataset_details.csv", index=False)

print("\nDone!")
print("Saved as: clean_dataset_details.csv")