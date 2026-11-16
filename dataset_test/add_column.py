import pandas as pd

# Load your cleaned file (or the merged one)
file_path = 'merged_output.csv'  # Update this path if needed
df = pd.read_csv(file_path)

# 1. Define your Mapping Logic
# You can expand these lists based on the 15+ UCI IDs we discussed
def categorize_dataset(name):
    name = str(name).lower()
    
    # High Dimensional (e.g., Image data or many features)
    if any(word in name for word in ['digits', 'optical', 'covertype', 'letter', 'high_dim']):
        return 'high_dim'
    
    # Real World (Standard UCI datasets)
    elif any(word in name for word in ['iris', 'wine', 'glass', 'cancer', 'bean', 'rice', 'stroke']):
        return 'real'
    
    # Synthetic (If you manually generated blobs/moons/circles)
    elif any(word in name for word in ['blobs', 'moons', 'circles', 'density']):
        return 'synthetic'
    
    else:
        return 'other'

# 2. Apply the Category Column
df['Dataset_Type'] = df['Dataset'].apply(categorize_dataset)

# 3. Calculate Performance by Type
# Grouping by Type and Algorithm to see the average ARI/Silhouette
type_analysis = df.groupby(['Dataset_Type', 'Algorithm'])[['ARI', 'Silhouette']].mean().reset_index()

# 4. Save the categorized results
df.to_csv('categorized_benchmarks.csv', index=False)

print("--- Performance Summary by Dataset Type ---")
print(type_analysis)

print("\n✅ Saved categorized results to 'categorized_benchmarks.csv'")