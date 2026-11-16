import pandas as pd
import io

# 1. Load the data
# Replacing the string below with your actual file path: df = pd.read_csv('your_file.csv')


# For demonstration, I will assume the data is in a variable called 'data_string'
# df = pd.read_csv(io.StringIO(data_string))

# 2. Define the Mapping Dictionary to resolve duplicates
name_mapping = {
    # Novel Divisive (Order)
    'Novel divisive clustering algorithm(order based)': 'Novel Divisive (Order)',
    
    # Novel Divisive (Prox)
    'Novel Divisive Clustering (proximity based': 'Novel Divisive (Prox)',
    'Novel Divisive Clustering (proximity based)': 'Novel Divisive (Prox)',
    
    # Agglomerative Ward
    'AgglomerativeClustering (ward)': 'Agglo (ward)',
    
    # Agglomerative Complete
    'AgglomerativeClustering (complete)': 'Agglo (complete)',
    
    # Agglomerative Average
    'AgglomerativeClustering (average)': 'Agglo (average)',
    'Agglo(average)': 'Agglo (average)',
    
    # Agglomerative Single
    'AgglomerativeClustering (single)': 'Agglo (single)',
    'Agglo(single)': 'Agglo (single)',
    
    # Standardizing casing/spacing for density-based
    'density-based': 'Density-Based'
}

def clean_and_export(df):
    # Apply the mapping
    df['Algorithm'] = df['Algorithm'].replace(name_mapping)
    
    # Remove any leading/trailing whitespace that might cause hidden duplicates
    df['Algorithm'] = df['Algorithm'].str.strip()
    df['Dataset'] = df['Dataset'].str.strip()

    # 3. Handle Duplicate rows
    # If a Dataset + Algorithm combination appears twice after renaming, 
    # we take the mean of the scores (or you can use .first())
    df_cleaned = df.groupby(['Dataset', 'Algorithm', 'Dataset_Type'], as_index=False).agg({
        'ARI': 'mean',
        'Silhouette': 'mean'
    })

    # 4. Save to CSV
    df_cleaned.to_csv('cleaned_clustering_results.csv', index=False)
    print("File 'cleaned_clustering_results.csv' has been created successfully.")
    return df_cleaned

# Example usage:
df = pd.read_csv('categorized_benchmarks-selected_1.csv')
clean_df = clean_and_export(df)
