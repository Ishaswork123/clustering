import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Custom divisive clustering implementations
class DivisiveClusteringBenchmark:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def kmeans_divisive(self, X: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
        """Divisive clustering using KMeans binary splits"""
        labels = np.zeros(len(X), dtype=int)
        
        def split_cluster(cluster_indices, cluster_id, depth):
            if len(cluster_indices) <= 1 or cluster_id >= n_clusters:
                return
            
            X_cluster = X[cluster_indices]
            
            if len(X_cluster) <= 2:
                labels[cluster_indices] = cluster_id
                return
            
            # Binary split using KMeans
            kmeans = KMeans(n_clusters=2, random_state=self.random_state, max_iter=max_iter)
            sub_labels = kmeans.fit_predict(X_cluster)
            
            # Assign new cluster IDs
            mask_0 = cluster_indices[sub_labels == 0]
            mask_1 = cluster_indices[sub_labels == 1]
            
            labels[mask_0] = cluster_id
            
            # Recursively split the larger cluster
            if len(mask_0) > len(mask_1):
                split_cluster(mask_0, cluster_id + 1, depth + 1)
                labels[mask_1] = cluster_id + 2
                split_cluster(mask_1, cluster_id + 3, depth + 1)
            else:
                split_cluster(mask_1, cluster_id + 1, depth + 1)
                labels[mask_0] = cluster_id + 2
                split_cluster(mask_0, cluster_id + 3, depth + 1)
        
        split_cluster(np.arange(len(X)), 0, 0)
        
        # Remap labels to be consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])
    
    def kmedoids_divisive(self, X: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
        """Divisive clustering using K-Medoids binary splits"""
        from sklearn_extra.cluster import KMedoids
        
        labels = np.zeros(len(X), dtype=int)
        
        def split_cluster(cluster_indices, cluster_id, depth):
            if len(cluster_indices) <= 1 or cluster_id >= n_clusters:
                return
            
            X_cluster = X[cluster_indices]
            
            if len(X_cluster) <= 2:
                labels[cluster_indices] = cluster_id
                return
            
            # Binary split using KMedoids
            try:
                kmedoids = KMedoids(n_clusters=2, random_state=self.random_state, max_iter=max_iter)
                sub_labels = kmedoids.fit_predict(X_cluster)
                
                mask_0 = cluster_indices[sub_labels == 0]
                mask_1 = cluster_indices[sub_labels == 1]
                
                labels[mask_0] = cluster_id
                
                if len(mask_0) > len(mask_1):
                    split_cluster(mask_0, cluster_id + 1, depth + 1)
                    labels[mask_1] = cluster_id + 2
                    split_cluster(mask_1, cluster_id + 3, depth + 1)
                else:
                    split_cluster(mask_1, cluster_id + 1, depth + 1)
                    labels[mask_0] = cluster_id + 2
                    split_cluster(mask_0, cluster_id + 3, depth + 1)
                    
            except Exception as e:
                # Fallback to KMeans if KMedoids fails
                kmeans = KMeans(n_clusters=2, random_state=self.random_state)
                sub_labels = kmeans.fit_predict(X_cluster)
        
        split_cluster(np.arange(len(X)), 0, 0)
        
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels])
    
    def diana_divisive(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """DIANA (DIvisive ANAlysis) algorithm implementation"""
        from scipy.spatial.distance import cdist
        
        labels = np.zeros(len(X), dtype=int)
        cluster_queue = [np.arange(len(X))]
        current_label = 0
        
        while cluster_queue and current_label < n_clusters - 1:
            cluster_indices = cluster_queue.pop(0)
            
            if len(cluster_indices) <= 1:
                labels[cluster_indices] = current_label
                current_label += 1
                continue
            
            X_cluster = X[cluster_indices]
            
            # Find the most dissimilar point (splinter group seed)
            pairwise_dist = cdist(X_cluster, X_cluster, metric='cosine')
            avg_distance = pairwise_dist.mean(axis=1)
            seed_idx = np.argmax(avg_distance)
            
            splinter_group = [seed_idx]
            remaining_group = [i for i in range(len(cluster_indices)) if i != seed_idx]
            
            # Iteratively move points to splinter group
            changed = True
            while changed and len(splinter_group) < len(cluster_indices):
                changed = False
                
                # Calculate average distances
                if len(splinter_group) > 0 and len(remaining_group) > 0:
                    splinter_dist = pairwise_dist[remaining_group][:, splinter_group].mean(axis=1)
                    remaining_dist = pairwise_dist[remaining_group][:, remaining_group].mean(axis=1)
                    
                    # Find points closer to splinter group
                    move_indices = np.where(splinter_dist < remaining_dist)[0]
                    
                    if len(move_indices) > 0:
                        # Move one point at a time (largest difference)
                        diff = remaining_dist[move_indices] - splinter_dist[move_indices]
                        move_idx = move_indices[np.argmax(diff)]
                        
                        splinter_group.append(remaining_group[move_idx])
                        remaining_group.pop(move_idx)
                        changed = True
            
            # Convert indices back to original indices
            splinter_original = cluster_indices[splinter_group]
            remaining_original = cluster_indices[remaining_group]
            
            # Assign labels
            if len(splinter_original) > 0:
                labels[splinter_original] = current_label
                current_label += 1
                if len(splinter_original) > 1 and current_label < n_clusters:
                    cluster_queue.append(splinter_original)
            
            if len(remaining_original) > 0:
                if current_label < n_clusters:
                    labels[remaining_original] = current_label
                    current_label += 1
                    if len(remaining_original) > 1:
                        cluster_queue.append(remaining_original)
                else:
                    labels[remaining_original] = current_label - 1
        
        return labels
    
    def bisecting_kmeans(self, X: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
        """Bisecting K-Means divisive clustering"""
        labels = np.zeros(len(X), dtype=int)
        cluster_sets = [np.arange(len(X))]
        
        for k in range(1, n_clusters):
            # Find the largest cluster to split
            largest_idx = np.argmax([len(c) for c in cluster_sets])
            cluster_indices = cluster_sets.pop(largest_idx)
            
            if len(cluster_indices) <= 1:
                cluster_sets.append(cluster_indices)
                continue
            
            X_cluster = X[cluster_indices]
            
            # Split using KMeans
            kmeans = KMeans(n_clusters=2, random_state=self.random_state, max_iter=max_iter)
            sub_labels = kmeans.fit_predict(X_cluster)
            
            # Create two new clusters
            for i in range(2):
                new_cluster = cluster_indices[sub_labels == i]
                cluster_sets.append(new_cluster)
        
        # Assign final labels
        for label, cluster_indices in enumerate(cluster_sets):
            labels[cluster_indices] = label
        
        return labels

class BenchmarkExperiment:
    def __init__(self):
        self.benchmark = DivisiveClusteringBenchmark()
        self.results = []
        
    def generate_datasets(self):
        """Generate synthetic datasets for benchmarking"""
        datasets = {}
        
        # Well-separated Gaussian blobs
        X1, y1 = make_blobs(n_samples=1000, centers=5, cluster_std=0.8, 
                           random_state=42, n_features=2)
        datasets['Gaussian_Blobs'] = (X1, y1)
        
        # Different density clusters
        X2, y2 = make_blobs(n_samples=1000, centers=5, cluster_std=[1.0, 0.5, 1.5, 0.3, 0.8],
                           random_state=42, n_features=2)
        datasets['Variable_Density'] = (X2, y2)
        
        # Non-spherical clusters (moons)
        X3, y3 = make_moons(n_samples=1000, noise=0.1, random_state=42)
        datasets['Moons'] = (X3, y3)
        
        # Concentric circles
        X4, y4 = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=42)
        datasets['Circles'] = (X4, y4)
        
        # High-dimensional data
        X5, y5 = make_blobs(n_samples=1000, centers=5, n_features=10, 
                           cluster_std=1.0, random_state=42)
        datasets['High_Dim'] = (X5, y5)
        
        return datasets
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality using multiple metrics"""
        n_clusters = len(np.unique(labels))
        
        if n_clusters < 2 or n_clusters >= len(X):
            return {
                'silhouette': -1,
                'calinski_harabasz': -1,
                'davies_bouldin': float('inf')
            }
        
        try:
            silhouette = silhouette_score(X, labels)
        except:
            silhouette = -1
            
        try:
            calinski = calinski_harabasz_score(X, labels)
        except:
            calinski = -1
            
        try:
            davies = davies_bouldin_score(X, labels)
        except:
            davies = float('inf')
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        }
    
    def run_experiment(self, dataset_name: str, X: np.ndarray, y_true: np.ndarray, 
                      n_clusters: int = 5, n_runs: int = 5):
        """Run benchmarking experiment on a single dataset"""
        
        algorithms = {
            'KMeans_Divisive': lambda: self.benchmark.kmeans_divisive(X, n_clusters),
            'KMedoids_Divisive': lambda: self.benchmark.kmedoids_divisive(X, n_clusters),
            'DIANA': lambda: self.benchmark.diana_divisive(X, n_clusters),
            'Bisecting_KMeans': lambda: self.benchmark.bisecting_kmeans(X, n_clusters),
            'Agglomerative_Ward': lambda: AgglomerativeClustering(
                n_clusters=n_clusters, linkage='ward').fit_predict(X),
            'Agglomerative_Complete': lambda: AgglomerativeClustering(
                n_clusters=n_clusters, linkage='complete').fit_predict(X)
        }
        
        for algo_name, algo_func in algorithms.items():
            print(f"  Testing {algo_name}...")
            
            metrics_list = []
            times = []
            
            for run in range(n_runs):
                start_time = time.time()
                
                try:
                    labels = algo_func()
                    exec_time = time.time() - start_time
                    times.append(exec_time)
                    
                    # Evaluate clustering
                    metrics = self.evaluate_clustering(X, labels)
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    print(f"    Error in {algo_name}, run {run}: {str(e)}")
                    continue
            
            if metrics_list and times:
                # Average metrics across runs
                avg_metrics = {}
                for key in metrics_list[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in metrics_list])
                
                avg_time = np.mean(times)
                
                self.results.append({
                    'dataset': dataset_name,
                    'algorithm': algo_name,
                    'silhouette': avg_metrics['silhouette'],
                    'calinski_harabasz': avg_metrics['calinski_harabasz'],
                    'davies_bouldin': avg_metrics['davies_bouldin'],
                    'time_seconds': avg_time,
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'true_clusters': len(np.unique(y_true))
                })
    
    def run_all_experiments(self, n_clusters: int = 5, n_runs: int = 3):
        """Run experiments on all datasets"""
        print("Generating datasets...")
        datasets = self.generate_datasets()
        
        print("Running experiments...")
        for name, (X, y) in datasets.items():
            print(f"\nDataset: {name}")
            print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.run_experiment(name, X_scaled, y, n_clusters, n_runs)
        
        return pd.DataFrame(self.results)
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Create comprehensive visualizations of results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Silhouette Scores by Algorithm
        ax1 = axes[0]
        silhouette_data = results_df.pivot_table(
            index='algorithm', columns='dataset', values='silhouette'
        )
        silhouette_data.plot(kind='bar', ax=ax1)
        ax1.set_title('Silhouette Scores by Algorithm and Dataset')
        ax1.set_ylabel('Silhouette Score (higher is better)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Execution Time Comparison
        ax2 = axes[1]
        time_data = results_df.pivot_table(
            index='algorithm', columns='dataset', values='time_seconds'
        )
        time_data.plot(kind='bar', ax=ax2, logy=True)
        ax2.set_title('Execution Time by Algorithm (log scale)')
        ax2.set_ylabel('Time (seconds, log scale)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Davies-Bouldin Index
        ax3 = axes[2]
        db_data = results_df.pivot_table(
            index='algorithm', columns='dataset', values='davies_bouldin'
        )
        db_data.plot(kind='bar', ax=ax3)
        ax3.set_title('Davies-Bouldin Index by Algorithm')
        ax3.set_ylabel('DB Index (lower is better)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Algorithm Performance Heatmap
        ax4 = axes[3]
        # Normalize metrics for comparison
        normalized_scores = results_df.copy()
        for metric in ['silhouette', 'calinski_harabasz']:
            normalized_scores[metric] = normalized_scores.groupby('dataset')[metric].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
        normalized_scores['davies_bouldin'] = normalized_scores.groupby('dataset')['davies_bouldin'].transform(
            lambda x: 1 - ((x - x.min()) / (x.max() - x.min()))
        )
        
        # Average normalized score
        normalized_scores['overall_score'] = normalized_scores[
            ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        ].mean(axis=1)
        
        heatmap_data = normalized_scores.pivot_table(
            index='algorithm', columns='dataset', values='overall_score'
        )
        im = ax4.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(heatmap_data.columns)))
        ax4.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax4.set_yticks(range(len(heatmap_data.index)))
        ax4.set_yticklabels(heatmap_data.index)
        ax4.set_title('Overall Performance Heatmap')
        plt.colorbar(im, ax=ax4, label='Normalized Score')
        
        # 5. Time vs Quality Scatter
        ax5 = axes[4]
        for dataset in results_df['dataset'].unique():
            subset = results_df[results_df['dataset'] == dataset]
            ax5.scatter(subset['time_seconds'], subset['silhouette'], 
                       label=dataset, s=100, alpha=0.6)
        
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Silhouette Score')
        ax5.set_title('Time vs Quality Trade-off')
        ax5.set_xscale('log')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Algorithm Ranking
        ax6 = axes[5]
        algorithm_scores = results_df.groupby('algorithm').agg({
            'silhouette': 'mean',
            'time_seconds': 'mean'
        })
        algorithm_scores['combined_score'] = (
            algorithm_scores['silhouette'] / algorithm_scores['time_seconds']
        )
        algorithm_scores['combined_score'].sort_values().plot(kind='barh', ax=ax6)
        ax6.set_xlabel('Silhouette/Time Ratio')
        ax6.set_title('Algorithm Efficiency Ranking')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary table
        summary = results_df.groupby('algorithm').agg({
            'silhouette': ['mean', 'std'],
            'time_seconds': ['mean', 'std'],
            'davies_bouldin': ['mean', 'std']
        }).round(4)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(summary)
        
        return summary

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("HIERARCHICAL DIVISIVE CLUSTERING BENCHMARK STUDY")
    print("="*80)
    
    # Initialize experiment
    experiment = BenchmarkExperiment()
    
    # Run all experiments
    results_df = experiment.run_all_experiments(n_clusters=5, n_runs=3)
    
    # Save raw results
    results_df.to_csv('clustering_benchmark_results.csv', index=False)
    print("\nRaw results saved to 'clustering_benchmark_results.csv'")
    
    # Display results
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS")
    print("="*80)
    print(results_df.to_string())
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    summary = experiment.visualize_results(results_df)
    
    # Generate LaTeX table for conference paper
    print("\n" + "="*80)
    print("LATEX TABLE FOR CONFERENCE PAPER")
    print("="*80)
    
    latex_table = results_df.pivot_table(
        index='algorithm',
        columns='dataset',
        values=['silhouette', 'time_seconds'],
        aggfunc='mean'
    ).round(4)
    
    print(latex_table.to_latex())
    
    # Additional analysis
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find best algorithm for each dataset
    for dataset in results_df['dataset'].unique():
        subset = results_df[results_df['dataset'] == dataset]
        best_silhouette = subset.loc[subset['silhouette'].idxmax()]
        fastest = subset.loc[subset['time_seconds'].idxmin()]
        
        print(f"\nDataset: {dataset}")
        print(f"  Best quality: {best_silhouette['algorithm']} "
              f"(Silhouette: {best_silhouette['silhouette']:.4f})")
        print(f"  Fastest: {fastest['algorithm']} "
              f"(Time: {fastest['time_seconds']:.4f}s)")