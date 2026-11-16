"""
Enhanced Hierarchical Divisive Clustering Benchmark
Added comprehensive visualization and analysis functions
"""

import time
import argparse
import os
import math
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools

from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Try to import KMedoids
try:
    from sklearn_extra.cluster import KMedoids
    _have_kmedoids = 'sklearn_extra'
except Exception:
    _have_kmedoids = None
    try:
        from pyclustering.cluster.kmedoids import kmedoids as py_kmedoids
        from pyclustering.utils.metric import type_metric, distance_metric
        _have_kmedoids = 'pyclustering'
    except Exception:
        _have_kmedoids = None

# ----- Utility functions -----

def purity_score(y_true, y_pred):
    """Calculate clustering purity"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_pred)
    total = len(y_true)
    correct = 0
    for l in labels:
        mask = (y_pred == l)
        if mask.sum() == 0:
            continue
        true_labels, counts = np.unique(y_true[mask], return_counts=True)
        correct += counts.max()
    return correct / total

# DIANA (simplified) implementation
def diana_bipartition(X, dist_matrix=None):
    """DIANA algorithm for binary partition"""
    n = X.shape[0]
    if dist_matrix is None:
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X)
    else:
        D = dist_matrix
    C = set(range(n))
    S = set()
    avg_diss = D.mean(axis=1)
    seed = int(np.argmax(avg_diss))
    S.add(seed)
    C.remove(seed)
    moved = True
    while moved:
        moved = False
        for i in list(C):
            diss_to_S = np.mean([D[i, s] for s in S]) if len(S) > 0 else np.inf
            if len(C) > 1:
                diss_to_C = np.mean([D[i, c] for c in C if c != i])
            else:
                diss_to_C = 0.0
            if diss_to_S > diss_to_C:
                S.add(i)
                C.remove(i)
                moved = True
    labels = np.zeros(n, dtype=int)
    for s in S:
        labels[s] = 1
    return labels

# Divisive Tree class (recursive binary splits)
class DivisiveTree:
    def __init__(self, splitter, max_depth=100, min_size=2, name='divisive'):
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_size = min_size
        self.name = name
        self.tree = {}
        self.next_id = 0

    def fit(self, X, desired_clusters=2):
        n = X.shape[0]
        root = self._new_node(list(range(n)))
        import heapq
        heap = [(-len(self.tree[root][0]), root, 0)]
        clusters = 1
        while heap and clusters < desired_clusters:
            size, node_id, depth = heapq.heappop(heap)
            size = -size
            indices, l, r = self.tree[node_id]
            if depth >= self.max_depth or size < self.min_size*2:
                continue
            subX = X[indices]
            try:
                split_labels = self.splitter(subX)
            except Exception:
                from sklearn.metrics import pairwise_distances
                D = pairwise_distances(subX)
                split_labels = self.splitter(subX, dist_matrix=D)
            if len(np.unique(split_labels)) < 2:
                continue
            left_idx = [indices[i] for i in range(len(indices)) if split_labels[i] == 0]
            right_idx = [indices[i] for i in range(len(indices)) if split_labels[i] == 1]
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue
            left_id = self._new_node(left_idx)
            right_id = self._new_node(right_idx)
            self.tree[node_id] = (indices, left_id, right_id)
            heapq.heappush(heap, (-len(self.tree[left_id][0]), left_id, depth+1))
            heapq.heappush(heap, (-len(self.tree[right_id][0]), right_id, depth+1))
            clusters += 1
        labels = np.zeros(n, dtype=int)
        leaves = self._get_leaves()
        for cid, node_id in enumerate(leaves):
            for idx in self.tree[node_id][0]:
                labels[idx] = cid
        return labels

    def _new_node(self, indices):
        nid = self.next_id
        self.tree[nid] = (indices, None, None)
        self.next_id += 1
        return nid

    def _get_leaves(self):
        leaves = [nid for nid, v in self.tree.items() if v[1] is None and v[2] is None]
        return leaves

# Splitter implementations
def splitter_kmeans(subX, dist_matrix=None):
    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    lab = km.fit_predict(subX)
    return lab

def splitter_kmedoids(subX, dist_matrix=None):
    if _have_kmedoids == 'sklearn_extra':
        km = KMedoids(n_clusters=2, method='pam', random_state=0)
        return km.fit_predict(subX)
    elif _have_kmedoids == 'pyclustering':
        n = subX.shape[0]
        if n <= 2:
            return np.arange(n)
        import random
        initial = random.sample(range(n), 2)
        metric = distance_metric(type_metric.EUCLIDEAN)
        try:
            result = py_kmedoids(subX.tolist(), initial, metric=metric)
            result.process()
            clusters = result.get_clusters()
            labels = np.zeros(n, dtype=int)
            for i, cl in enumerate(clusters):
                for idx in cl:
                    labels[idx] = i
            if len(np.unique(labels)) < 2:
                return splitter_kmeans(subX)
            return labels
        except Exception:
            return splitter_kmeans(subX)
    else:
        return splitter_kmeans(subX)

def splitter_diana(subX, dist_matrix=None):
    return diana_bipartition(subX, dist_matrix)

def splitter_spectral(subX, dist_matrix=None):
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    sc = SpectralClustering(n_clusters=2, random_state=0, n_init=5, assign_labels='kmeans')
    try:
        lab = sc.fit_predict(subX)
    except Exception:
        lab = splitter_kmeans(subX)
    return lab

def splitter_ward_binary(subX, dist_matrix=None):
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    ac = AgglomerativeClustering(n_clusters=2, linkage='ward')
    try:
        return ac.fit_predict(subX)
    except Exception:
        return splitter_kmeans(subX)

def splitter_density(subX, dist_matrix=None):
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(10, max(2, subX.shape[0]-1))).fit(subX)
        distances, _ = nbrs.kneighbors(subX)
        eps = np.median(distances[:, -1])
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(subX)
        unique = np.unique(labels[labels >= 0])
        if len(unique) >= 1:
            counts = {lab: np.sum(labels == lab) for lab in unique}
            largest = max(counts.items(), key=lambda x: x[1])[0]
            out = np.zeros(subX.shape[0], dtype=int)
            out[labels == largest] = 1
            if len(np.unique(out)) < 2:
                return splitter_kmeans(subX)
            return out
        else:
            return splitter_kmeans(subX)
    except Exception:
        return splitter_kmeans(subX)

def splitter_gmm(subX, dist_matrix=None):
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    try:
        gm = GaussianMixture(n_components=2, random_state=0)
        lab = gm.fit_predict(subX)
        return lab
    except Exception:
        return splitter_kmeans(subX)

# Agglomerative baseline
def agglomerative_flat(X, n_clusters=2, linkage='ward'):
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return ac.fit_predict(X)

# Enhanced evaluation metrics
def evaluate_flat(true_labels, pred_labels, data):
    res = {}
    res['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    res['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
    res['Purity'] = purity_score(true_labels, pred_labels)
    try:
        if len(np.unique(pred_labels)) > 1 and len(np.unique(pred_labels)) < len(pred_labels):
            res['Silhouette'] = silhouette_score(data, pred_labels)
            res['Calinski_Harabasz'] = calinski_harabasz_score(data, pred_labels)
            res['Davies_Bouldin'] = davies_bouldin_score(data, pred_labels)
        else:
            res['Silhouette'] = np.nan
            res['Calinski_Harabasz'] = np.nan
            res['Davies_Bouldin'] = np.nan
    except Exception:
        res['Silhouette'] = np.nan
        res['Calinski_Harabasz'] = np.nan
        res['Davies_Bouldin'] = np.nan
    return res

# Enhanced benchmark functions
def run_divisive_experiment(X, y, method_name, splitter_fn, ks=[2,3,4], repeats=3):
    records = []
    Xs = StandardScaler().fit_transform(X)
    for rep in range(repeats):
        for k in ks:
            start = time.time()
            tree = DivisiveTree(splitter_fn, name=method_name)
            lab = tree.fit(Xs, desired_clusters=k)
            elapsed = time.time() - start
            stats = evaluate_flat(y, lab, Xs)
            stats.update({'method': method_name, 'k': k, 'repeat': rep, 
                         'n_samples': X.shape[0], 'n_features': X.shape[1], 
                         'time': elapsed, 'time_per_sample': elapsed / X.shape[0]})
            records.append(stats)
    return records

def run_agglomerative_baseline(X, y, linkage, ks=[2,3,4], repeats=3):
    records = []
    Xs = StandardScaler().fit_transform(X)
    for rep in range(repeats):
        for k in ks:
            start = time.time()
            lab = agglomerative_flat(Xs, n_clusters=k, linkage=linkage)
            elapsed = time.time() - start
            stats = evaluate_flat(y, lab, Xs)
            stats.update({'method': f'Agglo-{linkage}', 'k': k, 'repeat': rep, 
                         'n_samples': X.shape[0], 'n_features': X.shape[1], 
                         'time': elapsed, 'time_per_sample': elapsed / X.shape[0]})
            records.append(stats)
    return records

# Enhanced dataset generation
def make_datasets():
    datasets = []
    # Synthetic datasets
    X1, y1 = make_blobs(n_samples=500, centers=4, random_state=0, cluster_std=1.0)
    datasets.append(('blobs_4', X1, y1))
    
    X2, y2 = make_moons(n_samples=500, noise=0.08, random_state=0)
    datasets.append(('moons', X2, y2))
    
    X3, y3 = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=0)
    datasets.append(('circles', X3, y3))
    
    # More challenging synthetic datasets
    X4, y4 = make_blobs(n_samples=500, centers=5, random_state=42, cluster_std=[0.5, 1.5, 0.8, 1.2, 0.3])
    datasets.append(('variable_density', X4, y4))
    
    # Real datasets
    iris = load_iris()
    datasets.append(('iris', iris.data, iris.target))
    
    wine = load_wine()
    datasets.append(('wine', wine.data, wine.target))
    
    digits = load_digits()
    Xd = digits.data
    yd = digits.target
    sel = np.random.RandomState(0).choice(len(yd), size=500, replace=False)
    datasets.append(('digits500', Xd[sel], yd[sel]))
    
    # High-dimensional dataset
    X5, y5 = make_blobs(n_samples=300, centers=3, n_features=20, random_state=42)
    datasets.append(('high_dim', X5, y5))
    
    return datasets

# ==================== ENHANCED VISUALIZATION FUNCTIONS ====================

def create_comprehensive_visualizations(df, output_dir='comprehensive_plots'):
    """Create comprehensive visualizations for conference paper"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Overall Performance Summary
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Average ARI across datasets
    ax1 = plt.subplot(3, 3, 1)
    avg_performance = df.groupby('method').agg({'ARI': 'mean', 'time': 'mean'}).reset_index()
    avg_performance = avg_performance.sort_values('ARI', ascending=False)
    
    # Color bars based on algorithm type
    divisive_methods = [m for m in avg_performance['method'] if m.startswith('Div-')]
    agglo_methods = [m for m in avg_performance['method'] if m.startswith('Agglo-')]
    
    colors = []
    for method in avg_performance['method']:
        if method.startswith('Div-'):
            colors.append('#1f77b4')  # Blue for divisive
        elif method.startswith('Agglo-'):
            colors.append('#ff7f0e')  # Orange for agglomerative
        else:
            colors.append('#2ca02c')  # Green for others
    
    bars = ax1.barh(avg_performance['method'], avg_performance['ARI'], color=colors)
    ax1.set_xlabel('Average ARI')
    ax1.set_title('Overall Algorithm Performance', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, 1)
    
    # Add ARI values on bars
    for bar, ari in zip(bars, avg_performance['ARI']):
        ax1.text(ari + 0.01, bar.get_y() + bar.get_height()/2, f'{ari:.3f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add legend for algorithm types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Divisive Methods'),
        Patch(facecolor='#ff7f0e', label='Agglomerative Methods')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Plot 2: Runtime comparison
    ax2 = plt.subplot(3, 3, 2)
    runtime_data = df.groupby(['method', 'k'])['time'].mean().unstack()
    
    # Sort methods by average runtime
    runtime_data['avg_time'] = runtime_data.mean(axis=1)
    runtime_data = runtime_data.sort_values('avg_time', ascending=False)
    runtime_data = runtime_data.drop('avg_time', axis=1)
    
    # Create bar plot with different colors for each k
    x = np.arange(len(runtime_data.index))
    width = 0.2
    k_values = sorted(df['k'].unique())
    
    for i, k in enumerate(k_values):
        if k in runtime_data.columns:
            offset = width * (i - len(k_values)/2 + 0.5)
            ax2.bar(x + offset, runtime_data[k], width, label=f'k={k}')
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Runtime by Method and K', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(runtime_data.index, rotation=45, ha='right', fontsize=9)
    ax2.legend(title='K')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Performance heatmap
    ax3 = plt.subplot(3, 3, 3)
    heatmap_data = df.pivot_table(index='method', columns='dataset', values='ARI', aggfunc='mean')
    
    # Sort methods by average performance
    heatmap_data['avg_ari'] = heatmap_data.mean(axis=1)
    heatmap_data = heatmap_data.sort_values('avg_ari', ascending=False)
    heatmap_data = heatmap_data.drop('avg_ari', axis=1)
    
    im = ax3.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax3.set_xticks(range(len(heatmap_data.columns)))
    ax3.set_xticklabels(heatmap_data.columns, rotation=45, ha='right', fontsize=9)
    ax3.set_yticks(range(len(heatmap_data.index)))
    ax3.set_yticklabels(heatmap_data.index, fontsize=9)
    ax3.set_title('Performance Heatmap (ARI)', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('ARI Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            text = ax3.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Plot 4: Quality metrics correlation
    ax4 = plt.subplot(3, 3, 4)
    metrics = ['ARI', 'NMI', 'Purity', 'Silhouette']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        corr_data = df[available_metrics].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax4, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax4.set_title('Metrics Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Plot 5: Time vs Quality scatter
    ax5 = plt.subplot(3, 3, 5)
    methods = df['method'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    
    # Calculate average time and ARI for each method
    avg_stats = df.groupby('method').agg({'time': 'mean', 'ARI': 'mean'}).reset_index()
    
    for i, row in avg_stats.iterrows():
        method = row['method']
        avg_time = row['time']
        avg_ari = row['ARI']
        
        # Determine marker based on algorithm type
        marker = 'o' if method.startswith('Div-') else 's'
        size = 120 if method.startswith('Div-') else 100
        
        ax5.scatter(avg_time, avg_ari, s=size, alpha=0.8, 
                   label=method, color=colors[i], marker=marker, 
                   edgecolors='black', linewidth=0.5)
        
        # Add method name
        ax5.annotate(method, (avg_time, avg_ari), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.8)
    
    ax5.set_xlabel('Average Time (s)', fontsize=10)
    ax5.set_ylabel('Average ARI', fontsize=10)
    ax5.set_title('Time-Quality Trade-off', fontsize=12, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance by dataset type
    ax6 = plt.subplot(3, 3, 6)
    dataset_types = {
        'Synthetic': ['blobs_4', 'moons', 'circles', 'variable_density'],
        'Real': ['iris', 'wine', 'digits500'],
        'High-dim': ['high_dim']
    }
    
    performance_by_type = {}
    for dtype, dnames in dataset_types.items():
        subset = df[df['dataset'].isin(dnames)]
        if not subset.empty:
            performance_by_type[dtype] = subset.groupby('method')['ARI'].mean()
    
    # Create grouped bar chart
    if performance_by_type:
        methods_for_chart = list(avg_stats.sort_values('ARI', ascending=False)['method'].head(8))
        x = np.arange(len(methods_for_chart))
        width = 0.25
        
        for i, (dtype, perf) in enumerate(performance_by_type.items()):
            offset = (i - 1) * width
            values = [perf.get(m, 0) for m in methods_for_chart]
            ax6.bar(x + offset, values, width, label=dtype, alpha=0.8)
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(methods_for_chart, rotation=45, ha='right', fontsize=9)
        ax6.set_ylabel('Average ARI', fontsize=10)
        ax6.set_title('Performance by Dataset Type', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Scalability analysis
    ax7 = plt.subplot(3, 3, 7)
    if 'n_samples' in df.columns:
        sample_data = df.groupby(['method', 'n_samples']).agg({'time': 'mean'}).reset_index()
        
        # Get top 5 methods by ARI
        top_methods = df.groupby('method')['ARI'].mean().nlargest(5).index
        
        for method in top_methods:
            method_data = sample_data[sample_data['method'] == method]
            if len(method_data) > 1:
                ax7.plot(method_data['n_samples'], method_data['time'], 
                        'o-', label=method, linewidth=2, markersize=6)
        
        ax7.set_xlabel('Sample Size', fontsize=10)
        ax7.set_ylabel('Time (s)', fontsize=10)
        ax7.set_title('Scalability Analysis (Top 5 Methods)', fontsize=12, fontweight='bold')
        ax7.set_xscale('log')
        ax7.set_yscale('log')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: K-value impact
    ax8 = plt.subplot(3, 3, 8)
    k_performance = df.groupby(['method', 'k'])['ARI'].mean().reset_index()
    
    # Get top 3 methods
    top_methods_k = df.groupby('method')['ARI'].mean().nlargest(3).index
    
    for method in top_methods_k:
        method_data = k_performance[k_performance['method'] == method]
        if len(method_data) > 0:
            ax8.plot(method_data['k'], method_data['ARI'], 
                    'o-', label=method, linewidth=2, markersize=8)
    
    ax8.set_xlabel('K (number of clusters)', fontsize=10)
    ax8.set_ylabel('Average ARI', fontsize=10)
    ax8.set_title('Impact of K on Performance (Top 3)', fontsize=12, fontweight='bold')
    ax8.set_xticks(sorted(df['k'].unique()))
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Statistical significance heatmap
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate win rates between methods
    methods_for_matrix = list(avg_stats.sort_values('ARI', ascending=False)['method'].head(6))
    win_matrix = np.zeros((len(methods_for_matrix), len(methods_for_matrix)))
    
    for i, m1 in enumerate(methods_for_matrix):
        for j, m2 in enumerate(methods_for_matrix):
            if i != j:
                m1_scores = df[df['method'] == m1]['ARI'].dropna().values
                m2_scores = df[df['method'] == m2]['ARI'].dropna().values
                
                if len(m1_scores) > 1 and len(m2_scores) > 1:
                    try:
                        t_stat, p_val = stats.mannwhitneyu(m1_scores, m2_scores, 
                                                         alternative='greater')
                        if p_val < 0.05:
                            win_matrix[i, j] = 1
                    except:
                        pass
    
    im = ax9.imshow(win_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
    ax9.set_xticks(range(len(methods_for_matrix)))
    ax9.set_xticklabels(methods_for_matrix, rotation=45, ha='right', fontsize=9)
    ax9.set_yticks(range(len(methods_for_matrix)))
    ax9.set_yticklabels(methods_for_matrix, fontsize=9)
    ax9.set_title('Statistical Wins Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(methods_for_matrix)):
        for j in range(len(methods_for_matrix)):
            if i != j:
                text = ax9.text(j, i, 'W' if win_matrix[i, j] == 1 else '',
                              ha="center", va="center", 
                              color="white" if win_matrix[i, j] == 1 else "black", 
                              fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'comprehensive_summary.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # 2. Dataset-specific visualizations
    print("\nCreating dataset-specific visualizations...")
    for dataset_name in df['dataset'].unique():
        dataset_subset = df[df['dataset'] == dataset_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Available metrics
        available_metrics = ['ARI', 'NMI', 'Purity', 'Silhouette', 'Calinski_Harabasz', 'Davies_Bouldin']
        available_metrics = [m for m in available_metrics if m in dataset_subset.columns]
        
        # Create one plot for each available metric
        for idx, metric in enumerate(available_metrics[:6]):
            ax = axes[idx]
            
            # Pivot data for plotting
            metric_data = dataset_subset.pivot_table(
                index='method', 
                columns='k', 
                values=metric, 
                aggfunc='mean'
            )
            
            # Sort methods by average metric value
            metric_data['avg'] = metric_data.mean(axis=1)
            metric_data = metric_data.sort_values('avg', ascending=False)
            metric_data = metric_data.drop('avg', axis=1)
            
            # Plot
            metric_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{metric} for {dataset_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='K', fontsize=9, title_fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Improve y-axis for Davies_Bouldin (lower is better)
            if metric == 'Davies_Bouldin':
                ax.invert_yaxis()
        
        # If we have fewer than 6 metrics, hide empty subplots
        for idx in range(len(available_metrics), 6):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Dataset: {dataset_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dataset_{dataset_name}_metrics.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'dataset_{dataset_name}_metrics.pdf'), 
                    bbox_inches='tight')
        plt.close()
    
    # 3. Create interactive visualizations with plotly
    print("Creating interactive visualizations...")
    create_interactive_visualizations(df, output_dir)
    
    # 4. Additional specialized visualizations
    create_additional_visualizations(df, output_dir)
    
    print(f"\n✓ Comprehensive visualizations saved to {output_dir}")

def create_additional_visualizations(df, output_dir):
    """Create additional specialized visualizations"""
    
    # 1. Performance distribution by method
    fig, ax = plt.subplots(figsize=(14, 8))
    
    methods = df['method'].unique()
    ari_data = [df[df['method'] == m]['ARI'].dropna().values for m in methods]
    
    # Create violin plot
    parts = ax.violinplot(ari_data, showmeans=True, showmedians=True)
    
    # Customize colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plt.cm.tab20(i / len(methods)))
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('ARI Score', fontsize=12)
    ax.set_title('Distribution of ARI Scores by Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Runtime vs Sample Size
    if 'n_samples' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by method and sample size
        runtime_data = df.groupby(['method', 'n_samples'])['time'].mean().unstack()
        
        # Plot lines for each method
        for method in runtime_data.index:
            ax.plot(runtime_data.columns, runtime_data.loc[method], 
                   'o-', label=method, linewidth=2, markersize=6)
        
        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Runtime vs Sample Size', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'runtime_vs_samples.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Algorithm ranking by efficiency (ARI/Time)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    efficiency_data = df.groupby('method').agg({
        'ARI': 'mean',
        'time': 'mean'
    })
    efficiency_data['efficiency'] = efficiency_data['ARI'] / efficiency_data['time']
    efficiency_data = efficiency_data.sort_values('efficiency', ascending=True)
    
    bars = ax.barh(range(len(efficiency_data)), efficiency_data['efficiency'])
    
    # Color bars
    for i, (method, _) in enumerate(efficiency_data.iterrows()):
        if method.startswith('Div-'):
            bars[i].set_color('#1f77b4')
        elif method.startswith('Agglo-'):
            bars[i].set_color('#ff7f0e')
    
    ax.set_yticks(range(len(efficiency_data)))
    ax.set_yticklabels(efficiency_data.index, fontsize=10)
    ax.set_xlabel('Efficiency (ARI / Time)', fontsize=12)
    ax.set_title('Algorithm Efficiency Ranking', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add efficiency values
    for i, (bar, eff) in enumerate(zip(bars, efficiency_data['efficiency'])):
        ax.text(eff, i, f' {eff:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_ranking.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_visualizations(df, output_dir):
    """Create interactive plotly visualizations"""
    try:
        # Interactive performance comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ARI vs NMI', 'Time vs Quality', 
                           'Performance by K', 'Dataset Comparison'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Plot 1: ARI vs NMI
        avg_metrics = df.groupby('method').agg({'ARI': 'mean', 'NMI': 'mean'}).reset_index()
        fig.add_trace(
            go.Scatter(
                x=avg_metrics['ARI'],
                y=avg_metrics['NMI'],
                mode='markers+text',
                text=avg_metrics['method'],
                marker=dict(
                    size=15, 
                    color=avg_metrics['ARI'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='ARI')
                ),
                textposition='top center'
            ),
            row=1, col=1
        )
        
        # Plot 2: Time vs Quality
        # Create categorical color mapping
        methods = df['method'].unique()
        color_map = {method: i for i, method in enumerate(methods)}
        
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df['ARI'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=[color_map[m] for m in df['method']],
                    colorscale='Set1',  # This is a valid colorscale name
                    showscale=True,
                    colorbar=dict(
                        title='Method',
                        tickvals=list(range(len(methods))),
                        ticktext=list(methods)
                    )
                ),
                text=df['method'] + '<br>' + df['dataset'],
                hoverinfo='text'
            ),
            row=1, col=2
        )
        
        # Plot 3: Performance by K
        k_performance = df.groupby(['k', 'method'])['ARI'].mean().reset_index()
        colors = px.colors.qualitative.Set1
        
        for i, method in enumerate(k_performance['method'].unique()):
            subset = k_performance[k_performance['method'] == method]
            color_idx = i % len(colors)
            fig.add_trace(
                go.Scatter(
                    x=subset['k'],
                    y=subset['ARI'],
                    mode='lines+markers',
                    name=method,
                    line=dict(color=colors[color_idx], width=2),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
        
        # Plot 4: Dataset comparison
        dataset_performance = df.groupby(['dataset', 'method'])['ARI'].mean().reset_index()
        pivot_data = dataset_performance.pivot(index='method', columns='dataset', values='ARI')
        
        fig.add_trace(
            go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn',
                zmin=0, 
                zmax=1,
                colorbar=dict(title='ARI')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800, 
            width=1200,
            showlegend=True, 
            title_text="Interactive Benchmark Analysis",
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="ARI", row=1, col=1)
        fig.update_yaxes(title_text="NMI", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="ARI", row=1, col=2)
        fig.update_xaxes(title_text="K", row=2, col=1)
        fig.update_yaxes(title_text="ARI", row=2, col=1)
        fig.update_xaxes(title_text="Dataset", row=2, col=2)
        fig.update_yaxes(title_text="Method", row=2, col=2)
        
        # Save interactive plot
        fig.write_html(os.path.join(output_dir, "interactive_analysis.html"))
        
        # Create parallel coordinates plot
        if len(df) > 0:
            # Prepare data for parallel coordinates
            parallel_data = df.copy()
            
            # Convert method to categorical for coloring
            parallel_data['method_cat'] = parallel_data['method'].astype('category').cat.codes
            
            # Select metrics for parallel coordinates
            metrics_for_parallel = ['ARI', 'NMI', 'Purity', 'Silhouette', 'time']
            available_metrics = [m for m in metrics_for_parallel if m in parallel_data.columns]
            
            if available_metrics:
                parallel_fig = px.parallel_coordinates(
                    parallel_data,
                    dimensions=available_metrics,
                    color='ARI',
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    labels={col: col for col in available_metrics},
                    title='Parallel Coordinates Plot of Clustering Performance'
                )
                parallel_fig.write_html(os.path.join(output_dir, "parallel_coordinates.html"))
        
        # Create 3D scatter plot (if we have appropriate data)
        if 'n_features' in df.columns and 'time' in df.columns:
            scatter_3d_data = df.groupby(['method', 'n_samples', 'n_features']).agg({
                'ARI': 'mean',
                'time': 'mean'
            }).reset_index()
            
            if len(scatter_3d_data) > 0:
                scatter_3d = px.scatter_3d(
                    scatter_3d_data,
                    x='n_samples',
                    y='n_features',
                    z='ARI',
                    color='method',
                    size='time',
                    hover_name='method',
                    title='3D Performance Analysis',
                    labels={
                        'n_samples': 'Sample Size',
                        'n_features': 'Number of Features',
                        'ARI': 'Average ARI',
                        'time': 'Time (s)'
                    }
                )
                scatter_3d.write_html(os.path.join(output_dir, "3d_analysis.html"))
        
        print(f"Interactive visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Warning: Could not create interactive visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

# ==================== ENHANCED ANALYSIS FUNCTIONS ====================

def perform_statistical_analysis(df, output_dir='statistical_analysis'):
    """Perform comprehensive statistical analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # 1. Friedman test for algorithm comparison
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*80)
    
    methods = df['method'].unique()
    datasets = df['dataset'].unique()
    
    # Prepare data for Friedman test
    friedman_data = []
    for dataset in datasets:
        dataset_scores = []
        for method in methods:
            score = df[(df['dataset'] == dataset) & 
                      (df['method'] == method)]['ARI'].mean()
            dataset_scores.append(score)
        friedman_data.append(dataset_scores)
    
    if len(friedman_data) > 1 and len(friedman_data[0]) > 1:
        friedman_stat, p_value = stats.friedmanchisquare(*friedman_data)
        
        print(f"\nFriedman Test Results:")
        print(f"  Statistic: {friedman_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("  Significant differences exist between algorithms (p < 0.05)")
            
            # Post-hoc Nemenyi test
            k = len(methods)
            N = len(datasets)
            q_alpha = 2.948  # For α=0.05, k=8
            CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
            
            print(f"\nPost-hoc Nemenyi Test:")
            print(f"  Critical Distance (CD): {CD:.4f}")
            
            # Calculate average ranks
            ranks = []
            for dataset_scores in friedman_data:
                rank_data = stats.rankdata([-s for s in dataset_scores])
                ranks.append(rank_data)
            
            avg_ranks = np.mean(ranks, axis=0)
            
            print("\nAverage Ranks (1 = best):")
            for method, rank in zip(methods, avg_ranks):
                print(f"  {method}: {rank:.2f}")
            
            # Find significantly different pairs
            print("\nSignificantly Different Pairs (difference > CD):")
            for i, j in itertools.combinations(range(len(methods)), 2):
                if abs(avg_ranks[i] - avg_ranks[j]) > CD:
                    print(f"  {methods[i]} (rank {avg_ranks[i]:.2f}) vs "
                          f"{methods[j]} (rank {avg_ranks[j]:.2f})")
                    
                    # Perform pairwise t-test
                    scores_i = df[df['method'] == methods[i]]['ARI'].values
                    scores_j = df[df['method'] == methods[j]]['ARI'].values
                    t_stat, p_val = stats.ttest_ind(scores_i, scores_j, equal_var=False)
                    
                    results.append({
                        'test': 'pairwise_t_test',
                        'method1': methods[i],
                        'method2': methods[j],
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'mean_diff': np.mean(scores_i) - np.mean(scores_j)
                    })
        else:
            print("  No significant differences between algorithms")
    
    # 2. Performance by dataset type
    print("\n" + "-"*80)
    print("PERFORMANCE BY DATASET TYPE")
    print("-"*80)
    
    dataset_types = {
        'synthetic': ['blobs_4', 'moons', 'circles', 'variable_density'],
        'real': ['iris', 'wine', 'digits500'],
        'high_dim': ['high_dim']
    }
    
    for dtype, dnames in dataset_types.items():
        subset = df[df['dataset'].isin(dnames)]
        if not subset.empty:
            best_method = subset.groupby('method')['ARI'].mean().idxmax()
            best_score = subset.groupby('method')['ARI'].mean().max()
            print(f"\n{dtype.upper()} datasets:")
            print(f"  Best method: {best_method} (ARI: {best_score:.4f})")
            
            # Compare divisive vs agglomerative
            divisive_methods = [m for m in methods if m.startswith('Div-')]
            agglo_methods = [m for m in methods if m.startswith('Agglo-')]
            
            div_scores = subset[subset['method'].isin(divisive_methods)]['ARI'].mean()
            agg_scores = subset[subset['method'].isin(agglo_methods)]['ARI'].mean()
            
            print(f"  Average divisive: {div_scores:.4f}")
            print(f"  Average agglomerative: {agg_scores:.4f}")
            print(f"  Difference: {div_scores - agg_scores:.4f}")
    
    # 3. Scalability analysis
    print("\n" + "-"*80)
    print("SCALABILITY ANALYSIS")
    print("-"*80)
    
    if 'n_samples' in df.columns:
        scalability_results = []
        for method in methods:
            subset = df[df['method'] == method]
            if len(subset) > 1:
                # Estimate time complexity
                samples = subset['n_samples'].unique()
                times = [subset[subset['n_samples'] == s]['time'].mean() for s in samples]
                
                if len(samples) > 1 and np.all(np.array(times) > 0):
                    # Fit log-log model: log(time) = a + b*log(n)
                    coeffs = np.polyfit(np.log(samples), np.log(times), 1)
                    exponent = coeffs[0]
                    complexity = f"O(n^{exponent:.2f})"
                    
                    print(f"\n{method}:")
                    print(f"  Time complexity: {complexity}")
                    print(f"  Samples tested: {samples}")
                    print(f"  Average times: {times}")
                    
                    scalability_results.append({
                        'method': method,
                        'exponent': exponent,
                        'complexity': complexity
                    })
        
        # Save scalability results
        scalability_df = pd.DataFrame(scalability_results)
        if not scalability_df.empty:
            scalability_df = scalability_df.sort_values('exponent')
            scalability_df.to_csv(os.path.join(output_dir, 'scalability_analysis.csv'), index=False)
            print(f"\nScalability results saved to {output_dir}/scalability_analysis.csv")
    
    # 4. Save statistical results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'statistical_results.csv'), index=False)
    
    print(f"\nStatistical analysis complete. Results saved to {output_dir}")

def generate_conference_tables(df, output_dir='conference_tables'):
    """Generate publication-ready LaTeX tables"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary table of best performers
    summary = df.groupby(['dataset', 'method']).agg({
        'ARI': ['mean', 'std'],
        'NMI': ['mean', 'std'],
        'time': ['mean', 'std']
    }).round(3)
    
    # Find best method for each dataset
    best_per_dataset = []
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        best_method = subset.groupby('method')['ARI'].mean().idxmax()
        best_ari = subset.groupby('method')['ARI'].mean().max()
        best_per_dataset.append({
            'dataset': dataset,
            'best_method': best_method,
            'ARI': best_ari,
            'avg_time': subset[subset['method'] == best_method]['time'].mean()
        })
    
    best_df = pd.DataFrame(best_per_dataset)
    
    # 2. Generate LaTeX tables
    # Table 1: Overall performance
    overall_perf = df.groupby('method').agg({
        'ARI': 'mean',
        'NMI': 'mean',
        'Purity': 'mean',
        'time': 'mean'
    }).round(3).sort_values('ARI', ascending=False)
    
    latex_table1 = overall_perf.to_latex(
        caption='Overall performance of clustering algorithms',
        label='tab:overall_performance',
        position='htbp'
    )
    
    with open(os.path.join(output_dir, 'table_overall.tex'), 'w') as f:
        f.write(latex_table1)
    
    # Table 2: Best method per dataset
    latex_table2 = best_df.to_latex(
        columns=['dataset', 'best_method', 'ARI', 'avg_time'],
        caption='Best performing algorithm for each dataset',
        label='tab:best_per_dataset',
        index=False,
        float_format="%.3f"
    )
    
    with open(os.path.join(output_dir, 'table_best_per_dataset.tex'), 'w') as f:
        f.write(latex_table2)
    
    # Table 3: Statistical comparison
    methods = df['method'].unique()
    comparison_matrix = pd.DataFrame(index=methods, columns=methods)
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                comparison_matrix.loc[m1, m2] = '-'
            else:
                scores1 = df[df['method'] == m1]['ARI'].values
                scores2 = df[df['method'] == m2]['ARI'].values
                t_stat, p_val = stats.ttest_ind(scores1, scores2, equal_var=False)
                if p_val < 0.05:
                    comparison_matrix.loc[m1, m2] = 'W' if np.mean(scores1) > np.mean(scores2) else 'L'
                else:
                    comparison_matrix.loc[m1, m2] = 'D'
    
    latex_table3 = comparison_matrix.to_latex(
        caption='Pairwise comparison of algorithms (W: win, L: loss, D: draw)',
        label='tab:pairwise_comparison'
    )
    
    with open(os.path.join(output_dir, 'table_pairwise.tex'), 'w') as f:
        f.write(latex_table3)
    
    print(f"Conference tables saved to {output_dir}")

def create_performance_profiles(df, output_dir='performance_profiles'):
    """Create performance profiles for algorithm comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['ARI', 'NMI', 'Purity', 'Silhouette']
    methods = df['method'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if metric in df.columns:
            ax = axes[idx]
            
            # Calculate performance ratios
            dataset_methods = []
            performance_ratios = []
            
            for dataset in df['dataset'].unique():
                dataset_data = df[df['dataset'] == dataset]
                if not dataset_data.empty:
                    best_score = dataset_data[metric].max()
                    if best_score > 0:
                        for method in methods:
                            method_score = dataset_data[dataset_data['method'] == method][metric].mean()
                            if not np.isnan(method_score):
                                ratio = method_score / best_score
                                dataset_methods.append((dataset, method, ratio))
            
            # Create performance profile
            ratios_df = pd.DataFrame(dataset_methods, columns=['dataset', 'method', 'ratio'])
            
            # Plot cumulative distribution
            for method in methods:
                method_ratios = ratios_df[ratios_df['method'] == method]['ratio'].values
                if len(method_ratios) > 0:
                    method_ratios.sort()
                    y = np.arange(1, len(method_ratios) + 1) / len(method_ratios)
                    ax.plot(method_ratios, y, label=method, linewidth=2)
            
            ax.set_xlabel(f'{metric} Ratio (algorithm/best)')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'Performance Profile: {metric}')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_profiles.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance profiles saved to {output_dir}")

# ==================== MAIN ENHANCED RUNNER ====================

def enhanced_main(args):
    """Enhanced main function with comprehensive analysis"""
    
    # Generate datasets
    datasets = make_datasets()
    
    # Define algorithms
    divisive_methods = [
        ('Div-KMeans', splitter_kmeans),
        ('Div-KMedoids', splitter_kmedoids) if _have_kmedoids else None,
        ('DIANA', splitter_diana),
        ('Div-Spectral', splitter_spectral),
        ('Div-Ward', splitter_ward_binary),
        ('Div-Density', splitter_density),
        ('Div-GMM', splitter_gmm)
    ]
    
    # Remove None entries
    divisive_methods = [m for m in divisive_methods if m is not None]
    
    agg_linkages = ['ward', 'complete', 'average', 'single']
    ks = [2, 3, 4, 5]
    
    all_records = []
    tasks = []
    
    # Prepare tasks
    print("Preparing experiment tasks...")
    for name, X, y in datasets:
        for mname, fn in divisive_methods:
            tasks.append(('div', name, X, y, mname, fn))
        for link in agg_linkages:
            tasks.append(('agg', name, X, y, link, None))
    
    def worker(task):
        typ, dname, X, y, m, fn = task
        if typ == 'div':
            recs = run_divisive_experiment(X, y, m, fn, ks=ks, repeats=args.repeats)
        else:
            recs = run_agglomerative_baseline(X, y, m, ks=ks, repeats=args.repeats)
        for r in recs:
            r['dataset'] = dname
        return recs
    
    # Run experiments
    print(f"Running {len(tasks)} clustering experiments...")
    if args.n_jobs == 1:
        for t in tqdm(tasks):
            all_records.extend(worker(t))
    else:
        n_jobs = mp.cpu_count() if args.n_jobs == -1 else args.n_jobs
        print(f"Using {n_jobs} parallel jobs")
        results = Parallel(n_jobs=n_jobs)(delayed(worker)(t) for t in tqdm(tasks))
        for r in results:
            all_records.extend(r)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    df.to_csv(args.out, index=False)
    print(f"\n✓ Results saved to {args.out}")
    print(f"  Total records: {len(df)}")
    print(f"  Methods tested: {df['method'].nunique()}")
    print(f"  Datasets: {df['dataset'].nunique()}")
    
    # Generate comprehensive visualizations
    print("\nGenerating comprehensive visualizations...")
    create_comprehensive_visualizations(df, args.plots)
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    perform_statistical_analysis(df, 'statistical_analysis')
    
    # Generate conference tables
    print("\nGenerating conference tables...")
    generate_conference_tables(df, args.tex_dir)
    
    # Create performance profiles
    print("\nCreating performance profiles...")
    create_performance_profiles(df, 'performance_profiles')
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE - SUMMARY")
    print("="*80)
    
    # Top performers
    top_methods = df.groupby('method')['ARI'].mean().nlargest(5)
    print("\nTop 5 performing algorithms (by ARI):")
    for method, score in top_methods.items():
        avg_time = df[df['method'] == method]['time'].mean()
        print(f"  {method}: ARI={score:.3f}, Time={avg_time:.3f}s")
    
    # Best for each dataset type
    print("\nBest algorithms by dataset type:")
    dataset_types = {
        'Synthetic': ['blobs_4', 'moons', 'circles', 'variable_density'],
        'Real': ['iris', 'wine', 'digits500'],
        'High-dim': ['high_dim']
    }
    
    for dtype, dnames in dataset_types.items():
        subset = df[df['dataset'].isin(dnames)]
        if not subset.empty:
            best_method = subset.groupby('method')['ARI'].mean().idxmax()
            best_score = subset.groupby('method')['ARI'].mean().max()
            avg_time = subset[subset['method'] == best_method]['time'].mean()
            print(f"  {dtype}: {best_method} (ARI={best_score:.3f}, Time={avg_time:.3f}s)")
    
    # Fastest algorithm
    fastest = df.groupby('method')['time'].mean().nsmallest(1)
    print(f"\nFastest algorithm: {fastest.index[0]} (Time={fastest.values[0]:.3f}s)")
    
    # Most consistent algorithm (lowest std in ARI)
    consistency = df.groupby('method')['ARI'].std().nsmallest(1)
    print(f"Most consistent algorithm: {consistency.index[0]} (Std={consistency.values[0]:.3f})")
    
    print(f"\nAll results and visualizations are available in:")
    print(f"  - Results CSV: {args.out}")
    print(f"  - Visualizations: {args.plots}/")
    print(f"  - Statistical analysis: statistical_analysis/")
    print(f"  - Conference tables: {args.tex_dir}/")
    print(f"  - Performance profiles: performance_profiles/")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Hierarchical Divisive Clustering Benchmark')
    parser.add_argument('--out', type=str, default='enhanced_benchmark_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--plots', type=str, default='enhanced_plots',
                       help='Directory to save comprehensive plots')
    parser.add_argument('--tex_dir', type=str, default='conference_tables',
                       help='Directory to save LaTeX tables')
    parser.add_argument('--repeats', type=int, default=3,
                       help='Number of repetitions for each experiment')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Specific datasets to run (optional)')
    
    args = parser.parse_args()
    
    # Run enhanced benchmark
    results_df = enhanced_main(args)