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
from sklearn.model_selection import train_test_split
import openpyxl
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
from sklearn.metrics import pairwise_distances ,silhouette_score
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


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


# ==================== ENHANCED VISUALIZATION FUNCTIONS ====================


  
    
class NovelDivisiveClustering:
    """
    Novel divisive clustering algorithms with order-based and proximity-based splitting criteria.
    
    Parameters:
    -----------
    split_cri : str, either 'rbs' or 'pbs'
        Splitting criterion to use
    max_depth : int, default=100
        Maximum depth of divisive tree
    min_size : int, default=2
        Minimum cluster size for splitting
    """
    
    def __init__(self, split_cri='rbs', max_depth=100, min_size=2):
        self.split_cri = split_cri
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = {}
        self.next_id = 0
        self.D = None  # Distance matrix
        
    def fit(self, X, desired_clusters=2):
        """
        Fit the divisive clustering model.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        desired_clusters : int
            Number of clusters desired
            
        Returns:
        --------
        labels : array of shape (n_samples,)
            Cluster labels
        """
        n_samples = X.shape[0]
        
        # Precompute distance matrix
        self.D = pairwise_distances(X, metric='euclidean')
        
        # Initialize with all indices
        root_indices = list(range(n_samples))
        root_id = self._create_node(root_indices)
        
        # Priority queue based on cluster size
        import heapq
        heap = [(-len(root_indices), root_id, 0)]  # Negative for max-heap
        
        current_clusters = 1
        
        while heap and current_clusters < desired_clusters:
            neg_size, node_id, depth = heapq.heappop(heap)
            size = -neg_size
            indices, left_id, right_id = self.tree[node_id]
            
            # Stop if depth limit reached or cluster too small
            if depth >= self.max_depth or size < self.min_size * 2:
                continue
            
            # Split the cluster
            try:
                child0_indices, child1_indices = self._split_cluster(indices)
            except Exception as e:
                print(f"Split failed: {e}")
                continue
            
            if len(child0_indices) == 0 or len(child1_indices) == 0:
                continue
            
            # Create child nodes
            child0_id = self._create_node(child0_indices)
            child1_id = self._create_node(child1_indices)
            
            # Update tree
            self.tree[node_id] = (indices, child0_id, child1_id)
            
            # Push children to heap
            heapq.heappush(heap, (-len(child0_indices), child0_id, depth + 1))
            heapq.heappush(heap, (-len(child1_indices), child1_id, depth + 1))
            
            current_clusters += 1
        
        # Generate final labels
        labels = np.zeros(n_samples, dtype=int)
        leaves = self._get_leaves()
        for cluster_id, node_id in enumerate(leaves):
            for idx in self.tree[node_id][0]:
                labels[idx] = cluster_id
        
        return labels
    
    def _split_cluster(self, parent_indices):
        """
        Split a cluster using either order-based or proximity-based criterion.
        
        Parameters:
        -----------
        parent_indices : list
            Indices of points in the current cluster
            
        Returns:
        --------
        child0, child1 : lists
            Indices for the two child clusters
        """
        n_elem = len(parent_indices)
        
        if n_elem < 2:
            return parent_indices, []
        
        # Extract submatrix for current cluster
        sub_mat = self.D[np.ix_(parent_indices, parent_indices)]
        
        # Find the two most distant points as seeds
        ind_0, ind_1 = np.unravel_index(np.argmax(sub_mat), sub_mat.shape)
        elem_0 = parent_indices[ind_0]
        elem_1 = parent_indices[ind_1]
        
        # Initialize children with seeds
        child_0 = [elem_0]
        child_1 = [elem_1]
        
        # Remove seeds from parent list (working copy)
        remaining = list(parent_indices)
        remaining.remove(elem_0)
        remaining.remove(elem_1)
        
        if len(remaining) == 0:
            return child_0, child_1
        
        # Get indices of remaining elements
        sel_inds = (np.arange(n_elem) != ind_0) & (np.arange(n_elem) != ind_1)
        
        # Extract distance rows to each seed
        row_0 = sub_mat[ind_0, sel_inds]
        row_1 = sub_mat[ind_1, sel_inds]
        
        # Get remaining indices in parent order
        remaining_indices = [i for i in range(n_elem) if i != ind_0 and i != ind_1]
        remaining_elements = [parent_indices[i] for i in remaining_indices]
        
        # Sort elements by proximity to each seed
        sort_order_0 = np.argsort(row_0)
        sort_order_1 = np.argsort(row_1)
        
        prox_elem_0 = [remaining_elements[i] for i in sort_order_0]
        prox_elem_1 = [remaining_elements[i] for i in sort_order_1]
        
        # Sorted distance values
        row_sort_0 = np.sort(row_0)
        row_sort_1 = np.sort(row_1)
        
        if self.split_cri == 'rbs':
            """
            Rank-Based Split:
            Points are assigned based on their rank ordering relative to each seed.
            
            For each point, we compare its position in the sorted proximity list
            to seed 0 versus seed 1. If it ranks higher (closer) to seed 0,
            it joins child_0. If ranks are equal, the actual distance values
            determine assignment.
            
            Rationale: This method is robust to outliers because it uses ordering
            rather than absolute distances, making it suitable for data with
            varying densities.
            """
            processed = set()
            
            for i, elem in enumerate(prox_elem_0):
                if elem in processed:
                    continue
                    
                # Find position in other list
                try:
                    pos_in_1 = prox_elem_1.index(elem)
                except ValueError:
                    continue
                
                # Compare ranks
                if pos_in_1 > i:
                    child_0.append(elem)
                elif pos_in_1 < i:
                    child_1.append(elem)
                else:
                    # Equal ranks: compare actual distances
                    if row_sort_1[i] >= row_sort_0[i]:
                        child_0.append(elem)
                    else:
                        child_1.append(elem)
                
                processed.add(elem)
            
        elif self.split_cri == 'pbs':
            """
            Proximity-Based Split:
            Points are assigned by directly comparing their distances to each seed.
            
            For each point, we compare its distance to seed 0 versus seed 1.
            If distance to seed 0 is smaller, it joins child_0. When distances
            are equal, we fall back to rank comparison.
            
            Rationale: This method is more sensitive to the actual distance
            magnitudes, making it better for well-separated clusters where
            absolute distances provide clear separation.
            """
            processed = set()
            
            for i, elem in enumerate(prox_elem_0):
                if elem in processed:
                    continue
                
                # Compare distances
                if row_sort_1[i] > row_sort_0[i]:
                    child_0.append(elem)
                elif row_sort_1[i] < row_sort_0[i]:
                    child_1.append(elem)
                else:
                    # Equal distances: use rank comparison
                    try:
                        pos_in_1 = prox_elem_1.index(elem)
                        if pos_in_1 >= i:
                            child_0.append(elem)
                        else:
                            child_1.append(elem)
                    except ValueError:
                        child_0.append(elem)
                
                processed.add(elem)
        
        # Add any remaining elements (shouldn't happen with proper processing)
        remaining = [e for e in remaining_elements if e not in processed]
        if remaining:
            # Assign arbitrarily or split evenly
            split_point = len(remaining) // 2
            child_0.extend(remaining[:split_point])
            child_1.extend(remaining[split_point:])
        
        return child_0, child_1
    
    def _create_node(self, indices):
        """Create a new node in the tree."""
        node_id = self.next_id
        self.tree[node_id] = (indices, None, None)
        self.next_id += 1
        return node_id
    
    def _get_leaves(self):
        """Get all leaf nodes."""
        return [node_id for node_id, (_, left, right) in self.tree.items() 
                if left is None and right is None]


# Wrapper functions for benchmark integration
def splitter_order_based(subX, dist_matrix=None):
    """
    Splitter function for order-based novel divisive algorithm.
    To be used with DivisiveTree class from benchmark.
    """
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    
    # Create temporary clustering object
    temp_cluster = NovelDivisiveClustering(split_cri='order')
    temp_cluster.D = pairwise_distances(subX) if dist_matrix is None else dist_matrix
    
    # Get split for this node
    child0, child1 = temp_cluster._split_cluster(list(range(subX.shape[0])))
    
    # Create labels
    labels = np.zeros(subX.shape[0], dtype=int)
    for idx in child0:
        labels[idx] = 0
    for idx in child1:
        labels[idx] = 1
    
    return labels


def splitter_proximity_based(subX, dist_matrix=None):
    """
    Splitter function for proximity-based novel divisive algorithm.
    To be used with DivisiveTree class from benchmark.
    """
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    
    # Create temporary clustering object
    temp_cluster = NovelDivisiveClustering(split_cri='prox')
    temp_cluster.D = pairwise_distances(subX) if dist_matrix is None else dist_matrix
    
    # Get split for this node
    child0, child1 = temp_cluster._split_cluster(list(range(subX.shape[0])))
    
    # Create labels
    labels = np.zeros(subX.shape[0], dtype=int)
    for idx in child0:
        labels[idx] = 0
    for idx in child1:
        labels[idx] = 1
    
    return labels


def splitter_order_based_fallback(subX, dist_matrix=None):
    """
    Wrapper with fallback to KMeans if novel method fails.
    """
    try:
        return splitter_order_based(subX, dist_matrix)
    except Exception as e:
        print(f"Order-based split failed, falling back to KMeans: {e}")
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(subX)


def splitter_proximity_based_fallback(subX, dist_matrix=None):
    """
    Wrapper with fallback to KMeans if novel method fails.
    """
    try:
        return splitter_proximity_based(subX, dist_matrix)
    except Exception as e:
        print(f"Proximity-based split failed, falling back to KMeans: {e}")
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(subX)


# --- Algorithm Wrappers ---
def run_benchmarks(dataset_name, X, y_true):
    n_clusters = len(np.unique(y_true))
    results = []

    # Map your table headers to the actual code implementations
    methods = [
        ('Divisive K-Means', DivisiveTree(splitter_kmeans)),
        ('Divisive Ward', DivisiveTree(splitter_ward_binary)),
        ('Divisive GMM', DivisiveTree(splitter_gmm)),
        ('DIANA', DivisiveTree(splitter_diana)),
        ('density-based', DivisiveTree(splitter_density)),
        ('AgglomerativeClustering (ward)', AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')),
        ('AgglomerativeClustering (complete)', AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')),
        ('AgglomerativeClustering (average)', AgglomerativeClustering(n_clusters=n_clusters, linkage='average')),
        ('AgglomerativeClustering (single)', AgglomerativeClustering(n_clusters=n_clusters, linkage='single')),
        ('Novel divisive clustering algorithm(order based)', NovelDivisiveClustering(split_cri='rbs')),
        ('Novel Divisive Clustering (proximity based', NovelDivisiveClustering(split_cri='pbs'))
    ]

    for algo_name, model in methods:
        try:
            # Fit/Predict logic
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                # For DivisiveTree and NovelDivisiveClustering
                labels = model.fit(X, desired_clusters=n_clusters)
            
            ari = adjusted_rand_score(y_true, labels)
            sil = silhouette_score(X, labels)

            results.append({
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'ARI': round(ari, 4),
                'Silhouette': round(sil, 4)
            })
            print(f"Dataset: {name} | Algo: {algo_name}")
            print(f"  -> Predicted Unique Labels: {np.unique(labels)}")
            print(f"  -> ARI: {ari:.4f} | Silhouette: {sil:.4f}")
        except Exception as e:
            print(f"Error {algo_name} on {dataset_name}: {e}")
    
    return results

# --- Main Execution ---
# Add your 50 UCI IDs here
# uci_ids = [176,357,292,352,519,484,763,544] applied
# uci_ids=[235 ,468 ,296, 519] # applied
# uci_ids=[ 488, 760 ,591, 551, 485]
# uci_ids = [ 25, 109, 148, 186, 52 ,372,270,537,342,312] #nott applied
# uci_ids = [ 25, 109, 148, 186, 52 ,372,537,342,312] #nott applied
# uci_ids=[257,300,19,697,350,144,73]
# uci_ids=[1,235,891,159,15,27,20,80]

# uci_ids=[ 488, 760 ,591, 551, 485]
# final_storage = []
# uci_ids = [53, 45, 2, 222, 267, 571]  applied
CSV_NAME = 'clustering_results_table.csv'
final_storage=[]

for uci_id in uci_ids:
    try:
        repo = fetch_ucirepo(id=uci_id)
        name = repo.metadata.name
        X_df = repo.data.features
        y_true = repo.data.targets.values.ravel()
        y_df = repo.data.targets

        # 1. SAMPLING LOGIC (Limit to 2000)
        n_samples = X_df.shape[0]
        if n_samples > 2000:
            print(f"Sampling 2000 instances from {n_samples} for {name}...")
            # Use stratify to keep the class balance the same
            X_df, _, y_true, _ = train_test_split(
                X_df, 
                y_df.values.ravel(), 
                train_size=2000, 
                stratify=y_df.values.ravel(), 
                random_state=42
            )
        else:
            print(f"Using full dataset for {name} ({n_samples} samples).")
            y_true = y_df.values.ravel()

       # 2. ROBUST PREPROCESSING (Fixes the 'median' error)
        X_df = pd.DataFrame(X_df)
        
        # Numeric: Fill with median
        num_cols = X_df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            X_df[num_cols] = X_df[num_cols].fillna(X_df[num_cols].median())
            
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns
        categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns

        # Categorical: Fill with mode, then Label Encode
        cat_cols = X_df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            X_df[col] = X_df[col].fillna(X_df[col].mode()[0] if not X_df[col].mode().empty else "Unknown")
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
            
        X_scaled = StandardScaler().fit_transform(X_df.values)

        print(f"Testing Dataset ID {uci_id}: {name}")
        dataset_records = run_benchmarks(name, X_scaled, y_true)
       
        final_storage.extend(dataset_records)

    except Exception as e:
        print(f"ID {uci_id} failed: {e}")

# --- 4. SMART SAVING LOGIC (Handles Empty or Missing Files) ---
if final_storage:
    new_results_df = pd.DataFrame(final_storage)
    
    # Check if file exists AND is not empty
    if os.path.exists(CSV_NAME) and os.path.getsize(CSV_NAME) > 0:
        try:
            existing_df = pd.read_csv(CSV_NAME)
            combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
            
            # Use 'Dataset' and 'Algorithm' if those are your column names
            # Adjust these names to match your run_benchmarks output keys
            if 'Dataset' in combined_df.columns and 'Algorithm' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['Dataset', 'Algorithm'], keep='last')
            
            combined_df.to_csv(CSV_NAME, index=False)
            print(f"\n✓ Updated {CSV_NAME} with new results.")
        except Exception as e:
            print(f"Error merging with existing file: {e}. Saving as new file instead.")
            new_results_df.to_csv(CSV_NAME, index=False)
    else:
        # File doesn't exist or is 0 bytes, so just create it
        new_results_df.to_csv(CSV_NAME, index=False)
        print(f"\n✓ Created fresh {CSV_NAME}.")
else:
    print("\n! No new results were generated by the algorithms.")