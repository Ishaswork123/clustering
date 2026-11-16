import numpy as np
from sklearn.metrics import pairwise_distances,silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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


# Example usage in benchmark
if __name__ == "__main__":
    # Test the algorithms
    from sklearn.datasets import make_blobs
    
    # Generate sample data
    X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    
    # Test order-based split
    cluster_ord = NovelDivisiveClustering(split_cri='rbs', max_depth=5)
    labels_ord = cluster_ord.fit(X, desired_clusters=3)
    print(f"Order-based split - Unique labels: {np.unique(labels_ord)}")
    
    # Test proximity-based split
    cluster_prox = NovelDivisiveClustering(split_cri='pbs', max_depth=5)
    labels_prox = cluster_prox.fit(X, desired_clusters=3)
    print(f"Proximity-based split - Unique labels: {np.unique(labels_prox)}")
    
#check ari score
# Convert targets to a 1D array if they aren't already
    y_true = y.flatten() 

# ARI scores from -1 to 1 (1 is a perfect match)
    ari_ord = adjusted_rand_score(y_true, labels_ord)
    ari_prox = adjusted_rand_score(y_true, labels_prox)

    print(f"Order-Based ARI: {ari_ord:.4f}")
    print(f"Proximity-Based ARI: {ari_prox:.4f}")
    sil_ord = silhouette_score(X, labels_ord)
    sil_prox = silhouette_score(X, labels_prox)

    print(f"Order-Based Silhouette Score: {sil_ord:.4f}")
    print(f"Proximity-Based Silhouette Score: {sil_prox:.4f}")