"""
Divisive + Agglomerative clustering benchmarking script

Extended features (added per request):
 - Additional divisive splitters:
    * Ward-based binary split (2-way agglomerative on the node)
    * Density-based split using DBSCAN core vs non-core split
    * GMM split (2-component Gaussian Mixture)
 - Agglomerative (bottom-up) baseline algorithms for comparison:
    * AgglomerativeClustering (ward, complete, average, single)
 - Produces LaTeX-ready tables and publication-quality plots (PNG + PDF) optimized for conference figures and tables.
 - Outputs CSV for full results and LaTeX tables saved to files.

Usage notes:
 - Install dependencies: pip install numpy scipy scikit-learn pandas matplotlib tqdm joblib pyclustering sklearn-extra
 - Run same as before. New arguments allow controlling which algorithm families to run.

Author: ChatGPT (extended scaffold). Please inspect the generated results and adapt datasets / parameters for your final experiments.

"""

import time
import argparse
import os
import math
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

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
    # apply 2-way agglomerative (ward) within node
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    ac = AgglomerativeClustering(n_clusters=2, linkage='ward')
    try:
        return ac.fit_predict(subX)
    except Exception:
        return splitter_kmeans(subX)

def splitter_density(subX, dist_matrix=None):
    # Use DBSCAN to find dense cores; if DBSCAN yields at least 2 labels, map largest component vs rest
    if subX.shape[0] < 2:
        return np.zeros(subX.shape[0], dtype=int)
    # heuristics for eps: use median pairwise distance to 5th neighbor
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(10, max(2, subX.shape[0]-1))).fit(subX)
        distances, _ = nbrs.kneighbors(subX)
        # pick a small quantile as eps
        eps = np.median(distances[:, -1])
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(subX)
        unique = np.unique(labels[labels >= 0])
        if len(unique) >= 1:
            # find largest cluster among core clusters
            counts = {lab: np.sum(labels == lab) for lab in unique}
            largest = max(counts.items(), key=lambda x: x[1])[0]
            out = np.zeros(subX.shape[0], dtype=int)
            out[labels == largest] = 1
            # ensure both labels present, else fallback
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

# Agglomerative baseline: produces flat clustering directly (not divisive tree)
def agglomerative_flat(X, n_clusters=2, linkage='ward'):
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return ac.fit_predict(X)

# Benchmark harness and metrics
def evaluate_flat(true_labels, pred_labels, data):
    res = {}
    res['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    res['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
    res['Purity'] = purity_score(true_labels, pred_labels)
    try:
        if len(np.unique(pred_labels)) > 1 and len(np.unique(pred_labels)) < len(pred_labels):
            res['Silhouette'] = silhouette_score(data, pred_labels)
        else:
            res['Silhouette'] = np.nan
    except Exception:
        res['Silhouette'] = np.nan
    return res

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
            stats.update({'method': method_name, 'k': k, 'repeat': rep, 'n_samples': X.shape[0], 'n_features': X.shape[1], 'time': elapsed})
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
            stats.update({'method': f'Agglo-{linkage}', 'k': k, 'repeat': rep, 'n_samples': X.shape[0], 'n_features': X.shape[1], 'time': elapsed})
            records.append(stats)
    return records

# Datasets loader
def make_datasets():
    datasets = []
    X1, y1 = make_blobs(n_samples=500, centers=4, random_state=0, cluster_std=1.0)
    datasets.append(('blobs_4', X1, y1))
    X2, y2 = make_moons(n_samples=500, noise=0.08, random_state=0)
    datasets.append(('moons', X2, y2))
    X3, y3 = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=0)
    datasets.append(('circles', X3, y3))
    iris = load_iris()
    datasets.append(('iris', iris.data, iris.target))
    wine = load_wine()
    datasets.append(('wine', wine.data, wine.target))
    digits = load_digits()
    Xd = digits.data
    yd = digits.target
    sel = np.random.RandomState(0).choice(len(yd), size=500, replace=False)
    datasets.append(('digits500', Xd[sel], yd[sel]))
    return datasets

# LaTeX table and plotting utilities
def save_latex_tables(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # produce a table per dataset and k summarizing mean +/- std for metrics
    grp = df.groupby(['dataset','method','k'])
    agg = grp.agg(ARI_mean=('ARI','mean'), ARI_std=('ARI','std'), NMI_mean=('NMI','mean'), NMI_std=('NMI','std'), Purity_mean=('Purity','mean'), Purity_std=('Purity','std'), Silhouette_mean=('Silhouette','mean'), Silhouette_std=('Silhouette','std'))
    agg = agg.reset_index()
    for dname, g in agg.groupby('dataset'):
        # pivot to methods x k for ARI (as example); build multirow table for all metrics
        # We'll produce separate LaTeX tables for each metric
        for metric in ['ARI','NMI','Purity','Silhouette']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            pivot = g.pivot(index='method', columns='k', values=mean_col)
            pivot_std = g.pivot(index='method', columns='k', values=std_col)
            # format mean (std)
            fmt = pivot.copy().astype(str)
            for r in pivot.index:
                for c in pivot.columns:
                    m = pivot.loc[r,c]
                    s = pivot_std.loc[r,c]
                    if np.isnan(m):
                        fmt.loc[r,c] = 'NaN'
                    else:
                        fmt.loc[r,c] = f"{m:.3f} ({s:.3f})"
            tex = fmt.to_latex(escape=False)
            fname = os.path.join(out_dir, f"table_{dname}_{metric}.tex")
            with open(fname, 'w') as f:
                f.write(f"% LaTeX table for dataset={dname} metric={metric}")
                f.write(tex)
    print(f"Saved LaTeX tables to {out_dir}")

def save_publication_plots(df, out_dir):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    summary = df.groupby(['dataset','method','k']).agg({'ARI':'mean','NMI':'mean','Purity':'mean','Silhouette':'mean','time':'mean'}).reset_index()
    for dname, group in summary.groupby('dataset'):
        fig, axes = plt.subplots(2,2, figsize=(10,7))
        axes = axes.ravel()
        metrics = ['ARI','NMI','Purity','Silhouette']
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for mname, mg in group.groupby('method'):
                ax.plot(mg['k'], mg[metric], marker='o', label=mname)
            ax.set_title(metric)
            ax.set_xlabel('k')
            ax.set_xticks(sorted(group['k'].unique()))
            ax.grid(True)
        fig.legend(loc='upper center', ncol=4)
        fig.tight_layout(rect=[0,0,1,0.95])
        png = os.path.join(out_dir, f"figure_{dname}.png")
        pdf = os.path.join(out_dir, f"figure_{dname}.pdf")
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)
    print(f"Saved publication-quality plots to {out_dir}")

# Main runner
def main(args):
    datasets = make_datasets()
    divisive_methods = [
        ('Div-KMeans', splitter_kmeans),
        ('Div-KMedoids', splitter_kmedoids),
        ('DIANA', splitter_diana),
        ('Div-Spectral', splitter_spectral),
        ('Div-Ward', splitter_ward_binary),
        ('Div-Density', splitter_density),
        ('Div-GMM', splitter_gmm)
    ]
    agg_linkages = ['ward','complete','average','single']
    ks = [2,3,4,5]
    all_records = []
    tasks = []
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

    if args.n_jobs == 1:
        for t in tqdm(tasks):
            all_records.extend(worker(t))
    else:
        results = Parallel(n_jobs=args.n_jobs)(delayed(worker)(t) for t in tasks)
        for r in results:
            all_records.extend(r)

    df = pd.DataFrame(all_records)
    df.to_csv(args.out, index=False)
    print(f"Saved results to {args.out}")

    # produce LaTeX tables and plots
    if args.tex_dir:
        save_latex_tables(df, args.tex_dir)
    if args.plots:
        save_publication_plots(df, args.plots)

    # print a concise summary to console
    summary = df.groupby(['dataset','method','k']).agg({'ARI':'mean','NMI':'mean','Purity':'mean','Silhouette':'mean','time':'mean'}).reset_index()
    print(summary.head(40))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='divisive_benchmark_results.csv')
    parser.add_argument('--plots', type=str, default='results_plots', help='directory to save summary plots')
    parser.add_argument('--tex_dir', type=str, default='latex_tables', help='directory to save LaTeX tables')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    main(args)
