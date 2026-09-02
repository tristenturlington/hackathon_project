"""
Compare the set of genes selected by two different methods:
  1. ANOVA F-test (SelectKBest, k=1000)  -- the feature selection step used
     before PCA in svm_baseline.py
  2. NSC shrinkage at the Delta that yields ~1000 active genes -- from
     nearest_shrunken_centroids.py

Both use the identical data loading, cleaning, and train/test split, so the
comparison is apples-to-apples: same 640 training samples decide both gene sets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# ----------------------------------------------------------------------
# 1. Load data (identical to both other scripts)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "data.csv"
LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.csv"
TABLES_DIRECTORY = PROJECT_ROOT / "results" / "tables"

X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)

X = X.loc[y_df.index]
y = y_df.iloc[:, 0]

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes.")

# ----------------------------------------------------------------------
# 2. Drop zero-variance genes (identical to both other scripts)
# ----------------------------------------------------------------------
var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)
gene_names = X.columns[var_filter.get_support()]
print(f"After removing zero-variance genes: {X_filtered.shape[1]} genes remain.")

# ----------------------------------------------------------------------
# 3. Train/test split (identical random_state=42 to both other scripts)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_train)
n_train = X_train.shape[0]

# ----------------------------------------------------------------------
# 4. Method 1: ANOVA F-test top 1000 genes (same as svm_baseline.py)
# ----------------------------------------------------------------------
K = 1000
anova_selector = SelectKBest(score_func=f_classif, k=K)
anova_selector.fit(X_train, y_train)
anova_mask = anova_selector.get_support()
anova_genes = set(gene_names[anova_mask])

print(f"\nANOVA F-test selected {len(anova_genes)} genes.")

# ----------------------------------------------------------------------
# 5. Method 2: NSC shrinkage at the delta that yields ~1000 active genes
#    (reusing the same NSC math from nearest_shrunken_centroids.py)
# ----------------------------------------------------------------------
def fit_nsc(X_train, y_train, classes):
    n_total, n_feat = X_train.shape
    x_bar = X_train.mean(axis=0)
    class_centroids = {}
    class_sizes = {}
    sum_sq_within = np.zeros(n_feat)
    for k in classes:
        mask = (y_train == k)
        Xk = X_train[mask]
        nk = Xk.shape[0]
        class_sizes[k] = nk
        centroid_k = Xk.mean(axis=0)
        class_centroids[k] = centroid_k
        sum_sq_within += ((Xk - centroid_k) ** 2).sum(axis=0)
    n_classes = len(classes)
    s_i = np.sqrt(sum_sq_within / (n_total - n_classes))
    s0 = np.median(s_i)
    m_k = {k: np.sqrt(1.0 / class_sizes[k] - 1.0 / n_total) for k in classes}
    return x_bar, class_centroids, s_i, s0, m_k, class_sizes


def compute_d_scores(x_bar, class_centroids, s_i, s0, m_k, classes):
    d_scores = {}
    for k in classes:
        d_scores[k] = (class_centroids[k] - x_bar) / (m_k[k] * (s_i + s0))
    return d_scores


def soft_threshold(d_scores, delta):
    shrunken = {}
    for k, d in d_scores.items():
        shrunken[k] = np.sign(d) * np.maximum(np.abs(d) - delta, 0)
    return shrunken


def count_active_genes(shrunken_d):
    n_features = next(iter(shrunken_d.values())).shape[0]
    active = np.zeros(n_features, dtype=bool)
    for k, d in shrunken_d.items():
        active |= (d != 0)
    return active


def find_delta_for_target_genes(d_scores, target, tol=1e-3, max_iter=100):
    lo, hi = 0.0, 100.0
    mid = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        n_active = count_active_genes(soft_threshold(d_scores, mid)).sum()
        if n_active == target:
            break
        elif n_active > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return mid


x_bar, class_centroids, s_i, s0, m_k, class_sizes = fit_nsc(X_train, y_train.values, classes)
d_scores = compute_d_scores(x_bar, class_centroids, s_i, s0, m_k, classes)

nsc_delta = find_delta_for_target_genes(d_scores, target=K)
nsc_active_mask = count_active_genes(soft_threshold(d_scores, nsc_delta))
nsc_genes = set(gene_names[nsc_active_mask])

print(f"NSC (delta={nsc_delta:.4f}) selected {len(nsc_genes)} genes.")

# ----------------------------------------------------------------------
# 6. Compare the two gene sets
# ----------------------------------------------------------------------
overlap = anova_genes & nsc_genes
anova_only = anova_genes - nsc_genes
nsc_only = nsc_genes - anova_genes
union = anova_genes | nsc_genes

print("\n" + "=" * 50)
print("GENE SET COMPARISON")
print("=" * 50)
print(f"ANOVA F-test genes:     {len(anova_genes)}")
print(f"NSC genes:              {len(nsc_genes)}")
print(f"Overlap (both methods): {len(overlap)}")
print(f"ANOVA only:             {len(anova_only)}")
print(f"NSC only:               {len(nsc_only)}")
print(f"Union (either method):  {len(union)}")

jaccard = len(overlap) / len(union) if union else 0
pct_of_anova_in_overlap = len(overlap) / len(anova_genes) * 100 if anova_genes else 0
pct_of_nsc_in_overlap = len(overlap) / len(nsc_genes) * 100 if nsc_genes else 0

print(f"\nJaccard similarity (overlap / union): {jaccard:.3f}")
print(f"% of ANOVA's genes also picked by NSC: {pct_of_anova_in_overlap:.1f}%")
print(f"% of NSC's genes also picked by ANOVA: {pct_of_nsc_in_overlap:.1f}%")

# ----------------------------------------------------------------------
# 7. Save results for the visualization teammate
# ----------------------------------------------------------------------
summary_df = pd.DataFrame([{
    "anova_genes": len(anova_genes),
    "nsc_genes": len(nsc_genes),
    "overlap": len(overlap),
    "anova_only": len(anova_only),
    "nsc_only": len(nsc_only),
    "union": len(union),
    "jaccard_similarity": jaccard,
    "pct_anova_in_overlap": pct_of_anova_in_overlap,
    "pct_nsc_in_overlap": pct_of_nsc_in_overlap,
}])
summary_path = TABLES_DIRECTORY / "gene_set_comparison_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved summary to {summary_path}")

# Save full gene lists with a column showing which method(s) selected each gene
all_genes_df = pd.DataFrame({"gene": sorted(union)})
all_genes_df["selected_by_anova"] = all_genes_df["gene"].isin(anova_genes)
all_genes_df["selected_by_nsc"] = all_genes_df["gene"].isin(nsc_genes)
all_genes_df["selected_by_both"] = all_genes_df["gene"].isin(overlap)
full_results_path = TABLES_DIRECTORY / "gene_set_comparison_full.csv"
all_genes_df.to_csv(full_results_path, index=False)
print(f"Saved full gene-by-gene comparison to {full_results_path}")
print("(columns: gene, selected_by_anova, selected_by_nsc, selected_by_both)")

print("\nThis data is ready for a Venn diagram: "
      f"ANOVA-only={len(anova_only)}, NSC-only={len(nsc_only)}, Overlap={len(overlap)}")
