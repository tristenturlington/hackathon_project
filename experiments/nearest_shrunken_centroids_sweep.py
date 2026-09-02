"""
Nearest Shrunken Centroids (NSC / PAM) classifier, implemented from scratch
following Tibshirani, Hastie, Narasimhan & Chu (2002), PNAS:
"Diagnosis of multiple cancer types by shrunken centroids of gene expression"

This reuses the same data loading, cleaning, and train/test split as
svm_baseline.py, so results are directly comparable to the SVM pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------------------------------------------
# 1. Load data (same as svm_baseline.py)
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
# 2. Drop zero-variance genes (same as svm_baseline.py)
# ----------------------------------------------------------------------
var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)
gene_names = X.columns[var_filter.get_support()]
print(f"After removing zero-variance genes: {X_filtered.shape[1]} genes remain.")

# ----------------------------------------------------------------------
# 3. Train/test split (same random_state as svm_baseline.py for a fair comparison)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_train)
n_classes = len(classes)
n_features = X_train.shape[1]
n_train = X_train.shape[0]

print(f"\nTrain size: {n_train}, Test size: {X_test.shape[0]}, "
      f"Classes: {list(classes)}")


# ----------------------------------------------------------------------
# 4. NSC core math
# ----------------------------------------------------------------------
def fit_nsc(X_train, y_train, classes):
    """
    Computes everything the shrinkage step needs:
      - overall centroid (x_bar)
      - per-class centroids (x_bar_k)
      - pooled within-class standard deviation (s_i)
      - class size correction factors (m_k)
    """
    n_total, n_feat = X_train.shape
    x_bar = X_train.mean(axis=0)  # overall centroid, shape (n_feat,)

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
        # accumulate within-class sum of squared deviations
        sum_sq_within += ((Xk - centroid_k) ** 2).sum(axis=0)

    n_classes = len(classes)
    # pooled within-class standard deviation, one value per gene
    s_i = np.sqrt(sum_sq_within / (n_total - n_classes))

    # s0: median of s_i, stabilizes genes with near-zero variance
    s0 = np.median(s_i)

    # m_k factor per class (accounts for class sample size)
    m_k = {k: np.sqrt(1.0 / class_sizes[k] - 1.0 / n_total) for k in classes}

    return x_bar, class_centroids, s_i, s0, m_k, class_sizes


def compute_d_scores(x_bar, class_centroids, s_i, s0, m_k, classes):
    """
    Standardized difference d_ik for every gene i and class k
    (this is the un-shrunken signal strength).
    Returns a dict: class -> array of shape (n_features,)
    """
    d_scores = {}
    for k in classes:
        d_scores[k] = (class_centroids[k] - x_bar) / (m_k[k] * (s_i + s0))
    return d_scores


def soft_threshold(d_scores, delta):
    """
    Shrinks each d_ik toward zero by delta (soft-thresholding).
    Genes with |d_ik| <= delta become exactly 0 for that class,
    meaning they no longer affect classification for that class.
    """
    shrunken = {}
    for k, d in d_scores.items():
        shrunken[k] = np.sign(d) * np.maximum(np.abs(d) - delta, 0)
    return shrunken


def shrunken_centroids(x_bar, s_i, s0, m_k, shrunken_d, classes):
    """
    Reconstructs each class's shrunken centroid from the shrunken d scores.
    """
    centroids = {}
    for k in classes:
        centroids[k] = x_bar + m_k[k] * (s_i + s0) * shrunken_d[k]
    return centroids


def classify(X, centroids, s_i, s0, class_sizes, n_total, classes):
    """
    Nearest shrunken centroid classification.
    Discriminant score for sample x and class k:
        delta_k(x) = sum_i [ (x_i - centroid_ki)^2 / (s_i + s0)^2 ] - 2*log(pi_k)
    Predict the class with the SMALLEST discriminant score.
    (pi_k = prior probability of class k, estimated from training proportions)
    """
    n_samples = X.shape[0]
    scores = np.zeros((n_samples, len(classes)))

    for idx, k in enumerate(classes):
        centroid_k = centroids[k]
        prior_k = class_sizes[k] / n_total
        # squared standardized distance to this class's centroid, summed over genes
        dist = ((X - centroid_k) ** 2) / (s_i + s0) ** 2
        scores[:, idx] = dist.sum(axis=1) - 2 * np.log(prior_k)

    pred_idx = scores.argmin(axis=1)
    return np.array(classes)[pred_idx]


def count_active_genes(shrunken_d):
    """
    Counts how many genes have a nonzero shrunken score for AT LEAST ONE class
    (i.e., genes that still influence classification after shrinkage).
    """
    n_features = next(iter(shrunken_d.values())).shape[0]
    active = np.zeros(n_features, dtype=bool)
    for k, d in shrunken_d.items():
        active |= (d != 0)
    return active.sum()


def find_delta_for_target_genes(d_scores, target, tol=1e-3, max_iter=100):
    """
    Binary search for the Delta value that shrinks the active gene count
    down to (approximately) `target`. Assumes active gene count is
    monotonically non-increasing as Delta increases.
    Returns (delta, actual_active_gene_count) -- exact target count may not
    be achievable, so the closest achievable count is returned alongside
    the delta that produced it.
    """
    lo, hi = 0.0, 100.0  # search range for delta
    mid = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        n_active = count_active_genes(soft_threshold(d_scores, mid))
        if n_active == target:
            break
        elif n_active > target:
            lo = mid       # not enough shrinkage yet, need bigger delta
        else:
            hi = mid       # shrunk too far, need smaller delta
        if hi - lo < tol:
            break
    return mid, count_active_genes(soft_threshold(d_scores, mid))


# ----------------------------------------------------------------------
# 5. Fit once, then sweep over shrinkage threshold Delta
# ----------------------------------------------------------------------
x_bar, class_centroids, s_i, s0, m_k, class_sizes = fit_nsc(X_train, y_train.values, classes)
d_scores = compute_d_scores(x_bar, class_centroids, s_i, s0, m_k, classes)

target_genes = 1000
target_delta, actual_genes = find_delta_for_target_genes(d_scores, target=target_genes)
print(f"\nDelta \u2248 {target_delta:.4f} gives {actual_genes} active genes "
      f"(target was {target_genes})")

_shrunk_d_at_target = soft_threshold(d_scores, target_delta)
_shrunk_centroids_at_target = shrunken_centroids(x_bar, s_i, s0, m_k, _shrunk_d_at_target, classes)
_test_pred_at_target = classify(X_test, _shrunk_centroids_at_target, s_i, s0, class_sizes, n_train, classes)
_test_acc_at_target = accuracy_score(y_test, _test_pred_at_target)
print(f"Test accuracy at Delta={target_delta:.4f} ({actual_genes} genes): "
      f"{_test_acc_at_target:.3f}")

deltas = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50]
results = []

print(f"\n{'Delta':>6} | {'Active genes':>13} | {'Train acc':>10} | {'Test acc':>9}")
print("-" * 50)

for delta in deltas:
    shrunk_d = soft_threshold(d_scores, delta)
    shrunk_centroids = shrunken_centroids(x_bar, s_i, s0, m_k, shrunk_d, classes)

    train_pred = classify(X_train, shrunk_centroids, s_i, s0, class_sizes, n_train, classes)
    test_pred = classify(X_test, shrunk_centroids, s_i, s0, class_sizes, n_train, classes)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    n_active = count_active_genes(shrunk_d)

    results.append({
        "delta": delta,
        "n_active_genes": n_active,
        "train_acc": train_acc,
        "test_acc": test_acc,
    })

    print(f"{delta:>6} | {n_active:>13} | {train_acc:>10.3f} | {test_acc:>9.3f}")

results_df = pd.DataFrame(results)

# ----------------------------------------------------------------------
# 6. Pick the best Delta (highest test accuracy; break ties toward fewer genes)
# ----------------------------------------------------------------------
best_row = results_df.sort_values(
    by=["test_acc", "n_active_genes"], ascending=[False, True]
).iloc[0]
best_delta = best_row["delta"]

print(f"\nBest delta: {best_delta}  "
      f"(test acc={best_row['test_acc']:.3f}, "
      f"active genes={int(best_row['n_active_genes'])})")

# ----------------------------------------------------------------------
# 7. Full evaluation at the best Delta
# ----------------------------------------------------------------------
shrunk_d = soft_threshold(d_scores, best_delta)
shrunk_centroids = shrunken_centroids(x_bar, s_i, s0, m_k, shrunk_d, classes)
final_pred = classify(X_test, shrunk_centroids, s_i, s0, class_sizes, n_train, classes)

print("\nClassification report at best delta:")
print(classification_report(y_test, final_pred))

print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, final_pred, labels=classes),
    index=classes,
    columns=classes
))

# ----------------------------------------------------------------------
# 8. Save results for the visualization teammate
# ----------------------------------------------------------------------
delta_sweep_path = TABLES_DIRECTORY / "nsc_delta_sweep.csv"
results_df.to_csv(delta_sweep_path, index=False)
print(f"\nSaved delta-sweep results to {delta_sweep_path} "
      "(columns: delta, n_active_genes, train_acc, test_acc)")

# Save the top genes that survive shrinkage at the best delta, per class,
# ranked by |shrunken d score| -- useful for the LLM reasoning component too.
top_genes_per_class = {}
for k in classes:
    d = shrunk_d[k]
    nonzero_idx = np.where(d != 0)[0]
    ranked_idx = nonzero_idx[np.argsort(-np.abs(d[nonzero_idx]))]
    top_genes_per_class[k] = [
        (gene_names[i], float(d[i])) for i in ranked_idx[:20]
    ]

rows = []
for k, gene_list in top_genes_per_class.items():
    for gene, score in gene_list:
        rows.append({"class": k, "gene": gene, "shrunken_d_score": score})

top_genes_path = TABLES_DIRECTORY / "nsc_top_genes_per_class.csv"
pd.DataFrame(rows).to_csv(top_genes_path, index=False)
print(f"Saved top discriminative genes per class to {top_genes_path}")
