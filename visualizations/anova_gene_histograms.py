"""
For the top genes most predictive of cancer type (by ANOVA F-test, same
selection as svm_baseline.py), plot overlapping/translucent histograms
of that gene's expression - one histogram per cancer type - to visually
show WHY that gene is predictive (i.e. do the cancer types' distributions
actually separate from each other?).

Expects the same two files as svm_baseline.py:
  - dataset/data.csv   : samples (rows) x genes (columns), first col = sample ID
  - dataset/labels.csv : sample ID -> cancer type (BRCA, KIRC, COAD, LUAD, PRAD)

Produces one PDF:
  - gene_histograms_by_cancer_type.pdf
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "data.csv"
LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.csv"

N_GENES_TO_PLOT = 6   # how many top genes to show (one subplot each)
BINS = 20

OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "gene_histogram_by_cancer_type.pdf"

RIBBON_COLORS = {
    "BRCA": "#FF69B4",  # breast cancer - pink
    "PRAD": "#87CEFA",  # prostate cancer - light blue
    "COAD": "#00008B",  # colorectal cancer - dark blue
    "KIRC": "#FFA500",  # kidney cancer - orange
    "LUAD": "#D3D3D3",  # lung cancer - light gray (stand-in for white ribbon)
}

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)
X = X.loc[y_df.index]
y = y_df.iloc[:, 0]

print(f"Loaded {X.shape[0]} samples x {X.shape[1]} genes.")

# ----------------------------------------------------------------------
# 3. Same feature selection as svm_baseline.py - find the top genes
# ----------------------------------------------------------------------
var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)
kept_gene_names = X.columns[var_filter.get_support()]

selector = SelectKBest(score_func=f_classif, k=N_GENES_TO_PLOT)
selector.fit(X_filtered, y)
selected_mask = selector.get_support()
top_genes = kept_gene_names[selected_mask]
scores = selector.scores_[selected_mask]

order_by_score = np.argsort(scores)[::-1]
top_genes = top_genes[order_by_score]
scores = scores[order_by_score]

print(f"Plotting histograms for top {N_GENES_TO_PLOT} genes:")
for gene, score in zip(top_genes, scores):
    print(f"  {gene}: F={score:.1f}")

# ----------------------------------------------------------------------
# 4. Plot: one subplot per gene, overlapping translucent histograms
#    per cancer type
# ----------------------------------------------------------------------
classes = sorted(y.unique())

n_cols = 3
n_rows = int(np.ceil(N_GENES_TO_PLOT / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = np.array(axes).reshape(-1)  # flatten in case of a single row

for i, gene in enumerate(top_genes):
    ax = axes[i]
    for cls in classes:
        values = X.loc[y == cls, gene]
        ax.hist(
            values, bins=BINS, alpha=0.5,
            color=RIBBON_COLORS.get(cls, "gray"),
            label=cls, edgecolor="none"
        )
    ax.set_yscale("log")   # <-- add this line
    ax.set_title(f"{gene}\n(F={scores[i]:.0f})")
    ax.set_xlabel("Expression level (log-transformed)")
    ax.set_ylabel("Number of samples")

# Turn off any unused subplot axes (if N_GENES_TO_PLOT doesn't fill the grid)
for j in range(len(top_genes), len(axes)):
    axes[j].axis("off")

# One shared legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, title="Cancer type",
    loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(classes)
)

fig.suptitle(
    "Expression distributions of the most cancer-type-predictive genes",
    y=1.08, fontsize=14
)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
