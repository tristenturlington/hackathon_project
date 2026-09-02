"""
Visualize the genes that the SVM pipeline (svm_baseline.py) actually
selected as most associated with cancer type, as a heatmap grouped by class.

This mirrors Part 4 of svm_baseline.py (SelectKBest + f_classif) so the
genes you see here are the same ones feeding the model - not just a
high-variance or arbitrary slice of the data.

Expects the same two files as svm_baseline.py:
  - dataset/data.csv   : samples (rows) x genes (columns), first col = sample ID
  - dataset/labels.csv : sample ID -> cancer type (BRCA, KIRC, COAD, LUAD, PRAD)

Produces one PDF:
  - expression_heatmap_selected_genes.pdf
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "data.csv"
LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.csv"

N_SAMPLES_PER_CLASS = 10**9  # effectively "all available"
K = 1000                      # matches svm_baseline.py's K=1000

OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "heatmap_selected_genes.pdf"

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)
X = X.loc[y_df.index]
y = y_df.iloc[:, 0]

print(f"Loaded {X.shape[0]} samples x {X.shape[1]} genes.")
print("Class counts:\n", y.value_counts())

# ----------------------------------------------------------------------
# 3. Same feature selection as svm_baseline.py Part 2 + Part 4
#    (run on the FULL dataset here, just for visualization purposes -
#    your actual model still fits selection only on the training split,
#    to avoid leakage; this script is just for looking at the data)
# ----------------------------------------------------------------------
var_filter = VarianceThreshold(threshold=0.0)
X_filtered = var_filter.fit_transform(X)
kept_gene_names = X.columns[var_filter.get_support()]
print(f"After removing zero-variance genes: {X_filtered.shape[1]} genes remain.")

selector = SelectKBest(score_func=f_classif, k=K)
selector.fit(X_filtered, y)
selected_mask = selector.get_support()
selected_genes = kept_gene_names[selected_mask]
scores = selector.scores_[selected_mask]

# Sort selected genes by F-score, highest first, for a cleaner plot
order_by_score = np.argsort(scores)[::-1]
selected_genes = selected_genes[order_by_score]

print(f"Selected top {K} genes by ANOVA F-test.")
print("Top 5 genes by F-score:")
for gene, score in zip(selected_genes[:5], scores[order_by_score][:5]):
    print(f"  {gene}: F={score:.1f}")

# ----------------------------------------------------------------------
# 4. Pick a balanced set of samples across all classes (not just the
#    first N rows - this avoids accidentally grabbing only one class
#    if the CSV happens to be grouped by cancer type)
# ----------------------------------------------------------------------
sampled_idx = (
    y.groupby(y)
     .apply(lambda s: s.sample(
         n=min(N_SAMPLES_PER_CLASS, len(s)), random_state=42
     ))
     .index.get_level_values(1)
)

y_sub = y.loc[sampled_idx]
order = y_sub.sort_values().index  # group same-class samples together
subset = X.loc[order, selected_genes]
labels_sorted = y_sub.loc[order]

# ----------------------------------------------------------------------
# 5. Plot: color strip (cancer type) + heatmap of selected genes
# ----------------------------------------------------------------------
classes = sorted(labels_sorted.unique())

# Colors based on real cancer awareness ribbon colors, so the legend is
# intuitive at a glance. Lung cancer's actual ribbon is white/pearl, which
# would be invisible on a white PDF background, so a light gray stands in
# for it here instead.
RIBBON_COLORS = {
    "BRCA": "#FF69B4",  # breast cancer - pink
    "PRAD": "#87CEFA",  # prostate cancer - light blue
    "COAD": "#00008B",  # colorectal cancer - dark blue
    "KIRC": "#FFA500",  # kidney cancer - orange
    "LUAD": "#D3D3D3",  # lung cancer - light gray (stand-in for white ribbon)
}
class_to_color = {c: RIBBON_COLORS.get(c, "black") for c in classes}
row_colors = [to_rgba(class_to_color[c]) for c in labels_sorted]

fig, (ax_strip, ax_heat) = plt.subplots(
    1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [0.03, 1]}
)

ax_strip.imshow(np.array(row_colors).reshape(-1, 1, 4), aspect="auto")
ax_strip.set_xticks([])
ax_strip.set_yticks([])
ax_strip.set_ylabel(f"All {len(labels_sorted)} samples, grouped by cancer type")

im = ax_heat.imshow(subset.values, aspect="auto", cmap="viridis")
ax_heat.set_title(
    "Gene expression by cancer type\n"
    f"(top {K} genes most predictive of cancer type, via ANOVA F-test)"
)
ax_heat.set_xlabel("Genes (most predictive -> least predictive, left to right)")
ax_heat.set_xticks([])
ax_heat.set_yticks([])
cbar = fig.colorbar(im, ax=ax_heat)
cbar.set_label("Expression level (log-transformed)")

legend_handles = [Patch(color=class_to_color[c], label=c) for c in classes]
ax_heat.legend(
    handles=legend_handles, title="Cancer type",
    bbox_to_anchor=(1.2, 1), loc="upper left", borderaxespad=0.
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, format="pdf")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")
