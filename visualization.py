"""
Visualize a slice of the gene expression matrix as a heatmap and save it as a PDF.

Expects the same two files as train_svm.py:
  - data.csv   : samples (rows) x genes (columns), first column = sample ID
  - labels.csv : sample ID -> cancer type (BRCA, KIRC, COAD, LUAD, PRAD)

Produces two PDFs:
  1. expression_heatmap_plain.pdf   - raw heatmap, no label info, with a colorbar
  2. expression_heatmap_labeled.pdf - same heatmap, but rows are sorted and
                                       color-tagged by cancer type, so you can
                                       see whether expression patterns cluster
                                       by class

Update the file paths / sample & gene counts below as needed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
DATA_PATH = "dataset/data.csv"
LABELS_PATH = "dataset/labels.csv"

N_SAMPLES = 50       # how many samples (rows) to show
N_GENES = 50          # how many genes (columns) to show
GENE_SELECTION = "most_variable"  # "first" or "most_variable"

OUTPUT_PLAIN = "heatmap_plain.pdf"
OUTPUT_LABELED = "heatmap_sorted.pdf"

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
X = pd.read_csv(DATA_PATH, index_col=0)
y_df = pd.read_csv(LABELS_PATH, index_col=0)
X = X.loc[y_df.index]
y = y_df.iloc[:, 0]

print(f"Loaded {X.shape[0]} samples x {X.shape[1]} genes.")

# ----------------------------------------------------------------------
# 3. Pick which genes to show
# ----------------------------------------------------------------------
if GENE_SELECTION == "most_variable":
    top_genes = X.var(axis=0).sort_values(ascending=False).index[:N_GENES]
else:
    top_genes = X.columns[:N_GENES]

# ----------------------------------------------------------------------
# 4. Plain heatmap (first N_SAMPLES rows, as originally shown)
# ----------------------------------------------------------------------
subset_plain = X.iloc[:N_SAMPLES][top_genes]

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(subset_plain.values, aspect="auto", cmap="viridis")
ax.set_title(f"Gene expression, raw (top {N_GENES} most variable genes x "
             f"{N_SAMPLES} samples)")
ax.set_xlabel("Genes")
ax.set_ylabel("Samples")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Expression level (log-transformed)")

fig.tight_layout()
fig.savefig(OUTPUT_PLAIN, format="pdf")
plt.close(fig)
print(f"Saved {OUTPUT_PLAIN}")

# ----------------------------------------------------------------------
# 5. Labeled heatmap: sort samples by cancer type, add a color strip
# ----------------------------------------------------------------------
# Take a balanced-ish sample across classes, then sort by class
y_sub = y.iloc[:N_SAMPLES]
order = y_sub.sort_values().index
subset_labeled = X.loc[order][top_genes]
labels_sorted = y_sub.loc[order]

classes = sorted(labels_sorted.unique())
class_colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
class_to_color = dict(zip(classes, class_colors))
row_colors = [class_to_color[c] for c in labels_sorted]

fig, (ax_strip, ax_heat) = plt.subplots(
    1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [0.03, 1]}
)

# Color strip showing cancer type per row
ax_strip.imshow(
    np.array(row_colors).reshape(-1, 1, 4), aspect="auto"
)
ax_strip.set_xticks([])
ax_strip.set_yticks([])
ax_strip.set_ylabel("Samples (grouped by cancer type)")

# Heatmap
im2 = ax_heat.imshow(subset_labeled.values, aspect="auto", cmap="viridis")
ax_heat.set_title(f"Gene expression grouped by cancer type "
                   f"(top {N_GENES} most variable genes)")
ax_heat.set_xlabel("Genes")
ax_heat.set_xticks([])
ax_heat.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax_heat)
cbar2.set_label("Expression level (log-transformed)")

legend_handles = [Patch(color=class_to_color[c], label=c) for c in classes]
ax_heat.legend(
    handles=legend_handles, title="Cancer type",
    bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0.
)

fig.tight_layout()
fig.savefig(OUTPUT_LABELED, format="pdf")
plt.close(fig)
print(f"Saved {OUTPUT_LABELED}")