"""
Heatmap of the top 20 most discriminative genes per cancer type, from
teammate's Nearest Shrunken Centroid (NSC) analysis.

Unlike the ANOVA F-test selection (which just asks "does this gene separate
SOME classes"), NSC d-scores are per-class and SIGNED:
  - positive score = gene is elevated (higher expression) in that class
  - negative score = gene is suppressed (lower expression) in that class
So this heatmap uses a diverging (red/blue) colormap centered at zero,
rather than the sequential viridis colormap used for the raw expression
heatmaps.

Also cross-references against the top genes previously found via
SelectKBest/ANOVA F-test (visualize_selected_genes.py / plot_gene_histograms.py)
and marks any overlapping genes with a star, since agreement between two
independent methods is a strong signal those genes are real markers.

Input:
  - nsc_top_genes_per_class.csv : class, gene, shrunken_d_score

Output:
  - nsc_top_genes_heatmap.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------
DATA_PATH = "nsc_top_genes_per_class.csv"
OUTPUT_PATH = "nsc_top_genes_heatmap.pdf"

CLASS_ORDER = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]

# Genes previously found via ANOVA F-test (top 6, from plot_gene_histograms.py)
# used here just to flag overlap - update this list if your top-K changes.
FTEST_TOP_GENES = {
    "gene_9175", "gene_9176", "gene_220", "gene_219",
    "gene_18135", "gene_13818",
}

# ----------------------------------------------------------------------
# 2. Load data
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["abs_score"] = df["shrunken_d_score"].abs()

# ----------------------------------------------------------------------
# 3. Build column order: genes grouped by the class they were selected
#    for, ranked by |score| within that class. A gene that appears in
#    more than one class's top-20 (e.g. gene_7965 in both BRCA and COAD)
#    is placed once, under whichever class it appears in first.
# ----------------------------------------------------------------------
gene_order = []
gene_origin_class = {}
for cls in CLASS_ORDER:
    sub = df[df["class"] == cls].sort_values("abs_score", ascending=False)
    for gene in sub["gene"]:
        if gene not in gene_origin_class:
            gene_order.append(gene)
            gene_origin_class[gene] = cls

print(f"{len(gene_order)} unique genes across all 5 classes' top 20 lists "
      f"({5 * 20} listed, so {5 * 20 - len(gene_order)} overlap between classes).")

# ----------------------------------------------------------------------
# 4. Build the class x gene matrix. A gene only has a known score for
#    the class(es) it was listed under; everywhere else is left at 0
#    (NSC shrinks most class/gene scores to exactly 0, so this is a
#    reasonable - though not guaranteed - stand-in for "not selected").
# ----------------------------------------------------------------------
matrix = pd.DataFrame(0.0, index=CLASS_ORDER, columns=gene_order)
for _, row in df.iterrows():
    matrix.loc[row["class"], row["gene"]] = row["shrunken_d_score"]

# ----------------------------------------------------------------------
# 5. Plot
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(24, 10))

vmax = matrix.values.max()
vmin = matrix.values.min()
abs_max = max(abs(vmax), abs(vmin))

im = ax.imshow(
    matrix.values, aspect="auto", cmap="coolwarm",
    vmin=-abs_max, vmax=abs_max
)

ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_yticklabels(CLASS_ORDER)
ax.set_xticks([])
ax.set_title(
    "Top discriminative genes per cancer type (Nearest Shrunken Centroid)\n"
    "Red = elevated expression in that class, Blue = suppressed, "
    "\u2605 = also found by ANOVA F-test",
    fontsize=13
)

# Vertical divider lines between each class's block of genes, and a
# centered label underneath each block
boundary = 0
for cls in CLASS_ORDER:
    n = sum(1 for g in gene_order if gene_origin_class[g] == cls)
    center = boundary + n / 2 - 0.5
    ax.text(center, len(CLASS_ORDER) - 0.3, cls,
            ha="center", va="top", fontsize=11, fontweight="bold")
    boundary += n
    if boundary < len(gene_order):
        ax.axvline(boundary - 0.5, color="black", linewidth=1)

# Star markers above columns for genes that also showed up in the
# ANOVA F-test top genes - i.e. two independent methods agreeing
for i, gene in enumerate(gene_order):
    if gene in FTEST_TOP_GENES:
        ax.annotate(
            "\u2605", xy=(i, 1.08), xycoords=("data", "axes fraction"),
            ha="center", va="bottom", fontsize=12, color="black",
            annotation_clip=False
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Shrunken d-score")

fig.tight_layout()
ax.set_title(
    "Top discriminative genes per cancer type (Nearest Shrunken Centroid)\n"
    "Red = elevated expression in that class, Blue = suppressed, "
    "\u2605 = also found by ANOVA F-test",
    fontsize=13, pad=20
)
fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PATH}")