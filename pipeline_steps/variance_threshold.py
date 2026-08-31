"""Remove genes that have no variance."""

from sklearn.feature_selection import VarianceThreshold


def variance_threshold(genes, threshold: float = 0.001):
    """Remove uninformative genes whose values never change."""
    variance_filter = VarianceThreshold(threshold)
    filtered_genes = variance_filter.fit_transform(genes)
    print(
        f"After removing {genes.shape[1] - filtered_genes.shape[1]} low-variance genes: ",
        f"{filtered_genes.shape[1]} genes remain."
    )
    return filtered_genes
