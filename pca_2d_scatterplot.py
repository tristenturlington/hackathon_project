import matplotlib.pyplot as plt
import numpy as np


def plot_pca(reduced_features, labels, predictions=None, axes_1 = 0, axes_2 = 1):
    """Plot patients using their first two PCA component values."""
    labels = np.asarray(labels)

    figure, axes = plt.subplots(figsize=(10, 7))

    for cancer_type in np.unique(labels):
        class_mask = labels == cancer_type

        axes.scatter(
            reduced_features[class_mask, axes_1],
            reduced_features[class_mask, axes_2],
            label=cancer_type,
            alpha=0.75,
            s=50,
        )

    if predictions is not None:
        predictions = np.asarray(predictions)
        incorrect_mask = predictions != labels

        axes.scatter(
            reduced_features[incorrect_mask, axes_1],
            reduced_features[incorrect_mask, axes_2],
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            s=150,
            label="Misclassified",
        )

    axes.set_title("Cancer Samples in PCA Space")
    axes.set_xlabel("Principal Component 1")
    axes.set_ylabel("Principal Component 2")
    axes.legend()
    axes.grid(alpha=0.2)

    figure.tight_layout()
    plt.show()