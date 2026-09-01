"""Load and align gene-expression data and cancer labels."""

from pathlib import Path

import pandas as pd


def load_data(data_path: Path, labels_path: Path):
    """Load the feature matrix and its matching class labels."""
    genes = pd.read_csv(data_path, index_col=0)
    patients = pd.read_csv(labels_path, index_col=0)

    missing_samples = patients.index.difference(genes.index)
    if not missing_samples.empty:
        raise ValueError(
            f"The data file is missing {len(missing_samples)} labeled samples."
        )

    genes = genes.loc[patients.index]
    patients = patients.iloc[:, 0]

    print(f"Loaded {genes.shape[0]} samples with {genes.shape[1]} genes.")
    print("Class counts:\n", patients.value_counts())
    return genes, patients
