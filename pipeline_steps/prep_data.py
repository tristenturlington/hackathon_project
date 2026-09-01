"""Load and transform the cancer gene data for model training."""

from pathlib import Path

from .anova_test import anova_test
from .load_data import load_data
from .principal_component_analysis import principal_component_analysis
from .stratified_train_test_split import stratified_train_test_split
from .variance_threshold import variance_threshold

DATASET_DIRECTORY = Path(__file__).resolve().parent.parent / "dataset"
DATA_PATH = DATASET_DIRECTORY / "data.csv"
LABELS_PATH = DATASET_DIRECTORY / "labels.csv"


def prep_data():
    """Load and prepare the data for analysis."""
    features, labels = load_data(DATA_PATH, LABELS_PATH)
    filtered_features = variance_threshold(features, 0.01)
    training_features, test_features, training_labels, test_labels = (
        stratified_train_test_split(filtered_features, labels)
    )
    selected_training_features, selected_test_features = anova_test(
        training_features, test_features, training_labels
    )
    reduced_training_features, reduced_test_features = (
        principal_component_analysis(
            selected_training_features,
            selected_test_features,
        )
    )

    return (
        reduced_training_features,
        reduced_test_features,
        training_labels,
        test_labels,
    )
