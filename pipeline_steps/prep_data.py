from hackathon_project.pipeline_steps import all as pipeline_steps

from pathlib import Path

DATASET_DIRECTORY = Path(__file__).resolve().parent / "dataset"
DATA_PATH = DATASET_DIRECTORY / "data.csv"
LABELS_PATH = DATASET_DIRECTORY / "labels.csv"

def prep_data():
    """Load and prepare the data for analysis."""
    features, labels = pipeline_steps.load_data(DATA_PATH, LABELS_PATH)
    filtered_features = pipeline_steps.variance_threshold(features, 0.01)
    training_features, test_features, training_labels, test_labels = (
        pipeline_steps.stratified_train_test_split(filtered_features, labels)
    )
    selected_training_features, selected_test_features = pipeline_steps.anova_test(
        training_features, test_features, training_labels
    )
    reduced_training_features, reduced_test_features = pipeline_steps.principal_component_analysis(
        selected_training_features, selected_test_features
    )

    return (
        reduced_training_features,
        reduced_test_features,
        training_labels,
        test_labels,
    )