"""Run the cancer gene classification pipeline.

Tristen Turlington, Ben Anderson, Mariah, Addi Bruening
8/31/2026
"""

from pathlib import Path

from sklearn.pipeline import Pipeline

from pipeline_steps import (
    anova_test,
    load_data,
    model_evaluation,
    principal_component_analysis,
    stratified_train_test_split,
    support_vector_machine,
    variance_threshold,
)


DATASET_DIRECTORY = Path(__file__).resolve().parent / "dataset"
DATA_PATH = DATASET_DIRECTORY / "data.csv"
LABELS_PATH = DATASET_DIRECTORY / "labels.csv"


def main():
    """Run all seven steps in order."""
    features, labels = load_data(DATA_PATH, LABELS_PATH)
    filtered_features = variance_threshold(features, 0.01)
    training_features, test_features, training_labels, test_labels = (
        stratified_train_test_split(filtered_features, labels)
    )
    selected_training_features, selected_test_features = anova_test(
        training_features, test_features, training_labels
    )
    reduced_training_features, reduced_test_features = principal_component_analysis(
        selected_training_features, selected_test_features
    )  
    classifier = support_vector_machine(reduced_training_features, training_labels)
    return model_evaluation(classifier, reduced_test_features, test_labels)

if __name__ == "__main__":
    main()
