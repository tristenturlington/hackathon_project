"""Run the cancer gene classification pipeline.

Tristen Turlington, Ben Anderson, Mariah, Addi Bruening
8/31/2026
"""

from pipeline_steps import (
    logistic_regression,
    model_evaluation,
    prep_data,
)


def main():
    """Run all seven steps in order."""
    (
        reduced_training_features,
        reduced_test_features,
        training_labels,
        test_labels,
    ) = prep_data()

    classifier = logistic_regression(reduced_training_features, training_labels)
    return model_evaluation(classifier, reduced_test_features, test_labels)


if __name__ == "__main__":
    main()
