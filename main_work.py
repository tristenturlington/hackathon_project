"""Run the cancer gene classification pipeline.

Tristen Turlington, Ben Anderson, Mariah, Addi Bruening
8/31/2026
"""



from pca_2d_scatterplot import plot_pca

from pipeline_steps import (
    anova_test,
    load_data,
    model_evaluation,
    principal_component_analysis,
    stratified_train_test_split,
    support_vector_machine,
    variance_threshold,
    prep_data,
)

import matplotlib.pyplot as plt








def main():
    """Run all seven steps in order."""
    (
        reduced_training_features,
        reduced_test_features,
        training_labels,
        test_labels
    ) = prep_data()

    classifier = support_vector_machine(reduced_training_features[:, :i], training_labels)
    model_evaluation(classifier, reduced_test_features[:, :i], test_labels)


if __name__ == "__main__":
    main()
