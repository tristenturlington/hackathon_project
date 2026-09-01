"""Select genes associated with cancer type using an ANOVA F-test."""

from sklearn.feature_selection import SelectKBest, f_classif


def anova_test(
    training_features,
    test_features,
    training_labels,
    number_of_genes: int = 1000,
):
    """Fit ANOVA selection on training data and transform both feature sets."""
    selected_count = min(number_of_genes, training_features.shape[1])
    selector = SelectKBest(score_func=f_classif, k=selected_count)
    selected_training_features = selector.fit_transform(
        training_features, training_labels
    )
    selected_test_features = selector.transform(test_features)
    print(f"Selected top {selected_count} genes by ANOVA F-test.")
    return selected_training_features, selected_test_features
