"""Cross-validate and train a support-vector classifier."""

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def support_vector_machine(
    training_features,
    training_labels,
    cross_validation_folds: int = 5,
    random_state: int = 42,
):
    """Cross-validate the linear SVM, fit it, and return the trained model."""
    classifier = SVC(kernel="linear", random_state=random_state)
    scores = cross_val_score(
        classifier,
        training_features,
        training_labels,
        cv=cross_validation_folds,
    )
    print(
        f"\n{cross_validation_folds}-fold CV accuracy on training set: "
        f"{scores.mean():.3f} (+/- {scores.std():.3f})"
    )
    classifier.fit(training_features, training_labels)
    return classifier
