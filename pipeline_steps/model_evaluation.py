"""Evaluate a fitted classifier on held-out data."""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def model_evaluation(classifier, test_features, test_labels):
    """Print test metrics and return the predicted cancer classes."""
    predictions = classifier.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\nTest set accuracy: {accuracy:.3f}\n")
    print("Classification report:")
    print(classification_report(test_labels, predictions))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(
                test_labels,
                predictions,
                labels=classifier.classes_,
            ),
            index=classifier.classes_,
            columns=classifier.classes_,
        )
    )
    return predictions, accuracy
