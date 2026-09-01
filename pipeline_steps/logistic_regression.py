"""Multiclass logistic regression implemented with NumPy."""

import numpy as np


class LogisticRegressionClassifier:
    """A multiclass softmax classifier trained with gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        number_of_iterations: int = 1000,
        tolerance: float = 1e-7,
    ):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.tolerance = tolerance
        self.weights_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.loss_ = None
        self.n_iter_ = 0

    def fit(self, features, labels):
        """Learn one set of softmax weights for each cancer class."""
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)

        if features.ndim != 2:
            raise ValueError("Training features must be a two-dimensional matrix.")
        if labels.ndim != 1:
            raise ValueError("Training labels must be a one-dimensional array.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must contain the same samples.")
        if features.shape[0] == 0:
            raise ValueError("Cannot train logistic regression with no samples.")

        self.classes_, encoded_labels = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Logistic regression requires at least two classes.")

        number_of_samples, self.n_features_in_ = features.shape
        features_with_bias = self._add_bias_column(features)
        one_hot_labels = np.eye(len(self.classes_))[encoded_labels]
        self.weights_ = np.zeros(
            (features_with_bias.shape[1], len(self.classes_)),
            dtype=float,
        )

        for iteration in range(self.number_of_iterations):
            scores = features_with_bias @ self.weights_
            probabilities = self._softmax(scores)
            error = probabilities - one_hot_labels
            gradient = features_with_bias.T @ error / number_of_samples
            self.weights_ -= self.learning_rate * gradient
            self.n_iter_ = iteration + 1

            if np.linalg.norm(gradient) <= self.tolerance:
                break

        final_probabilities = self._softmax(features_with_bias @ self.weights_)
        self.loss_ = -np.mean(
            np.sum(
                one_hot_labels * np.log(final_probabilities + 1e-12),
                axis=1,
            )
        )
        return self

    def predict_proba(self, features):
        """Return the probability assigned to every class for each sample."""
        features = self._validate_prediction_features(features)
        scores = self._add_bias_column(features) @ self.weights_
        return self._softmax(scores)

    def predict(self, features):
        """Return the most probable cancer class for each sample."""
        probabilities = self.predict_proba(features)
        predicted_class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_class_indices]

    def _validate_prediction_features(self, features):
        if self.weights_ is None or self.classes_ is None:
            raise ValueError("The classifier must be fitted before prediction.")

        features = np.asarray(features, dtype=float)
        if features.ndim != 2:
            raise ValueError("Prediction features must be a two-dimensional matrix.")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, "
                f"but received {features.shape[1]}."
            )
        return features

    @staticmethod
    def _add_bias_column(features):
        return np.column_stack((np.ones(features.shape[0]), features))

    @staticmethod
    def _softmax(scores):
        stabilized_scores = scores - np.max(scores, axis=1, keepdims=True)
        exponentials = np.exp(stabilized_scores)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def logistic_regression(
    training_features,
    training_labels,
    learning_rate: float = 0.1,
    number_of_iterations: int = 1000,
):
    """Train and return the custom multiclass logistic classifier."""
    classifier = LogisticRegressionClassifier(
        learning_rate=learning_rate,
        number_of_iterations=number_of_iterations,
    )
    return classifier.fit(training_features, training_labels)
