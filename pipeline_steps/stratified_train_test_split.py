"""Create stratified training and test sets."""

from sklearn.model_selection import train_test_split


def stratified_train_test_split(
    features,
    labels,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Split data while preserving the relative size of each cancer class."""
    return train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels, # stratified ensures that the proportion of each class is preserved in both the training and test sets
        random_state=seed,
    )
