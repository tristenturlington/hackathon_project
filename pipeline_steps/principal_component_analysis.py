"""Scale selected genes and reduce them with principal component analysis."""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def principal_component_analysis(
    training_features,
    test_features,
    number_of_components: int = 50,
    random_state: int = 42,
):
    """Fit scaling and PCA on training data and transform both feature sets."""
    scaler = StandardScaler()
    scaled_training_features = scaler.fit_transform(training_features)
    scaled_test_features = scaler.transform(test_features)

    component_count = min(
        number_of_components,
        scaled_training_features.shape[0],
        scaled_training_features.shape[1],
    )
    pca = PCA(n_components=component_count, random_state=random_state)
    reduced_training_features = pca.fit_transform(scaled_training_features)
    reduced_test_features = pca.transform(scaled_test_features)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(
        f"PCA: {component_count} components explain "
        f"{explained_variance:.1%} of variance."
    )
    return reduced_training_features, reduced_test_features
