from matplotlib import pyplot as plt

from pipeline_steps import model_evaluation, prep_data, support_vector_machine


def pca_accuracy_eval() -> None:
    (
        reduced_training_features,
        reduced_test_features,
        training_labels,
        test_labels,
    ) = prep_data()

    accuracy_list: list[tuple[int, float]] = []
    i: int = 50
    accuracy_threshold: float = 0.95
    while i > 0 and accuracy_threshold > 0.90:
        classifier = support_vector_machine(
            reduced_training_features[:, :i], training_labels
        )
        _, accuracy_threshold = model_evaluation(
            classifier, reduced_test_features[:, :i], test_labels
        )
        print(f"Accuracy threshold for PCA of {i} components: {accuracy_threshold}")
        accuracy_list.append((i, accuracy_threshold))
        i -= 1

    plt.plot([x[0] for x in accuracy_list], [x[1] for x in accuracy_list])
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Accuracy")
    plt.title("PCA Component Selection")
    plt.show()


if __name__ == "__main__":
    pca_accuracy_eval()
