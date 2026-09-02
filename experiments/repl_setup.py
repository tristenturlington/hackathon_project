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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_ROOT / "dataset"
DATA_PATH = DATASET_DIRECTORY / "data.csv"
LABELS_PATH = DATASET_DIRECTORY / "labels.csv"
