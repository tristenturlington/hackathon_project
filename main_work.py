"""

Tristen Turlington, Ben Anderson, Mariah, Addi Bruening
8/31/2026

Our main goals from the job title:
    - LLM Expertise: Experience developing and training large-scale machine learning models, including post-training techniques to enhance domain knowledge, reasoning capabilities, and model alignment.
    - You focus on the execution of defined projects. You are responsible for writing clean, efficient code to test specific hypotheses regarding reasoning and alignment.
    - Experience with molecular modalities (e.g., protein sequences, chemical graphs, and structured molecular data).
    - A passion for applying frontier AI to drug discovery.

STEPS:
    1.
"""

import csv
from pathlib import Path

import numpy as np


def get_data():
    """Return the gene data as a matrix and the class labels as a list."""
    dataset_directory = Path(__file__).resolve().parent / "dataset"
    data_path = dataset_directory / "data.csv"
    labels_path = dataset_directory / "labels.csv"

    with data_path.open(newline="") as data_file:
        number_of_columns = len(next(csv.reader(data_file)))

    data_matrix = np.loadtxt(
        data_path,
        delimiter=",",
        skiprows=1,
        usecols=range(1, number_of_columns),
    )

    with labels_path.open(newline="") as labels_file:
        rows = csv.reader(labels_file)
        next(rows)
        labels = [row[1] for row in rows]

    if data_matrix.shape[0] != len(labels):
        raise ValueError("The data and label CSV files contain different sample counts.")

    return data_matrix, labels




def step_1():
    pass


 # curl -L -o ~/breast-cancer-detection-unet.zip https://www.kaggle.com/api/v1/datasets/download/utkarshsaxenadn/breast-cancer-detection-unet

if __name__ == "__main__":
    print(lets_go())