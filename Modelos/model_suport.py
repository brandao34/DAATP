import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd

def load_best_score():
    # Path to the file that stores the best score
    best_score_file = 'best_score.txt'

    # Initialize default values
    best_score_ever = 0.0
    best_model_name = "N/A"

    # Check if the file exists and load the stored best score and model name
    if os.path.exists(best_score_file):
        with open(best_score_file, 'r') as file:
            lines = file.readlines()
            best_score_ever = float(lines[0].strip())
            best_model_name = lines[1].strip() if len(lines) > 1 else "N/A"

    # Print current best score and model name
    print(f"Current Best Score Stored: {best_score_ever * 100:.20f}%")
    print(f"Model with Best Score: {best_model_name}\n")

    return best_score_ever, best_model_name

label_mapping = {
    'CN-CN': 0,
    'AD-AD': 1,
    'MCI-AD': 2,
    'MCI-MCI': 3, 
#    'CN-MCI' : 4

}
def plot_confusion_matrix_with_labels(confusion_matrix):
    # Criar um mapeamento inverso
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Aplicar o mapeamento inverso na matriz de confusão
    cm_with_labels = np.zeros_like(confusion_matrix, dtype=object)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            cm_with_labels[i, j] = f"{reverse_label_mapping[i]} (Pred: {reverse_label_mapping[j]})"

    # Criar um DataFrame para facilitar a visualização
    df_cm = pd.DataFrame(confusion_matrix, index=reverse_label_mapping.values(), columns=reverse_label_mapping.values())
    
    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix with Labels')
    plt.show()