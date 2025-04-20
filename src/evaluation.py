import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score
import matplotlib.pyplot as plt


def evaluate_clusters(cluster: np.ndarray, labels: pd.DataFrame) -> dict:
    scores = {}
    for label_col in labels.columns:
        notna = labels[label_col].notna()
        ytrue = labels[label_col][notna]
        ypred = cluster[notna]
        scores[label_col] = homogeneity_score(ytrue, ypred)
    return scores

def plot_with_label(embeddings: np.ndarray, labels: pd.Series, title: str):
    cmap = plt.colormaps.get_cmap('tab20')
    for i, g in enumerate(labels.dropna().unique()):
        if pd.isna(g):
            continue
        mask = labels == g
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=g, color=cmap(i))
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.show()
