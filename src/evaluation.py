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
    cmap = plt.cm.get_cmap('tab20', 20)
    for i, g in enumerate(sorted(labels.dropna().unique())):
        if pd.isna(g):
            continue
        mask = labels == g
        marker = 'o'
        if i > 19:
            marker = '*'
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=g, color=cmap(i % 20), marker=marker)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.show()

def plot_with_label_continuous(embeddings: np.ndarray, labels: pd.Series, title: str):
    # plt.scatter(embeddings[labels.notna(), 0], embeddings[labels.notna(), 1], c=labels.dropna())
    cmap = plt.cm.get_cmap('viridis', labels.nunique())
    for i, g in enumerate(sorted(labels.dropna().unique())):
        if pd.isna(g):
            continue
        mask = labels == g
        marker = 'o'
        if i > 19:
            marker = '*'
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=g, color=cmap(i % 20), marker=marker)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title(title)
    plt.show()
