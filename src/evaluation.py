import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_score


def evaluate_clusters(cluster: np.ndarray, labels: pd.DataFrame) -> dict:
    scores = {}
    for label_col in labels.columns:
        notna = labels[label_col].notna()
        ytrue = labels[label_col][notna]
        ypred = cluster[notna]
        scores[label_col] = homogeneity_score(ytrue, ypred)
    return scores
