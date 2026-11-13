"""Métriques d'évaluation des classifieurs.

- accuracy : part de prédictions exactes ;
- balanced_accuracy : moyenne des rappels par classe, insensible au déséquilibre ;
- f1_macro : moyenne des F1 par classe, chaque classe pèse autant (métrique principale) ;
- within_one : part des prédictions à au plus une classe d'écart. NOVA et
  Nutri-Score sont ordonnés : confondre B et C est moins grave que A et E.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true, y_pred) -> dict:
    """Métriques sur des classes encodées de 0 à k-1 (ordre naturel des scores)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "within_one": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "n_test": int(len(y_true)),
    }


def per_class_f1(y_true, y_pred, labels: list) -> pd.Series:
    """F1 de chaque classe, indexé par le libellé de la classe."""
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(len(labels))))
    return pd.Series(scores, index=labels, name="f1")


def confusion_shares(y_true, y_pred, labels: list) -> pd.DataFrame:
    """Matrice de confusion en % de chaque classe réelle (lignes : réel, colonnes : prédit)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(100 * matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    return pd.DataFrame(shares, index=labels, columns=labels)
