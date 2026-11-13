"""Modèle entraîné, prêt à être enregistré et rechargé.

Ce module est séparé du script d'entraînement : une classe définie dans un script
lancé avec `python -m` serait enregistrée sous le nom `__main__`, et le modèle
ne pourrait plus être rechargé depuis un notebook ou l'application.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from nutrilys.models.dataset import NUTRIENT_FEATURES


def shift_for_sparse(x) -> np.ndarray:
    """Décale les nutriments de +1 pour XGBoost.

    Dans une matrice creuse, XGBoost traite les zéros comme des valeurs manquantes :
    « 0 g de sucre » se confondrait avec « sucre inconnu ». Le décalage garde les
    vrais zéros distincts ; les arbres sont insensibles à ce changement d'échelle.
    """
    return np.asarray(x, dtype=float) + 1.0


@dataclass
class FittedModel:
    """Tout ce qu'il faut pour prédire : vectoriseur, prétraitement numérique, classifieur."""

    task: str
    model: str
    classes: list
    vectorizer: TfidfVectorizer
    numeric: object | None
    classifier: object
    selector: object | None = None  # sélection de termes (XGBoost)
    timings: dict = field(default_factory=dict)

    def features(self, df: pd.DataFrame):
        x = self.vectorizer.transform(df["text"])
        selector = getattr(self, "selector", None)  # absent des modèles enregistrés avant
        if selector is not None:
            x = selector.transform(x)
        if self.numeric is not None:
            num = self.numeric.transform(df[NUTRIENT_FEATURES])
            x = sparse.hstack([x, sparse.csr_matrix(num)], format="csr")
        return x

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(self.features(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Classes prédites, dans leur forme d'origine (1-4 ou a-e)."""
        codes = self.predict_proba(df).argmax(axis=1)
        return np.asarray(self.classes)[codes]
