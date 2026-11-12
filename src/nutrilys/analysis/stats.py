"""Tests statistiques et tailles d'effet.

Avec des centaines de milliers de produits, presque toute différence est
« significative » : la p-valeur ne suffit pas. On rapporte donc toujours une
taille d'effet et son interprétation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AssociationResult:
    test: str
    statistic: float
    p_value: float
    effect_size: float
    effect_name: str
    interpretation: str
    n: int


def cramers_v(contingency: pd.DataFrame) -> AssociationResult:
    """Chi² d'indépendance et V de Cramér sur un tableau de contingence (effectifs)."""
    table = np.asarray(contingency, dtype=float)
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = int(table.sum())
    k = min(table.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n and k else float("nan")
    return AssociationResult(
        test="Chi² d'indépendance",
        statistic=float(chi2),
        p_value=float(p_value),
        effect_size=v,
        effect_name="V de Cramér",
        interpretation=interpret_cramers_v(v, k),
        n=n,
    )


def interpret_cramers_v(v: float, k: int) -> str:
    """Seuils de Cohen, ajustés au nombre de degrés de liberté min(lignes, colonnes) - 1."""
    small, medium, large = (np.array([0.1, 0.3, 0.5]) / np.sqrt(max(k, 1))).tolist()
    if v >= large:
        return "forte"
    if v >= medium:
        return "moyenne"
    if v >= small:
        return "faible"
    return "négligeable"


def spearman(x: pd.Series, y: pd.Series) -> AssociationResult:
    """Corrélation de rang de Spearman (relation monotone entre deux variables ordonnées)."""
    mask = x.notna() & y.notna()
    rho, p_value = stats.spearmanr(x[mask], y[mask])
    return AssociationResult(
        test="Corrélation de Spearman",
        statistic=float(rho),
        p_value=float(p_value),
        effect_size=float(rho),
        effect_name="rho",
        interpretation=interpret_correlation(float(rho)),
        n=int(mask.sum()),
    )


def interpret_correlation(r: float) -> str:
    """Seuils de Cohen pour un coefficient de corrélation."""
    size = abs(r)
    if size >= 0.5:
        strength = "forte"
    elif size >= 0.3:
        strength = "moyenne"
    elif size >= 0.1:
        strength = "faible"
    else:
        return "négligeable"
    return f"{strength}, {'positive' if r > 0 else 'négative'}"


def describe(result: AssociationResult) -> str:
    """Phrase de synthèse, prête pour un sous-titre de graphique."""
    p = "p < 0,001" if result.p_value < 0.001 else f"p = {result.p_value:.3f}".replace(".", ",")
    effect = f"{result.effect_size:.2f}".replace(".", ",")
    n = f"{result.n:,}".replace(",", " ")
    return (
        f"{result.test} : {p}, {result.effect_name} = {effect} "
        f"(association {result.interpretation}), n = {n}"
    )
