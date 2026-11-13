"""Comparaison des modèles de référence (étape 6) et du transformer (étape 7).

Les résultats de Colab sont déposés ainsi :
    reports/models/<tâche>_camembert_metrics.json
    data/processed/predictions/<tâche>_camembert.parquet
"""

import json

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from nutrilys import config

TRANSFORMER = "camembert"
MODEL_LABELS = {
    "lr": "Régression logistique",
    "xgb": "XGBoost",
    TRANSFORMER: "CamemBERT (fine-tuning)",
}
MIN_GAIN = 0.02  # gain de F1 macro en dessous duquel le modèle simple reste préféré


def transformer_metrics() -> pd.DataFrame:
    """Métriques renvoyées par Colab, une ligne par tâche."""
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(config.MODEL_REPORTS_DIR.glob(f"*_{TRANSFORMER}_metrics.json"))
    ]
    return pd.DataFrame(rows)


def all_metrics() -> pd.DataFrame:
    """Métriques des modèles de référence et du transformer réunies."""
    baseline = pd.read_csv(config.MODEL_REPORTS_DIR / "baseline_metrics.csv")
    return pd.concat([baseline, transformer_metrics()], ignore_index=True)


def predictions(task: str, model: str) -> pd.DataFrame:
    return pd.read_parquet(config.PREDICTIONS_DIR / f"{task}_{model}.parquet")


def same_test_set(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Vrai si les deux modèles ont été évalués sur exactement les mêmes produits."""
    return set(a["code"]) == set(b["code"])


def mcnemar_test(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Test de McNemar : les deux modèles se trompent-ils sur des produits différents
    au point qu'un écart de précision ne puisse pas être dû au hasard ?

    Seuls comptent les désaccords : produits bien classés par l'un et pas par l'autre.
    """
    if not same_test_set(a, b):
        raise ValueError("Les deux modèles n'ont pas été évalués sur le même jeu de test.")
    merged = a.merge(b, on="code", suffixes=("_a", "_b"))
    ok_a = merged["y_pred_a"] == merged["y_true_a"]
    ok_b = merged["y_pred_b"] == merged["y_true_b"]
    table = [
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ]
    disagreements = table[0][1] + table[1][0]
    if disagreements == 0:  # aucun désaccord : aucune différence à tester
        statistic, p_value = 0.0, 1.0
    else:
        # Test exact (binomial) pour peu de désaccords, sinon approximation du chi²
        result = mcnemar(table, exact=disagreements < 25, correction=True)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "only_a_correct": table[0][1],
        "only_b_correct": table[1][0],
        "statistic": statistic,
        "p_value": p_value,
        "accuracy_gap": float(ok_b.mean() - ok_a.mean()),
    }


def choose_model(metrics: pd.DataFrame, task: str, p_value: float | None = None) -> tuple:
    """Modèle retenu pour l'application et justification.

    Le transformer n'est retenu que s'il gagne au moins MIN_GAIN de F1 macro, avec un
    écart significatif (McNemar). Sinon la régression logistique, plus rapide et
    explicable, fonctionnant sans GPU, est conservée.
    """
    rows = metrics[metrics["task"] == task].set_index("model")["f1_macro"]
    lr = rows.get("lr")
    tr = rows.get(TRANSFORMER)
    if tr is None or lr is None:
        return "lr", "Pas de résultat du transformer pour cette tâche."
    gain = tr - lr
    significant = p_value is not None and p_value < 0.05
    gap = f"{gain:+.3f}".replace(".", ",")
    if gain >= MIN_GAIN and significant:
        return TRANSFORMER, f"Écart de F1 macro de {gap} en faveur de CamemBERT, significatif."
    if not significant:
        why = "écart non significatif"
    elif gain < 0:
        why = "la régression logistique est significativement meilleure"
    else:
        why = f"gain significatif mais inférieur au seuil de {MIN_GAIN:+.2f}".replace(".", ",")
    reason = (
        f"Écart de F1 macro de {gap} pour CamemBERT, {why} : la régression logistique, "
        "rapide, explicable et sans GPU, est conservée."
    )
    return "lr", reason
