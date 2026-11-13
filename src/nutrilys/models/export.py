"""Export des données des tâches « texte seul » pour l'entraînement sur Google Colab.

Un fichier par tâche, avec le texte d'entrée, la cible et le découpage figé de
l'étape 6 : CamemBERT est ainsi évalué sur exactement le même jeu de test que
les modèles de référence.

Usage :
    uv run python -m nutrilys.models.export
"""

import logging

import pandas as pd

from nutrilys import config
from nutrilys.db import products_connection
from nutrilys.models.dataset import TASKS, assign_splits, load_task_frame
from nutrilys.viz.fmt import fmt_int

logger = logging.getLogger(__name__)

COLUMNS = ["code", "text", "target", "split"]


def export_task(con, name: str) -> pd.DataFrame:
    """Données d'une tâche prêtes pour Colab (le découpage existant est réutilisé)."""
    task = TASKS[name]
    df = assign_splits(load_task_frame(con, task), task)[COLUMNS]
    if df["split"].isna().any():
        raise ValueError(f"{name} : produits sans découpage, relancer l'étape 6.")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.COLAB_DIR.mkdir(parents=True, exist_ok=True)
    con = products_connection()
    for name in config.TRANSFORMER_TASKS:
        df = export_task(con, name)
        path = config.COLAB_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        counts = df["split"].value_counts()
        print(
            f"{path.name} : {fmt_int(len(df))} produits (train {fmt_int(counts['train'])}, "
            f"valid {fmt_int(counts['valid'])}, test {fmt_int(counts['test'])}), "
            f"{path.stat().st_size / 1024**2:.1f} Mo"
        )
    con.close()
    print(f"\nFichiers à charger dans Colab : {config.COLAB_DIR}")


if __name__ == "__main__":
    main()
