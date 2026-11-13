"""Jeux de données des modèles : textes d'entrée, tâches et découpage figé.

Quatre tâches, deux par cible. Dans chaque paire, les deux scénarios portent
sur exactement les mêmes produits et le même découpage, pour que l'écart de
performance ne mesure que l'effet des informations ajoutées.

    nova_name        NOVA à partir du nom et des catégories (scénario A)
    nova_full        NOVA avec en plus la liste d'ingrédients (scénario B)
    nutriscore_text  Nutri-Score à partir du nom, des catégories et des ingrédients (A)
    nutriscore_full  Nutri-Score avec en plus les valeurs nutritionnelles (B)

Le scénario A répond à la vraie question (que permet de deviner le texte ?),
le scénario B sert de contrôle : NOVA et Nutri-Score sont calculés par des
règles à partir des ingrédients et des nutriments, qu'un modèle peut retrouver.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from nutrilys import config

NUTRIENT_FEATURES = [
    "energy_kcal_100g",
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
]


@dataclass(frozen=True)
class Task:
    name: str
    target: str
    population: str  # condition SQL sur la vue products
    with_ingredients: bool
    with_nutrients: bool
    description: str

    @property
    def classes(self) -> list:
        return config.NOVA_GROUPS if self.target == "nova_group" else config.NUTRISCORE_GRADES

    @property
    def pair(self) -> str:
        """Identifiant commun aux deux scénarios d'une même cible (même découpage)."""
        return self.target


NOVA_POPULATION = "has_nova AND has_ingredients"
NUTRISCORE_POPULATION = "has_nutriscore AND has_ingredients AND has_nutrients"

TASKS = {
    "nova_name": Task(
        "nova_name", "nova_group", NOVA_POPULATION, False, False,
        "NOVA à partir du nom et des catégories",
    ),
    "nova_full": Task(
        "nova_full", "nova_group", NOVA_POPULATION, True, False,
        "NOVA avec la liste d'ingrédients",
    ),
    "nutriscore_text": Task(
        "nutriscore_text", "nutriscore_grade", NUTRISCORE_POPULATION, True, False,
        "Nutri-Score à partir du texte",
    ),
    "nutriscore_full": Task(
        "nutriscore_full", "nutriscore_grade", NUTRISCORE_POPULATION, True, True,
        "Nutri-Score avec les valeurs nutritionnelles",
    ),
}  # fmt: skip


def tag_words(tags) -> str:
    """['en:breakfast-cereals', 'fr:pates-a-tartiner'] -> 'breakfast cereals, pates a tartiner'."""
    # Une liste absente arrive sous plusieurs formes (None, pd.NA, NaN) : tout ce qui
    # n'est pas une séquence est traité comme « pas de catégorie »
    if not isinstance(tags, list | tuple | np.ndarray) or len(tags) == 0:
        return ""
    return ", ".join(str(t).split(":", 1)[-1].replace("-", " ") for t in tags)


def build_text(df: pd.DataFrame, with_ingredients: bool) -> pd.Series:
    """Texte d'entrée : nom, dénomination, catégories et, selon le scénario, ingrédients."""
    parts = [
        df["product_name"].fillna(""),
        df["generic_name"].fillna(""),
        "catégories : " + df["categories_tags"].map(tag_words),
    ]
    if with_ingredients:
        parts.append("ingrédients : " + df["ingredients_text"].fillna(""))
    text = parts[0]
    for part in parts[1:]:
        text = text + " | " + part
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def load_task_frame(con, task: Task, sample: int | None = None) -> pd.DataFrame:
    """Produits de la population de la tâche, avec texte, nutriments et cible."""
    nutrients = ", ".join(NUTRIENT_FEATURES)
    df = con.execute(
        f"""
        SELECT code, product_name, generic_name, categories_tags, ingredients_text,
               {nutrients}, is_organic, {task.target} AS target
        FROM products
        WHERE {task.population}
        ORDER BY code
        """
    ).df()
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=config.RANDOM_SEED).sort_values("code")
    df["text"] = build_text(df, task.with_ingredients)
    return df.reset_index(drop=True)


def make_splits(codes: pd.Series, target: pd.Series) -> pd.Series:
    """Découpage train / valid / test stratifié sur la cible, reproductible."""
    test_size = config.TEST_SIZE
    valid_size = config.VALID_SIZE / (1 - test_size)
    idx = np.arange(len(codes))
    rest, test = train_test_split(
        idx, test_size=test_size, stratify=target, random_state=config.RANDOM_SEED
    )
    train, valid = train_test_split(
        rest, test_size=valid_size, stratify=target.iloc[rest], random_state=config.RANDOM_SEED
    )
    split = pd.Series("train", index=codes.index)
    split.iloc[valid] = "valid"
    split.iloc[test] = "test"
    return split


def assign_splits(df: pd.DataFrame, task: Task, path: Path | None = None) -> pd.DataFrame:
    """Ajoute la colonne split, en réutilisant le découpage déjà enregistré pour la cible.

    Le fichier splits.parquet garde une ligne par (cible, produit) : les deux
    scénarios d'une cible et l'étape 7 (Colab) utilisent ainsi le même jeu de test.
    """
    path = path or config.SPLITS_PATH  # lu à l'appel, pas au chargement du module
    saved = (
        pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["pair", "code", "split"])
    )
    known = saved[saved["pair"] == task.pair]
    if set(df["code"]) <= set(known["code"]):
        return df.merge(known[["code", "split"]], on="code", how="left")
    split = make_splits(df["code"], df["target"])
    new = pd.DataFrame({"pair": task.pair, "code": df["code"], "split": split.values})
    saved = pd.concat([saved[saved["pair"] != task.pair], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    saved.to_parquet(path, index=False)
    return df.assign(split=split.values)
