"""Schéma de la table de référence data/processed/products.parquet.

Ce dictionnaire sert à la fois de contrat (vérifié par les tests) et de
documentation (exporté dans docs/data_dictionary.md).
"""

from pathlib import Path

from nutrilys import config

# nom de colonne -> (type, description)
PRODUCTS_SCHEMA: dict[str, tuple[str, str]] = {
    "code": ("texte", "Code-barres du produit, identifiant unique."),
    "product_name": ("texte", "Nom du produit, langue principale ou à défaut français."),
    "generic_name": ("texte", "Dénomination générique (« biscuits fourrés au chocolat »)."),
    "ingredients_text": ("texte", "Liste des ingrédients telle que saisie sur Open Food Facts."),
    "brand": ("texte", "Première marque déclarée, pour l'affichage."),
    "brand_tag": (
        "texte",
        "Première marque normalisée par Open Food Facts, pour les regroupements.",
    ),
    "categories_tags": ("liste", "Catégories Open Food Facts, de la plus large à la plus précise."),
    "main_category": (
        "texte",
        "Catégorie de référence utilisée par Open Food Facts pour comparer.",
    ),
    "food_group": ("texte", "Groupe alimentaire de niveau 1 (PNNS), ex. sugary-snacks."),
    "food_subgroup": (
        "texte",
        "Sous-groupe alimentaire de niveau 2 (PNNS), ex. biscuits-and-cakes.",
    ),
    "labels_tags": ("liste", "Labels déclarés (bio, AOP, commerce équitable...)."),
    "countries_tags": ("liste", "Pays de vente."),
    "is_organic": ("booléen", "Vrai si un label bio est déclaré (variable est_bio)."),
    "nutriscore_grade": (
        "texte",
        "Nutri-Score de a à e (algorithme 2023), vide si non calculable.",
    ),
    "nutriscore_score": ("entier", "Score numérique sous-jacent au Nutri-Score."),
    "nova_group": ("entier", "Groupe NOVA de 1 (brut) à 4 (ultra-transformé), vide si inconnu."),
    "green_score_grade": ("texte", "Green-Score, ex-Eco-Score (a-plus à f), affichage seulement."),
    "n_additives": ("entier", "Nombre d'additifs, vide si la liste d'ingrédients est absente."),
    "additives": ("liste", "Codes des additifs détectés (E322, E471...)."),
    "n_ingredients": ("entier", "Nombre d'ingrédients détectés."),
    "has_palm_oil": ("booléen", "Vrai si au moins un ingrédient issu de l'huile de palme."),
    "energy_kcal_100g": ("réel", "Énergie en kcal pour 100 g."),
    "fat_100g": ("réel", "Matières grasses en g pour 100 g."),
    "saturated_fat_100g": ("réel", "Acides gras saturés en g pour 100 g."),
    "carbohydrates_100g": ("réel", "Glucides en g pour 100 g."),
    "sugars_100g": ("réel", "Sucres en g pour 100 g."),
    "fiber_100g": ("réel", "Fibres en g pour 100 g."),
    "proteins_100g": ("réel", "Protéines en g pour 100 g."),
    "salt_100g": ("réel", "Sel en g pour 100 g."),
    "has_ingredients": ("booléen", "Liste d'ingrédients disponible."),
    "has_nutriscore": ("booléen", "Nutri-Score valide (a à e) disponible."),
    "has_nova": ("booléen", "Groupe NOVA valide (1 à 4) disponible."),
    "has_nutrients": ("booléen", "Sucres, sel et graisses saturées tous renseignés."),
    "completeness": ("réel", "Complétude de la fiche selon Open Food Facts (0 à 1)."),
    "n_scans": ("entier", "Nombre de scans uniques, indicateur de popularité."),
    "last_modified": ("date", "Date de dernière modification de la fiche."),
}

PRODUCTS_COLUMNS = list(PRODUCTS_SCHEMA)


def write_data_dictionary(path: Path = config.ROOT_DIR / "docs" / "data_dictionary.md") -> Path:
    """Exporte le schéma en tableau Markdown."""
    lines = [
        "# Dictionnaire de données",
        "",
        "Table de référence : `data/processed/products.parquet`, une ligne par produit.",
        "Source : Open Food Facts (licence ODbL). Généré par `nutrilys.data.clean`.",
        "",
        "| Colonne | Type | Description |",
        "|---|---|---|",
    ]
    lines += [f"| `{name}` | {kind} | {desc} |" for name, (kind, desc) in PRODUCTS_SCHEMA.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
