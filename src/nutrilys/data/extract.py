"""Extraction des produits utiles depuis l'export brut d'Open Food Facts.

Le fichier brut fait plusieurs Go : DuckDB le lit en flux et n'écrit que les
produits des pays retenus et les colonnes utiles, sans tout charger en mémoire.

Usage :
    uv run python -m nutrilys.data.extract
"""

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd

from nutrilys import config

logger = logging.getLogger(__name__)

# Colonnes reprises telles quelles
PLAIN_COLUMNS = [
    "code",
    "brands",
    "brands_tags",
    "categories_tags",
    "food_groups_tags",
    "compared_to_category",
    "labels_tags",
    "countries_tags",
    "additives_n",
    "additives_tags",
    "ingredients_n",
    "ingredients_analysis_tags",
    "ingredients_from_palm_oil_n",
    "nutriscore_grade",
    "nutriscore_score",
    "nova_group",
    "environmental_score_grade",
    "no_nutrition_data",
    "completeness",
    "unique_scans_n",
    "last_modified_t",
]

# Colonnes multilingues : liste de {lang, text}
TEXT_COLUMNS = ["product_name", "generic_name", "ingredients_text"]


def _text_expr(column: str) -> str:
    """Texte de la langue principale du produit, à défaut le texte en français."""
    main = f"list_filter({column}, lambda x: x.lang = 'main')[1].text"
    french = f"list_filter({column}, lambda x: x.lang = 'fr')[1].text"
    return f"coalesce({main}, {french}) AS {column}"


def _nutrient_expr(nutrient: str) -> str:
    """Valeur pour 100 g d'un nutriment, extraite de la liste nutriments."""
    alias = nutrient.replace("-", "_") + "_100g"
    value = f"list_filter(nutriments, lambda x: x.name = '{nutrient}')[1]['100g']"
    return f"TRY_CAST({value} AS DOUBLE) AS {alias}"


def build_extract_query(source: Path, countries: list[str]) -> str:
    """Construit la requête SELECT d'extraction."""
    columns = (
        PLAIN_COLUMNS[:1]
        + [_text_expr(c) for c in TEXT_COLUMNS]
        + PLAIN_COLUMNS[1:]
        + [_nutrient_expr(n) for n in config.NUTRIENTS]
    )
    countries_sql = "[" + ", ".join(f"'{c}'" for c in countries) + "]"
    return (
        "SELECT\n    "
        + ",\n    ".join(columns)
        + f"\nFROM read_parquet('{source.as_posix()}')"
        + f"\nWHERE list_has_any(countries_tags, {countries_sql})"
        + "\n  AND coalesce(obsolete, false) = false"
    )


def extract_off(
    source: Path = config.OFF_RAW_PATH,
    dest: Path = config.OFF_INTERIM_PATH,
    countries: list[str] | None = None,
) -> int:
    """Écrit l'extrait filtré au format Parquet et renvoie son nombre de lignes."""
    countries = countries or config.COUNTRIES
    if not source.exists():
        raise FileNotFoundError(
            f"{source} introuvable. Lancer d'abord : uv run python -m nutrilys.data.download"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)

    query = build_extract_query(source, countries)
    with duckdb.connect() as con:
        con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads = {config.DUCKDB_THREADS}")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"COPY ({query}) TO '{dest.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        count_query = f"SELECT count(*) FROM read_parquet('{dest.as_posix()}')"
        n_rows = con.execute(count_query).fetchone()[0]

    logger.info("Extrait écrit : %s (%d produits, pays : %s)", dest, n_rows, ", ".join(countries))
    return n_rows


def summarize(path: Path = config.OFF_INTERIM_PATH) -> pd.DataFrame:
    """Taux de remplissage des colonnes clés de l'extrait."""
    key_columns = [
        "product_name",
        "ingredients_text",
        "brands",
        "categories_tags",
        "nutriscore_grade",
        "nova_group",
        "additives_n",
        "sugars_100g",
        "salt_100g",
        "saturated_fat_100g",
    ]
    exprs = ", ".join(f"round(100 * avg(({c} IS NOT NULL)::int), 1) AS {c}" for c in key_columns)
    with duckdb.connect() as con:
        row = con.execute(f"SELECT {exprs} FROM read_parquet('{path.as_posix()}')").df()
    return row.T.rename(columns={0: "rempli_%"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrait les produits utiles d'Open Food Facts.")
    parser.add_argument("--countries", nargs="+", help="tags pays, par défaut config.COUNTRIES")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_off(countries=args.countries)
    print("\nTaux de remplissage des colonnes clés :")
    print(summarize().to_string())


if __name__ == "__main__":
    main()
