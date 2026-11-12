"""Construction de la table de référence à partir de l'extrait Open Food Facts.

Étapes :
    1. dédoublonnage par code-barres (on garde la fiche la plus récente) ;
    2. retrait des produits sans nom ;
    3. nettoyage des valeurs : scores hors domaine et nutriments aberrants mis à vide ;
    4. ajout des variables dérivées (bio, groupe alimentaire, indicateurs de disponibilité).

Les produits ne sont retirés qu'aux étapes 1 et 2. Ailleurs, une valeur douteuse
est mise à vide plutôt que de supprimer tout le produit.

Usage :
    uv run python -m nutrilys.data.clean
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from nutrilys import config
from nutrilys.data.schema import PRODUCTS_COLUMNS, write_data_dictionary
from nutrilys.db import connect, sql_path

logger = logging.getLogger(__name__)

NUTRIENT_COLUMNS = [n.replace("-", "_") + "_100g" for n in config.NUTRIENTS]

# Règles « sous-total <= total » : (colonne contrôlée, colonne de référence)
NUTRIENT_SUBTOTALS = [
    ("saturated_fat_100g", "fat_100g"),
    ("sugars_100g", "carbohydrates_100g"),
]


def _sql_list(values: list[str]) -> str:
    return "[" + ", ".join(f"'{v}'" for v in values) + "]"


def _sql_tuple(values: list[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _bounded(column: str) -> str:
    """Valeur conservée seulement si elle est dans les bornes physiques."""
    upper = config.NUTRIENT_MAX_100G.get(column, config.DEFAULT_NUTRIENT_MAX_100G)
    return f"CASE WHEN {column} BETWEEN 0 AND {upper} THEN {column} END AS {column}"


def _consistent(column: str) -> str:
    """Valeur mise à vide si elle dépasse le total dont elle fait partie."""
    for part, total in NUTRIENT_SUBTOTALS:
        if column == part:
            tol = config.NUTRIENT_TOLERANCE_G
            return f"CASE WHEN {part} > {total} + {tol} THEN NULL ELSE {part} END AS {part}"
    return column


def _final_select() -> str:
    """Sélection finale : colonnes nettoyées et variables dérivées, dans l'ordre du schéma."""
    grades = _sql_tuple(config.NUTRISCORE_GRADES)
    green = _sql_tuple(config.GREEN_SCORE_GRADES)
    organic = _sql_list(config.ORGANIC_LABELS)
    nutrients = ",\n        ".join(NUTRIENT_COLUMNS)
    return f"""
    SELECT
        code,
        product_name,
        generic_name,
        ingredients_text,
        brand,
        brand_tag,
        categories_tags,
        main_category,
        food_group,
        food_subgroup,
        labels_tags,
        countries_tags,
        coalesce(list_has_any(labels_tags, {organic}), false) AS is_organic,
        nutriscore_grade,
        CASE WHEN nutriscore_grade IS NOT NULL THEN nutriscore_score END AS nutriscore_score,
        nova_group,
        green_score_grade,
        CASE WHEN ingredients_text IS NOT NULL
             THEN coalesce(additives_n, len(additives)) END AS n_additives,
        additives,
        CASE WHEN ingredients_text IS NOT NULL THEN ingredients_n END AS n_ingredients,
        coalesce(ingredients_from_palm_oil_n, 0) > 0 AS has_palm_oil,
        {nutrients},
        ingredients_text IS NOT NULL AS has_ingredients,
        nutriscore_grade IS NOT NULL AS has_nutriscore,
        nova_group IS NOT NULL AS has_nova,
        (sugars_100g IS NOT NULL AND salt_100g IS NOT NULL
         AND saturated_fat_100g IS NOT NULL) AS has_nutrients,
        completeness,
        unique_scans_n AS n_scans,
        last_modified
    FROM (
        SELECT
            * EXCLUDE (nutriscore_grade, nova_group, green_score_grade),
            CASE WHEN lower(nutriscore_grade) IN {grades}
                 THEN lower(nutriscore_grade) END AS nutriscore_grade,
            CASE WHEN nova_group BETWEEN 1 AND 4 THEN nova_group END AS nova_group,
            CASE WHEN lower(green_score_grade) IN {green}
                 THEN lower(green_score_grade) END AS green_score_grade
        FROM normalized
    )
    """


def build_products(
    source: Path = config.OFF_INTERIM_PATH,
    dest: Path = config.PRODUCTS_PATH,
    report_path: Path | None = config.CLEANING_REPORT_PATH,
) -> dict:
    """Construit la table de référence et renvoie le rapport de nettoyage."""
    if not source.exists():
        raise FileNotFoundError(
            f"{source} introuvable. Lancer d'abord : uv run python -m nutrilys.data.extract"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    with connect() as con:
        con.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{sql_path(source)}')")
        steps.append({"step": "Extrait France", "rows": _count(con, "source")})

        # 1. Dédoublonnage : une ligne par code-barres, la plus récente
        con.execute("""
            CREATE TEMP TABLE dedup AS
            SELECT * REPLACE (trim(code) AS code)
            FROM source
            WHERE nullif(trim(code), '') IS NOT NULL
            QUALIFY row_number() OVER (
                PARTITION BY trim(code) ORDER BY last_modified_t DESC NULLS LAST
            ) = 1
        """)
        steps.append({"step": "Un produit par code-barres", "rows": _count(con, "dedup")})

        # 2. Produits sans nom : inutilisables dans l'app comme dans les analyses
        con.execute("""
            CREATE TEMP TABLE named AS
            SELECT * FROM dedup WHERE nullif(trim(product_name), '') IS NOT NULL
        """)
        steps.append({"step": "Avec un nom de produit", "rows": _count(con, "named")})

        # 3. Normalisation des textes et bornes physiques des nutriments
        bounded = ",\n                ".join(_bounded(c) for c in NUTRIENT_COLUMNS)
        con.execute(f"""
            CREATE TEMP TABLE bounded AS
            SELECT
                code,
                regexp_replace(trim(product_name), '\\s+', ' ', 'g') AS product_name,
                nullif(trim(generic_name), '') AS generic_name,
                nullif(trim(ingredients_text), '') AS ingredients_text,
                nullif(trim(split_part(brands, ',', 1)), '') AS brand,
                brands_tags[1] AS brand_tag,
                categories_tags,
                nullif(replace(compared_to_category, 'en:', ''), '') AS main_category,
                replace(food_groups_tags[1], 'en:', '') AS food_group,
                replace(food_groups_tags[2], 'en:', '') AS food_subgroup,
                labels_tags,
                countries_tags,
                nutriscore_grade,
                nutriscore_score,
                nova_group,
                environmental_score_grade AS green_score_grade,
                additives_n,
                -- en:e322i -> E322i (seule la lettre E en majuscule)
                list_transform(
                    coalesce(additives_tags, []), lambda x: 'E' || replace(x, 'en:e', '')
                ) AS additives,
                ingredients_n,
                ingredients_from_palm_oil_n,
                {bounded},
                completeness,
                unique_scans_n,
                to_timestamp(last_modified_t) AS last_modified
            FROM named
        """)

        # Cohérence entre nutriments (sous-total <= total)
        consistent = ",\n                ".join(_consistent(c) for c in NUTRIENT_COLUMNS)
        others = ", ".join(NUTRIENT_COLUMNS)
        con.execute(f"""
            CREATE TEMP VIEW normalized AS
            SELECT * EXCLUDE ({others}), {consistent}
            FROM bounded
        """)

        # 4. Variables dérivées et écriture
        con.execute(f"CREATE TEMP TABLE products AS {_final_select()}")
        con.execute(f"COPY products TO '{sql_path(dest)}' (FORMAT parquet, COMPRESSION zstd)")
        steps.append({"step": "Table de référence", "rows": _count(con, "products")})

        nullified = {
            c: _count_not_null(con, "named", c) - _count_not_null(con, "products", c)
            for c in NUTRIENT_COLUMNS
        }
        invalid_scores = {
            c: _count_not_null(con, "named", c) - _count_not_null(con, "products", c)
            for c in ("nutriscore_grade", "nova_group")
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source.name,
        "output": dest.name,
        "steps": steps,
        "invalid_scores_set_to_null": invalid_scores,
        "nutrient_values_set_to_null": nullified,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Table de référence écrite : %s (%d produits)", dest, steps[-1]["rows"])
    return report


def _count(con, table: str) -> int:
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _count_not_null(con, table: str, column: str) -> int:
    return con.execute(f"SELECT count({column}) FROM {table}").fetchone()[0]


def check_columns(path: Path = config.PRODUCTS_PATH) -> None:
    """Vérifie que la table produite respecte le schéma documenté."""
    with connect() as con:
        columns = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')"
            ).fetchall()
        ]
    if columns != PRODUCTS_COLUMNS:
        raise ValueError(f"Colonnes inattendues : {columns}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = build_products()
    check_columns()
    write_data_dictionary()

    print("\nÉtapes du nettoyage :")
    for step in report["steps"]:
        print(f"  {step['step']:<30} {step['rows']:>10,}".replace(",", " "))
    print("\nScores hors domaine mis à vide :")
    for column, n in report["invalid_scores_set_to_null"].items():
        print(f"  {column:<30} {n:>10,}".replace(",", " "))
    print("\nValeurs nutritionnelles aberrantes mises à vide :")
    for column, n in report["nutrient_values_set_to_null"].items():
        print(f"  {column:<30} {n:>10,}".replace(",", " "))


if __name__ == "__main__":
    main()
