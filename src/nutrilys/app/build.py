"""Table légère de l'application : une ligne par produit, déjà jointe et triée.

Elle réunit la table de référence, les signaux de controverse et l'estimation NOVA
des produits sans liste d'ingrédients. Le texte de recherche est préparé une fois
pour toutes (minuscules, sans accents), et les produits sont triés par popularité :
les premiers résultats d'une recherche sont ainsi les produits les plus scannés.

Usage :
    uv run python -m nutrilys.app.build
"""

import logging
from pathlib import Path

from nutrilys import config
from nutrilys.db import connect, sql_path

logger = logging.getLogger(__name__)


def build_app_table(
    products: Path = config.PRODUCTS_PATH,
    controversies: Path = config.CONTROVERSIES_PATH,
    nova_estimated: Path = config.NOVA_ESTIMATED_PATH,
    dest: Path = config.APP_PRODUCTS_PATH,
) -> int:
    """Écrit la table de l'application et renvoie son nombre de lignes."""
    if not products.exists():
        raise FileNotFoundError(f"{products} introuvable : lancer nutrilys.data.clean.")
    with connect() as con:
        con.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{sql_path(products)}')")
        # Tables facultatives : l'application fonctionne sans, avec moins d'informations
        if controversies.exists():
            con.execute(
                f"""CREATE VIEW c AS SELECT code, controversy_score, reasons,
                    risky_additive_names FROM read_parquet('{sql_path(controversies)}')"""
            )
        else:
            con.execute(
                "CREATE VIEW c AS SELECT NULL::VARCHAR AS code, NULL::INT AS controversy_score, "
                "NULL::VARCHAR AS reasons, NULL::VARCHAR[] AS risky_additive_names WHERE false"
            )
        if nova_estimated.exists():
            con.execute(
                f"""CREATE VIEW e AS SELECT code, predicted AS nova_estimated,
                    confidence AS nova_confidence FROM read_parquet('{sql_path(nova_estimated)}')"""
            )
        else:
            con.execute(
                "CREATE VIEW e AS SELECT NULL::VARCHAR AS code, NULL::INT AS nova_estimated, "
                "NULL::DOUBLE AS nova_confidence WHERE false"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"""
            COPY (
                SELECT
                    p.code, p.product_name, p.generic_name, p.brand, p.brand_tag,
                    p.food_group, p.food_subgroup, p.main_category, p.categories_tags,
                    p.countries_tags, p.is_organic,
                    p.nutriscore_grade, p.nutriscore_score, p.nova_group, p.green_score_grade,
                    e.nova_estimated::INT AS nova_estimated, e.nova_confidence,
                    p.n_additives, p.additives, p.has_palm_oil, p.ingredients_text,
                    p.energy_kcal_100g, p.fat_100g, p.saturated_fat_100g, p.carbohydrates_100g,
                    p.sugars_100g, p.fiber_100g, p.proteins_100g, p.salt_100g,
                    coalesce(c.controversy_score, 0) AS controversy_score,
                    c.reasons, coalesce(c.risky_additive_names, []) AS risky_additive_names,
                    coalesce(p.n_scans, 0) AS n_scans,
                    strip_accents(lower(
                        p.product_name || ' ' || coalesce(p.brand, '') || ' ' || p.code
                    )) AS search_text
                FROM p
                LEFT JOIN c USING (code)
                LEFT JOIN e USING (code)
                ORDER BY n_scans DESC, p.code
            ) TO '{sql_path(dest)}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 100000)
            """
        )
        n = con.execute(f"SELECT count(*) FROM read_parquet('{sql_path(dest)}')").fetchone()[0]
    logger.info("Table de l'application écrite : %s (%d produits)", dest, n)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_app_table()


if __name__ == "__main__":
    main()
