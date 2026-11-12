"""Produits et marques controversés (objectif 2).

Un produit est signalé quand ses indicateurs se contredisent ou appellent la
vigilance. Cinq signaux, chacun explicable en une phrase :

    good_score_ultra      Nutri-Score A ou B, mais ultra-transformé (NOVA 4)
    good_score_additives  Nutri-Score A ou B, mais au moins MANY_ADDITIVES additifs
    risky_additives       au moins un additif à risque élevé de surexposition (EFSA)
    recalled              code-barres présent dans RappelConso
    organic_ultra         label bio, mais ultra-transformé (NOVA 4)

Le score de controverse est le nombre de signaux (0 à 5). Il est volontairement
simple : chaque point se justifie par un fait vérifiable sur la fiche du produit.

Usage :
    uv run python -m nutrilys.analysis.controversies
"""

import logging
from pathlib import Path

import pandas as pd

from nutrilys import config
from nutrilys.data.additives import build_additives
from nutrilys.data.recalls import build_recalls
from nutrilys.db import products_connection

logger = logging.getLogger(__name__)

SIGNALS = {
    "good_score_ultra": "Bien noté mais ultra-transformé",
    "good_score_additives": f"Bien noté mais {config.MANY_ADDITIVES} additifs ou plus",
    "risky_additives": "Additif à risque élevé de surexposition",
    "recalled": "Rappel de produit",
    "organic_ultra": "Bio mais ultra-transformé",
}


def _register(con, additives: pd.DataFrame, recalls: pd.DataFrame) -> None:
    con.register("additives_ref", additives[["additive", "name_fr", "overexposure_risk"]])
    con.register("recalls_ref", recalls)


def compute_signals(con, additives: pd.DataFrame, recalls: pd.DataFrame) -> pd.DataFrame:
    """Signaux de tous les produits ayant au moins un signal, avec l'explication."""
    _register(con, additives, recalls)
    many = int(config.MANY_ADDITIVES)
    df = con.execute(
        f"""
        WITH risky AS (
            SELECT code, list(name_fr ORDER BY additive) AS risky_additive_names
            FROM (
                SELECT code, unnest(additives) AS additive
                FROM products WHERE has_ingredients
            )
            JOIN additives_ref USING (additive)
            WHERE overexposure_risk = 'high'
            GROUP BY code
        ),
        signals AS (
            SELECT
                p.code, p.product_name, p.brand, p.brand_tag, p.food_group, p.food_subgroup,
                p.nutriscore_grade, p.nova_group, p.n_additives, p.is_organic, p.n_scans,
                coalesce(r.risky_additive_names, []) AS risky_additive_names,
                rc.recall_date, rc.recall_reason, rc.recall_theme, rc.recall_risk, rc.recall_link,
                coalesce(p.nutriscore_grade IN ('a', 'b') AND p.nova_group = 4, false)
                    AS good_score_ultra,
                coalesce(p.nutriscore_grade IN ('a', 'b') AND p.n_additives >= {many}, false)
                    AS good_score_additives,
                r.code IS NOT NULL AS risky_additives,
                rc.gtin_key IS NOT NULL AS recalled,
                coalesce(p.is_organic AND p.nova_group = 4, false) AS organic_ultra
            FROM products p
            LEFT JOIN risky r USING (code)
            LEFT JOIN recalls_ref rc ON ltrim(p.code, '0') = rc.gtin_key
        )
        SELECT *,
               good_score_ultra::INT + good_score_additives::INT + risky_additives::INT
               + recalled::INT + organic_ultra::INT AS controversy_score
        FROM signals
        WHERE good_score_ultra OR good_score_additives OR risky_additives
              OR recalled OR organic_ultra
        """
    ).df()
    df["reasons"] = df.apply(explain, axis=1)
    return df


def explain(row: pd.Series) -> str:
    """Une phrase par signal, séparées par un point-virgule."""
    parts = []
    grade = str(row.get("nutriscore_grade") or "").upper()
    if row["good_score_ultra"]:
        parts.append(f"Nutri-Score {grade} mais ultra-transformé (NOVA 4)")
    if row["good_score_additives"]:
        parts.append(f"Nutri-Score {grade} mais {int(row['n_additives'])} additifs")
    if row["risky_additives"]:
        names = ", ".join(row["risky_additive_names"])
        parts.append(f"contient {names}, à risque élevé de surexposition selon l'EFSA")
    if row["recalled"]:
        date = row.get("recall_date")
        when = f" le {date:%d/%m/%Y}" if pd.notna(date) else ""
        # Famille du motif si elle est reconnue, sinon début du motif tel quel
        theme, reason = row.get("recall_theme"), row.get("recall_reason")
        if isinstance(theme, str) and theme != "Autre motif":
            detail = f" ({theme.lower() if theme != 'E. coli' else theme})"
        elif isinstance(reason, str) and reason.strip():
            detail = f" ({reason.strip()[:80]})"
        else:
            detail = ""
        parts.append(f"rappelé{when}{detail}")
    if row["organic_ultra"]:
        parts.append("bio mais ultra-transformé (NOVA 4)")
    text = " ; ".join(parts)
    return text[:1].upper() + text[1:] + "." if text else ""


def signal_counts(signals: pd.DataFrame) -> pd.DataFrame:
    """Nombre de produits concernés par chaque signal."""
    counts = signals[list(SIGNALS)].sum().rename("products").rename_axis("signal").reset_index()
    counts["label"] = counts["signal"].map(SIGNALS)
    return counts.sort_values("products", ascending=False).reset_index(drop=True)


def risky_additives_frequency(con, additives: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Additifs évalués à risque élevé ou modéré, par fréquence d'utilisation.

    Population : produits avec liste d'ingrédients.
    """
    con.register("additives_ref", additives)
    return con.execute(
        f"""
        WITH base AS (SELECT additives FROM products WHERE has_ingredients),
             total AS (SELECT count(*) AS n FROM base)
        SELECT a.additive, a.name_fr, a.risk_label, count(*) AS products,
               100.0 * count(*) / any_value(total.n) AS share
        FROM (SELECT unnest(list_distinct(additives)) AS additive FROM base)
        JOIN additives_ref a USING (additive)
        CROSS JOIN total
        WHERE a.overexposure_risk IN ('high', 'moderate')
        GROUP BY ALL
        ORDER BY products DESC
        LIMIT {int(n)}
        """
    ).df()


def risky_share_by_nova(con, signals: pd.DataFrame) -> pd.DataFrame:
    """Part des produits contenant un additif à risque élevé, par groupe NOVA."""
    con.register("risky_codes", signals.loc[signals["risky_additives"], ["code"]])
    return con.execute(
        """
        SELECT p.nova_group, count(*) AS products,
               100.0 * count(r.code) / count(*) AS share
        FROM products p LEFT JOIN risky_codes r USING (code)
        WHERE p.has_nova AND p.has_ingredients
        GROUP BY 1 ORDER BY 1
        """
    ).df()


def good_score_ultra_by_subgroup(con, min_products: int = 200) -> pd.DataFrame:
    """Part d'ultra-transformés parmi les produits notés A ou B, par sous-groupe."""
    return con.execute(
        f"""
        SELECT food_subgroup, count(*) AS products,
               sum((nova_group = 4)::INT) AS ultra,
               100.0 * avg((nova_group = 4)::INT) AS share
        FROM products
        WHERE nutriscore_grade IN ('a', 'b') AND has_nova AND food_subgroup IS NOT NULL
        GROUP BY 1
        HAVING count(*) >= {int(min_products)}
        ORDER BY share DESC
        """
    ).df()


def brand_profiles(con, min_products: int = config.MIN_BRAND_PRODUCTS) -> pd.DataFrame:
    """Profil des marques ayant au moins `min_products` produits notés (Nutri-Score et NOVA).

    halo : part d'ultra-transformés parmi les produits de la marque notés A ou B.
    Calculé seulement si la marque a au moins 10 produits notés A ou B.
    """
    return con.execute(
        f"""
        SELECT brand_tag,
               mode(brand) AS brand,
               count(*) AS products,
               100.0 * avg((nutriscore_grade IN ('a', 'b'))::INT) AS share_ab,
               100.0 * avg((nova_group = 4)::INT) AS share_nova4,
               100.0 * avg(is_organic::INT) AS share_organic,
               sum((nutriscore_grade IN ('a', 'b'))::INT) AS n_ab,
               CASE WHEN sum((nutriscore_grade IN ('a', 'b'))::INT) >= 10
                    THEN 100.0 * sum((nutriscore_grade IN ('a', 'b') AND nova_group = 4)::INT)
                         / sum((nutriscore_grade IN ('a', 'b'))::INT)
               END AS halo
        FROM products
        WHERE has_nutriscore AND has_nova AND brand_tag IS NOT NULL
        GROUP BY brand_tag
        HAVING count(*) >= {int(min_products)}
        """
    ).df()


def build_controversies(
    products_path: Path = config.PRODUCTS_PATH, dest: Path = config.CONTROVERSIES_PATH
) -> pd.DataFrame:
    """Construit les tables additifs et rappels, puis la table des produits signalés."""
    additives = build_additives()
    recalls = build_recalls()
    con = products_connection(products_path)
    signals = compute_signals(con, additives, recalls)
    con.close()
    signals.to_parquet(dest, index=False)
    logger.info("Produits signalés écrits : %s (%d produits)", dest, len(signals))
    return signals


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signals = build_controversies()
    print("\nProduits concernés par signal :")
    for _, row in signal_counts(signals).iterrows():
        print(f"  {row['label']:<45} {row['products']:>9,}".replace(",", " "))
    print("\nRépartition du score de controverse :")
    print(signals["controversy_score"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
