"""Agrégats de l'analyse de la qualité nutritionnelle (objectif 1).

Toutes les fonctions prennent une connexion qui expose la vue `products`
(voir `nutrilys.db.products_connection`) et renvoient un DataFrame.
Chaque calcul porte sur la sous-population qui dispose des informations
nécessaires : elle est précisée dans la docstring.
"""

import duckdb
import pandas as pd

# Groupes alimentaires affichés en français
FOOD_GROUP_LABELS = {
    "sugary-snacks": "Produits sucrés",
    "fish-meat-eggs": "Viandes, poissons, œufs",
    "milk-and-dairy-products": "Produits laitiers",
    "cereals-and-potatoes": "Céréales et pommes de terre",
    "beverages": "Boissons",
    "fruits-and-vegetables": "Fruits et légumes",
    "fats-and-sauces": "Matières grasses et sauces",
    "composite-foods": "Plats composés",
    "salty-snacks": "Snacks salés",
    "alcoholic-beverages": "Boissons alcoolisées",
    "baby-foods-and-milks": "Aliments pour bébés",
}


def _df(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    return con.execute(query).df()


def nutriscore_distribution(con) -> pd.DataFrame:
    """Répartition des Nutri-Score. Population : produits avec Nutri-Score valide."""
    return _df(
        con,
        """
        SELECT nutriscore_grade, count(*) AS products,
               100.0 * count(*) / sum(count(*)) OVER () AS share
        FROM products WHERE has_nutriscore
        GROUP BY 1 ORDER BY 1
        """,
    )


def nova_distribution(con) -> pd.DataFrame:
    """Répartition des groupes NOVA. Population : produits avec NOVA."""
    return _df(
        con,
        """
        SELECT nova_group, count(*) AS products,
               100.0 * count(*) / sum(count(*)) OVER () AS share
        FROM products WHERE has_nova
        GROUP BY 1 ORDER BY 1
        """,
    )


def nutriscore_by_nova(con) -> pd.DataFrame:
    """Répartition des Nutri-Score dans chaque groupe NOVA (format long).

    Population : produits avec Nutri-Score et NOVA.
    """
    return _df(
        con,
        """
        SELECT nova_group, nutriscore_grade, count(*) AS products,
               100.0 * count(*) / sum(count(*)) OVER (PARTITION BY nova_group) AS share
        FROM products WHERE has_nutriscore AND has_nova
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
    )


def contingency(long: pd.DataFrame, rows: str, columns: str, values: str) -> pd.DataFrame:
    """Tableau de contingence (effectifs) à partir d'un format long."""
    return long.pivot_table(index=rows, columns=columns, values=values, fill_value=0, aggfunc="sum")


def top_additives(con, n: int = 15) -> pd.DataFrame:
    """Additifs les plus fréquents : part des produits qui en contiennent.

    Population : produits avec liste d'ingrédients.
    """
    return _df(
        con,
        f"""
        WITH base AS (SELECT additives FROM products WHERE has_ingredients),
             total AS (SELECT count(*) AS n FROM base)
        SELECT additive, count(*) AS products, 100.0 * count(*) / any_value(total.n) AS share
        FROM (SELECT unnest(list_distinct(additives)) AS additive FROM base)
        CROSS JOIN total
        GROUP BY additive
        ORDER BY products DESC
        LIMIT {int(n)}
        """,
    )


def additives_by_nova(con) -> pd.DataFrame:
    """Nombre d'additifs par groupe NOVA. Population : produits avec NOVA et ingrédients."""
    return _df(
        con,
        """
        SELECT nova_group,
               count(*) AS products,
               avg(n_additives) AS mean_additives,
               median(n_additives) AS median_additives,
               100.0 * avg((n_additives = 0)::INT) AS share_without_additives
        FROM products WHERE has_nova AND n_additives IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """,
    )


def nova_additives_pairs(con) -> pd.DataFrame:
    """Couples (NOVA, nombre d'additifs) pour la corrélation de rang."""
    return _df(
        con,
        """
        SELECT nova_group, n_additives FROM products
        WHERE has_nova AND n_additives IS NOT NULL
        """,
    )


def nutriscore_by_group(con, min_products: int = 1000) -> pd.DataFrame:
    """Répartition des Nutri-Score par groupe alimentaire (format long).

    Population : produits avec Nutri-Score et groupe alimentaire. Les groupes de
    moins de `min_products` produits notés sont écartés. La colonne share_ab
    (part de A et B dans le groupe) sert à trier les groupes.
    """
    return _df(
        con,
        f"""
        WITH counts AS (
            SELECT food_group, nutriscore_grade, count(*) AS products
            FROM products WHERE has_nutriscore AND food_group IS NOT NULL
            GROUP BY 1, 2
        ),
        shares AS (
            SELECT *,
                   sum(products) OVER (PARTITION BY food_group) AS group_products,
                   100.0 * products / sum(products) OVER (PARTITION BY food_group) AS share
            FROM counts
        )
        SELECT *,
               sum(CASE WHEN nutriscore_grade IN ('a', 'b') THEN share ELSE 0 END)
                   OVER (PARTITION BY food_group) AS share_ab
        FROM shares
        WHERE group_products >= {int(min_products)}
        ORDER BY share_ab DESC, food_group, nutriscore_grade
        """,
    )


def nutrients_by_group(con, min_products: int = 1000) -> pd.DataFrame:
    """Médianes de sucre, sel et graisses saturées par groupe alimentaire.

    Population : produits avec groupe alimentaire ; chaque médiane est calculée
    sur les produits où le nutriment est renseigné. La médiane résiste mieux que
    la moyenne aux valeurs extrêmes.
    """
    return _df(
        con,
        f"""
        SELECT food_group,
               count(*) AS products,
               median(sugars_100g) AS sugars,
               median(salt_100g) AS salt,
               median(saturated_fat_100g) AS saturated_fat,
               count(sugars_100g) AS n_sugars,
               count(salt_100g) AS n_salt,
               count(saturated_fat_100g) AS n_saturated_fat
        FROM products WHERE food_group IS NOT NULL
        GROUP BY 1
        HAVING count(*) >= {int(min_products)}
        ORDER BY 1
        """,
    )


def label_food_groups(df: pd.DataFrame, column: str = "food_group") -> pd.DataFrame:
    """Ajoute une colonne food_group_label en français."""
    out = df.copy()
    out["food_group_label"] = out[column].map(FOOD_GROUP_LABELS).fillna(out[column])
    return out
