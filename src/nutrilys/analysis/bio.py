"""Comparaison bio / conventionnel à catégorie égale (objectif 5).

Principe : on ne compare jamais le bio et le conventionnel tous rayons confondus,
car le bio est sur-représenté dans certains rayons (céréales, fruits et légumes)
qui sont naturellement mieux notés. On compare donc à l'intérieur de chaque
sous-groupe alimentaire (niveau 2 : biscuits, fromages, jus de fruits...).

Deux niveaux de résultat :
- par sous-groupe : test, taille d'effet, correction pour tests multiples ;
- global : taux standardisés, c'est-à-dire recalculés comme si bio et
  conventionnel avaient la même répartition entre sous-groupes.

Limite structurante : un produit sans label bio déclaré est compté comme
conventionnel. Les étiquettes manquantes diluent donc les écarts vers zéro.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

STRATUM = "food_subgroup"
MIN_PER_ARM = 100
MIN_EFFECT = 0.1  # |r| en dessous : écart jugé négligeable même si significatif
ALPHA = 0.05

# Indicateurs continus : (colonne, libellé, unité). Plus bas = meilleur pour tous.
CONTINUOUS_OUTCOMES = {
    "nutriscore_score": ("Score Nutri-Score", "points"),
    "n_additives": ("Nombre d'additifs", "additifs"),
    "sugars_100g": ("Sucres", "g/100 g"),
    "salt_100g": ("Sel", "g/100 g"),
    "saturated_fat_100g": ("Graisses saturées", "g/100 g"),
}

# Indicateurs binaires pour la standardisation : (libellé, plus haut = meilleur ?)
BINARY_OUTCOMES = {
    "is_ab": ("Nutri-Score A ou B", True),
    "is_de": ("Nutri-Score D ou E", False),
    "is_nova4": ("Ultra-transformé (NOVA 4)", False),
    "has_additive": ("Au moins un additif", False),
}

# Sous-groupes alimentaires (niveau 2) en français ; à défaut, le code est mis en forme
SUBGROUP_LABELS = {
    "appetizers": "Apéritifs",
    "artificially-sweetened-beverages": "Boissons édulcorées",
    "biscuits-and-cakes": "Biscuits et gâteaux",
    "bread": "Pain",
    "breakfast-cereals": "Céréales du petit-déjeuner",
    "cereals": "Céréales et pâtes",
    "cheese": "Fromages",
    "chocolate-products": "Chocolats",
    "dairy-desserts": "Desserts lactés",
    "dressings-and-sauces": "Sauces et assaisonnements",
    "dried-fruits": "Fruits secs",
    "eggs": "Œufs",
    "fats": "Matières grasses",
    "fish-and-seafood": "Poissons et fruits de mer",
    "fruit-juices": "Jus de fruits",
    "fruit-nectars": "Nectars",
    "fruits": "Fruits",
    "ice-cream": "Glaces",
    "legumes": "Légumineuses",
    "meat": "Viandes",
    "milk-and-yogurt": "Lait et yaourts",
    "nuts": "Fruits à coque",
    "offals": "Abats",
    "one-dish-meals": "Plats préparés",
    "pastries": "Viennoiseries",
    "pizza-pies-and-quiches": "Pizzas, tartes et quiches",
    "plant-based-milk-substitutes": "Boissons végétales",
    "potatoes": "Pommes de terre",
    "processed-meat": "Charcuteries",
    "salty-and-fatty-products": "Produits salés et gras",
    "sandwiches": "Sandwichs",
    "soups": "Soupes",
    "sweetened-beverages": "Boissons sucrées",
    "sweets": "Confiseries et sucreries",
    "teas-and-herbal-teas-and-coffees": "Thés, tisanes et cafés",
    "unsweetened-beverages": "Boissons non sucrées",
    "vegetables": "Légumes",
    "waters-and-flavored-waters": "Eaux",
}


def subgroup_label(code: str) -> str:
    return SUBGROUP_LABELS.get(code, str(code).replace("-", " ").capitalize())


def load_frame(con) -> pd.DataFrame:
    """Produits rattachés à un sous-groupe, avec les indicateurs nécessaires."""
    df = con.execute(
        """
        SELECT code, food_group, food_subgroup, is_organic,
               nutriscore_grade, nutriscore_score, nova_group, n_additives,
               sugars_100g, salt_100g, saturated_fat_100g
        FROM products
        WHERE food_subgroup IS NOT NULL
        """
    ).df()
    # Indicateurs binaires : vides quand l'information de base est absente
    grade = df["nutriscore_grade"]
    df["is_ab"] = grade.isin(["a", "b"]).where(grade.notna())
    df["is_de"] = grade.isin(["d", "e"]).where(grade.notna())
    df["is_nova4"] = (df["nova_group"] == 4).where(df["nova_group"].notna())
    df["has_additive"] = (df["n_additives"] > 0).where(df["n_additives"].notna())
    return df


def composition(df: pd.DataFrame) -> pd.DataFrame:
    """Part de bio par groupe alimentaire de niveau 1 : montre le biais de composition."""
    out = (
        df.groupby("food_group")
        .agg(products=("code", "size"), organic=("is_organic", "sum"))
        .assign(share_organic=lambda d: 100 * d["organic"] / d["products"])
        .sort_values("share_organic", ascending=False)
        .reset_index()
    )
    return out


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    """Taille d'effet du Mann-Whitney : > 0 si le premier échantillon (bio) est plus élevé."""
    return 2 * u / (n1 * n2) - 1


def compare_continuous(
    df: pd.DataFrame, outcomes: list[str] | None = None, min_per_arm: int = MIN_PER_ARM
) -> pd.DataFrame:
    """Mann-Whitney bio vs conventionnel dans chaque sous-groupe, pour chaque indicateur."""
    outcomes = outcomes or list(CONTINUOUS_OUTCOMES)
    rows = []
    for outcome in outcomes:
        data = df[df[outcome].notna()]
        for stratum, sub in data.groupby(STRATUM):
            bio = sub.loc[sub["is_organic"], outcome].to_numpy()
            conv = sub.loc[~sub["is_organic"], outcome].to_numpy()
            if len(bio) < min_per_arm or len(conv) < min_per_arm:
                continue
            u, p = stats.mannwhitneyu(bio, conv, alternative="two-sided")
            rows.append(
                {
                    "outcome": outcome,
                    STRATUM: stratum,
                    "food_group": sub["food_group"].iloc[0],
                    "n_bio": len(bio),
                    "n_conv": len(conv),
                    "median_bio": float(np.median(bio)),
                    "median_conv": float(np.median(conv)),
                    "mean_bio": float(bio.mean()),
                    "mean_conv": float(conv.mean()),
                    "effect": _rank_biserial(u, len(bio), len(conv)),
                    "p_value": float(p),
                }
            )
    return _finalize(pd.DataFrame(rows))


def compare_grades(
    df: pd.DataFrame,
    outcome: str = "nutriscore_grade",
    min_per_arm: int = MIN_PER_ARM,
) -> pd.DataFrame:
    """Chi² bio vs conventionnel sur une variable en classes (Nutri-Score, NOVA)."""
    rows = []
    data = df[df[outcome].notna()]
    for stratum, sub in data.groupby(STRATUM):
        n_bio = int(sub["is_organic"].sum())
        n_conv = len(sub) - n_bio
        if n_bio < min_per_arm or n_conv < min_per_arm:
            continue
        table = pd.crosstab(sub["is_organic"], sub[outcome])
        table = table.loc[:, table.sum() > 0]
        if table.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(table)
        v = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))
        rows.append(
            {
                "outcome": outcome,
                STRATUM: stratum,
                "food_group": sub["food_group"].iloc[0],
                "n_bio": n_bio,
                "n_conv": n_conv,
                "effect": v,
                "p_value": float(p),
            }
        )
    return _finalize(pd.DataFrame(rows))


def _finalize(res: pd.DataFrame) -> pd.DataFrame:
    """Correction de Benjamini-Hochberg et verdict lisible."""
    if res.empty:
        return res
    res["q_value"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["significant"] = (res["q_value"] < ALPHA) & (res["effect"].abs() >= MIN_EFFECT)
    res["label"] = res[STRATUM].map(subgroup_label)
    if "median_bio" in res:
        res["verdict"] = np.select(
            [~res["significant"], res["effect"] < 0],
            ["Pas d'écart notable", "Plus bas en bio"],
            default="Plus haut en bio",
        )
    else:
        res["verdict"] = np.where(
            res["significant"], "Répartition différente", "Pas d'écart notable"
        )
    return res.reset_index(drop=True)


def verdict_counts(res: pd.DataFrame) -> pd.DataFrame:
    """Nombre de sous-groupes par verdict, pour chaque indicateur."""
    return (
        res.pivot_table(index="outcome", columns="verdict", values=STRATUM, aggfunc="count")
        .fillna(0)
        .astype(int)
    )


def standardized_rates(df: pd.DataFrame, indicator: str, min_per_arm: int = MIN_PER_ARM) -> dict:
    """Taux bruts et standardisés (standardisation directe par sous-groupe).

    Taux standardisé : moyenne des taux de chaque sous-groupe, pondérée par la
    taille totale du sous-groupe (bio + conventionnel). Bio et conventionnel sont
    ainsi ramenés à la même composition. Seuls les sous-groupes où les deux
    populations atteignent `min_per_arm` sont retenus, pour les deux calculs.
    """
    data = df[df[indicator].notna()].copy()
    data[indicator] = data[indicator].astype(float)
    by = data.groupby([STRATUM, "is_organic"])[indicator].agg(["mean", "size"]).unstack()
    keep = (by["size"][True] >= min_per_arm) & (by["size"][False] >= min_per_arm)
    by = by[keep]
    weights = by["size"][True] + by["size"][False]
    kept = data[data[STRATUM].isin(by.index)]

    crude = kept.groupby("is_organic")[indicator].mean()
    std_bio = float(np.average(by["mean"][True], weights=weights))
    std_conv = float(np.average(by["mean"][False], weights=weights))
    return {
        "indicator": indicator,
        "strata": int(keep.sum()),
        "n_bio": int(by["size"][True].sum()),
        "n_conv": int(by["size"][False].sum()),
        "crude_bio": 100 * float(crude[True]),
        "crude_conv": 100 * float(crude[False]),
        "std_bio": 100 * std_bio,
        "std_conv": 100 * std_conv,
    }


def standardized_table(df: pd.DataFrame, min_per_arm: int = MIN_PER_ARM) -> pd.DataFrame:
    """Taux bruts et standardisés pour tous les indicateurs binaires."""
    rows = [standardized_rates(df, ind, min_per_arm) for ind in BINARY_OUTCOMES]
    out = pd.DataFrame(rows)
    out["label"] = out["indicator"].map(lambda i: BINARY_OUTCOMES[i][0])
    out["crude_gap"] = out["crude_bio"] - out["crude_conv"]
    out["std_gap"] = out["std_bio"] - out["std_conv"]
    return out
