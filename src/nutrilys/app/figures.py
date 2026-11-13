"""Figures de la page Analyses, calculées une fois et mises en cache.

Elles reprennent les calculs des notebooks 02 à 06, avec les mêmes titres
construits à partir des données.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nutrilys import config
from nutrilys.analysis import bio, controversies, nutrition
from nutrilys.analysis.stats import cramers_v, describe
from nutrilys.db import products_connection
from nutrilys.models.dataset import TASKS
from nutrilys.viz import charts
from nutrilys.viz.fmt import fmt_int, fmt_num, fmt_pct
from nutrilys.viz.theme import NOVA_COLORS, NOVA_LABELS, NUTRISCORE_COLORS, SERIES, use_theme


def _con(path: str):
    return products_connection(Path(path))


@st.cache_data(show_spinner="Calcul des répartitions...")
def quality_figures(path: str) -> list[go.Figure]:
    use_theme()
    con = _con(path)
    ns = nutrition.nutriscore_distribution(con)
    share = ns.set_index("nutriscore_grade")["share"]
    figs = [
        charts.category_bars(
            ns,
            "nutriscore_grade",
            "share",
            NUTRISCORE_COLORS,
            counts="products",
            title=f"{fmt_pct(share[['d', 'e']].sum())} des produits notés sont classés D ou E",
            subtitle=f"Répartition du Nutri-Score, {fmt_int(ns['products'].sum())} produits",
        )
    ]
    nova = nutrition.nova_distribution(con)
    s4 = nova.set_index("nova_group")["share"].get(4, 0)
    figs.append(
        charts.category_bars(
            nova,
            "nova_group",
            "share",
            NOVA_COLORS,
            labels=NOVA_LABELS,
            counts="products",
            title=f"{fmt_pct(s4)} des produits renseignés sont ultra-transformés",
            subtitle=f"Répartition des groupes NOVA, {fmt_int(nova['products'].sum())} produits",
        )
    )
    cross = nutrition.nutriscore_by_nova(con)
    assoc = cramers_v(nutrition.contingency(cross, "nova_group", "nutriscore_grade", "products"))
    ab4 = cross.query("nova_group == 4 and nutriscore_grade in ['a', 'b']")["share"].sum()
    figs.append(
        charts.stacked_shares(
            cross,
            "nova_group",
            "nutriscore_grade",
            "share",
            NUTRISCORE_COLORS,
            segment_order=list("abcde"),
            row_order=[1, 2, 3, 4],
            row_labels=NOVA_LABELS,
            title=f"{fmt_pct(ab4)} des produits ultra-transformés sont pourtant classés A ou B",
            subtitle=describe(assoc),
        )
    )
    groups = nutrition.label_food_groups(nutrition.nutriscore_by_group(con))
    order = groups.drop_duplicates("food_group")["food_group_label"].tolist()
    figs.append(
        charts.stacked_shares(
            groups,
            "food_group_label",
            "nutriscore_grade",
            "share",
            NUTRISCORE_COLORS,
            segment_order=list("abcde"),
            row_order=order,
            title="Le Nutri-Score varie d'abord selon le rayon",
            subtitle="Répartition du Nutri-Score par groupe alimentaire",
        )
    )
    con.close()
    return figs  # fmt: skip


@st.cache_data(show_spinner="Comparaison bio et conventionnel...")
def organic_figures(path: str) -> list[go.Figure]:
    use_theme()
    con = _con(path)
    df = bio.load_frame(con)
    con.close()
    try:
        std = bio.standardized_table(df)
    except (ValueError, KeyError):  # trop peu de produits bio pour comparer
        return []
    ab = std.set_index("indicator").loc["is_ab"]
    return [
        charts.dumbbell_panels(
            std, "label",
            panels=[("Comparaison brute", "crude_conv", "crude_bio"),
                    ("À catégorie égale", "std_conv", "std_bio")],
            names=("Conventionnel", "Bio"), colors=(SERIES[0], SERIES[2]),
            title=f"Nutri-Score A ou B : écart bio de {fmt_num(ab['crude_gap'], 0)} points en "
                  f"brut, {fmt_num(ab['std_gap'], 0)} à catégorie égale",
            subtitle=f"Part des produits (%), {int(ab['strata'])} sous-groupes comparables",
        )
    ]  # fmt: skip


@st.cache_data(show_spinner="Chargement des controverses...")
def controversy_figures(products_path: str, controversies_path: str) -> list[go.Figure]:
    use_theme()
    if not Path(controversies_path).exists():
        return []
    signals = pd.read_parquet(controversies_path)
    counts = controversies.signal_counts(signals)
    counts["text"] = counts["products"].map(fmt_int)
    fig = charts.ranked_bars(
        counts, "label", "products", text="text",
        title=f"{fmt_int(len(signals))} produits présentent au moins un signal de controverse",
        subtitle="Nombre de produits concernés par chaque signal", value_label="Produits",
    )  # fmt: skip
    fig.update_xaxes(tickformat=",d")
    con = _con(products_path)
    by_sub = controversies.good_score_ultra_by_subgroup(con).head(12)
    con.close()
    by_sub["label"] = by_sub["food_subgroup"].map(bio.subgroup_label)
    fig2 = charts.ranked_bars(
        by_sub, "label", "share", suffix=" %", value_format=".0f",
        title="Bien notés mais ultra-transformés : les rayons les plus concernés",
        subtitle="Part de NOVA 4 parmi les produits notés A ou B",
        value_label="Part de NOVA 4 (%)",
    )  # fmt: skip
    return [fig, fig2]


@st.cache_data(show_spinner=False)
def model_figure(reports_dir: str) -> go.Figure | None:
    use_theme()
    path = Path(reports_dir) / "baseline_metrics.csv"
    if not path.exists():
        return None
    from nutrilys.models import compare

    metrics = compare.all_metrics()
    metrics["task_label"] = metrics["task"].map({k: t.description for k, t in TASKS.items()})
    colors = {"lr": SERIES[0], "xgb": SERIES[1], "camembert": SERIES[2]}
    colors = {k: v for k, v in colors.items() if k in set(metrics["model"])}
    return charts.grouped_bars(
        metrics, "task_label", "model", "f1_macro", colors, series_labels=compare.MODEL_LABELS,
        title="Prédire NOVA et le Nutri-Score à partir du texte",
        subtitle="F1 macro sur le jeu de test, par tâche et par modèle",
        value_label="F1 macro", value_range=(0, 1.08), value_format=".2f",
    )  # fmt: skip


def all_paths() -> dict:
    return {
        "products": str(config.PRODUCTS_PATH),
        "controversies": str(config.CONTROVERSIES_PATH),
        "reports": str(config.MODEL_REPORTS_DIR),
    }
