import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from nutrilys.analysis import nutrition
from nutrilys.analysis.stats import cramers_v, describe, interpret_correlation, spearman
from nutrilys.data.clean import build_products
from nutrilys.db import products_connection
from nutrilys.viz import charts
from nutrilys.viz.fmt import fmt_int, fmt_num, fmt_pct
from nutrilys.viz.theme import NOVA_COLORS, NUTRISCORE_COLORS, use_theme


@pytest.fixture
def con(interim_factory, tmp_path):
    """Petite table de référence : 6 produits aux profils connus."""
    rows = [
        {"code": "1", "nutriscore_grade": "a", "nova_group": 1, "ingredients_text": "pomme",
         "additives_n": 0, "food_groups_tags": ["en:fruits-and-vegetables"], "sugars_100g": 10.0},
        {"code": "2", "nutriscore_grade": "a", "nova_group": 1, "ingredients_text": "carotte",
         "additives_n": 0, "food_groups_tags": ["en:fruits-and-vegetables"], "sugars_100g": 5.0},
        {"code": "3", "nutriscore_grade": "e", "nova_group": 4, "ingredients_text": "sucre",
         "additives_n": 3, "additives_tags": ["en:e322", "en:e471", "en:e322"],
         "food_groups_tags": ["en:sugary-snacks"], "sugars_100g": 50.0},
        {"code": "4", "nutriscore_grade": "d", "nova_group": 4, "ingredients_text": "sucre",
         "additives_n": 2, "additives_tags": ["en:e322"],
         "food_groups_tags": ["en:sugary-snacks"], "sugars_100g": 40.0},
        {"code": "5", "nutriscore_grade": "unknown", "nova_group": 3},
        {"code": "6", "nutriscore_grade": "c"},
    ]  # fmt: skip
    dest = tmp_path / "products.parquet"
    build_products(interim_factory(rows), dest, report_path=None)
    connection = products_connection(dest)
    yield connection
    connection.close()


def test_nutriscore_distribution_only_counts_valid_grades(con):
    df = nutrition.nutriscore_distribution(con)
    assert df.set_index("nutriscore_grade")["products"].to_dict() == {
        "a": 2,
        "c": 1,
        "d": 1,
        "e": 1,
    }
    assert df["share"].sum() == pytest.approx(100)


def test_nova_distribution(con):
    df = nutrition.nova_distribution(con).set_index("nova_group")
    assert df["products"].to_dict() == {1: 2, 3: 1, 4: 2}


def test_nutriscore_by_nova_shares_sum_to_100_per_group(con):
    df = nutrition.nutriscore_by_nova(con)
    assert df.groupby("nova_group")["share"].sum().tolist() == pytest.approx([100, 100])
    table = nutrition.contingency(df, "nova_group", "nutriscore_grade", "products")
    assert table.loc[4, "e"] == 1


def test_top_additives_counts_each_product_once(con):
    df = nutrition.top_additives(con).set_index("additive")
    assert df.loc["E322", "products"] == 2  # présent deux fois dans le produit 3
    assert df.loc["E322", "share"] == pytest.approx(50)  # 2 produits sur 4 avec ingrédients


def test_additives_by_nova(con):
    df = nutrition.additives_by_nova(con).set_index("nova_group")
    assert df.loc[4, "mean_additives"] == pytest.approx(2.5)
    assert df.loc[1, "share_without_additives"] == pytest.approx(100)


def test_group_aggregates(con):
    by_group = nutrition.nutriscore_by_group(con, min_products=1)
    assert by_group.iloc[0]["food_group"] == "fruits-and-vegetables"  # 100 % de A
    nutrients = nutrition.nutrients_by_group(con, min_products=1).set_index("food_group")
    assert nutrients.loc["sugary-snacks", "sugars"] == pytest.approx(45)
    labelled = nutrition.label_food_groups(by_group)
    assert "Fruits et légumes" in set(labelled["food_group_label"])


def test_cramers_v_extremes():
    perfect = pd.DataFrame([[50, 0], [0, 50]])
    independent = pd.DataFrame([[25, 25], [25, 25]])
    assert cramers_v(perfect).effect_size == pytest.approx(1, abs=0.05)
    assert cramers_v(independent).effect_size == pytest.approx(0, abs=1e-9)
    assert cramers_v(perfect).interpretation == "forte"


def test_spearman_and_description():
    x = pd.Series(np.arange(100))
    result = spearman(x, x * 2)
    assert result.effect_size == pytest.approx(1)
    assert "p < 0,001" in describe(result)
    assert interpret_correlation(-0.35) == "moyenne, négative"
    assert interpret_correlation(0.05) == "négligeable"


def test_charts_build_figures(con):
    use_theme()
    dist = nutrition.nutriscore_distribution(con)
    fig = charts.category_bars(dist, "nutriscore_grade", "share", NUTRISCORE_COLORS, title="t")
    assert isinstance(fig, go.Figure) and len(fig.data) == 1

    cross = nutrition.nutriscore_by_nova(con)
    fig = charts.stacked_shares(
        cross, "nova_group", "nutriscore_grade", "share", NUTRISCORE_COLORS,
        segment_order=list("abcde"), row_order=[1, 4],
    )  # fmt: skip
    assert len(fig.data) == 5

    fig = charts.ranked_bars(nutrition.top_additives(con), "additive", "share")
    assert len(fig.data) == 1
    assert set(NOVA_COLORS) == {1, 2, 3, 4}


def test_french_formatting():
    assert fmt_int(1171729) == "1 171 729"
    assert fmt_pct(40.24, 1) == "40,2 %"
    assert fmt_num(2.36, 1) == "2,4"
