from itertools import cycle

import numpy as np
import pandas as pd
import pytest

from nutrilys.analysis import bio
from nutrilys.data.clean import build_products
from nutrilys.db import products_connection


def _frame(spec: list[tuple[str, bool, int, dict]]) -> pd.DataFrame:
    """Construit un jeu de produits : (sous-groupe, bio, nombre, valeurs)."""
    rows, k = [], 0
    for subgroup, organic, n, values in spec:
        for _ in range(n):
            k += 1
            rows.append(
                {"code": str(k), "food_group": "g", "food_subgroup": subgroup,
                 "is_organic": organic, **{c: v() if callable(v) else v for c, v in values.items()}}
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_continuous_comparison_detects_a_real_gap_only(rng):
    # Même distribution exacte des deux côtés pour les fromages : aucun écart attendu
    cheese_bio, cheese_conv = cycle([0, 1, 2, 3, 4]), cycle([0, 1, 2, 3, 4])
    df = _frame(
        [
            ("biscuits-and-cakes", True, 150, {"n_additives": lambda: rng.poisson(1)}),
            ("biscuits-and-cakes", False, 300, {"n_additives": lambda: rng.poisson(4)}),
            ("cheese", True, 150, {"n_additives": lambda: next(cheese_bio)}),
            ("cheese", False, 300, {"n_additives": lambda: next(cheese_conv)}),
            ("fruits", True, 20, {"n_additives": 0}),  # trop peu de bio : écarté
            ("fruits", False, 300, {"n_additives": 0}),
        ]
    )
    res = bio.compare_continuous(df, ["n_additives"]).set_index("food_subgroup")
    assert set(res.index) == {"biscuits-and-cakes", "cheese"}
    assert res.loc["biscuits-and-cakes", "verdict"] == "Plus bas en bio"
    assert res.loc["biscuits-and-cakes", "effect"] < -0.5
    assert res.loc["cheese", "verdict"] == "Pas d'écart notable"
    assert res.loc["biscuits-and-cakes", "label"] == "Biscuits et gâteaux"
    assert (res["q_value"] >= res["p_value"]).all()


def test_grade_comparison(rng):
    grades = list("abcde")
    df = _frame(
        [
            ("cereals", True, 200, {"nutriscore_grade": lambda: rng.choice(grades[:2])}),
            ("cereals", False, 200, {"nutriscore_grade": lambda: rng.choice(grades[2:])}),
        ]
    )
    res = bio.compare_grades(df, "nutriscore_grade")
    assert res.loc[0, "verdict"] == "Répartition différente"
    assert res.loc[0, "effect"] > 0.9


def test_standardization_removes_composition_bias():
    # Même taux de A/B pour bio et conventionnel dans chaque rayon,
    # mais le bio est concentré dans le rayon le mieux noté.
    df = _frame(
        [
            ("vegetables", True, 400, {"is_ab": 1.0}),
            ("vegetables", False, 100, {"is_ab": 1.0}),
            ("sweets", True, 100, {"is_ab": 0.0}),
            ("sweets", False, 400, {"is_ab": 0.0}),
        ]
    )
    rates = bio.standardized_rates(df, "is_ab")
    assert rates["crude_bio"] == pytest.approx(80)
    assert rates["crude_conv"] == pytest.approx(20)
    assert rates["std_bio"] == pytest.approx(rates["std_conv"])
    assert rates["strata"] == 2


def test_verdict_counts(rng):
    df = _frame(
        [
            ("biscuits-and-cakes", True, 150, {"sugars_100g": lambda: rng.normal(20, 3)}),
            ("biscuits-and-cakes", False, 150, {"sugars_100g": lambda: rng.normal(30, 3)}),
        ]
    )
    counts = bio.verdict_counts(bio.compare_continuous(df, ["sugars_100g"]))
    assert counts.loc["sugars_100g", "Plus bas en bio"] == 1


def test_load_frame_builds_indicators(interim_factory, tmp_path):
    rows = [
        {"code": "1", "food_groups_tags": ["en:sugary-snacks", "en:sweets"],
         "labels_tags": ["en:organic"], "nutriscore_grade": "a", "nova_group": 4,
         "ingredients_text": "sucre", "additives_n": 0},
        {"code": "2", "food_groups_tags": ["en:sugary-snacks", "en:sweets"],
         "nutriscore_grade": "unknown"},
        {"code": "3"},  # sans sous-groupe : exclu
    ]  # fmt: skip
    dest = tmp_path / "products.parquet"
    build_products(interim_factory(rows), dest, report_path=None)
    con = products_connection(dest)
    df = bio.load_frame(con).set_index("code")
    con.close()

    assert list(df.index) == ["1", "2"] or set(df.index) == {"1", "2"}
    assert df.loc["1", "is_ab"] == True  # noqa: E712
    assert df.loc["1", "is_nova4"] == True  # noqa: E712
    assert df.loc["1", "has_additive"] == False  # noqa: E712
    assert pd.isna(df.loc["2", "is_ab"])
    comp = bio.composition(df.reset_index())
    assert comp.loc[0, "share_organic"] == pytest.approx(50)
