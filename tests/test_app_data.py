import pandas as pd
import pytest

from nutrilys.app import queries
from nutrilys.app.build import build_app_table
from nutrilys.app.queries import Filters
from nutrilys.data.clean import build_products

ROWS = [
    {"code": "3017620422003", "product_name": "Pâte à tartiner aux noisettes", "brands": "Nutella",
     "brands_tags": ["nutella"], "nutriscore_grade": "e", "nova_group": 4,
     "food_groups_tags": ["en:sugary-snacks", "en:sweets"], "ingredients_text": "sucre",
     "unique_scans_n": 900, "sugars_100g": 56.3},
    {"code": "0000000000001", "product_name": "Pâte à tartiner bio", "brands": "Bio Village",
     "brands_tags": ["bio-village"], "nutriscore_grade": "d", "nova_group": 3,
     "labels_tags": ["en:organic"], "food_groups_tags": ["en:sugary-snacks", "en:sweets"],
     "ingredients_text": "noisettes", "unique_scans_n": 50},
    {"code": "0000000000002", "product_name": "Jambon blanc", "brands": "Fleury",
     "brands_tags": ["fleury"], "nutriscore_grade": "b",
     "food_groups_tags": ["en:fish-meat-eggs", "en:processed-meat"], "unique_scans_n": 300},
    {"code": "0000000000003", "product_name": "Épinards hachés", "brands": "Bonduelle",
     "nutriscore_grade": "a", "nova_group": 1, "ingredients_text": "épinards",
     "food_groups_tags": ["en:fruits-and-vegetables", "en:vegetables"], "unique_scans_n": 10},
]  # fmt: skip


@pytest.fixture
def app_path(interim_factory, tmp_path):
    products = tmp_path / "products.parquet"
    build_products(interim_factory(ROWS), products, report_path=None)
    controversies = tmp_path / "controversies.parquet"
    pd.DataFrame(
        {
            "code": ["3017620422003"],
            "controversy_score": [1],
            "reasons": ["Contient E322."],
            "risky_additive_names": [["E322 - Lécithines"]],
        }
    ).to_parquet(controversies)
    estimated = tmp_path / "nova_estimated.parquet"
    pd.DataFrame({"code": ["0000000000002"], "predicted": [4], "confidence": [0.9]}).to_parquet(
        estimated
    )
    dest = tmp_path / "app_products.parquet"
    assert build_app_table(products, controversies, estimated, dest) == 4
    return dest


@pytest.fixture
def con(app_path):
    connection = queries.open_app(app_path)
    yield connection
    connection.close()


def test_table_is_sorted_by_popularity_and_joined(app_path):
    df = pd.read_parquet(app_path)
    assert df["code"].tolist()[0] == "3017620422003"  # le plus scanné en premier
    row = df.set_index("code").loc["0000000000002"]
    assert row["nova_estimated"] == 4 and pd.isna(row["nova_group"])
    assert df.set_index("code").loc["3017620422003", "controversy_score"] == 1


def test_search_ignores_case_and_accents(con):
    rows, total = queries.search(con, Filters(text="PATE a tartiner"))
    assert total == 2
    assert rows["code"].tolist() == ["3017620422003", "0000000000001"]
    _, total = queries.search(con, Filters(text="epinards"))
    assert total == 1
    _, total = queries.search(con, Filters(text="3017620422003"))  # code-barres
    assert total == 1


def test_search_filters(con):
    assert queries.search(con, Filters(organic_only=True))[1] == 1
    assert queries.search(con, Filters(nutriscore=["a", "b"]))[1] == 2
    assert queries.search(con, Filters(nova=[4]))[1] == 1
    assert queries.search(con, Filters(nova=[4], include_estimated=True))[1] == 2
    assert queries.search(con, Filters(flagged_only=True))[1] == 1
    assert queries.search(con, Filters(food_group="sugary-snacks"))[1] == 2
    assert queries.search(con, Filters(brand="Fleury"))[1] == 1


def test_search_is_safe_with_special_characters(con):
    rows, total = queries.search(con, Filters(text="'; DROP TABLE app; --"))
    assert total == 0 and rows.empty


def test_pagination(con):
    first, total = queries.search(con, Filters(), limit=2, offset=0)
    second, _ = queries.search(con, Filters(), limit=2, offset=2)
    assert total == 4 and not set(first["code"]) & set(second["code"])


def test_product_and_lists(con):
    p = queries.product(con, "3017620422003")
    assert p["brand"] == "Nutella" and p["nutriscore_grade"] == "e"
    assert queries.product(con, "inconnu") is None
    assert "sugary-snacks" in queries.food_groups(con)
    assert queries.top_brands(con)[0] in {"Nutella", "Bio Village", "Fleury", "Bonduelle"}
    figures = queries.key_figures(con)
    assert figures["products"] == 4 and figures["flagged"] == 1
    assert figures["share_nova4"] == pytest.approx(100 / 3)
