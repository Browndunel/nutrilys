import json

import pandas as pd
import pytest

from nutrilys.data.clean import build_products, check_columns
from nutrilys.data.schema import PRODUCTS_COLUMNS, write_data_dictionary


@pytest.fixture
def run(interim_factory, tmp_path):
    """Construit la table de référence et renvoie (table indexée par code, rapport)."""

    def _run(rows):
        source = interim_factory(rows)
        dest = tmp_path / "products.parquet"
        report = build_products(source, dest, report_path=tmp_path / "report.json")
        return pd.read_parquet(dest).set_index("code"), report

    return _run


def test_duplicates_keep_most_recent_record(run):
    df, report = run(
        [
            {"code": "123", "product_name": "Ancien", "last_modified_t": 1},
            {"code": " 123 ", "product_name": "Récent", "last_modified_t": 2},
            {"code": "456"},
        ]
    )
    assert len(df) == 2
    assert df.loc["123", "product_name"] == "Récent"
    assert [s["rows"] for s in report["steps"]] == [3, 2, 2, 2]


def test_products_without_name_or_code_are_dropped(run):
    df, _ = run(
        [
            {"code": "1", "product_name": "Biscuit"},
            {"code": "2", "product_name": "   "},
            {"code": "3", "product_name": None},
            {"code": "", "product_name": "Sans code"},
        ]
    )
    assert list(df.index) == ["1"]


def test_invalid_scores_become_null(run):
    df, report = run(
        [
            {"code": "1", "nutriscore_grade": "B", "nutriscore_score": 2, "nova_group": 4},
            {"code": "2", "nutriscore_grade": "unknown", "nutriscore_score": 5, "nova_group": 0},
            {"code": "3", "nutriscore_grade": "not-applicable"},
        ]
    )
    assert df.loc["1", "nutriscore_grade"] == "b"
    assert df.loc["1", "has_nutriscore"]
    assert pd.isna(df.loc["2", "nutriscore_grade"])
    assert pd.isna(df.loc["2", "nutriscore_score"])
    assert pd.isna(df.loc["2", "nova_group"])
    assert not df.loc["2", "has_nova"]
    assert report["invalid_scores_set_to_null"] == {"nutriscore_grade": 2, "nova_group": 1}


def test_out_of_range_nutrients_become_null(run):
    df, report = run(
        [
            {"code": "1", "sugars_100g": 150.0, "salt_100g": -1.0, "energy_kcal_100g": 950.0},
            {"code": "2", "sugars_100g": 30.0, "salt_100g": 1.2, "energy_kcal_100g": 500.0},
        ]
    )
    assert df.loc["1", ["sugars_100g", "salt_100g", "energy_kcal_100g"]].isna().all()
    assert df.loc["2", "sugars_100g"] == 30.0
    assert report["nutrient_values_set_to_null"]["sugars_100g"] == 1


def test_subtotal_above_total_becomes_null(run):
    df, _ = run(
        [
            {"code": "1", "fat_100g": 10.0, "saturated_fat_100g": 20.0},
            {"code": "2", "fat_100g": 10.0, "saturated_fat_100g": 10.5},  # tolérance d'arrondi
            {"code": "3", "carbohydrates_100g": 5.0, "sugars_100g": 40.0},
            {"code": "4", "saturated_fat_100g": 3.0},  # total inconnu : on garde
        ]
    )
    assert pd.isna(df.loc["1", "saturated_fat_100g"])
    assert df.loc["1", "fat_100g"] == 10.0
    assert df.loc["2", "saturated_fat_100g"] == 10.5
    assert pd.isna(df.loc["3", "sugars_100g"])
    assert df.loc["4", "saturated_fat_100g"] == 3.0


def test_organic_detection(run):
    df, _ = run(
        [
            {"code": "1", "labels_tags": ["en:organic", "en:eu-organic"]},
            {"code": "2", "labels_tags": ["fr:ab-agriculture-biologique"]},
            {"code": "3", "labels_tags": ["en:fair-trade"]},
            {"code": "4", "labels_tags": None},
        ]
    )
    assert df["is_organic"].to_dict() == {"1": True, "2": True, "3": False, "4": False}


def test_derived_columns(run):
    df, _ = run(
        [
            {
                "code": "1",
                "product_name": "  Pâte   à tartiner ",
                "brands": "Ferrero, Nutella",
                "brands_tags": ["ferrero", "nutella"],
                "food_groups_tags": ["en:sugary-snacks", "en:sweets"],
                "compared_to_category": "en:hazelnut-spreads",
                "ingredients_text": "sucre, huile de palme, noisettes",
                "additives_n": 1,
                "additives_tags": ["en:e322"],
                "ingredients_from_palm_oil_n": 1,
                "sugars_100g": 56.3,
                "salt_100g": 0.1,
                "saturated_fat_100g": 10.6,
            },
            {"code": "2", "additives_n": 3},
        ]
    )
    spread = df.loc["1"]
    assert spread["product_name"] == "Pâte à tartiner"
    assert spread["brand"] == "Ferrero"
    assert spread["brand_tag"] == "ferrero"
    assert spread["food_group"] == "sugary-snacks"
    assert spread["food_subgroup"] == "sweets"
    assert spread["main_category"] == "hazelnut-spreads"
    assert list(spread["additives"]) == ["E322"]
    assert spread["n_additives"] == 1
    assert spread["has_palm_oil"]
    assert spread["has_ingredients"] and spread["has_nutrients"]

    # Sans ingrédients, le nombre d'additifs est inconnu, pas nul
    assert pd.isna(df.loc["2", "n_additives"])
    assert not df.loc["2", "has_nutrients"]


def test_output_matches_documented_schema(interim_factory, tmp_path):
    dest = tmp_path / "products.parquet"
    build_products(interim_factory([{"code": "1"}]), dest, report_path=None)
    check_columns(dest)

    df = pd.read_parquet(dest)
    assert list(df.columns) == PRODUCTS_COLUMNS


def test_report_is_written(interim_factory, tmp_path):
    report_path = tmp_path / "report.json"
    build_products(interim_factory([{"code": "1"}]), tmp_path / "p.parquet", report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["steps"][-1]["rows"] == 1


def test_data_dictionary_lists_every_column(tmp_path):
    path = write_data_dictionary(tmp_path / "dict.md")
    content = path.read_text(encoding="utf-8")
    assert all(f"`{column}`" in content for column in PRODUCTS_COLUMNS)


def test_missing_source_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="nutrilys.data.extract"):
        build_products(tmp_path / "absent.parquet", tmp_path / "p.parquet", report_path=None)
