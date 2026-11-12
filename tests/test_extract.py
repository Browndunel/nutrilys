"""Test de l'extraction sur un mini-fichier qui reproduit la structure d'Open Food Facts."""

import duckdb
import pandas as pd
import pytest

from nutrilys.data.extract import extract_off, summarize

FAKE_OFF_SQL = """
SELECT * FROM (VALUES
    ('3017620422003',
     [{'lang': 'main', 'text': 'Pâte à tartiner'}, {'lang': 'fr', 'text': 'Pâte à tartiner'}],
     [],
     [{'lang': 'main', 'text': 'sucre, huile de palme, noisettes'}],
     ['en:france', 'en:belgium'], false,
     [{'name': 'sugars', '100g': 56.3}, {'name': 'salt', '100g': 0.107}],
     'e', 4),
    ('0000000000001',
     [{'lang': 'fr', 'text': 'Yaourt nature'}],
     [],
     [],
     ['en:france'], NULL,
     [{'name': 'sugars', '100g': 4.0}],
     'a', 1),
    ('0000000000002',
     [{'lang': 'main', 'text': 'Produit retiré'}],
     [], [], ['en:france'], true, [], 'c', 3),
    ('0000000000003',
     [{'lang': 'main', 'text': 'Tomaten'}],
     [], [], ['en:germany'], false, [], 'b', 1)
) AS t(code, product_name, generic_name, ingredients_text, countries_tags, obsolete,
       nutriments, nutriscore_grade, nova_group)
"""

OTHER_COLUMNS = {
    "brands": "NULL::VARCHAR",
    "brands_tags": "[]::VARCHAR[]",
    "categories_tags": "[]::VARCHAR[]",
    "food_groups_tags": "[]::VARCHAR[]",
    "compared_to_category": "NULL::VARCHAR",
    "labels_tags": "[]::VARCHAR[]",
    "additives_n": "0",
    "additives_tags": "[]::VARCHAR[]",
    "ingredients_n": "0",
    "ingredients_analysis_tags": "[]::VARCHAR[]",
    "ingredients_from_palm_oil_n": "0",
    "nutriscore_score": "NULL::INTEGER",
    "environmental_score_grade": "NULL::VARCHAR",
    "no_nutrition_data": "false",
    "completeness": "0.5",
    "unique_scans_n": "0",
    "last_modified_t": "0",
}


@pytest.fixture
def fake_off(tmp_path):
    path = tmp_path / "food.parquet"
    extra = ", ".join(f"{expr} AS {name}" for name, expr in OTHER_COLUMNS.items())
    duckdb.execute(f"COPY (SELECT *, {extra} FROM ({FAKE_OFF_SQL})) TO '{path.as_posix()}'")
    return path


def test_extract_keeps_only_selected_countries_and_active_products(fake_off, tmp_path):
    dest = tmp_path / "out.parquet"
    n_rows = extract_off(source=fake_off, dest=dest, countries=["en:france"])

    df = pd.read_parquet(dest).set_index("code")
    assert n_rows == 2
    assert set(df.index) == {"3017620422003", "0000000000001"}


def test_extract_flattens_texts_and_nutrients(fake_off, tmp_path):
    dest = tmp_path / "out.parquet"
    extract_off(source=fake_off, dest=dest, countries=["en:france"])
    df = pd.read_parquet(dest).set_index("code")

    spread = df.loc["3017620422003"]
    assert spread["product_name"] == "Pâte à tartiner"
    assert spread["ingredients_text"] == "sucre, huile de palme, noisettes"
    assert spread["sugars_100g"] == pytest.approx(56.3)
    assert pd.isna(spread["fiber_100g"])

    # Sans entrée "main", on se rabat sur le français
    assert df.loc["0000000000001", "product_name"] == "Yaourt nature"


def test_extract_handles_several_countries(fake_off, tmp_path):
    dest = tmp_path / "out.parquet"
    n_rows = extract_off(source=fake_off, dest=dest, countries=["en:france", "en:germany"])
    assert n_rows == 3


def test_summarize_reports_fill_rates(fake_off, tmp_path):
    dest = tmp_path / "out.parquet"
    extract_off(source=fake_off, dest=dest, countries=["en:france"])
    summary = summarize(dest)
    assert summary.loc["product_name", "rempli_%"] == 100.0
    assert summary.loc["ingredients_text", "rempli_%"] == 50.0


def test_extract_fails_clearly_without_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="nutrilys.data.download"):
        extract_off(source=tmp_path / "absent.parquet", dest=tmp_path / "out.parquet")
