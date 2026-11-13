"""Exécution de chaque page de l'application sur un petit jeu de données temporaire."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nutrilys import config
from nutrilys.analysis import controversies
from nutrilys.app.build import build_app_table
from nutrilys.data.additives import parse_taxonomy
from nutrilys.data.clean import build_products
from nutrilys.db import products_connection
from nutrilys.models import baseline

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
WORDS = {1: "épinards hachés", 2: "huile d'olive", 3: "fromage affiné", 4: "biscuits fourrés"}


@pytest.fixture(scope="module")
def app_data(tmp_path_factory):
    """Jeu complet : produits, signaux, modèles et table de l'application."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import write_interim

    tmp = tmp_path_factory.mktemp("app")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        nova = int(rng.integers(1, 5))
        rows.append({
            "code": f"{3000000000000 + i}", "product_name": f"{WORDS[nova]} {i}",
            "brands": "Marque A" if i % 2 else "Marque B", "brands_tags": ["marque"],
            "categories_tags": [f"en:cat-{nova}"],
            "food_groups_tags": ["en:sugary-snacks", "en:biscuits-and-cakes"],
            "ingredients_text": f"{WORDS[nova]}, eau", "nova_group": nova,
            "additives_n": nova - 1, "additives_tags": ["en:e250"] if nova == 4 else [],
            "nutriscore_grade": "abcde"[nova - 1 + i % 2], "sugars_100g": 10.0 * nova,
            "salt_100g": 0.5, "saturated_fat_100g": 2.0, "unique_scans_n": i,
            "labels_tags": ["en:organic"] if i % 7 == 0 else [],
        })  # fmt: skip
    interim = tmp / "interim.parquet"
    write_interim(rows, interim)
    paths = {
        "PRODUCTS_PATH": tmp / "products.parquet",
        "ADDITIVES_PATH": tmp / "additives.parquet",
        "CONTROVERSIES_PATH": tmp / "controversies.parquet",
        "NOVA_ESTIMATED_PATH": tmp / "nova_estimated.parquet",
        "APP_PRODUCTS_PATH": tmp / "app_products.parquet",
        "MODELS_DIR": tmp / "models",
        "MODEL_REPORTS_DIR": tmp / "reports",
        "PREDICTIONS_DIR": tmp / "predictions",
        "SPLITS_PATH": tmp / "splits.parquet",
    }
    build_products(interim, paths["PRODUCTS_PATH"], report_path=None)
    additives = parse_taxonomy(
        {"en:e250": {"name": {"fr": "E250 - Nitrite de sodium"},
                     "efsa_evaluation_overexposure_risk": {"en": "en:high"}}}
    )  # fmt: skip
    additives.to_parquet(paths["ADDITIVES_PATH"])
    recalls = pd.DataFrame(columns=["gtin_key", "recall_date", "recall_reason", "recall_theme",
                                    "recall_risk", "recall_category", "recall_brand",
                                    "recall_label", "recall_link"])  # fmt: skip
    con = products_connection(paths["PRODUCTS_PATH"])
    controversies.compute_signals(con, additives, recalls).to_parquet(paths["CONTROVERSIES_PATH"])
    con.close()

    mp = pytest.MonkeyPatch()
    for name, value in paths.items():
        mp.setattr(config, name, value)
    mp.setattr(config, "TFIDF_MIN_DF", 1)
    mp.setattr(baseline, "products_connection", lambda: products_connection(paths["PRODUCTS_PATH"]))
    baseline.run(["nova_name", "nutriscore_text"], ["lr"])
    pd.DataFrame({"code": ["x"], "predicted": [4], "confidence": [0.8]}).to_parquet(
        paths["NOVA_ESTIMATED_PATH"]
    )
    build_app_table(paths["PRODUCTS_PATH"], paths["CONTROVERSIES_PATH"],
                    paths["NOVA_ESTIMATED_PATH"], paths["APP_PRODUCTS_PATH"])  # fmt: skip
    yield paths
    mp.undo()


def _run(page: str | None = None, **params) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    if page:
        at.switch_page(page)
        for key, value in params.items():
            at.query_params[key] = value
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_home(app_data):
    at = _run()
    assert at.title[0].value == "NUTRILYS"


def test_explorer_search_and_filters(app_data):
    at = _run("views/explorer.py", q="biscuits")
    table = at.dataframe[0].value
    assert len(table) > 0 and table["product_name"].str.contains("biscuits").all()


def test_product_sheet(app_data):
    at = _run("views/fiche.py", code="3000000000004")
    html = " ".join(m.value for m in at.markdown)
    assert "Valeurs nutritionnelles" in html and "Nutri-Score" in html


def test_product_sheet_unknown_code(app_data):
    at = _run("views/fiche.py", code="0000")
    assert at.warning


def test_analyses(app_data):
    at = _run("views/analyses.py")
    assert at.tabs


def test_prediction_with_explanation(app_data):
    at = _run("views/predire.py")
    at.text_input[0].input("biscuits fourrés au chocolat")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert any("pesé" in m.value for m in at.markdown)
