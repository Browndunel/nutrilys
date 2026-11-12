import pandas as pd
import pytest

from nutrilys.analysis import controversies as ctv
from nutrilys.data.additives import code_from_key, parse_taxonomy
from nutrilys.data.clean import build_products
from nutrilys.data.recalls import build_recalls, detect_columns, gtin_key
from nutrilys.db import products_connection

TAXONOMY = {
    "en:e250": {
        "name": {"en": "E250 - Sodium nitrite", "fr": "E250 - Nitrite de sodium"},
        "additives_classes": {"en": "en:preservative"},
        "efsa_evaluation_overexposure_risk": {"en": "en:high"},
        "anses_additives_of_interest": {"en": "yes"},
    },
    "en:e322i": {
        "name": {"en": "E322i - Lecithin"},
        "additives_classes": {"en": "en:emulsifier, en:antioxidant"},
    },
    "en:e330": {
        "name": {"en": "E330 - Citric acid", "fr": "E330 - Acide citrique"},
        "efsa_evaluation_overexposure_risk": {"en": "en:no"},
    },
    "en:not-an-additive": {"name": {"en": "x"}},
}


def test_parse_taxonomy():
    df = parse_taxonomy(TAXONOMY).set_index("additive")
    assert list(df.index) == ["E250", "E322i", "E330"]
    assert df.loc["E250", "name_fr"] == "E250 - Nitrite de sodium"
    assert df.loc["E250", "overexposure_risk"] == "high"
    assert df.loc["E250", "risk_label"] == "Élevé"
    assert df.loc["E250", "anses_interest"]
    assert df.loc["E322i", "name_fr"] == "E322i - Lecithin"  # repli sur l'anglais
    assert df.loc["E322i", "classes"] == ["emulsifier", "antioxidant"]
    assert pd.isna(df.loc["E322i", "overexposure_risk"])
    assert code_from_key("en:e160a") == "E160a"


def test_detect_recall_columns():
    columns = [
        "Numéro de fiche", "GTIN", "Marque de salubrité", "Marque du produit",
        "Motif du rappel", "Risques encourus", "Date de publication",
        "Lien vers la fiche rappel", "Sous-catégorie de produit", "Catégorie de produit",
    ]  # fmt: skip
    found = detect_columns(columns)
    assert found["gtin"] == "GTIN"
    assert found["brand"] == "Marque du produit"
    assert found["category"] == "Catégorie de produit"
    assert found["date"] == "Date de publication"
    with pytest.raises(ValueError, match="gtin"):
        detect_columns(["motif", "date"])


def test_gtin_key():
    assert gtin_key("03017620422003") == ["3017620422003"]
    assert gtin_key("3017620422003 3017620425035") == ["3017620422003", "3017620425035"]
    assert gtin_key(None) == []


def test_build_recalls_keeps_latest_per_code(tmp_path):
    raw = pd.DataFrame(
        {
            "gtin": ["0123456789012", "123456789012", "999999999999"],
            "date_publication": ["2024-01-01", "2025-06-01", "2023-03-03"],
            "motif_rappel": ["ancien", "Listeria", "Salmonelle"],
        }
    )
    source = tmp_path / "rappelconso.parquet"
    raw.to_parquet(source)
    df = build_recalls(source, tmp_path / "recalls.parquet").set_index("gtin_key")
    assert len(df) == 2
    assert df.loc["123456789012", "recall_reason"] == "Listeria"


@pytest.fixture
def con(interim_factory, tmp_path):
    rows = [
        # Bien noté, ultra-transformé, beaucoup d'additifs dont un à risque, bio
        {"code": "1", "nutriscore_grade": "a", "nova_group": 4, "ingredients_text": "x",
         "additives_n": 6, "labels_tags": ["en:organic"], "brands": "Marque A",
         "brands_tags": ["marque-a"],
         "additives_tags": ["en:e250", "en:e330", "en:e322i", "en:e1", "en:e2", "en:e3"]},
        # Rappelé uniquement (code avec zéro de tête dans RappelConso)
        {"code": "123456789012", "nutriscore_grade": "c", "nova_group": 3,
         "ingredients_text": "y", "additives_n": 0},
        # Aucun signal
        {"code": "3", "nutriscore_grade": "b", "nova_group": 1, "ingredients_text": "z",
         "additives_n": 0, "additives_tags": []},
    ]  # fmt: skip
    dest = tmp_path / "products.parquet"
    build_products(interim_factory(rows), dest, report_path=None)
    connection = products_connection(dest)
    yield connection
    connection.close()


@pytest.fixture
def refs():
    additives = parse_taxonomy(TAXONOMY)
    recalls = pd.DataFrame(
        {
            "gtin_key": ["123456789012"],
            "recall_date": [pd.Timestamp("2025-06-01")],
            "recall_reason": ["Présence de Listeria"],
            "recall_risk": [None],
            "recall_category": [None],
            "recall_brand": [None],
            "recall_label": [None],
            "recall_link": [None],
        }
    )
    return additives, recalls


def test_signals_and_explanations(con, refs):
    signals = ctv.compute_signals(con, *refs).set_index("code")
    assert set(signals.index) == {"1", "123456789012"}

    first = signals.loc["1"]
    assert first["good_score_ultra"] and first["good_score_additives"]
    assert first["risky_additives"] and first["organic_ultra"]
    assert not first["recalled"]
    assert first["controversy_score"] == 4
    assert list(first["risky_additive_names"]) == ["E250 - Nitrite de sodium"]
    assert first["reasons"].startswith("Nutri-Score A mais ultra-transformé (NOVA 4)")
    assert "6 additifs" in first["reasons"] and "EFSA" in first["reasons"]

    recalled = signals.loc["123456789012"]
    assert recalled["recalled"] and recalled["controversy_score"] == 1
    assert recalled["reasons"] == "Rappelé le 01/06/2025 (Présence de Listeria)."

    counts = ctv.signal_counts(signals.reset_index()).set_index("signal")
    assert counts.loc["recalled", "products"] == 1


def test_aggregates(con, refs):
    additives, recalls = refs
    signals = ctv.compute_signals(con, additives, recalls)
    freq = ctv.risky_additives_frequency(con, additives)
    assert freq["additive"].tolist() == ["E250"]
    assert freq.loc[0, "share"] == pytest.approx(100 / 3)

    by_nova = ctv.risky_share_by_nova(con, signals).set_index("nova_group")
    assert by_nova.loc[4, "share"] == 100 and by_nova.loc[1, "share"] == 0

    sub = ctv.good_score_ultra_by_subgroup(con, min_products=1)
    assert sub.empty  # aucun produit n'a de sous-groupe dans ce jeu

    brands = ctv.brand_profiles(con, min_products=1).set_index("brand_tag")
    assert brands.loc["marque-a", "brand"] == "Marque A"
    assert pd.isna(brands.loc["marque-a", "halo"])  # moins de 10 produits A ou B
