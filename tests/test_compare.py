import json

import pandas as pd
import pytest

from nutrilys import config
from nutrilys.data.clean import build_products
from nutrilys.db import products_connection
from nutrilys.models import compare
from nutrilys.models.export import COLUMNS, export_task


def _preds(correct: list[bool], codes=None) -> pd.DataFrame:
    codes = codes or [str(i) for i in range(len(correct))]
    return pd.DataFrame({"code": codes, "y_true": 1, "y_pred": [1 if ok else 2 for ok in correct]})


def test_mcnemar_detects_a_clear_difference():
    a = _preds([True] * 50 + [False] * 50)
    b = _preds([True] * 95 + [False] * 5)
    result = compare.mcnemar_test(a, b)
    assert result["only_b_correct"] == 45 and result["only_a_correct"] == 0
    assert result["p_value"] < 0.001
    assert result["accuracy_gap"] == pytest.approx(0.45)


def test_mcnemar_identical_models_and_few_disagreements():
    a = _preds([True] * 60 + [False] * 40)
    assert compare.mcnemar_test(a, a)["p_value"] == 1.0  # aucun désaccord
    b = _preds([True] * 63 + [False] * 37)  # 3 désaccords : test exact
    assert compare.mcnemar_test(a, b)["p_value"] > 0.2


def test_mcnemar_no_difference_and_test_set_check():
    a = _preds([True, False] * 50)
    b = _preds([False, True] * 50)
    assert compare.mcnemar_test(a, b)["p_value"] > 0.5
    with pytest.raises(ValueError, match="même jeu de test"):
        compare.mcnemar_test(a, _preds([True] * 100, codes=[f"x{i}" for i in range(100)]))


@pytest.mark.parametrize(
    ("f1_camembert", "p_value", "expected"),
    [(0.90, 0.001, "camembert"), (0.90, 0.40, "lr"), (0.81, 0.001, "lr")],
)
def test_choose_model(f1_camembert, p_value, expected):
    metrics = pd.DataFrame(
        {"task": ["nova_name", "nova_name"], "model": ["lr", "camembert"],
         "f1_macro": [0.80, f1_camembert]}
    )  # fmt: skip
    model, reason = compare.choose_model(metrics, "nova_name", p_value)
    assert model == expected and reason


def test_choose_model_reason_when_baseline_wins():
    metrics = pd.DataFrame(
        {"task": ["nova_name", "nova_name"], "model": ["lr", "camembert"],
         "f1_macro": [0.833, 0.826]}
    )  # fmt: skip
    model, reason = compare.choose_model(metrics, "nova_name", 0.001)
    assert model == "lr"
    assert "-0,007" in reason and "significativement meilleure" in reason


def test_metrics_are_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_REPORTS_DIR", tmp_path)
    pd.DataFrame({"task": ["nova_name"], "model": ["lr"], "f1_macro": [0.8]}).to_csv(
        tmp_path / "baseline_metrics.csv", index=False
    )
    (tmp_path / "nova_name_camembert_metrics.json").write_text(
        json.dumps({"task": "nova_name", "model": "camembert", "f1_macro": 0.85}), encoding="utf-8"
    )
    merged = compare.all_metrics()
    assert sorted(merged["model"]) == ["camembert", "lr"]


def test_export_reuses_the_frozen_split(interim_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPLITS_PATH", tmp_path / "splits.parquet")
    rows = [
        {"code": f"{i:04d}", "ingredients_text": "x", "nova_group": 1 + i % 4,
         "categories_tags": ["en:sodas"]}
        for i in range(200)
    ]  # fmt: skip
    dest = tmp_path / "products.parquet"
    build_products(interim_factory(rows), dest, report_path=None)
    con = products_connection(dest)
    first = export_task(con, "nova_name")
    again = export_task(con, "nova_name")
    con.close()
    assert list(first.columns) == COLUMNS
    assert set(first["split"]) == {"train", "valid", "test"}
    assert first.equals(again)
