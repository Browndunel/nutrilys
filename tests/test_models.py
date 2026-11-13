import numpy as np
import pandas as pd
import pytest

from nutrilys import config
from nutrilys.data.clean import build_products
from nutrilys.db import products_connection
from nutrilys.models import baseline
from nutrilys.models.dataset import (
    TASKS,
    assign_splits,
    build_text,
    load_task_frame,
    make_splits,
    tag_words,
)
from nutrilys.models.evaluate import classification_metrics, confusion_shares, per_class_f1

# Chaque mot est associé à un groupe NOVA et à un Nutri-Score : signal net, toutes classes présentes
GRADES = {"pomme": "a", "carotte": "a", "lentilles": "a", "riz complet": "b", "huile olive": "c",
          "beurre doux": "d", "sel de mer": "c", "sucre roux": "e", "pain campagne": "b",
          "fromage affine": "d", "conserve thon": "b", "jambon sec": "e", "soda cola": "e",
          "biscuit fourre": "e", "nuggets panes": "d", "barre chocolatee": "e"}  # fmt: skip
WORDS = {
    1: ["pomme", "carotte", "lentilles", "riz complet"],
    2: ["huile olive", "beurre doux", "sel de mer", "sucre roux"],
    3: ["pain campagne", "fromage affine", "conserve thon", "jambon sec"],
    4: ["soda cola", "biscuit fourre", "nuggets panes", "barre chocolatee"],
}


@pytest.fixture
def con(interim_factory, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(800):
        nova = int(rng.integers(1, 5))
        word = str(rng.choice(WORDS[nova]))
        grade = GRADES[word]
        rows.append(
            {
                "code": f"{i:06d}",
                "product_name": f"{word} produit {i}",
                "categories_tags": [f"en:{word.replace(' ', '-')}"],
                "ingredients_text": f"{word}, eau" + (", arôme, e471" if nova == 4 else ""),
                "nova_group": nova,
                "nutriscore_grade": grade,
                "sugars_100g": float(10 * nova),
                "salt_100g": 0.0 if nova == 1 else 0.5,
                "saturated_fat_100g": float(nova),
            }
        )
    rows[0]["categories_tags"] = None  # catégorie absente, comme dans les vraies données
    dest = tmp_path / "products.parquet"
    build_products(interim_factory(rows), dest, report_path=None)
    monkeypatch.setattr(config, "SPLITS_PATH", tmp_path / "splits.parquet")
    monkeypatch.setattr(config, "TFIDF_MIN_DF", 1)
    connection = products_connection(dest)
    yield connection
    connection.close()


def test_tag_words_and_text():
    assert tag_words(["en:breakfast-cereals", "fr:pates-a-tartiner"]) == (
        "breakfast cereals, pates a tartiner"
    )
    for missing in (None, pd.NA, np.nan, [], np.array([])):
        assert tag_words(missing) == ""
    assert tag_words(np.array(["en:sodas"])) == "sodas"
    df = pd.DataFrame(
        {
            "product_name": ["Pâte  à tartiner"],
            "generic_name": [None],
            "categories_tags": [["en:spreads"]],
            "ingredients_text": ["Sucre, noisettes"],
        }
    )
    assert "ingrédients" not in build_text(df, with_ingredients=False)[0]
    assert build_text(df, with_ingredients=True)[0] == (
        "pâte à tartiner | | catégories : spreads | ingrédients : sucre, noisettes"
    )


def test_splits_are_stratified_and_reproducible():
    target = pd.Series([1] * 700 + [4] * 300)
    codes = pd.Series([f"{i}" for i in range(1000)])
    first, second = make_splits(codes, target), make_splits(codes, target)
    assert first.equals(second)
    assert first.value_counts(normalize=True)["test"] == pytest.approx(0.15, abs=0.01)
    test_share_4 = (target[first == "test"] == 4).mean()
    assert test_share_4 == pytest.approx(0.3, abs=0.02)


def test_pair_tasks_share_the_same_split(con):
    a = assign_splits(load_task_frame(con, TASKS["nova_name"]), TASKS["nova_name"])
    b = assign_splits(load_task_frame(con, TASKS["nova_full"]), TASKS["nova_full"])
    merged = a[["code", "split"]].merge(b[["code", "split"]], on="code")
    assert len(merged) == len(a) and (merged["split_x"] == merged["split_y"]).all()


def test_missing_class_is_reported(con):
    task = TASKS["nova_name"]
    df = assign_splits(load_task_frame(con, task), task)
    train = df[(df["split"] == "train") & (df["target"] != 2)]
    with pytest.raises(ValueError, match="absentes"):
        baseline.fit(task, "lr", train, df[df["split"] == "valid"])


def test_metrics():
    y_true, y_pred = np.array([0, 1, 2, 3]), np.array([0, 1, 3, 1])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.5 and m["within_one"] == 0.75
    shares = confusion_shares(y_true, y_pred, ["1", "2", "3", "4"])
    assert shares.loc["3", "4"] == 100
    assert per_class_f1(y_true, y_pred, ["1", "2", "3", "4"])["1"] == 1


@pytest.mark.parametrize("model", ["lr", "xgb"])
@pytest.mark.parametrize("task_name", ["nova_name", "nutriscore_full"])
def test_fit_and_evaluate(con, task_name, model):
    task = TASKS[task_name]
    df = assign_splits(load_task_frame(con, task), task)
    parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    fitted = baseline.fit(task, model, parts["train"], parts["valid"])
    metrics, preds = baseline.evaluate(fitted, parts["test"])
    assert metrics["f1_macro"] > 0.6  # signal fort dans les données simulées
    assert set(preds["y_pred"]) <= set(task.classes)
    assert len(fitted.predict(parts["test"].head(3))) == 3
    if model == "lr":
        terms = baseline.top_terms(fitted, n=3)
        assert len(terms) == 3 * len(task.classes)


def test_predict_products(con, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    task = TASKS["nova_name"]
    df = assign_splits(load_task_frame(con, task), task)
    fitted = baseline.fit(task, "lr", df[df["split"] == "train"], df[df["split"] == "valid"])
    out = baseline.predict_products(con, fitted, "has_nova")
    assert len(out) == len(df)
    assert set(out["predicted"]) <= {1, 2, 3, 4}
    assert out["confidence"].between(0, 1).all()


def test_charts_for_models():
    from nutrilys.viz import charts

    shares = confusion_shares(np.array([0, 1, 1]), np.array([0, 1, 0]), ["1", "2"])
    assert len(charts.confusion_heatmap(shares).layout.annotations) == 4
    df = pd.DataFrame(
        {"task": ["a", "a", "b"], "model": ["lr", "xgb", "lr"], "f1": [0.5, 0.6, 0.7]}
    )
    fig = charts.grouped_bars(df, "task", "model", "f1", {"lr": "#000", "xgb": "#111"})
    assert len(fig.data) == 2


def test_saved_model_reloads(con, tmp_path):
    import joblib

    task = TASKS["nutriscore_full"]
    df = assign_splits(load_task_frame(con, task), task)
    fitted = baseline.fit(task, "xgb", df[df["split"] == "train"], df[df["split"] == "valid"])
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    reloaded = joblib.load(path)
    assert type(reloaded).__module__ == "nutrilys.models.fitted"
    sample = df.head(5)
    assert (reloaded.predict(sample) == fitted.predict(sample)).all()


class _KeepOpen:
    """Connexion partagée que `run` ne doit pas fermer pendant le test."""

    def __init__(self, con):
        self._con = con

    def __getattr__(self, name):
        return getattr(self._con, name)

    def close(self):
        pass


def test_metrics_file_keeps_only_consistent_runs(con, tmp_path, monkeypatch):
    for name in ("MODELS_DIR", "PREDICTIONS_DIR", "MODEL_REPORTS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "NOVA_ESTIMATED_PATH", tmp_path / "nova_estimated.parquet")
    monkeypatch.setattr(baseline, "products_connection", lambda: _KeepOpen(con))
    path = tmp_path / "model_reports_dir" / "baseline_metrics.csv"

    # Fichier ancien, sans colonne « sample » : origine inconnue, doit être écarté
    path.parent.mkdir(parents=True)
    pd.DataFrame({"task": ["nova_full"], "model": ["xgb"], "f1_macro": [0.9]}).to_csv(path)
    baseline.run(["nova_name"], ["lr"])
    assert pd.read_csv(path)["model"].tolist() == ["lr"]

    baseline.run(["nova_name"], ["lr"], sample=300)  # essai sur échantillon
    baseline.run(["nova_name"], ["lr"])  # complet : l'essai est écarté
    assert pd.read_csv(path)["sample"].tolist() == [0]

    baseline.run(["nova_name"], ["xgb"])  # complément, mêmes conditions : conservé
    assert sorted(pd.read_csv(path)["model"]) == ["lr", "xgb"]


def test_limit_train_keeps_class_balance():
    train = pd.DataFrame({"target": [1] * 800 + [4] * 200, "code": range(1000)})
    small = baseline.limit_train(train, 100)
    assert len(small) == 100
    assert (small["target"] == 4).mean() == pytest.approx(0.2)
    assert baseline.limit_train(train, None) is train
    assert baseline.limit_train(train, 5000) is train
