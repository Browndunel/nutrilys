"""Modèles de référence : TF-IDF + régression logistique, TF-IDF + XGBoost.

Pour chaque tâche (voir `dataset.TASKS`) et chaque modèle :
- entraînement sur train, arrêt anticipé de XGBoost sur valid ;
- évaluation sur test uniquement ;
- enregistrement du modèle, des prédictions de test et des métriques.

Usage :
    uv run python -m nutrilys.models.baseline --sample 50000     # essai rapide
    uv run python -m nutrilys.models.baseline --models lr        # régression logistique, tout
    uv run python -m nutrilys.models.baseline --tasks nova_full nutriscore_full --models xgb

XGBoost n'est utile que pour les scénarios de contrôle (B), où il retrouve les règles
de calcul par seuils ; sur le texte seul, la régression logistique fait aussi bien,
bien plus vite (essai sur 50 000 produits du 13/11/2026).
"""

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from nutrilys import config
from nutrilys.db import products_connection
from nutrilys.models.dataset import (
    NUTRIENT_FEATURES,
    TASKS,
    Task,
    assign_splits,
    build_text,
    load_task_frame,
)
from nutrilys.models.evaluate import classification_metrics
from nutrilys.models.fitted import FittedModel, shift_for_sparse

logger = logging.getLogger(__name__)

MODELS = {"lr": "Régression logistique", "xgb": "XGBoost"}


def encode(target: pd.Series, classes: list) -> np.ndarray:
    mapping = {c: i for i, c in enumerate(classes)}
    return target.map(mapping).to_numpy()


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        # Libellés présents dans tous les textes (voir dataset.build_text) : sans information
        stop_words=["categories", "ingredients"],
    )


def _numeric(model: str):
    """Nutriments : imputation et mise à l'échelle pour la régression logistique ;
    XGBoost gère lui-même les valeurs manquantes."""
    if model == "lr":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
        )
    return FunctionTransformer(shift_for_sparse)


def fit(task: Task, model: str, train: pd.DataFrame, valid: pd.DataFrame) -> FittedModel:
    y_train = encode(train["target"], task.classes)
    y_valid = encode(valid["target"], task.classes)
    missing = set(range(len(task.classes))) - set(y_train)
    if missing:  # sinon les colonnes de probabilité ne correspondraient plus aux classes
        absent = [task.classes[i] for i in sorted(missing)]
        raise ValueError(f"Classes absentes du jeu d'entraînement : {absent}")
    start = time.perf_counter()

    vectorizer = _vectorizer().fit(train["text"])
    numeric = _numeric(model).fit(train[NUTRIENT_FEATURES]) if task.with_nutrients else None
    fitted = FittedModel(task.name, model, task.classes, vectorizer, numeric, None)
    if model == "xgb":
        # Les arbres n'ont pas besoin de tout le vocabulaire : on garde les termes les
        # plus liés à la cible (test du chi²). Gain de temps d'environ x12, F1 inchangé.
        x_text = vectorizer.transform(train["text"])
        k = min(config.XGB_TEXT_FEATURES, x_text.shape[1])
        fitted.selector = SelectKBest(chi2, k=k).fit(x_text, y_train)
    x_train, x_valid = fitted.features(train), fitted.features(valid)

    if model == "lr":
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(x_train, y_train)
    else:
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.2,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            tree_method="hist",
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            max_bin=64,
            n_jobs=-1,  # tous les cœurs du processeur
            random_state=config.RANDOM_SEED,
        )
        clf.fit(
            x_train,
            y_train,
            sample_weight=compute_sample_weight("balanced", y_train),
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
    fitted.classifier = clf
    fitted.timings["fit_seconds"] = round(time.perf_counter() - start, 1)
    return fitted


def evaluate(fitted: FittedModel, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Métriques sur le jeu de test et prédictions détaillées."""
    start = time.perf_counter()
    proba = fitted.predict_proba(test)
    elapsed = time.perf_counter() - start
    y_true = encode(test["target"], fitted.classes)
    y_pred = proba.argmax(axis=1)

    metrics = classification_metrics(y_true, y_pred)
    metrics.update(
        task=fitted.task,
        model=fitted.model,
        fit_seconds=fitted.timings.get("fit_seconds"),
        predict_ms_per_1000=round(1000 * elapsed / max(len(test), 1) * 1000, 1),
    )
    predictions = pd.DataFrame(
        {
            "code": test["code"].to_numpy(),
            "y_true": np.asarray(fitted.classes)[y_true],
            "y_pred": np.asarray(fitted.classes)[y_pred],
            "confidence": proba.max(axis=1),
        }
    )
    return metrics, predictions


def top_terms(fitted: FittedModel, n: int = 15) -> pd.DataFrame:
    """Termes les plus associés à chaque classe (régression logistique uniquement)."""
    if fitted.model != "lr":
        raise ValueError("Disponible seulement pour la régression logistique.")
    names = np.asarray(fitted.vectorizer.get_feature_names_out())
    coefs = fitted.classifier.coef_[:, : len(names)]
    rows = []
    for k, cls in enumerate(fitted.classes):
        best = np.argsort(coefs[k])[::-1][:n]
        rows += [{"class": cls, "term": names[i], "weight": float(coefs[k, i])} for i in best]
    return pd.DataFrame(rows)


def predict_products(con, fitted: FittedModel, where: str) -> pd.DataFrame:
    """Applique un modèle aux produits qui vérifient la condition SQL `where`.

    Sert par exemple à estimer le groupe NOVA des produits sans liste d'ingrédients
    avec le modèle « nom et catégories ».
    """
    task = TASKS[fitted.task]
    nutrients = ", ".join(NUTRIENT_FEATURES)
    df = con.execute(
        f"""
        SELECT code, product_name, generic_name, categories_tags, ingredients_text,
               food_group, {nutrients}
        FROM products WHERE {where}
        """
    ).df()
    df["text"] = build_text(df, task.with_ingredients)
    # Par paquets, pour borner la mémoire sur plusieurs centaines de milliers de produits
    proba = (
        np.vstack(
            [fitted.predict_proba(df.iloc[i : i + 100_000]) for i in range(0, len(df), 100_000)]
        )
        if len(df)
        else np.empty((0, len(fitted.classes)))
    )
    return pd.DataFrame(
        {
            "code": df["code"],
            "food_group": df["food_group"],
            "predicted": np.asarray(fitted.classes)[proba.argmax(axis=1)],
            "confidence": proba.max(axis=1),
        }
    )


def best_model(metrics: pd.DataFrame, task: str) -> str:
    """Modèle de meilleur F1 macro pour une tâche."""
    rows = metrics[metrics["task"] == task].sort_values("f1_macro")
    return str(rows.iloc[-1]["model"])


def estimate_missing_nova(metrics: pd.DataFrame) -> pd.DataFrame:
    """Estime le groupe NOVA des produits qui n'en ont pas, avec le meilleur modèle
    « nom et catégories ». Résultat enregistré pour l'application."""
    model = best_model(metrics, "nova_name")
    con = products_connection()
    out = predict_products(con, load_model("nova_name", model), "NOT has_nova")
    con.close()
    out["model"] = model
    out.to_parquet(config.NOVA_ESTIMATED_PATH, index=False)
    logger.info("NOVA estimé pour %d produits sans NOVA (%s)", len(out), model)
    return out


def load_model(task: str, model: str) -> FittedModel:
    return joblib.load(model_path(task, model))


def model_path(task: str, model: str) -> Path:
    return config.MODELS_DIR / f"{task}_{model}.joblib"


def run(tasks: list[str], models: list[str], sample: int | None = None) -> pd.DataFrame:
    """Entraîne et évalue chaque combinaison, enregistre modèles, prédictions et métriques."""
    config.ensure_dirs()
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    con = products_connection()
    for name in tasks:
        task = TASKS[name]
        start = time.perf_counter()
        df = assign_splits(load_task_frame(con, task, sample), task)
        logger.info("%s : données chargées en %.0f s", name, time.perf_counter() - start)
        parts = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
        logger.info(
            "%s : %d produits (train %d, valid %d, test %d)",
            name, len(df), len(parts["train"]), len(parts["valid"]), len(parts["test"]),
        )  # fmt: skip
        for model in models:
            fitted = fit(task, model, parts["train"], parts["valid"])
            metrics, predictions = evaluate(fitted, parts["test"])
            metrics["n_train"] = len(parts["train"])
            metrics["sample"] = sample or 0
            joblib.dump(fitted, model_path(name, model), compress=3)
            predictions.to_parquet(config.PREDICTIONS_DIR / f"{name}_{model}.parquet", index=False)
            results.append(metrics)
            del fitted, predictions
            gc.collect()  # libère la mémoire avant le modèle suivant
            logger.info(
                "%s / %s : F1 macro %.3f, accuracy %.3f (%.0f s)",
                name, model, metrics["f1_macro"], metrics["accuracy"], metrics["fit_seconds"],
            )  # fmt: skip
    con.close()

    new = pd.DataFrame(results)
    path = config.MODEL_REPORTS_DIR / "baseline_metrics.csv"
    if path.exists():
        # On garde les résultats déjà calculés dans les mêmes conditions (même échantillon,
        # donc même jeu de test) et on remplace ceux qui viennent d'être recalculés.
        old = pd.read_csv(path)
        # Une ligne sans information d'échantillon (fichier plus ancien) est écartée :
        # on ne peut pas savoir sur quel jeu de test elle a été calculée.
        if "sample" in old:
            same_run = old["sample"].eq(sample or 0)
        else:
            same_run = pd.Series(False, index=old.index)
        recomputed = old.set_index(["task", "model"]).index.isin(
            new.set_index(["task", "model"]).index
        )
        new = pd.concat([old[same_run & ~recomputed], new], ignore_index=True)
    new.to_csv(path, index=False)
    (config.MODEL_REPORTS_DIR / "baseline_run.json").write_text(
        json.dumps({"sample": sample, "tasks": tasks, "models": models}, indent=2),
        encoding="utf-8",
    )
    return new


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne les modèles de référence.")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--sample", type=int, help="nombre de produits par tâche (essai rapide)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = run(args.tasks, args.models, args.sample)
    if "nova_name" in args.tasks:
        estimate_missing_nova(metrics)
    columns = ["task", "model", "f1_macro", "accuracy", "within_one", "fit_seconds"]
    print("\n" + metrics[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
