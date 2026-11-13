"""Ressources partagées de l'application, chargées une fois puis mises en cache.

Une connexion DuckDB n'est pas faite pour être utilisée par plusieurs sessions en
même temps : chaque appel reçoit son propre curseur sur la connexion en cache.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from nutrilys import config
from nutrilys.app import queries


def require(path: Path, command: str) -> None:
    """Arrête la page avec une consigne claire si un fichier nécessaire manque."""
    if not path.exists():
        st.error(f"Fichier manquant : `{path.name}`. Le créer avec : `{command}`")
        st.stop()


@st.cache_resource(show_spinner=False)
def _app_connection(path: str):
    return queries.open_app(path)


def app_cursor():
    require(config.APP_PRODUCTS_PATH, "uv run python -m nutrilys.app.build")
    return _app_connection(str(config.APP_PRODUCTS_PATH)).cursor()


@st.cache_data(show_spinner=False)
def food_groups(path: str) -> list[str]:
    return queries.food_groups(_app_connection(path).cursor())


@st.cache_data(show_spinner=False)
def top_brands(path: str) -> list[str]:
    return queries.top_brands(_app_connection(path).cursor())


@st.cache_data(show_spinner=False)
def key_figures(path: str) -> dict:
    return queries.key_figures(_app_connection(path).cursor())


@st.cache_data(show_spinner=False)
def additives_table(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["additive", "name_fr", "risk_label", "classes"])
    return pd.read_parquet(path).set_index("additive")


@st.cache_resource(show_spinner=False)
def _model(path: str):
    import joblib

    return joblib.load(path)


def model(task: str, name: str = "lr"):
    """Modèle entraîné à l'étape 6, ou None s'il n'a pas été entraîné."""
    path = config.MODELS_DIR / f"{task}_{name}.joblib"
    return _model(str(path)) if path.exists() else None
