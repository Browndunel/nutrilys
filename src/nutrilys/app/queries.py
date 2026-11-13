"""Requêtes de l'application sur la table légère (vue `app`).

Toutes les valeurs saisies par l'utilisateur passent par des paramètres liés
(jamais concaténées dans le SQL).
"""

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from nutrilys.db import connect, sql_path

RESULT_COLUMNS = [
    "code", "product_name", "brand", "food_group", "nutriscore_grade", "nova_group",
    "nova_estimated", "is_organic", "n_additives", "controversy_score", "n_scans",
]  # fmt: skip


@dataclass
class Filters:
    text: str = ""
    food_group: str | None = None
    brand: str | None = None
    nutriscore: list[str] = field(default_factory=list)
    nova: list[int] = field(default_factory=list)
    include_estimated: bool = False  # NOVA estimé accepté quand le NOVA est absent
    organic_only: bool = False
    flagged_only: bool = False


def open_app(path) -> duckdb.DuckDBPyConnection:
    """Connexion avec la vue `app` sur la table légère."""
    con = connect()
    con.execute(f"CREATE VIEW app AS SELECT * FROM read_parquet('{sql_path(Path(path))}')")
    return con


def normalize(text: str) -> str:
    """Minuscules sans accents, comme la colonne search_text."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).strip()


def _where(f: Filters) -> tuple[str, list]:
    clauses, params = ["true"], []
    for token in normalize(f.text).split():
        clauses.append("search_text LIKE ?")
        params.append(f"%{token}%")
    if f.food_group:
        clauses.append("food_group = ?")
        params.append(f.food_group)
    if f.brand:
        clauses.append("brand = ?")
        params.append(f.brand)
    if f.nutriscore:
        clauses.append(f"nutriscore_grade IN ({', '.join('?' * len(f.nutriscore))})")
        params += list(f.nutriscore)
    if f.nova:
        marks = ", ".join("?" * len(f.nova))
        column = "coalesce(nova_group, nova_estimated)" if f.include_estimated else "nova_group"
        clauses.append(f"{column} IN ({marks})")
        params += [int(n) for n in f.nova]
    if f.organic_only:
        clauses.append("is_organic")
    if f.flagged_only:
        clauses.append("controversy_score > 0")
    return " AND ".join(clauses), params


def search(con, f: Filters, limit: int = 50, offset: int = 0) -> tuple[pd.DataFrame, int]:
    """Produits correspondant aux filtres, les plus populaires d'abord, et leur nombre total."""
    where, params = _where(f)
    total = con.execute(f"SELECT count(*) FROM app WHERE {where}", params).fetchone()[0]
    rows = con.execute(
        f"""SELECT {", ".join(RESULT_COLUMNS)} FROM app WHERE {where}
            ORDER BY n_scans DESC, code LIMIT ? OFFSET ?""",
        [*params, int(limit), int(offset)],
    ).df()
    return rows, int(total)


def product(con, code: str) -> dict | None:
    """Toutes les informations d'un produit, ou None s'il n'existe pas."""
    df = con.execute("SELECT * FROM app WHERE code = ?", [str(code)]).df()
    return None if df.empty else df.iloc[0].to_dict()


def food_groups(con) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT food_group FROM app WHERE food_group IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]


def top_brands(con, n: int = 500) -> list[str]:
    """Marques les plus représentées, pour la liste de choix."""
    return [
        r[0]
        for r in con.execute(
            "SELECT brand FROM app WHERE brand IS NOT NULL GROUP BY brand "
            "ORDER BY count(*) DESC LIMIT ?",
            [int(n)],
        ).fetchall()
    ]


def key_figures(con) -> dict:
    """Chiffres clés de l'accueil."""
    row = con.execute(
        """
        SELECT count(*) AS products,
               count(nutriscore_grade) AS with_nutriscore,
               100.0 * avg(CASE WHEN nutriscore_grade IS NOT NULL
                                THEN (nutriscore_grade IN ('d', 'e'))::INT END) AS share_de,
               count(nova_group) AS with_nova,
               100.0 * avg(CASE WHEN nova_group IS NOT NULL
                                THEN (nova_group = 4)::INT END) AS share_nova4,
               count(*) FILTER (WHERE controversy_score > 0) AS flagged,
               count(*) FILTER (WHERE controversy_score >= 2) AS flagged_multi,
               count(nova_estimated) AS with_estimate
        FROM app
        """
    ).df()
    return row.iloc[0].to_dict()
