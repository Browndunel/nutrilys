"""Connexion DuckDB commune, avec les réglages de mémoire du projet."""

from pathlib import Path

import duckdb

from nutrilys import config


def connect() -> duckdb.DuckDBPyConnection:
    """Connexion en mémoire, limitée en RAM et en threads."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads = {config.DUCKDB_THREADS}")
    con.execute("SET preserve_insertion_order = false")
    return con


def sql_path(path: Path) -> str:
    """Chemin utilisable dans une requête SQL, y compris sous Windows."""
    return path.as_posix()
