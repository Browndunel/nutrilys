"""Rappels de produits (RappelConso V2), ramenés à une ligne par code-barres.

Les noms de colonnes de l'export ont évolué au fil des versions : on les repère
par mot-clé plutôt que par nom exact, et on affiche ce qui a été trouvé.

Usage :
    uv run python -m nutrilys.data.recalls
"""

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from nutrilys import config

logger = logging.getLogger(__name__)

# Rôle -> mots-clés cherchés dans le nom de colonne (dans l'ordre de préférence)
COLUMN_KEYWORDS = {
    "gtin": ["gtin", "code_barre"],
    "date": ["date_publication", "date_de_publication"],
    "reason": ["motif"],
    "risk": ["risques_encourus", "risque"],
    "category": ["categorie_produit", "categorie_de_produit", "categorie"],
    "brand": ["marque_produit", "nom_de_la_marque", "marque"],
    "label": ["libelle", "nom_produit", "modeles_ou_references", "titre"],
    "link": ["lien_vers_la_fiche", "lien"],
}
# Colonnes à ne jamais retenir pour un rôle, même si un mot-clé correspond
EXCLUDED = {"brand": ["salubrite"], "category": ["sous_categorie"], "link": ["affichette", "image"]}
REQUIRED = ("gtin",)


def _normalize(name: str) -> str:
    """Nom de colonne sans accents, en minuscules, espaces remplacés par _."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def detect_columns(columns: list[str]) -> dict[str, str]:
    """Associe chaque rôle à la première colonne dont le nom contient un mot-clé."""
    normalized = {_normalize(c): c for c in columns}
    found: dict[str, str] = {}
    for role, keywords in COLUMN_KEYWORDS.items():
        for keyword in keywords:
            excluded = EXCLUDED.get(role, [])
            match = next(
                (
                    orig
                    for norm, orig in normalized.items()
                    if keyword in norm and not any(e in norm for e in excluded)
                ),
                None,
            )
            if match is not None and match not in found.values():
                found[role] = match
                break
    missing = [role for role in REQUIRED if role not in found]
    if missing:
        raise ValueError(
            f"Colonnes introuvables dans RappelConso : {missing}. Colonnes : {columns}"
        )
    return found


def gtin_key(value) -> list[str]:
    """Codes-barres d'une cellule, réduits aux chiffres et sans zéros de tête."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    codes = re.findall(r"\d{6,14}", str(value))
    return [c.lstrip("0") for c in codes if c.strip("0")]


def build_recalls(
    source: Path = config.RAPPELCONSO_RAW_PATH, dest: Path = config.RECALLS_PATH
) -> pd.DataFrame:
    """Une ligne par code-barres rappelé, avec le rappel le plus récent."""
    if not source.exists():
        raise FileNotFoundError(
            f"{source} introuvable. Lancer d'abord : "
            "uv run python -m nutrilys.data.download --only rappelconso"
        )
    raw = pd.read_parquet(source)
    cols = detect_columns(list(raw.columns))
    logger.info("Colonnes RappelConso utilisées : %s", cols)

    df = pd.DataFrame({role: raw[col] for role, col in cols.items()})
    df["gtin_key"] = df["gtin"].map(gtin_key)
    df = df.explode("gtin_key").dropna(subset=["gtin_key"])
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.sort_values("date", ascending=False)
    df = df.drop_duplicates("gtin_key").drop(columns="gtin")
    for role in COLUMN_KEYWORDS:
        if role != "gtin" and role not in df:
            df[role] = None
    df = df[["gtin_key", "date", "reason", "risk", "category", "brand", "label", "link"]]
    df.columns = ["gtin_key"] + [f"recall_{c}" for c in df.columns[1:]]

    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Rappels écrits : %s (%d codes-barres)", dest, len(df))
    return df.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_recalls()
    print(f"\n{len(df)} codes-barres rappelés")
    if df["recall_date"].notna().any():
        print(f"Période : {df['recall_date'].min():%d/%m/%Y} - {df['recall_date'].max():%d/%m/%Y}")


if __name__ == "__main__":
    main()
