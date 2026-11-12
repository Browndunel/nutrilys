"""Table de référence des additifs, à partir de la taxonomie d'Open Food Facts.

Pour chaque additif : nom en français, classes (colorant, émulsifiant...),
risque de surexposition selon les évaluations de l'EFSA, et appartenance à la
liste des additifs suivis par l'ANSES.

Le « risque de surexposition » indique si une partie de la population risque de
dépasser la dose journalière admissible. Ce n'est pas un jugement de toxicité
de l'additif en lui-même.

Usage :
    uv run python -m nutrilys.data.additives
"""

import json
import logging
from pathlib import Path

import pandas as pd

from nutrilys import config

logger = logging.getLogger(__name__)

RISK_LABELS = {"high": "Élevé", "moderate": "Modéré", "no": "Faible ou nul"}


def _value(entry: dict, key: str, lang: str = "en") -> str | None:
    """Valeur d'une propriété de taxonomie, de la forme {"en": "..."}."""
    field = entry.get(key)
    if isinstance(field, dict):
        return field.get(lang)
    return None


def _strip_prefix(value: str | None) -> str | None:
    return value.split(":", 1)[1] if value and ":" in value else value


def code_from_key(key: str) -> str:
    """en:e322i -> E322i (même format que la colonne `additives` des produits)."""
    return "E" + key.removeprefix("en:e")


def parse_taxonomy(taxonomy: dict) -> pd.DataFrame:
    """Transforme la taxonomie JSON en une ligne par additif."""
    rows = []
    for key, entry in taxonomy.items():
        if not key.startswith("en:e"):
            continue
        names = entry.get("name", {})
        classes = _value(entry, "additives_classes") or ""
        rows.append(
            {
                "additive": code_from_key(key),
                "name_fr": names.get("fr") or names.get("en") or code_from_key(key),
                "name_en": names.get("en"),
                "classes": [_strip_prefix(c.strip()) for c in classes.split(",") if c.strip()],
                "overexposure_risk": _strip_prefix(
                    _value(entry, "efsa_evaluation_overexposure_risk")
                ),
                "anses_interest": _value(entry, "anses_additives_of_interest") is not None,
                "efsa_url": _value(entry, "efsa_evaluation_url"),
            }
        )
    df = pd.DataFrame(rows).sort_values("additive").reset_index(drop=True)
    df["risk_label"] = df["overexposure_risk"].map(RISK_LABELS)
    return df


def build_additives(
    source: Path = config.ADDITIVES_RAW_PATH, dest: Path = config.ADDITIVES_PATH
) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(
            f"{source} introuvable. Lancer d'abord : "
            "uv run python -m nutrilys.data.download --only additives"
        )
    taxonomy = json.loads(source.read_text(encoding="utf-8"))
    df = parse_taxonomy(taxonomy)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("Table des additifs écrite : %s (%d additifs)", dest, len(df))
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_additives()
    print("\nRisque de surexposition (EFSA) :")
    print(df["risk_label"].fillna("Non évalué").value_counts().to_string())
    print(f"\nAdditifs suivis par l'ANSES : {int(df['anses_interest'].sum())}")


if __name__ == "__main__":
    main()
