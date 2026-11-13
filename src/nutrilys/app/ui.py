"""Composants visuels de l'application : feuille de style, pastilles, étiquette nutritionnelle.

Les couleurs reprennent le thème des graphiques (`nutrilys.viz.theme`). Une couleur
n'est jamais seule porteuse de sens : chaque pastille écrit aussi la lettre, le
groupe ou le niveau.
"""

import html

import pandas as pd

from nutrilys.viz.theme import (
    INK,
    NOVA_COLORS,
    NOVA_TEXT,
    NUTRISCORE_COLORS,
    NUTRISCORE_TEXT,
    SURFACE,
)

INK_SOFT = "#4f5753"
LINE = "#d9dcd6"
ACCENT = "#2a78d6"

NOVA_SHORT = {1: "brut ou peu transformé", 2: "ingrédient culinaire", 3: "transformé",
              4: "ultra-transformé"}  # fmt: skip

# Niveaux pour 100 g (repères utilisés par Open Food Facts et l'étiquetage britannique)
NUTRIENT_LEVELS = {
    "fat_100g": (3.0, 20.0),
    "saturated_fat_100g": (1.5, 5.0),
    "sugars_100g": (5.0, 12.5),
    "salt_100g": (0.3, 1.5),
}
LEVEL_STYLE = {
    "faible": ("#0ca30c", "Faible"),
    "modéré": ("#c98500", "Modéré"),
    "élevé": ("#d03b3b", "Élevé"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
/* Police du texte, sans toucher aux icônes de Streamlit (police d'icônes dédiée) */
.stApp, .stApp *:not([data-testid="stIconMaterial"]) {{
    font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {INK}; }}
h1 {{ font-size: 2.4rem; line-height: 1.15; }}
.block-container {{ max-width: 1180px; padding-top: 4.2rem; }}
.lead {{ font-size: 1.2rem; line-height: 1.55; color: {INK_SOFT}; max-width: 62ch; }}
.muted {{ color: {INK_SOFT}; font-size: 0.92rem; }}

.chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 1.2rem; }}
.chip {{
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.32rem 0.75rem; border-radius: 999px; font-weight: 700; font-size: 0.95rem;
}}
.chip small {{ font-weight: 400; opacity: 0.9; }}
.chip-outline {{ background: #fff; color: {INK}; border: 1.5px dashed {INK_SOFT}; }}
.chip-plain {{ background: #fff; color: {INK}; border: 1.5px solid {LINE}; }}

.finding {{ border-top: 3px solid {INK}; padding-top: 0.8rem; }}
.finding .figure {{ font-size: 2.1rem; font-weight: 700; line-height: 1.1; color: {INK}; }}
.finding p {{ margin: 0.35rem 0 0; color: {INK_SOFT}; line-height: 1.5; }}

/* Étiquette « Valeurs nutritionnelles », dans l'esprit du tableau imprimé sur l'emballage */
.label {{
    border: 2px solid {INK}; background: #fff; padding: 0.7rem 0.9rem 0.5rem;
    max-width: 470px; color: {INK};
}}
.label h4 {{
    margin: 0; font-size: 1.35rem; font-weight: 700; border-bottom: 8px solid {INK};
    padding-bottom: 0.25rem;
}}
.label .per {{ font-size: 0.9rem; padding: 0.25rem 0; border-bottom: 3px solid {INK}; }}
.label table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.label td {{ padding: 0.32rem 0; border-bottom: 1px solid {INK}; font-size: 0.98rem; }}
.label td.v {{ text-align: right; font-weight: 700; white-space: nowrap; }}
.label td.lvl {{ text-align: right; width: 6.5rem; font-size: 0.85rem; padding-right: 0.2rem;
                 white-space: nowrap; }}
.label tr.sub td:first-child {{ padding-left: 1.1rem; }}
.label tr:last-child td {{ border-bottom: none; }}
@media (max-width: 520px) {{
    .label {{ padding: 0.6rem 0.6rem 0.4rem; }}
    .label td {{ font-size: 0.9rem; }}
    .label td.lvl {{ width: 4.8rem; font-size: 0.76rem; }}
    .label .dot {{ margin-right: 0.2rem; }}
}}
.dot {{ display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 50%;
        margin-right: 0.35rem; vertical-align: middle; }}
.reasons li {{ margin-bottom: 0.45rem; line-height: 1.45; }}
a, a:visited {{ color: {ACCENT}; }}
:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
</style>
"""


def esc(value) -> str:
    return html.escape(str(value))


def is_missing(value) -> bool:
    """Vrai pour None, NaN et pd.NA ; faux pour les listes et tableaux."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # séquence : pd.isna renvoie un tableau
        return False


def nutriscore_chip(grade) -> str:
    if is_missing(grade):
        return '<span class="chip chip-plain">Nutri-Score non calculé</span>'
    g = str(grade).lower()
    return (
        f'<span class="chip" style="background:{NUTRISCORE_COLORS[g]};'
        f'color:{NUTRISCORE_TEXT[g]}">Nutri-Score {g.upper()}</span>'
    )


def nova_chip(group, estimated=None, confidence=None) -> str:
    if not is_missing(group):
        n = int(group)
        return (
            f'<span class="chip" style="background:{NOVA_COLORS[n]};color:{NOVA_TEXT[n]}">'
            f"NOVA {n} <small>{NOVA_SHORT[n]}</small></span>"
        )
    if not is_missing(estimated):
        n = int(estimated)
        conf = "" if is_missing(confidence) else f", confiance {round(100 * confidence)} %"
        return (
            f'<span class="chip chip-outline">NOVA {n} estimé '
            f"<small>{NOVA_SHORT[n]}{conf}</small></span>"
        )
    return '<span class="chip chip-plain">NOVA inconnu</span>'


def plain_chip(text: str) -> str:
    return f'<span class="chip chip-plain">{esc(text)}</span>'


def chips(*items: str) -> str:
    return f'<div class="chips">{"".join(items)}</div>'


def nutrient_level(column: str, value) -> str | None:
    """'faible', 'modéré' ou 'élevé' pour les nutriments qui ont des repères."""
    if column not in NUTRIENT_LEVELS or is_missing(value):
        return None
    low, high = NUTRIENT_LEVELS[column]
    if value <= low:
        return "faible"
    return "modéré" if value <= high else "élevé"


LABEL_ROWS = [
    ("energy_kcal_100g", "Énergie", "kcal", False),
    ("fat_100g", "Matières grasses", "g", False),
    ("saturated_fat_100g", "dont acides gras saturés", "g", True),
    ("carbohydrates_100g", "Glucides", "g", False),
    ("sugars_100g", "dont sucres", "g", True),
    ("fiber_100g", "Fibres alimentaires", "g", False),
    ("proteins_100g", "Protéines", "g", False),
    ("salt_100g", "Sel", "g", False),
]


def _number(value, unit: str) -> str:
    if is_missing(value):
        return "non renseigné"
    decimals = 0 if unit == "kcal" else (2 if value < 1 else 1)
    return f"{value:.{decimals}f}".replace(".", ",") + f" {unit}"


def nutrition_label(product: dict) -> str:
    """Tableau des valeurs nutritionnelles pour 100 g, avec un niveau pour 4 nutriments."""
    rows = []
    for column, name, unit, sub in LABEL_ROWS:
        value = product.get(column)
        level = nutrient_level(column, value)
        cell = ""
        if level:
            color, text = LEVEL_STYLE[level]
            cell = f'<span class="dot" style="background:{color}"></span>{text}'
        rows.append(
            f'<tr class="{"sub" if sub else ""}"><td>{name}</td>'
            f'<td class="v">{_number(value, unit)}</td><td class="lvl">{cell}</td></tr>'
        )
    return (
        '<div class="label"><h4>Valeurs nutritionnelles</h4>'
        f'<div class="per">pour 100 g</div><table>{"".join(rows)}</table></div>'
    )


def finding(figure: str, text: str) -> str:
    return f'<div class="finding"><div class="figure">{esc(figure)}</div><p>{esc(text)}</p></div>'


__all__ = ["CSS", "SURFACE", "chips", "finding", "nova_chip", "nutriscore_chip",
           "nutrition_label", "nutrient_level", "plain_chip", "esc"]  # fmt: skip
