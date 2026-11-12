"""Thème graphique commun aux notebooks et à l'application.

Couleurs des scores :
- Nutri-Score : couleurs officielles du logo, reconnues par le public. Elles ne
  séparent pas assez D et E pour tous les lecteurs : la lettre est donc toujours
  affichée à côté de la couleur (axe ou étiquette), jamais la couleur seule.
- NOVA : pas de charte officielle. Feu tricolore re-dosé pour rester distinguable
  par les lecteurs daltoniens (palette validée : écart CVD minimal 9,7, écart
  normal minimal 18,5). Le groupe est lui aussi toujours écrit.
"""

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from nutrilys import config

# Surfaces et encres
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Palette catégorielle, ordre fixe (validé pour les daltonismes)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PRIMARY = SERIES[0]

NUTRISCORE_COLORS = {"a": "#038141", "b": "#85bb2f", "c": "#fecb02", "d": "#ee8100", "e": "#e63e11"}
NOVA_COLORS = {1: "#1b9e4b", 2: "#d6aa00", 3: "#e05a14", 4: "#96182f"}

NOVA_LABELS = {
    1: "1 - Brut ou peu transformé",
    2: "2 - Ingrédient culinaire",
    3: "3 - Transformé",
    4: "4 - Ultra-transformé",
}

# Texte lisible posé sur chaque couleur de score
NUTRISCORE_TEXT = {"a": "#ffffff", "b": INK, "c": INK, "d": INK, "e": "#ffffff"}
NOVA_TEXT = {1: "#ffffff", 2: INK, 3: "#ffffff", 4: "#ffffff"}

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
TEMPLATE_NAME = "nutrilys"


def _template() -> go.layout.Template:
    axis = dict(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        linecolor=AXIS,
        showline=True,
        zeroline=False,
        ticks="",
        tickfont=dict(color=INK_SECONDARY, size=12),
        title=dict(font=dict(color=INK_SECONDARY, size=12)),
        automargin=True,
    )
    layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        title=dict(
            x=0,
            xanchor="left",
            xref="paper",
            font=dict(size=17, color=INK),
            subtitle=dict(font=dict(size=12, color=INK_SECONDARY)),
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES,
        separators=", ",
        xaxis=axis,
        yaxis=axis,
        bargap=0.3,
        barcornerradius=4,
        legend=dict(
            orientation="h",
            traceorder="normal",
            x=0,
            xanchor="left",
            y=1.0,
            yanchor="bottom",
            title=dict(text=""),
            font=dict(color=INK_SECONDARY, size=12),
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        margin=dict(l=10, r=30, t=95, b=40),
    )
    # Fin liseré de la couleur du fond entre segments empilés et barres voisines
    bar = go.Bar(marker=dict(line=dict(color=SURFACE, width=1.5)))
    return go.layout.Template(layout=layout, data=dict(bar=[bar]))


def use_theme() -> None:
    """Enregistre le thème et l'active par défaut pour toutes les figures."""
    pio.templates[TEMPLATE_NAME] = _template()
    pio.templates.default = TEMPLATE_NAME


def save_figure(fig: go.Figure, name: str, directory: Path = config.FIGURES_DIR) -> Path:
    """Enregistre la figure en HTML interactif (Plotly chargé depuis un CDN)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path
