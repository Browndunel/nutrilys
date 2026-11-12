"""Graphiques réutilisables (notebooks et application).

Chaque fonction reçoit un DataFrame déjà agrégé et renvoie une figure Plotly.
Le titre énonce le constat ; il est fourni par l'appelant, qui connaît les chiffres.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nutrilys.viz.theme import AXIS, INK_SECONDARY, PRIMARY, SURFACE


def _title(title: str, subtitle: str | None) -> dict:
    return dict(text=title, subtitle=dict(text=subtitle or ""))


def category_bars(
    df: pd.DataFrame,
    category: str,
    value: str,
    colors: dict,
    labels: dict | None = None,
    title: str = "",
    subtitle: str | None = None,
    value_label: str = "Part des produits (%)",
    counts: str | None = None,
) -> go.Figure:
    """Barres verticales, une couleur par modalité (Nutri-Score, NOVA)."""
    labels = labels or {}
    names = [labels.get(c, str(c).upper()) for c in df[category]]
    hover = "%{x}<br>%{y:.1f} %"
    if counts:
        hover += "<br>%{customdata:,.0f} produits"
    fig = go.Figure(
        go.Bar(
            x=names,
            y=df[value],
            marker_color=[colors[c] for c in df[category]],
            text=[f"{v:.0f} %" for v in df[value]],
            textposition="outside",
            textfont=dict(color=INK_SECONDARY),
            customdata=df[counts] if counts else None,
            hovertemplate=hover + "<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(title=_title(title, subtitle), height=420, showlegend=False)
    fig.update_yaxes(title_text=value_label, rangemode="tozero")
    fig.update_xaxes(showgrid=False)
    return fig


def stacked_shares(
    df: pd.DataFrame,
    row: str,
    segment: str,
    share: str,
    colors: dict,
    segment_order: list,
    row_order: list,
    segment_labels: dict | None = None,
    row_labels: dict | None = None,
    title: str = "",
    subtitle: str | None = None,
    min_label: float = 6.0,
) -> go.Figure:
    """Barres horizontales empilées à 100 % : répartition des segments dans chaque ligne.

    Les parts inférieures à `min_label` % ne sont pas étiquetées, pour éviter le bruit.
    """
    segment_labels = segment_labels or {}
    row_labels = row_labels or {}
    y = [row_labels.get(r, str(r)) for r in row_order]
    fig = go.Figure()
    for seg in segment_order:
        sub = df[df[segment] == seg].set_index(row).reindex(row_order)
        values = sub[share].fillna(0)
        name = segment_labels.get(seg, str(seg).upper())
        fig.add_bar(
            y=y,
            x=values,
            name=name,
            orientation="h",
            marker_color=colors[seg],
            text=[f"{v:.0f}" if v >= min_label else "" for v in values],
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"%{{y}}<br>{name} : %{{x:.1f}} %<extra></extra>",
        )
    fig.update_layout(
        barmode="stack",
        title=_title(title, subtitle),
        height=max(340, 130 + 42 * len(row_order)),
        margin=dict(t=125),
        legend=dict(traceorder="normal"),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def ranked_bars(
    df: pd.DataFrame,
    label: str,
    value: str,
    title: str = "",
    subtitle: str | None = None,
    value_label: str = "",
    value_format: str = ".1f",
    suffix: str = "",
    color: str = PRIMARY,
    text: str | None = None,
) -> go.Figure:
    """Barres horizontales triées, une seule série (classements).

    text : colonne d'étiquettes déjà formatées ; à défaut, la valeur formatée.
    """
    data = df.sort_values(value, ascending=True)
    labels = (
        data[text].tolist()
        if text
        else [f"{v:{value_format}}".replace(".", ",") + suffix for v in data[value]]
    )
    fig = go.Figure(
        go.Bar(
            y=data[label],
            x=data[value],
            orientation="h",
            marker_color=color,
            text=labels,
            textposition="outside",
            textfont=dict(color=INK_SECONDARY),
            hovertemplate=f"%{{y}}<br>%{{x:{value_format}}}{suffix}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        title=_title(title, subtitle),
        height=max(320, 70 + 30 * len(data)),
        showlegend=False,
    )
    fig.update_xaxes(title_text=value_label, rangemode="tozero")
    fig.update_yaxes(showgrid=False)
    return fig


def dumbbell_panels(
    df: pd.DataFrame,
    label: str,
    panels: list[tuple[str, str, str]],
    names: tuple[str, str],
    colors: tuple[str, str],
    title: str = "",
    subtitle: str | None = None,
    suffix: str = " %",
) -> go.Figure:
    """Haltères côte à côte : deux valeurs par ligne, un panneau par calcul.

    panels : liste de (titre du panneau, colonne de la 1re série, colonne de la 2e).
    """
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.06,
    )
    y = df[label].tolist()
    for col, (_, first, second) in enumerate(panels, start=1):
        for _, row in df.iterrows():
            fig.add_scatter(
                x=[row[first], row[second]],
                y=[row[label]] * 2,
                mode="lines",
                line=dict(color=AXIS, width=2),
                hoverinfo="skip",
                showlegend=False,
                row=1,
                col=col,
            )
        for column, name, color in ((first, names[0], colors[0]), (second, names[1], colors[1])):
            fig.add_scatter(
                x=df[column],
                y=y,
                mode="markers+text",
                name=name,
                marker=dict(color=color, size=12, line=dict(color=SURFACE, width=2)),
                text=[f"{v:.0f}" for v in df[column]],
                textposition="top center",
                textfont=dict(color=INK_SECONDARY, size=11),
                hovertemplate=f"%{{y}}<br>{name} : %{{x:.1f}}{suffix}<extra></extra>",
                legendgroup=name,
                showlegend=col == 1,
                row=1,
                col=col,
            )
        fig.update_xaxes(ticksuffix=suffix, rangemode="tozero", row=1, col=col)
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(color=INK_SECONDARY, size=13), x=annotation.x)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(
        title=_title(title, subtitle),
        height=max(380, 190 + 60 * len(df)),
        margin=dict(t=120, b=70),
        # Légende sous le graphique : au-dessus, elle gênerait les titres des panneaux
        legend=dict(y=-0.12, yanchor="top", x=0.5, xanchor="center"),
    )
    return fig


def effect_dots(
    df: pd.DataFrame,
    label: str,
    effect: str,
    category: str,
    colors: dict,
    title: str = "",
    subtitle: str | None = None,
    hover: str | None = None,
    x_title: str = "Taille d'effet",
) -> go.Figure:
    """Une taille d'effet par ligne, centrée sur zéro, colorée par verdict."""
    data = df.sort_values(effect)
    fig = go.Figure()
    for cat, color in colors.items():
        sub = data[data[category] == cat]
        if sub.empty:
            continue
        fig.add_scatter(
            x=sub[effect],
            y=sub[label],
            mode="markers",
            name=cat,
            marker=dict(color=color, size=11, line=dict(color=SURFACE, width=2)),
            customdata=sub[hover] if hover else None,
            hovertemplate="%{y}<br>effet : %{x:.2f}"
            + ("<br>%{customdata}" if hover else "")
            + "<extra></extra>",
        )
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.5))
    fig.update_xaxes(range=[-1.05, 1.05], title_text=x_title, zeroline=False)
    fig.update_yaxes(showgrid=True, categoryorder="array", categoryarray=data[label].tolist())
    fig.update_layout(
        title=_title(title, subtitle),
        height=max(360, 150 + 26 * len(data)),
        margin=dict(t=125),
    )
    return fig
