"""Graphiques réutilisables (notebooks et application).

Chaque fonction reçoit un DataFrame déjà agrégé et renvoie une figure Plotly.
Le titre énonce le constat ; il est fourni par l'appelant, qui connaît les chiffres.
"""

import pandas as pd
import plotly.graph_objects as go

from nutrilys.viz.theme import INK_SECONDARY, PRIMARY


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
) -> go.Figure:
    """Barres horizontales triées, une seule série (classements)."""
    data = df.sort_values(value, ascending=True)
    fig = go.Figure(
        go.Bar(
            y=data[label],
            x=data[value],
            orientation="h",
            marker_color=color,
            text=[f"{v:{value_format}}".replace(".", ",") + suffix for v in data[value]],
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
