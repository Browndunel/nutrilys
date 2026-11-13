import pandas as pd
import streamlit as st

from nutrilys import config
from nutrilys.app import state, ui
from nutrilys.models.baseline import explain_prediction
from nutrilys.models.dataset import build_text
from nutrilys.viz import charts
from nutrilys.viz.fmt import fmt_pct
from nutrilys.viz.theme import NOVA_COLORS, NOVA_LABELS, NUTRISCORE_COLORS, use_theme

use_theme()
st.title("Prédire à partir d'une description")
st.markdown(
    '<p class="lead">Les modèles entraînés sur Open Food Facts estiment le groupe NOVA ou le '
    "Nutri-Score d'un produit à partir de son seul texte. Ce sont des estimations, utiles "
    "quand la fiche est incomplète, pas des mesures.</p>",
    unsafe_allow_html=True,
)


def frame(name: str, categories: str, ingredients: str = "") -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "product_name": [name],
            "generic_name": [None],
            "categories_tags": [[c.strip() for c in categories.split(",") if c.strip()]],
            "ingredients_text": [ingredients or None],
        }
    )
    return df


def show(fitted, df, colors, labels, classes_label):
    proba = fitted.predict_proba(df)[0]
    result = pd.DataFrame({"cls": fitted.classes, "share": 100 * proba})
    best = result.loc[result["share"].idxmax()]
    fig = charts.category_bars(
        result, "cls", "share", colors, labels=labels,
        title=f"{classes_label} {labels.get(best['cls'], str(best['cls']).upper())}, "
              f"probabilité {fmt_pct(best['share'])}",
        subtitle="Probabilité de chaque classe selon le modèle",
        value_label="Probabilité (%)",
    )  # fmt: skip
    st.plotly_chart(fig, width="stretch")
    terms = explain_prediction(fitted, df)
    if not terms.empty:
        st.markdown("**Mots qui ont le plus pesé** : " + ", ".join(
            f"« {ui.esc(t)} »" for t in terms["term"]))  # fmt: skip


nova_tab, ns_tab = st.tabs(["Degré de transformation (NOVA)", "Nutri-Score"])

with nova_tab:
    fitted = state.model("nova_name")
    if fitted is None:
        st.info("Modèle absent : `uv run python -m nutrilys.models.baseline --models lr`")
    else:
        with st.form("nova"):
            name = st.text_input("Nom du produit", placeholder="Nuggets de poulet panés")
            cats = st.text_input("Catégories (facultatif, séparées par des virgules)",
                                 placeholder="plats préparés, volailles")  # fmt: skip
            go = st.form_submit_button("Estimer le groupe NOVA", type="primary")
        if go and name.strip():
            df = frame(name, cats)
            df["text"] = build_text(df, with_ingredients=False)
            labels = {g: NOVA_LABELS[g] for g in config.NOVA_GROUPS}
            show(fitted, df, NOVA_COLORS, labels, "Groupe")
        elif go:
            st.warning("Saisissez au moins le nom du produit.")

with ns_tab:
    fitted = state.model("nutriscore_text")
    if fitted is None:
        st.info("Modèle absent : `uv run python -m nutrilys.models.baseline --models lr`")
    else:
        with st.form("nutriscore"):
            name = st.text_input("Nom du produit", placeholder="Biscuits fourrés au chocolat")
            cats = st.text_input("Catégories (facultatif)", placeholder="biscuits, goûters")
            ingredients = st.text_area(
                "Liste des ingrédients", placeholder="Farine de blé, sucre, huile de palme..."
            )
            go = st.form_submit_button("Estimer le Nutri-Score", type="primary")
        if go and name.strip():
            df = frame(name, cats, ingredients)
            df["text"] = build_text(df, with_ingredients=True)
            show(fitted, df, NUTRISCORE_COLORS, {}, "Nutri-Score")
        elif go:
            st.warning("Saisissez au moins le nom du produit.")
