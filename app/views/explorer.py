import math

import streamlit as st

from nutrilys import config
from nutrilys.analysis.nutrition import FOOD_GROUP_LABELS
from nutrilys.app import state, ui
from nutrilys.app.queries import Filters, search
from nutrilys.viz.fmt import fmt_int

path = str(config.APP_PRODUCTS_PATH)
con = state.app_cursor()

st.title("Explorer les produits")

with st.sidebar:
    st.header("Filtres")
    text = st.text_input("Recherche", value=st.query_params.get("q", ""),
                         placeholder="Nom, marque ou code-barres")  # fmt: skip
    groups = state.food_groups(path)
    group = st.selectbox(
        "Rayon", [None, *groups], format_func=lambda g: "Tous les rayons" if g is None
        else FOOD_GROUP_LABELS.get(g, g),
    )  # fmt: skip
    brand = st.selectbox(
        "Marque", [None, *state.top_brands(path)],
        format_func=lambda b: "Toutes les marques" if b is None else b,
        help="Les 500 marques les plus représentées. Pour une autre marque, tapez son nom "
             "dans la recherche.",
    )  # fmt: skip
    nutriscore = st.multiselect(
        "Nutri-Score", list("abcde"), format_func=str.upper, placeholder="Toutes les notes"
    )
    nova = st.multiselect("Groupe NOVA", [1, 2, 3, 4], placeholder="Tous les groupes")
    include_estimated = st.checkbox(
        "Inclure les NOVA estimés", help="Pour les produits sans liste d'ingrédients, "
        "le groupe NOVA est estimé par un modèle à partir du nom et des catégories."
    )  # fmt: skip
    organic = st.toggle("Produits bio uniquement")
    flagged = st.toggle("Avec un signal de controverse")
    st.selectbox("Pays", ["France"], disabled=True, help="Autres pays en phase 2.")

filters = Filters(
    text=text, food_group=group, brand=brand, nutriscore=nutriscore, nova=nova,
    include_estimated=include_estimated, organic_only=organic, flagged_only=flagged,
)  # fmt: skip

page_size = config.APP_PAGE_SIZE
_, total = search(con, filters, limit=1)
pages = max(1, math.ceil(total / page_size))

head_left, head_right = st.columns([3, 1], vertical_alignment="bottom")
head_left.markdown(f"**{fmt_int(total)}** produit{'s' if total > 1 else ''}, les plus consultés "
                   "d'abord. Sélectionnez une ligne pour ouvrir la fiche.")  # fmt: skip
page = head_right.number_input("Page", min_value=1, max_value=pages, value=1, step=1)

if total == 0:
    st.info("Aucun produit ne correspond. Essayez un mot plus court ou retirez un filtre.")
    st.stop()

rows, _ = search(con, filters, limit=page_size, offset=(page - 1) * page_size)
table = rows.assign(
    nutriscore=rows["nutriscore_grade"].str.upper().fillna(""),
    nova=[
        str(int(g))
        if not ui.is_missing(g)
        else (f"{int(e)} (estimé)" if not ui.is_missing(e) else "")
        for g, e in zip(rows["nova_group"], rows["nova_estimated"], strict=True)
    ],
    rayon=rows["food_group"].map(FOOD_GROUP_LABELS).fillna(""),
    bio=rows["is_organic"].map({True: "Bio", False: ""}),
    signaux=rows["controversy_score"].where(rows["controversy_score"] > 0),
)
event = st.dataframe(
    table,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_order=["product_name", "brand", "rayon", "nutriscore", "nova", "bio",
                  "n_additives", "signaux"],
    column_config={
        "product_name": st.column_config.TextColumn("Produit", width="large"),
        "brand": "Marque",
        "rayon": "Rayon",
        "nutriscore": st.column_config.TextColumn("Nutri-Score", width="small"),
        "nova": st.column_config.TextColumn("NOVA", width="small"),
        "bio": st.column_config.TextColumn("Bio", width="small"),
        "n_additives": st.column_config.NumberColumn("Additifs", format="%d", width="small"),
        "signaux": st.column_config.NumberColumn(
            "Signaux", format="%d", width="small",
            help="Nombre de signaux de controverse (0 à 5)",
        ),
    },
)  # fmt: skip

selected = event.selection.rows if event and event.selection else []
if selected:
    st.switch_page("views/fiche.py", query_params={"code": rows.iloc[selected[0]]["code"]})
