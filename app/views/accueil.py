import streamlit as st

from nutrilys import config
from nutrilys.app import state, ui
from nutrilys.viz.fmt import fmt_int, fmt_pct

state.app_cursor()  # vérifie que la table de l'application existe
figures = state.key_figures(str(config.APP_PRODUCTS_PATH))

st.title("NUTRILYS")
st.markdown(
    f'<p class="lead">Ce que disent vraiment les étiquettes de {fmt_int(figures["products"])} '
    "produits alimentaires vendus en France : qualité nutritionnelle, degré de "
    "transformation, additifs et rappels, à partir des données ouvertes "
    "d'Open Food Facts.</p>",
    unsafe_allow_html=True,
)

with st.form("search", border=False):
    col_input, col_button = st.columns([5, 1], vertical_alignment="bottom")
    query = col_input.text_input(
        "Rechercher un produit",
        placeholder="Nom, marque ou code-barres : pâte à tartiner, Bonne Maman, 3017620422003",
    )
    if col_button.form_submit_button("Rechercher", type="primary", width="stretch"):
        st.switch_page("views/explorer.py", query_params={"q": query} if query else None)

st.write("")
c1, c2, c3 = st.columns(3, gap="large")
c1.markdown(
    ui.finding(
        fmt_pct(figures["share_nova4"]),
        "des produits dont on connaît les ingrédients sont ultra-transformés (NOVA 4).",
    ),
    unsafe_allow_html=True,
)
c2.markdown(
    ui.finding(
        fmt_pct(figures["share_de"]),
        "des produits notés ont un Nutri-Score D ou E, les deux moins bonnes notes.",
    ),
    unsafe_allow_html=True,
)
c3.markdown(
    ui.finding(
        fmt_int(figures["flagged"]),
        "produits présentent au moins un signal de controverse : bonne note mais "
        "ultra-transformé, additif à risque, rappel...",
    ),
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Deux scores, deux questions")
left, right = st.columns(2, gap="large")
left.markdown(
    "**Le Nutri-Score** répond à la question *« Que contient ce produit ? »*. "
    "Il note de A à E la composition pour 100 g : sucre, sel, graisses saturées et "
    "énergie d'un côté, fibres, protéines, fruits et légumes de l'autre."
)
right.markdown(
    "**La classification NOVA** répond à *« Comment ce produit a-t-il été fabriqué ? »*. "
    "Elle va de 1 (aliment brut) à 4 (ultra-transformé : arômes, émulsifiants, édulcorants...). "
    "Un produit peut être bien noté par l'un et mal par l'autre."
)
st.page_link("views/analyses.py", label="Voir les analyses détaillées")

st.divider()
st.markdown(
    '<p class="muted">Sources : Open Food Facts (licence ODbL), RappelConso (DGCCRF), '
    "évaluations des additifs de l'EFSA. Les données sont collaboratives : une information "
    "absente ne signifie pas qu'elle n'existe pas. Projet d'analyse SoGood.</p>",
    unsafe_allow_html=True,
)
