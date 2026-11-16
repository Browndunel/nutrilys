"""Point d'entrée de l'application NUTRILYS.

Lancement :
    uv run streamlit run app/app.py
"""

from pathlib import Path

import streamlit as st

from nutrilys.app.ui import CSS

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="NUTRILYS", layout="wide", initial_sidebar_state="auto")
st.markdown(CSS, unsafe_allow_html=True)
st.logo(str(ASSETS / "logo_horizontal.png"), icon_image=str(ASSETS / "logo_icon.png"), size="large")

pages = [
    st.Page("views/accueil.py", title="Accueil", default=True),
    st.Page("views/explorer.py", title="Explorer", url_path="explorer"),
    st.Page("views/analyses.py", title="Analyses", url_path="analyses"),
    st.Page("views/predire.py", title="Prédire", url_path="predire"),
    st.Page("views/fiche.py", title="Fiche produit", url_path="fiche", visibility="hidden"),
]
st.navigation(pages, position="top").run()
