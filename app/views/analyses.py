import streamlit as st

from nutrilys import config
from nutrilys.app import figures, state

st.title("Analyses")
st.markdown(
    '<p class="lead">Les résultats principaux des notebooks, recalculés sur les données '
    "actuelles. Chaque titre énonce le constat ; le détail des méthodes est dans le dépôt.</p>",
    unsafe_allow_html=True,
)
state.require(config.PRODUCTS_PATH, "uv run python -m nutrilys.data.clean")
paths = figures.all_paths()

quality, organic, controversy, models = st.tabs(
    ["Nutri-Score et NOVA", "Bio et conventionnel", "Controverses", "Modèles"]
)
with quality:
    for fig in figures.quality_figures(paths["products"]):
        st.plotly_chart(fig, width="stretch")
with organic:
    st.markdown(
        "Le bio est sur-représenté dans des rayons naturellement mieux notés. La comparaison "
        "de droite recalcule les taux comme si bio et conventionnel avaient la même "
        "répartition entre rayons : c'est elle qui mesure l'effet du label."
    )
    organic_figs = figures.organic_figures(paths["products"])
    if not organic_figs:
        st.info("Pas assez de produits bio par rayon pour une comparaison fiable.")
    for fig in organic_figs:
        st.plotly_chart(fig, width="stretch")
with controversy:
    figs = figures.controversy_figures(paths["products"], paths["controversies"])
    if not figs:
        st.info("Signaux non calculés : `uv run python -m nutrilys.analysis.controversies`")
    for fig in figs:
        st.plotly_chart(fig, width="stretch")
with models:
    fig = figures.model_figure(paths["reports"])
    if fig is None:
        st.info("Modèles non entraînés : `uv run python -m nutrilys.models.baseline --models lr`")
    else:
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "Les scénarios avec ingrédients ou nutriments servent de contrôle : NOVA et "
            "Nutri-Score sont calculés à partir de ces informations, le modèle retrouve "
            "la règle. Les scénarios « texte seul » mesurent ce que la description permet "
            "de deviner."
        )
