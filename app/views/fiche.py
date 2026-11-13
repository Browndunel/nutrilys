import streamlit as st

from nutrilys import config
from nutrilys.analysis.bio import subgroup_label
from nutrilys.analysis.nutrition import FOOD_GROUP_LABELS
from nutrilys.app import state, ui
from nutrilys.app.queries import product

code = st.query_params.get("code")
if not code:
    st.info("Choisissez un produit dans l'explorateur pour afficher sa fiche.")
    st.page_link("views/explorer.py", label="Aller à l'explorateur")
    st.stop()

p = product(state.app_cursor(), code)
if p is None:
    st.warning(f"Aucun produit avec le code-barres {code}.")
    st.page_link("views/explorer.py", label="Revenir à l'explorateur")
    st.stop()

st.page_link("views/explorer.py", label="Retour aux résultats")
st.title(p["product_name"])
context = [p.get("brand")]
if not ui.is_missing(p.get("food_subgroup")):
    context.append(subgroup_label(p["food_subgroup"]))
elif not ui.is_missing(p.get("food_group")):
    context.append(FOOD_GROUP_LABELS.get(p["food_group"], p["food_group"]))
context.append(f"code-barres {p['code']}")
st.markdown(
    f'<p class="muted">{ui.esc(", ".join(c for c in context if c and not ui.is_missing(c)))}</p>',
    unsafe_allow_html=True,
)

badges = [
    ui.nutriscore_chip(p.get("nutriscore_grade")),
    ui.nova_chip(p.get("nova_group"), p.get("nova_estimated"), p.get("nova_confidence")),
]
if p.get("is_organic"):
    badges.append(ui.plain_chip("Label bio"))
if not ui.is_missing(p.get("green_score_grade")):
    badges.append(ui.plain_chip(f"Green-Score {str(p['green_score_grade']).upper()}"))
st.markdown(ui.chips(*badges), unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="large")
with left:
    st.markdown(ui.nutrition_label(p), unsafe_allow_html=True)
    st.caption("Repères faible, modéré, élevé pour 100 g, utilisés par Open Food Facts.")

with right:
    st.subheader("Signaux de controverse")
    if p.get("controversy_score", 0) > 0 and isinstance(p.get("reasons"), str):
        reasons = [r.strip().rstrip(".") for r in p["reasons"].split(" ; ")]
        items = "".join(f"<li>{ui.esc(r[:1].upper() + r[1:])}.</li>" for r in reasons)
        st.markdown(f'<ul class="reasons">{items}</ul>', unsafe_allow_html=True)
    else:
        st.markdown("Aucun signal pour ce produit.")

    st.subheader("Additifs")
    codes = list(p.get("additives") if p.get("additives") is not None else [])
    if len(codes) == 0:
        known = not ui.is_missing(p.get("n_additives"))
        st.markdown("Aucun additif détecté." if known else
                    "Liste d'ingrédients absente : additifs inconnus.")  # fmt: skip
    else:
        ref = state.additives_table(str(config.ADDITIVES_PATH))
        rows = [
            {
                "Additif": ref["name_fr"].get(c, c) if not ref.empty else c,
                "Risque de surexposition (EFSA)": (ref["risk_label"].get(c) if not ref.empty
                                                   else None) or "Non évalué",
            }
            for c in codes
        ]  # fmt: skip
        st.dataframe(rows, hide_index=True, width="stretch")

    if not ui.is_missing(p.get("ingredients_text")):
        with st.expander("Liste des ingrédients"):
            st.write(p["ingredients_text"])

st.link_button(
    "Voir la fiche sur Open Food Facts", f"https://world.openfoodfacts.org/product/{p['code']}"
)
