"""Minimal Streamlit UI fixture: production i18n + production gateway, no ML model load."""
import streamlit as st
st.set_page_config(page_title="Car Price Lab · language test", layout="wide")
from pricing.i18n import render_language_selector, locale, tr
from pricing.international_gateway import render_market_gateway
from pricing.studio_design import render_hero

render_language_selector()
selected = render_market_gateway()
render_hero()
st.markdown("### Smoke market: " + selected)
st.markdown("### Smoke language: " + locale())
st.caption(tr("calculate"))

# Real production locale strings and a minimal form interaction; no ML predictions.
with st.form("i18n_smoke_form"):
    st.text_input(tr("make"), key="smoke_make")
    if st.form_submit_button(tr("calculate")):
        st.caption("Smoke form submitted: " + locale())
