"""Minimal Streamlit UI fixture: production i18n + production gateway, no ML model load."""
import streamlit as st
st.set_page_config(page_title="Car Price Lab · language test", layout="wide")
from pricing.i18n import render_language_selector, locale, tr
from pricing.international_gateway import render_market_gateway

render_language_selector()
selected = render_market_gateway()
st.markdown("### Smoke market: " + selected)
st.markdown("### Smoke language: " + locale())
st.caption(tr("calculate"))
