"""Consent-component UI fixture (no real analytics, database or geolocation).

Uses the same first-party HTML, JavaScript and translations as production;
the GA tag is intentionally invalid and geolocation is disabled for tests.
"""
from pathlib import Path
import streamlit as st
st.set_page_config(page_title="Consent translation smoke")
from pricing.i18n import render_language_selector, locale
from pricing.i18n_consent import modal_labels
from pricing.i18n_science_legal import copy
render_language_selector()
st.write("Consent locale: "+locale())
folder=Path(__file__).resolve().parents[1]/"pricing"/"consent"
component=st.components.v2.component(
    "consent_translation_ui_test",
    html=(folder/"notice.html").read_text(encoding="utf-8"),
    js=(folder/"notice.js").read_text(encoding="utf-8"),
    css=(folder/"notice.css").read_text(encoding="utf-8"),
)
component(key="consent_translation_ui_test",data={
    "enabled":True,
    "version":"consent-i18n-smoke-only",
    "current":None,
    "google_tag_id":"INVALID",
    "geo_mode":"local",
    "backend_status":"test",
    "controller":"Car Price Studio test",
    "contact":"test@example.invalid",
    "labels":{**modal_labels(),
              **{k:copy(k) for k in (
                  "welcome","service","consent","retention","sharing",
                  "rights","reference","original","operator")},
              "contact_copy":copy("contact")},
},on_consent_change=lambda:None)
