"""Serve measured QA evidence without importing inference or training code."""
from pathlib import Path
import streamlit as st


def render():
    root = Path(__file__).resolve().parents[1]
    report = root / 'reports/ACCESSIBILITY.md'
    st.markdown(report.read_text())
    st.download_button('Download the accessibility report', report.read_text(),
                       file_name='Car-Price-Studio-Accessibility.md', mime='text/markdown',key='accessibility_report_md')
    raw = root / 'reports/accessibility-responsive-chromium-app.json'
    st.download_button('Download raw Chromium results', raw.read_text(),
                       file_name='Car-Price-Studio-Accessibility.json', mime='application/json',key='accessibility_report_json')
    webkit = root / 'reports/accessibility-responsive-webkit-app.json'
    st.download_button('Download raw WebKit results', webkit.read_text(),
                       file_name='Car-Price-Studio-WebKit-Accessibility.json', mime='application/json',key='accessibility_report_webkit_json')
    firefox = root / 'reports/accessibility-responsive-firefox-app.json'
    st.download_button('Download raw Firefox results', firefox.read_text(),
                       file_name='Car-Price-Studio-Firefox-Accessibility.json', mime='application/json',key='accessibility_report_firefox_json')
