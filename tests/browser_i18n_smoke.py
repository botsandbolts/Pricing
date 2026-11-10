"""Browser smoke: real Streamlit components-v2/localStorage and gateway buttons.

Runs against tests/i18n_smoke_app.py without loading or retraining price models.
"""
import os
from playwright.sync_api import sync_playwright

BASE = os.environ.get("I18N_SMOKE_URL", "http://localhost:8776")
TIMEOUT = 45000

def expect_text(page, value):
    page.get_by_text(value, exact=False).first.wait_for(timeout=TIMEOUT)

def select_language(page, label):
    # Streamlit 1.66: first combobox is the top-of-page language selector.
    page.get_by_role("combobox").first.click()
    page.get_by_text(label, exact=True).last.click()

def smoke(engine):
    browser = engine.launch(headless=True)
    try:
        polish = browser.new_context(locale="pl-PL", viewport={"width": 1280, "height": 900})
        page = polish.new_page()
        page.goto(BASE, wait_until="domcontentloaded")
        expect_text(page, "Szwagier zna cenę.")
        expect_text(page, "Smoke language: pl")
        select_language(page, "🇪🇸 Español")
        expect_text(page, "Tu cuñado pone el precio.")
        expect_text(page, "Smoke language: es")
        page.wait_for_function("localStorage.getItem('car-price-studio.language.v1') === 'es'")
        page.reload(wait_until="domcontentloaded")
        expect_text(page, "Smoke language: es")
        # Market selection must not change language preference or induce locale redirection.
        page.get_by_role("button", name="🇪🇺 Abrir estudio europeo").click()
        expect_text(page, "Smoke market: eu")
        expect_text(page, "Smoke language: es")
        page.screenshot(path=f"/tmp/i18n_{engine.name}_es.png", full_page=True)
        polish.close()

        american = browser.new_context(locale="en-US")
        us = american.new_page()
        us.goto(BASE, wait_until="domcontentloaded")
        expect_text(us, "Your buddy knows the market.")
        expect_text(us, "Smoke language: en-US")
        american.close()

        british = browser.new_context(locale="en-GB")
        uk = british.new_page()
        uk.goto(BASE, wait_until="domcontentloaded")
        expect_text(uk, "Your mate has a theory.")
        expect_text(uk, "Smoke language: en-GB")
        british.close()
    finally:
        browser.close()

with sync_playwright() as p:
    for engine in (p.firefox, p.chromium):
        smoke(engine)
print("Browser locale detection, persistence and market routing: PASS")
