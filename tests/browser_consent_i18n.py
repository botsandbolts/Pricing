"""Firefox and Chromium smoke of localized production consent HTML/JS.

No real telemetry is sent: the fixture uses an invalid GA ID and local-only
geolocation setting. The test checks visible native language, consent actions
and navigation across sessions without relying on a real tracking service.
"""
import os
from playwright.sync_api import sync_playwright

BASE=os.environ.get("CONSENT_I18N_URL","http://127.0.0.1:8777")
TIMEOUT=45000

def expect(page,text):
    page.get_by_text(text,exact=False).first.wait_for(timeout=TIMEOUT)

def smoke(engine):
    browser=engine.launch(headless=True)
    try:
        context=browser.new_context(locale="nl-NL",
            viewport={"width":390,"height":844})
        page=context.new_page()
        page.goto(BASE+"/?lang=nl",wait_until="domcontentloaded")
        expect(page,"Consent locale: nl")
        page.get_by_role("button",name="Weiger statistieken").wait_for(timeout=TIMEOUT)
        page.get_by_role("button",name="Weiger statistieken").click()
        page.get_by_role("button",name="Privacy en cookies").click()
        page.get_by_role("button",name="Sta statistieken toe").wait_for(timeout=TIMEOUT)
        page.get_by_role("button",name="Sta statistieken toe").click()
        page.get_by_role("button",name="Privacy en cookies").click()
        expect(page,"Jouw gegevens. Jouw keuze.")
        expect(page,"Toestemmingscookies")
        page.screenshot(path=f"/tmp/consent_{engine.name}_nl.png",full_page=True)
        # Explicit language link remains independent of cookie consent.
        page.goto(BASE+"/?lang=fr",wait_until="domcontentloaded")
        expect(page,"Consent locale: fr")
        page.get_by_role("button",name="Confidentialité et cookies").wait_for(timeout=TIMEOUT)
        context.close()
    finally:
        browser.close()

with sync_playwright() as p:
    for engine in (p.firefox,p.chromium):
        smoke(engine)
print("Consent modal localisation in Firefox/Chromium: PASS")
