"""Production-Python browser audit for both BOHB Ask choices and the same Bid.

Uses the real app and model artifacts on a synthetic car only, with two
separate one-click form submissions. No feedback is sent to Turso. Captum is
intentionally not asserted for the new neural heads pending attribution parity.
"""
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT = Path("reports/qa/native-python314")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"mode": "real Python 3.14, synthetic used car, dual neural Ask",
              "synthetic_only": True, "english": False,
              "multilingual": False, "bid": False,
              "feedback_prompt": False, "screenshots": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900},
                                  reduced_motion="reduce")
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto("http://127.0.0.1:8765", wait_until="domcontentloaded",
                      timeout=120000)
            english = page.get_by_role("button", name=re.compile("Evaluate in English"))
            multilingual = page.get_by_role("button", name=re.compile("Evaluate in my language"))
            english.first.wait_for(state="visible", timeout=120000)
            multilingual.first.wait_for(state="visible", timeout=120000)
            if english.first.is_disabled() or multilingual.first.is_disabled():
                raise AssertionError("One or both winner checkpoint gates are unavailable")
            page.screenshot(path=str(OUTPUT / "two-enabled-buttons.png"),
                            full_page=True)
            english.first.scroll_into_view_if_needed()
            english.first.click(timeout=30000)
            page.get_by_text(re.compile("Calculation complete")).first.wait_for(
                state="visible", timeout=240000)
            page.get_by_text("Your car · estimated prices").first.wait_for(
                state="visible", timeout=30000)
            report["english"] = True
            report["bid"] = True
            page.get_by_text(re.compile("Your verdict: was our AI valuation accurate")).first.wait_for(
                state="visible", timeout=30000)
            report["feedback_prompt"] = True
            page.screenshot(path=str(OUTPUT / "bge-bid-and-rating.png"),
                            full_page=True)

            # A second genuinely distinct model, same form; French is not
            # translated to English, and both models may remain cached.
            description = page.get_by_label("Seller's description")
            description.fill("Une belle voiture en bon état, titre propre, pneus neufs.")
            multilingual.first.scroll_into_view_if_needed()
            multilingual.first.click(timeout=30000)
            page.get_by_text(re.compile("Calculation complete")).first.wait_for(
                state="visible", timeout=240000)
            page.get_by_text("Your car · estimated prices").first.wait_for(
                state="visible", timeout=30000)
            report["multilingual"] = True
            page.screenshot(path=str(OUTPUT / "granite-french-and-bid.png"),
                            full_page=True)
            if page.locator('[data-testid="stException"]').count():
                raise AssertionError("Streamlit displayed an uncaught exception")
            if errors:
                raise AssertionError("Browser page errors: " + str(errors[:3]))
            print("DUAL_BGE_GRANITE_INT8_NATIVE_PRICING_AND_FEEDBACK_UI_PASSED",
                  flush=True)
        finally:
            report["browser_errors"] = errors[:5]
            (OUTPUT / "result.json").write_text(json.dumps(report, indent=2))
            try:
                page.screenshot(path=str(OUTPUT / "final-state.png"),
                                full_page=True)
            except Exception:
                pass
            ctx.close()
            browser.close()


if __name__ == "__main__":
    main()
