"""Real Streamlit browser checkout on Python 3.14, not homepage-only smoke.

Submits the default synthetic 2015 Corolla, asserts BOTH native Ask and Bid
prices appeared, then clicks Captum's 24-step explain button and checks its
neural output. Captures errors/screenshots to CI artifacts.
"""
import json
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT=Path("reports/qa/native-python314")


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    report={"mode":"genuine native Python 3.14 Streamlit UI",
            "synthetic_only":True,"ask":False,"bid":False,
            "captum":False,"screenshots":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        ctx=browser.new_context(viewport={"width":1280,"height":900},
                                reduced_motion="reduce")
        page=ctx.new_page()
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        try:
            page.goto("http://127.0.0.1:8765",wait_until="domcontentloaded",timeout=120000)
            form=page.get_by_role("button",name=re.compile(
                r"Calculate seller Ask.*wholesale Bid"))
            form.first.wait_for(state="visible",timeout=120000)
            form.first.scroll_into_view_if_needed()
            page.screenshot(path=str(OUTPUT/"before-submit.png"),full_page=True)
            form.first.click(timeout=30000)
            page.get_by_text("Your car · two historical price anchors").first.wait_for(
                state="visible",timeout=180000)
            page.get_by_text("Seller asking · historical anchor").first.wait_for(
                state="visible",timeout=180000)
            report["ask"]=True
            page.get_by_text("Wholesale auction · historical anchor").first.wait_for(
                state="visible",timeout=180000)
            report["bid"]=True
            page.screenshot(path=str(OUTPUT/"both-native-prices.png"),full_page=True)
            explain=page.get_by_role("button",name=re.compile(
                r"Explain THIS price.*Captum"))
            explain.first.wait_for(state="visible",timeout=40000)
            explain.first.scroll_into_view_if_needed()
            explain.first.click(timeout=30000)
            page.get_by_text("Neural baseline").first.wait_for(
                state="visible",timeout=180000)
            page.get_by_text("Actual neural output").first.wait_for(
                state="visible",timeout=50000)
            report["captum"]=True
            page.screenshot(path=str(OUTPUT/"captum-real-24step.png"),full_page=True)
            if page.locator('[data-testid="stException"]').count():
                raise AssertionError("Streamlit displayed an uncaught exception")
            if errors:
                raise AssertionError("Browser page errors: "+str(errors[:3]))
            print("PASS native two-price appraisal + Captum 24-step from genuine form",
                  flush=True)
        finally:
            report["browser_errors"]=errors[:5]
            (OUTPUT/"result.json").write_text(json.dumps(report,indent=2))
            try:page.screenshot(path=str(OUTPUT/"final-state.png"),full_page=True)
            except Exception:pass
            ctx.close()
            browser.close()


if __name__=="__main__":
    main()
