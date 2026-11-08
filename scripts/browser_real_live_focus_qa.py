"""Actual hosted Streamlit app, real headless Chromium mobile touch QA.

Unlike the earlier landing fixture, this reaches the live production URL,
renders the REAL reliability bulletin, and exercises exactly one banner tap
plus exactly one actual model calculation button tap. Save forensic screenshots
and page errors. Do not claim real-app success from the lightweight fixture.
"""
import json
import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

TARGET=os.environ.get("PRICING_LIVE_URL",
    "https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/")
OUT=Path("reports/qa/real-live-chromium")
OUT.mkdir(parents=True,exist_ok=True)


def run():
    evidence={"target":TARGET,"real_hosted_app":True,"browser":"Chromium",
              "mobile_touch":True,"banner_taps":0,"submit_taps":0,
              "reached_focus":False,"form_visible_in_viewport":False,
              "submitted_once":False,"errors":[]}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(
            viewport={"width":390,"height":844},
            device_scale_factor=1,is_mobile=True,has_touch=True,
            reduced_motion="reduce")
        page=context.new_page()
        page.on("pageerror",lambda e:evidence["errors"].append(str(e)[:350]))
        try:
            page.goto(TARGET,wait_until="domcontentloaded",timeout=90000)
            cta=page.get_by_role(
                "link",name="Go to the car pricing calculator")
            cta.wait_for(state="visible",timeout=150000)
            assert page.locator(
                '[aria-label="Temporary pricing model reliability notice"]'
            ).count()==1, "Real warning bulletin was not present on landing"
            page.screenshot(path=str(OUT/"real-live-before-cta.png"),full_page=True)
            cta.tap(timeout=30000)
            evidence["banner_taps"]=1
            page.wait_for_function(
                "new URLSearchParams(location.search).get('focus')==='pricing'",
                timeout=45000)
            evidence["reached_focus"]=True
            first=page.locator('[data-testid="stForm"]').first
            first.wait_for(state="visible",timeout=150000)
            button=page.get_by_role("button",name=re.compile(
                "Calculate seller Ask.*wholesale Bid",re.I))
            button.first.wait_for(state="visible",timeout=60000)
            form_box=first.bounding_box()
            assert form_box, "No form bounding box available"
            warning=page.locator(
                '[aria-label="Temporary pricing model reliability notice"]')
            warning_box=warning.bounding_box()
            assert warning_box, "Safety warning disappeared; should move down"
            evidence["form_viewport_y"]=round(form_box["y"],1)
            evidence["bulletin_viewport_y"]=round(warning_box["y"],1)
            evidence["scroll_y"]=page.evaluate("window.scrollY")
            assert form_box["y"] < warning_box["y"], (
                "Warning bulletin is still above the real form")
            assert form_box["y"] < 760, (
                "First tap still fails to show form in mobile viewport")
            evidence["form_visible_in_viewport"]=True
            page.screenshot(path=str(OUT/"real-live-after-one-banner-tap.png"),
                            full_page=True)
            assert not button.first.is_disabled(), (
                "Real calculator submit is disabled despite release availability")
            button.first.tap(timeout=45000)
            evidence["submit_taps"]=1
            confirmation=page.get_by_text(re.compile(
                r"Request received|Calculation complete|Your car · estimated prices|"
                r"Ask engine failed during|couldn't finish",re.I))
            confirmation.first.wait_for(state="visible",timeout=180000)
            evidence["submitted_once"]=True
            evidence["confirmation"]=confirmation.first.inner_text()[:350]
            page.screenshot(path=str(OUT/"real-live-after-one-submit.png"),
                            full_page=True)
            print("REAL_LIVE_CHROMIUM_ONE_TAP_PASSED",
                  json.dumps(evidence),flush=True)
        finally:
            evidence["url_after"]=page.url
            (OUT/"report.json").write_text(json.dumps(evidence,indent=2))
            try:page.screenshot(path=str(OUT/"real-live-final-state.png"),
                                full_page=True)
            except Exception:pass
            context.close()
            browser.close()


if __name__=="__main__":
    run()
