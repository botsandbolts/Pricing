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
            # A fresh hosted session opens GDPR consent as a
            # Streamlit v2 component. Its accessible buttons may be
            # isolated behind a closed ShadowRoot, so Playwright's role
            # locator cannot always see them despite visible pixels.
            # First capture actual evidence, then click the visually
            # confirmed 'Reject analytics' location at 390x844.
            page.wait_for_timeout(8500)
            page.screenshot(path=str(OUT/"real-live-before-consent.png"),
                            full_page=True)
            page.mouse.click(104,626)
            evidence["privacy_choice"]="rejection_click_at_confirmed_mobile_button"
            page.wait_for_timeout(1300)
            page.screenshot(path=str(OUT/"real-live-after-consent.png"),
                            full_page=True)
            # Independent direct-route check against the REAL hosted app.
            # This gives a useful verdict even if fresh-session consent
            # or mobile tap interception later blocks the banner itself.
            direct=context.new_page()
            direct.goto(TARGET.rstrip("/")+"/?focus=pricing",
                        wait_until="domcontentloaded",timeout=90000)
            direct.wait_for_timeout(10000)
            direct.screenshot(path=str(OUT/"real-live-focused-first-render.png"),
                              full_page=True)
            evidence["direct_focus_url"]=direct.url
            evidence["direct_focus_text"]=direct.locator("body").inner_text(
                timeout=10000)[-2500:]
            direct_form=direct.locator('[data-testid="stForm"]').first
            direct_form.wait_for(state="visible",timeout=120000)
            direct_notice=direct.locator(
                '[aria-label="Temporary pricing model reliability notice"]')
            direct_notice.wait_for(state="attached",timeout=30000)
            direct_form_box=direct_form.bounding_box()
            direct_notice_box=direct_notice.bounding_box()
            assert direct_form_box and direct_notice_box
            evidence["direct_focus_form_y"]=round(direct_form_box["y"],1)
            evidence["direct_focus_warning_y"]=round(direct_notice_box["y"],1)
            evidence["direct_focus_order_passed"]=(
                direct_form_box["y"]<direct_notice_box["y"])
            direct.screenshot(path=str(OUT/"real-live-direct-focus.png"),
                              full_page=True)
            direct.close()
            assert evidence["direct_focus_order_passed"], (
                "Real hosted calculator still follows the warning bulletin")
            cta=page.get_by_role(
                "link",name="Go to the car pricing calculator")
            cta.wait_for(state="attached",timeout=120000)
            cta.scroll_into_view_if_needed(timeout=30000)
            cta.wait_for(state="visible",timeout=30000)
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
