"""Real Firefox QA: ONE click to the pricing section, ONE click to submit.

The fixture uses production branding/navigation/CSS but no heavyweight ML.
A successful st.form_submit_button proves the first event reached Streamlit.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/qa/landing-banner"


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen([
        sys.executable,"-m","streamlit","run",
        str(ROOT/"tests/landing_hero_fixture.py"),
        "--server.port","8766","--server.address","127.0.0.1",
        "--server.headless","true","--browser.gatherUsageStats","false"],
        cwd=ROOT,stdout=(OUT/"server.log").open("w"),stderr=subprocess.STDOUT)
    try:
        for _ in range(80):
            try:
                with urllib.request.urlopen(
                        "http://127.0.0.1:8766/_stcore/health",timeout=2) as resp:
                    if resp.status==200:
                        break
            except Exception:
                time.sleep(.4)
        else:
            raise AssertionError("Local Streamlit landing fixture not healthy")
        with sync_playwright() as p:
            browser=p.firefox.launch(headless=True)
            try:
                details=[]
                for width,height in ((1440,1000),(820,1000),(390,844),(320,800)):
                    mobile=width<=390
                    ctx=browser.new_context(
                        viewport={"width":width,"height":height},
                        device_scale_factor=1,
                        has_touch=mobile,
                        reduced_motion="reduce")
                    page=ctx.new_page()
                    page.goto("http://127.0.0.1:8766",
                              wait_until="domcontentloaded")
                    hero=page.locator(".garage-home-hero")
                    hero.wait_for(timeout=35000)
                    assert hero.is_visible(),f"Missing homepage hero at {width}"
                    assert "I KNOW WHAT I'VE GOT" in hero.locator("h1").inner_text()
                    cta=page.get_by_role(
                        "link",name="Go to the car pricing calculator")
                    assert cta.is_visible()
                    assert cta.get_attribute("href")=="#studio-calculator"
                    assert page.locator(".garage-home-assurances").is_visible()
                    assert page.locator(".garage-home-art svg").count()>=1
                    assert not page.evaluate(
                        "document.documentElement.scrollWidth > window.innerWidth"
                    ),f"Horizontal overflow at {width}"
                    privacy=page.locator(".garage-home-privacy")
                    privacy.locator("summary").click()
                    assert privacy.get_attribute("open") is not None
                    assert "Regulatory oversight is not an endorsement" in privacy.inner_text()
                    assert "Rejecting analytics doesn't disable the calculator" in privacy.inner_text()
                    page.screenshot(path=str(OUT/f"landing-{width}px.png"),full_page=True)
                    # Native hash scroll: no Streamlit rerun, no missing banners.
                    cta.click(timeout=15000)
                    page.wait_for_function(
                        "location.hash === '#studio-calculator'",timeout=15000)
                    assert hero.count()==1 and hero.is_visible()
                    assert page.get_by_label("Pick a sample car").is_visible()
                    # One activation must reveal the editable form, without
                    # scrolling to a not-yet-rendered React hash anchor.
                    heading=page.get_by_role(
                        "heading",name="Your pricing calculator")
                    heading.wait_for(state="visible",timeout=45000)
                    assert page.locator(".garage-home-hero").count()==1
                    submit=page.get_by_role(
                        "button",name="Calculate seller Ask + wholesale Bid")
                    submit.wait_for(state="visible",timeout=30000)
                    submit.click(timeout=30000)
                    page.get_by_text("One-tap calculation received").wait_for(
                        state="visible",timeout=30000)
                    page.screenshot(path=str(OUT/f"calculator-{width}px.png"),full_page=True)
                    details.append({
                        "width":width,"touch":mobile,
                        "one_activation_to_form":True,
                        "one_activation_to_submit":True,
                        "privacy_details":True,"overflow":False})
                    ctx.close()
                # Original navigation remains available when not in focus mode.
                ctx=browser.new_context(viewport={"width":1200,"height":900})
                page=ctx.new_page()
                page.goto("http://127.0.0.1:8766",
                          wait_until="domcontentloaded")
                page.get_by_text("Choose a panel",exact=True).wait_for()
                page.get_by_role("radio",name="The science",exact=True).click()
                page.get_by_role(
                    "heading",name="Other studio workspace").wait_for()
                assert page.locator(".garage-home-hero").count()==0
                ctx.close()
                report={
                    "status":"passed",
                    "tested_with":"real Firefox + Streamlit mobile viewport",
                    "viewport_results":details,
                    "single_tap_mobile_cta":True,
                    "single_tap_mobile_form_submit":True,
                    "heavy_price_models_loaded":False}
                (OUT/"browser-report.json").write_text(json.dumps(
                    report,indent=2))
                print("LANDING_ONE_CLICK_FIREFOX_QA_PASSED",
                      json.dumps(details),flush=True)
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__=="__main__":
    run()
