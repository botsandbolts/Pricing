"""Real Chromium visual/UI regression of public landing banner.

Runs against lightweight fixture (same production studio_design, CSS and
navigation modules) to avoid provisioning multi-GB BGE + XGBoost weights.
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
    proc=subprocess.Popen(
        [sys.executable,"-m","streamlit","run",
         str(ROOT/"tests/landing_hero_fixture.py"),
         "--server.port","8766","--server.address","127.0.0.1",
         "--server.headless","true","--browser.gatherUsageStats","false"],
        cwd=ROOT,stdout=(OUT/"server.log").open("w"),
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(70):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8766/_stcore/health",timeout=2) as resp:
                    if resp.status==200:break
            except Exception:time.sleep(.4)
        else:raise AssertionError("Local Streamlit landing fixture did not become healthy")
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True)
            try:
                details=[]
                for width,height in ((1440,1000),(820,1000),(390,844),(320,800)):
                    page=browser.new_page(viewport={"width":width,"height":height},
                        device_scale_factor=1)
                    page.goto("http://127.0.0.1:8766",wait_until="domcontentloaded")
                    hero=page.locator(".garage-home-hero")
                    hero.wait_for(timeout=35000)
                    assert hero.is_visible(),f"Missing homepage banner at {width}px"
                    heading=hero.locator("h1")
                    assert "I KNOW WHAT I'VE GOT" in heading.inner_text()
                    cta=page.get_by_role("link",name="Go to the car pricing calculator")
                    assert cta.is_visible()
                    assert cta.get_attribute("href")=="#studio-calculator"
                    assert page.locator(".garage-home-assurances").is_visible()
                    assert page.locator(".garage-home-art svg").count()>=1
                    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth"),f"Horizontal overflow at {width}"
                    page.screenshot(path=str(OUT/f"landing-{width}px.png"),full_page=True)
                    cta.focus()
                    assert cta.evaluate("(el)=>document.activeElement===el"),"Keyboard focus lost"
                    page.keyboard.press("Enter")
                    page.wait_for_function("location.hash==='#studio-calculator'",timeout=10000)
                    assert page.locator("#studio-calculator").count()==1
                    privacy=page.locator(".garage-home-privacy")
                    privacy.locator("summary").click()
                    assert privacy.get_attribute("open") is not None
                    assert "Regulatory oversight is not an endorsement" in privacy.inner_text()
                    assert "Rejecting analytics doesn't disable the calculator" in page.locator(".garage-home-trust").inner_text() or "Rejecting analytics doesn't disable the calculator" in privacy.inner_text()
                    details.append({"width":width,"overflow":False,"cta_hash":page.evaluate("location.hash"),"privacy_details":True})
                    page.close()
                # The full-page campaign must appear ONLY in the main panel.
                page=browser.new_page(viewport={"width":1200,"height":900})
                page.goto("http://127.0.0.1:8766",wait_until="domcontentloaded")
                page.get_by_text("Choose a panel",exact=True).wait_for()
                other=page.get_by_role("radio",name="The science",exact=True)
                other.click()
                page.get_by_role("heading",name="Other studio workspace").wait_for()
                assert page.locator(".garage-home-hero").count()==0,"Banner leaked into other panels"
                page.close()
                (OUT/"browser-report.json").write_text(json.dumps({
                    "status":"passed","tested_with":"real Chromium and Streamlit",
                    "viewport_results":details,"calculator_cta_functional":True,
                    "keyboard":True,"nonpricing_workspace_compact":True,
                    "independent_model_weights_loaded":False,
                },indent=2))
                print("LANDING_BANNER_BROWSER_QA_PASSED",json.dumps(details),flush=True)
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:proc.kill()


if __name__=="__main__":run()
