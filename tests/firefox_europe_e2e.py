"""Actual Firefox desktop + mobile smoke of the US/Europe landing and EU Ask.

Requires a running local Streamlit server and the release model bundle.
Screenshots and browser console issues are written to artifacts/ for audit.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

HOST=os.getenv("CPS_BASE_URL","http://127.0.0.1:8507")
ART=Path("artifacts/firefox-europe")
ART.mkdir(parents=True,exist_ok=True)

def diagnose(page,label):
    page.screenshot(path=str(ART/(label+".png")),full_page=True)
    errors=page.locator('[data-testid="stException"]').count()
    if errors:
        raise AssertionError(f"Streamlit exception detected on {label}")
    return page.evaluate("""() => ({
       width: document.documentElement.clientWidth,
       scrollWidth: document.documentElement.scrollWidth,
       headings: [...document.querySelectorAll('h1,h2')].map(x=>x.textContent).slice(0,15)
    })""")

def test_device(browser,device,width,height):
    page=browser.new_page(viewport={"width":width,"height":height},
                          device_scale_factor=1,locale="en-US")
    problems=[]
    page.on("pageerror",lambda err:problems.append(str(err)))
    page.goto(HOST+"/?market=eu",wait_until="domcontentloaded",timeout=120000)
    expect(page.get_by_text("From Hamburg to Rome.",exact=False)).to_be_visible(timeout=120000)
    expect(page.get_by_text("European Asking Price Lab",exact=False)).to_be_visible(timeout=90000)
    assert page.get_by_role("button",name=re.compile("Calculate European asking price")).count()==1
    diagnose(page,device+"-landing")
    model=page.get_by_role("textbox",name="Model",exact=True)
    if model.count():
        model.fill("golf")
    button=page.get_by_role("button",name=re.compile("Calculate European asking price"))
    assert button.is_enabled(), "Release is not model-ready"
    button.click(timeout=30000)
    expect(page.locator(".eu-result strong")).to_be_visible(timeout=180000)
    content=page.locator(".eu-result strong").inner_text()
    assert "€" in content and len(content)>2
    assert page.locator('[data-testid="stException"]').count()==0
    diagnose(page,device+"-valuation")
    assert not problems,f"Firefox JavaScript exceptions: {problems[:5]}"
    page.close()

def main():
    with sync_playwright() as api:
        browser=api.firefox.launch(headless=True)
        try:
            test_device(browser,"desktop",1365,850)
            test_device(browser,"mobile",390,844)
            page=browser.new_page(viewport={"width":1280,"height":850})
            page.goto(HOST+"/?market=us",wait_until="domcontentloaded",timeout=90000)
            expect(page.get_by_text("United States",exact=False).first).to_be_visible(timeout=120000)
            expect(page.get_by_text("European Studio",exact=False).first).to_be_visible(timeout=120000)
            diagnose(page,"us-still-available")
            page.close()
        finally:
            browser.close()
    print("FIREFOX_EUROPE_AND_US_UI_OK")

if __name__=="__main__":
    main()
