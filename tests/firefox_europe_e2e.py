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
    sizes=page.evaluate("""() => ({
       width: document.documentElement.clientWidth,
       scrollWidth: document.documentElement.scrollWidth,
       headings: [...document.querySelectorAll('h1,h2')].map(x=>x.textContent).slice(0,15)
    })""")
    if sizes["scrollWidth"] > sizes["width"]+8:
        raise AssertionError("Page has horizontal overflow on "+label+": "+repr(sizes))
    return sizes

def test_device(browser,device,width,height):
    page=browser.new_page(viewport={"width":width,"height":height},
                          device_scale_factor=1,locale="en-US")
    problems=[]
    page.on("pageerror",lambda err:problems.append(str(err)))
    page.goto(HOST+"/?market=eu",wait_until="domcontentloaded",timeout=120000)
    # Visual regression: the previous release accidentally displayed the
    # HTML hero as literal source text / an empty code block.
    international=page.locator(".cps-globe h1")
    expect(international).to_be_visible(timeout=120000)
    expect(international).to_have_text("Two continents. One garage. Zero crystal balls.")
    assert '<section class="cps-globe"' not in page.locator("body").inner_text()
    # Production banner must be visible ABOVE the European market hero, never on US.
    banner=page.locator(".st-key-eu_parody_banner")
    expect(banner).to_be_visible(timeout=120000)
    artwork=banner.locator("img").first
    expect(artwork).to_be_visible(timeout=30000)
    alt=artwork.get_attribute("alt") or ""
    for term in ("NICHT", "wirksufendeinaito", "WKDA", "Autohero", "AUTO1"):
        assert term in alt, f"Missing requested accessible text: {term}"
    page.wait_for_function("""() => {
        const img=document.querySelector(".st-key-eu_parody_banner img");
        return img && img.complete;
    }""",timeout=30000)
    loaded=artwork.evaluate("(img) => ({w:img.naturalWidth,h:img.naturalHeight,src:img.currentSrc})")
    assert loaded["w"]>500 and loaded["h"]>100, f"Banner asset did not render: {loaded}"
    banner_y=banner.bounding_box()["y"]
    hero_y=page.locator(".eu-hero").bounding_box()["y"]
    assert banner_y<hero_y, "European satire must be above European hero"
    expect(page.get_by_text("From Hamburg to Rome.",exact=False)).to_be_visible(timeout=120000)
    expect(page.get_by_text("European Asking Price Lab",exact=False)).to_be_visible(timeout=90000)
    expect(page.locator(".eu-hero")).to_contain_text("ResMLP")
    expect(page.locator(".eu-steps .eu-step")).to_have_count(3,timeout=90000)
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
    expect(page.locator(".eu-result")).to_contain_text("ResMLP")
    explain=page.get_by_role("button",name=re.compile("What moves this price"))
    expect(explain).to_be_enabled()
    explain.click(timeout=30000)
    expect(page.get_by_text("Each bar is a new ResMLP estimate",exact=False)).to_be_visible(timeout=90000)
    bars=page.locator(".eu-impact-graph .eu-impact-row")
    expect(bars).to_have_count(4,timeout=90000)
    widths=page.locator(".eu-impact-graph .eu-impact-bar").evaluate_all(
        "(els) => els.map(el => el.getBoundingClientRect().width)")
    assert any(w>4 for w in widths), "ResMLP price-impact graphics are empty"
    assert page.locator('[data-testid="stException"]').count()==0
    diagnose(page,device+"-valuation")
    # Exercise every European panel with the actual ResMLP receipt in memory.
    # One broken panel must not hide failures in the rest of the application.
    expected={
        "🔎 Listing Inspector":"Listing Inspector",
        "📉 Value over time":"Value over time",
        "📊 Evaluation":"European evaluation",
        "🧪 Model lab":"Ray Tune model lab",
        "🔬 The science":"Under the bonnet",
        "🕵️ Deal Detective":"Deal Detective",
        "🧹 Outlier lab":"Outlier Lab",
        "🌍 Market intelligence":"Market intelligence",
        "🏪 Merchant lens":"Merchant lens",
        "🚦 Pick your garage":"Pick your garage",
    }
    failures=[]
    for label,heading in expected.items():
        try:
            pill=page.get_by_role("button",name=label,exact=True)
            if not pill.count():
                pill=page.get_by_role("radio",name=label,exact=True)
            if not pill.count():
                raise AssertionError("Panel control is missing: "+label)
            pill.first.click(timeout=20000)
            expect(page.locator("h1,h2,h3").filter(has_text=heading).first).to_be_visible(timeout=60000)
            if page.locator('[data-testid="stException"]').count():
                raise AssertionError("Uncaught Streamlit error")
            box=page.evaluate("""() => ({
                width:document.documentElement.clientWidth,
                overflow:document.documentElement.scrollWidth
            })""")
            if box["overflow"]>box["width"]+8:
                raise AssertionError("Horizontal overflow: "+repr(box))
        except Exception as exc:
            failures.append(label+": "+str(exc)[:280])
            page.screenshot(path=str(ART/(device+"-panel-error-"+str(len(failures))+".png")),full_page=True)
    # Go back and check the valuation persists rather than resetting the user.
    try:
        pricing=page.get_by_role("button",name="🚗 Price my car",exact=True)
        if not pricing.count():
            pricing=page.get_by_role("radio",name="🚗 Price my car",exact=True)
        pricing.first.click(timeout=20000)
        expect(page.locator(".eu-result strong")).to_be_visible(timeout=60000)
        diagnose(page,device+"-return-to-pricing")
    except Exception as exc:
        failures.append("Return to pricing lost receipt: "+str(exc)[:280])
    assert not failures, "European workspace failures: "+" | ".join(failures)
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
            assert page.locator(".st-key-eu_parody_banner").count()==0, "European banner leaked into US pricing"
            expect(page.get_by_text("European Studio",exact=False).first).to_be_visible(timeout=120000)
            diagnose(page,"us-still-available")
            page.close()
        finally:
            browser.close()
    print("FIREFOX_EUROPE_AND_US_UI_OK")

if __name__=="__main__":
    main()
