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


def navigate_europe_panel(page,label):
    """Operate Streamlit selectbox by accessibility roles, not BaseWeb internals."""
    picker=page.locator('[data-testid="stSelectbox"]').filter(
        has_text="European Studio panels").first
    expect(picker).to_be_visible(timeout=20000)
    control=picker.get_by_role("combobox").first
    if not control.count():
        control=picker.get_by_role("button").first
    if not control.count():
        control=picker.locator("input").first
    if not control.count():
        control=picker.locator("[tabindex='0']").first
    if not control.count():
        print("EU_NAV_DOM",picker.evaluate("(n)=>n.outerHTML.slice(0,5000)"),flush=True)
        control=picker
    control.click(timeout=10000)
    # The displayed labels are translated, but the order and route IDs are
    # stable. Select the canonical option index rather than relying on emoji
    # and UK/US spelling ("indices" versus "indexes", "garage" vs "dealership").
    from pricing.europe_market import NAV
    index=NAV.index(label)
    options=page.get_by_role("option")
    expect(options).to_have_count(len(NAV),timeout=10000)
    options.nth(index).click(timeout=10000)

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
    expect(international).to_have_text("Two continents. One garage.")
    assert '<section class="cps-globe"' not in page.locator("body").inner_text()
    # The satirical advertising banner is German-only by design. English
    # browser sessions must not display the German parody or its legal copy.
    assert page.locator(".st-key-eu_parody_banner").count()==0, \
        "German-only parody is leaking into the English European calculator"
    expect(page.get_by_text("From Hamburg to Rome.",exact=False)).to_be_visible(timeout=120000)
    expect(page.get_by_text("Tell us about your car",exact=False)).to_be_visible(timeout=90000)
    expect(page.locator(".eu-hero")).to_contain_text("European")
    expect(page.locator(".eu-steps .eu-step")).to_have_count(3,timeout=90000)
    assert page.get_by_role("button",name=re.compile("Calculate asking price")).count()==1
    expect(page.get_by_text("Country · 8")).to_be_visible(timeout=90000)
    diagnose(page,device+"-landing")
    model=page.get_by_role("textbox",name="Model",exact=True)
    if model.count():
        model.fill("golf")
    button=page.get_by_role("button",name=re.compile("Calculate asking price"))
    assert button.is_enabled(), "Release is not model-ready"
    button.click(timeout=30000)
    expect(page.locator(".eu-result strong")).to_be_visible(timeout=180000)
    content=page.locator(".eu-result strong").inner_text()
    assert "€" in content and len(content)>2
    expect(page.locator(".eu-result")).to_contain_text("ASK")
    explain=page.get_by_role("button",name=re.compile("What moves this price"))
    expect(explain).to_be_enabled()
    explain.click(timeout=30000)
    expect(page.get_by_text("Each bar is a real model counterfactual",exact=False)).to_be_visible(timeout=90000)
    bars=page.locator(".eu-impact-graph .eu-impact-row")
    expect(bars).to_have_count(4,timeout=90000)
    widths=page.locator(".eu-impact-graph .eu-impact-bar").evaluate_all(
        "(els) => els.map(el => el.getBoundingClientRect().width)")
    assert any(w>4 for w in widths), "ResMLP price-impact graphics are empty"
    assert page.locator('[data-testid="stException"]').count()==0
    diagnose(page,device+"-valuation")
    # Exercise European panels with the ResMLP recent-market receipt in memory.
    # One broken panel must not hide failures in the rest of the application.
    expected={
        "🔎 Listing Inspector":"Listing Inspector",
        "📉 Value over time":"Value over time",
        "📈 Price Indices":"European Price Index Observatory",
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
            navigate_europe_panel(page,label)
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
        navigate_europe_panel(page,"🚗 Price my car")
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
