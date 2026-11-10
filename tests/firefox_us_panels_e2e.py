"""Local, genuine Firefox regression coverage for every US Streamlit workspace.

Runs against the production app entrypoint (not a fixture), preserving a single
session across panel navigation. Screenshots and a JSON audit always survive.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

BASE=os.environ.get("CPS_BASE_URL","http://127.0.0.1:8507")
OUTPUT=Path("artifacts/firefox-us-panels")
OUTPUT.mkdir(parents=True,exist_ok=True)
PANELS=[
    "🚗 Price my car","🔎 Listing Inspector","🚦 Pick your garage",
    "↔ Two-market garage","📅 Value over time","🧠 The science",
    "📊 Evaluation","🧪 Model lab","🕵️ Deal Detective",
    "🧹 Outlier lab","🏪 Merchant lens",
]

def audit():
    failures=[]
    report={"browser":"Firefox","market":"US","panels":[],"viewports":[],"javascript_errors":[]}
    with sync_playwright() as driver:
        browser=driver.firefox.launch(headless=True)
        context=browser.new_context(viewport={"width":1280,"height":900},
            locale="en-US",reduced_motion="reduce",accept_downloads=True)
        page=context.new_page()
        page.on("pageerror",lambda e:report["javascript_errors"].append(str(e)[:400]))
        try:
            page.goto(BASE+"/?market=us",wait_until="domcontentloaded",timeout=120000)
            expect(page.locator(".cps-globe h1")).to_be_visible(timeout=120000)
            expect(page.locator(".garage-home-hero")).to_be_visible(timeout=90000)
            assert page.locator(".st-key-eu_parody_banner").count()==0
            # Exercise real market switching without losing the American tools.
            europe=page.get_by_role("button",name="Open the European Studio")
            if europe.count():
                europe.click()
                expect(page.locator(".eu-hero")).to_be_visible(timeout=90000)
                page.get_by_role("button",name="Open the US Studio").click()
                expect(page.locator(".garage-home-hero")).to_be_visible(timeout=90000)
            # Responsive baseline without large inference resident in browser memory.
            for width,height in ((320,800),(390,844),(768,1024),(1440,900)):
                page.set_viewport_size({"width":width,"height":height})
                page.wait_for_timeout(450)
                geometry=page.evaluate("""() => ({
                    client:document.documentElement.clientWidth,
                    scroll:document.documentElement.scrollWidth
                })""")
                report["viewports"].append({"width":width,"height":height,**geometry})
                if geometry["scroll"]>geometry["client"]+8:
                    failures.append("Horizontal overflow at "+str(width)+"px")
            page.set_viewport_size({"width":1280,"height":900})
            for index,label in enumerate(PANELS):
                item={"name":label}
                try:
                    pill=page.locator('[data-testid="stPills"]').get_by_text(label,exact=True)
                    if not pill.count():
                        pill=page.get_by_role("button",name=label,exact=False)
                    if not pill.count():
                        pill=page.get_by_role("radio",name=label,exact=False)
                    if not pill.count():
                        pill=page.get_by_role("option",name=label,exact=False)
                    if not pill.count():
                        pill=page.get_by_text(label,exact=True)
                    if not pill.count():
                        raise AssertionError("Workspace navigation control missing")
                    pill.first.click(timeout=30000)
                    page.wait_for_timeout(1150)
                    item["exception_cards"]=page.locator('[data-testid="stException"]').count()
                    item["headings"]=page.locator("h1,h2,h3").all_inner_texts()[-10:]
                    item["body_characters"]=len(page.locator("body").inner_text())
                    item["visible"]=pill.first.is_visible()
                    item["warnings"]=page.locator('[data-testid="stAlert"]').count()
                    if item["exception_cards"]:
                        raise AssertionError("Streamlit exception rendered")
                    if item["body_characters"]<250:
                        raise AssertionError("Workspace did not render meaningful content")
                    if index in (0,1,5,8,10):
                        page.screenshot(path=str(OUTPUT/("us-panel-%02d.png"%index)),full_page=True)
                except Exception as exc:
                    failures.append(label+": "+str(exc)[:220])
                    item["error"]=str(exc)[:350]
                    try:page.screenshot(path=str(OUTPUT/("us-error-%02d.png"%index)),full_page=True)
                    except Exception:pass
                report["panels"].append(item)
            # Explicit legal routes must remain reachable from either market.
            for name in ("imprint","privacy"):
                check={"route":name}
                try:
                    page.goto(BASE+"/?legal="+name,wait_until="domcontentloaded",timeout=90000)
                    expect(page.locator('[data-testid="stApp"]')).to_be_visible(timeout=90000)
                    check["exception_cards"]=page.locator('[data-testid="stException"]').count()
                    check["body_characters"]=len(page.locator("body").inner_text())
                    if check["exception_cards"] or check["body_characters"]<250:
                        raise AssertionError("Legal page missing or threw an exception")
                except Exception as exc:
                    failures.append("Legal "+name+": "+str(exc)[:220])
                    check["error"]=str(exc)[:220]
                report.setdefault("legal",[]).append(check)
        except Exception as exc:
            failures.append("Fatal Firefox navigation error: "+str(exc)[:400])
            try:page.screenshot(path=str(OUTPUT/"fatal.png"),full_page=True)
            except Exception:pass
        finally:
            report["failures"]=failures
            (OUTPUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            context.close()
            browser.close()
    print(json.dumps({"panels_visited":len(report["panels"]),"viewport_tests":len(report["viewports"]),
                      "legal":report.get("legal"),"panels":report.get("panels"),
                      "failures":failures},indent=2),flush=True)
    if failures:raise SystemExit(1)

if __name__=="__main__":
    audit()
