"""Real-browser launch smoke test for Car Price Studio's main Ask/Bid page.

Runs against a local Streamlit server on a CI runner. Screenshots and a
machine-readable report are uploaded as artifacts; failures raise an error.
Never presses Predict, never loads the BGE weights or calls remote model APIs.
"""
import argparse
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

CASES=(
    ("chromium","desktop-1440",1440,900,False),
    ("chromium","tablet-768",768,1024,False),
    ("chromium","android-390",390,844,True),
    ("webkit","iphone-390",390,844,True),
    ("firefox","desktop-1280",1280,800,False),
)


def visit(browser,case,base,screens):
    engine,label,width,height,mobile=case
    context=browser.new_context(
        viewport={"width":width,"height":height},
        device_scale_factor=2 if mobile else 1,
        is_mobile=mobile if engine!="firefox" else False,
        has_touch=mobile if engine!="firefox" else False,
        reduced_motion="reduce",
    )
    page=context.new_page()
    failures=[]
    page.on("pageerror",lambda err:failures.append(str(err)))
    try:
        page.goto(base,wait_until="domcontentloaded",timeout=100000)
        page.locator(".exp-hero h1").wait_for(timeout=90000)
        required=("One car. Two prices.","Ask","Bid","CVaR95",
                  "Historical date","Tell us about the car",
                  "seller Ask + wholesale Bid",
                  "Your AI explanation is included")
        # Streamlit streams widgets AFTER rendering its hero. A hero-only
        # visibility condition is not evidence that the date control or
        # form has finished mounting on a slow mobile/webkit browser.
        # Wait for ALL original release contract labels without removing
        # any of them or disguising a genuine missing UI control.
        try:
            page.wait_for_function(
                "required => required.every(t => document.body.innerText.includes(t))",
                arg=list(required),timeout=90000)
        except Exception:
            screens.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screens/(label+"-failed.png")),full_page=True)
        text=page.locator("body").inner_text(timeout=20000)
        missing=[x for x in required if x not in text]
        if missing:
            raise AssertionError("Missing public-page message: "+", ".join(missing))
        banned=("Scan description","Fine-tune the details","BART-generated")
        present=[x for x in banned if x in text]
        if present:
            raise AssertionError("Retired public controls still visible: "+str(present))
        if page.locator('[data-testid="stException"]').count():
            raise AssertionError("Streamlit uncaught exception visible on homepage")
        button=page.get_by_role("button",name=re.compile(r"Calculate seller Ask.*wholesale Bid"))
        if not button.count():
            raise AssertionError("One-form Ask/Bid action not displayed")
        if not page.get_by_text("Historical date",exact=True).count():
            raise AssertionError("Historical pricing selector is missing")
        overflow=page.evaluate(
            "() => Math.max(document.documentElement.scrollWidth,document.body.scrollWidth) - window.innerWidth")
        if overflow>35:
            raise AssertionError(f"Horizontal viewport overflow {overflow}px at width {width}px")
        screens.mkdir(parents=True,exist_ok=True)
        screenshot=screens/(label+".png")
        page.screenshot(path=str(screenshot),full_page=True)
        # Historical mode is an explicit public control, not an advanced-only setting.
        page.get_by_text("Historical date",exact=True).first.click()
        page.get_by_text(
            "Historical valuation date (January 2014 to today)",exact=False
        ).first.wait_for(timeout=25000)
        page.screenshot(path=str(screens/(label+"-historical.png")),full_page=False)
        if page.locator('[data-testid="stException"]').count():
            raise AssertionError("Date switch triggered Streamlit exception")
        if failures:
            raise AssertionError("Browser execution errors: "+str(failures[:3]))
        return {"browser":engine,"viewport":label,"width":width,"height":height,
                "mobile":mobile,"status":"passed",
                "horizontal_overflow_px":overflow,
                "historical_date_control":"passed",
                "screenshot":str(screenshot)}
    finally:
        context.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",default="http://127.0.0.1:8765")
    p.add_argument("--output",default="reports/qa/main-pricing/browser")
    p.add_argument("--browsers",default="chromium,webkit,firefox")
    args=p.parse_args()
    dest=Path(args.output);screens=dest/"screenshots"
    dest.mkdir(parents=True,exist_ok=True)
    report={"tool":"Playwright / native browsers",
            "tested_scope":"Main Ask/Bid public UX without running expensive models",
            "result":"incomplete","cases":[],"privacy":"No user data sent"}
    enabled=set(args.browsers.split(","))
    failures=[]
    with sync_playwright() as playwright:
        for case in CASES:
            name=case[0]
            if name not in enabled:continue
            try:
                browser=getattr(playwright,name).launch(headless=True)
                try:r=visit(browser,case,args.base,screens)
                finally:browser.close()
                report["cases"].append(r)
                print("PASS",name,case[1],flush=True)
            except Exception as error:
                failures.append((name,case[1],str(error)))
                report["cases"].append({"browser":name,"viewport":case[1],
                                        "status":"failed","problem":str(error)})
                print("FAIL",name,case[1],str(error),flush=True)
    report["result"]="passed" if not failures and report["cases"] else "failed"
    report["passed"]=sum(x["status"]=="passed" for x in report["cases"])
    report["expected"]=sum(x[0] in enabled for x in CASES)
    (dest/"results.json").write_text(json.dumps(report,indent=2))
    if failures:
        raise SystemExit("Cross-browser QA failed: "+str(failures))
    print("CAMPAIGN_BROWSER_QA",report["passed"],report["expected"],flush=True)


if __name__=="__main__":
    main()
