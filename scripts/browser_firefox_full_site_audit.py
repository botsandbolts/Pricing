"""Firefox / Playwright end-to-end audit of the REAL hosted Car Price Studio.

Priority: pricing one-click journey; then the other public workspaces.
No mock predictions and no blind screen-coordinate consent dismissal.
Every failure writes inspectable screenshots and JSON evidence.
"""
import json
import os
import re
from pathlib import Path
from playwright.sync_api import sync_playwright

URL=os.environ.get("PRICING_LIVE_URL","https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/")
OUT=Path("reports/qa/real-live-firefox")
OUT.mkdir(parents=True,exist_ok=True)
PANELS=[
    "🚗 Price my car","🔎 Listing Inspector","🚦 Pick your garage",
    "↔ Two-market garage","📅 Value over time","🧠 The science",
    "📊 Evaluation","🧪 Model lab","🕵️ Deal Detective",
    "🧹 Outlier lab","🏪 Merchant lens",
]

def safe_shot(page,name):
    try:
        page.screenshot(path=str(OUT/name),full_page=True,timeout=30000)
    except Exception as exc:
        return str(exc)[:250]
    return None

def actual_streamlit_frame(page,report):
    """Streamlit Community Cloud wraps the app in an iframe; always descend."""
    for _ in range(50):
        for frame in page.frames:
            try:
                if (frame.locator('[aria-label="Car Price Studio model upgrade announcement"]').count()
                        or frame.locator(".garage-home-hero").count()
                        or frame.locator('[data-testid="stApp"]').count()):
                    report["app_frame_url"]=frame.url
                    report["frame_count"]=len(page.frames)
                    return frame
            except Exception:
                continue
        page.wait_for_timeout(1000)
    report["frame_urls"]=[f.url for f in page.frames]
    raise AssertionError("Streamlit application iframe did not attach")

def dismiss_consent(page,report):
    # Streamlit custom components may render in frames. Find controls by accessible name.
    for frame in page.frames:
        for label in ("Reject analytics","Reject optional analytics","Reject all",
                      "Only essential","Necessary only","Decline"):
            control=frame.get_by_role("button",name=label,exact=False)
            try:
                if control.count() and control.first.is_visible():
                    control.first.click(timeout=6000)
                    report["consent_action"]=label
                    return
            except Exception:
                continue
    report["consent_action"]="No visible accessible rejection button"

def axe_scan(frame,label,report):
    path=Path("node_modules/axe-core/axe.min.js")
    if not path.is_file():
        report.setdefault("notes",[]).append("axe-core missing; scan not executed")
        return
    try:
        frame.add_script_tag(path=str(path.resolve()))
        findings=frame.evaluate("""async () => {
            const root=document.querySelector('[data-testid="stMain"]')||document.body;
            const out=await axe.run(root,{runOnly:{type:'tag',values:['wcag2a','wcag2aa','wcag21aa']}});
            return out.violations.map(v=>({id:v.id,impact:v.impact,
                description:v.description,nodes:v.nodes.length}));
        }""")
        report.setdefault("axe",{})[label]=findings
    except Exception as exc:
        report.setdefault("axe",{})[label]={"scan_error":str(exc)[:250]}

def viewport_check(page,frame,width,height,report):
    page.set_viewport_size({"width":width,"height":height})
    page.wait_for_timeout(800)
    result=frame.evaluate("""() => ({
        innerWidth:innerWidth,
        scrollWidth:document.documentElement.scrollWidth,
        interactiveOverflow:[...document.querySelectorAll('input,textarea,button,[role="radio"]')]
            .filter(el=>{const r=el.getBoundingClientRect();
               return r.width>1 && r.height>1 && (r.left < -2 || r.right > innerWidth+2)})
            .map(el=>(el.getAttribute('aria-label')||el.textContent||'control').slice(0,75))
            .slice(0,15)
    })""")
    result["width"]=width
    result["height"]=height
    result["pass"]=result["scrollWidth"]<=width+2 and not result["interactiveOverflow"]
    report.setdefault("viewports",[]).append(result)
    safe_shot(page,"pricing-%spx.png"%width)
    return result["pass"]

def run():
    report={"target":URL,"browser":"Firefox","hosted":True,"pricing":{},
            "panels":[],"viewports":[],"errors":[]}
    failures=[]
    with sync_playwright() as p:
        browser=p.firefox.launch(headless=True)
        context=browser.new_context(viewport={"width":390,"height":844},
                                    reduced_motion="reduce",
                                    device_scale_factor=1)
        page=context.new_page()
        page.on("pageerror",lambda exc:report["errors"].append(str(exc)[:400]))
        try:
            page.goto(URL,wait_until="domcontentloaded",timeout=90000)
            # Consent is a true modal: dismiss before asserting visual layout.
            # No blind coordinate clicks; use its accessible button in every frame.
            page.wait_for_timeout(2500)
            app=actual_streamlit_frame(page,report)
            for _ in range(8):
                dismiss_consent(page,report)
                if report.get("consent_action")!="No visible accessible rejection button":
                    break
                page.wait_for_timeout(1200)
            page.wait_for_timeout(700)
            safe_shot(page,"after-consent-before-hero.png")
            # Streamlit uses a scrollable app container, not always window.
            # Some sections are virtualized until scrolled into view.
            for region in ('[data-testid="stAppViewContainer"]', '[data-testid="stMain"]'):
                try:
                    app.locator(region).first.evaluate('el=>{el.scrollTop=0}')
                except Exception:
                    pass
            hero=app.locator(".garage-home-hero")
            try:
                hero.wait_for(state="visible",timeout=55000)
            except Exception:
                report["hero_count"]=hero.count()
                report["initial_body_text"]=app.locator("body").inner_text()[:3500]
                report["dom_summary"]=app.evaluate("""() => ({
                  title:document.title,
                  url:location.href,
                  bodyChildren:document.body.children.length,
                  mainCount:document.querySelectorAll('main').length,
                  links:[...document.querySelectorAll('a')].slice(0,30).map(a=>a.textContent.trim().slice(0,80)),
                  scrollHeight:document.documentElement.scrollHeight,
                  appViewScrollHeight:document.querySelector('[data-testid="stAppViewContainer"]')?.scrollHeight,
                  appViewClientHeight:document.querySelector('[data-testid="stAppViewContainer"]')?.clientHeight
                })""")
                report["streamlit_exception_cards"]=app.locator('[data-testid="stException"]').count()
                raise
            release=app.locator('[aria-label="Car Price Studio model upgrade announcement"]')
            release.wait_for(state="attached",timeout=60000)
            report["pricing"]["hero_initial"]=hero.count()
            report["pricing"]["release_initial"]=release.count()
            safe_shot(page,"before-cta.png")
            cta=app.get_by_role("link",name="Go to the car pricing calculator")
            assert cta.get_attribute("href")=="#studio-calculator", "CTA still changes URL/routes"
            cta.click(timeout=30000)
            app.wait_for_function("location.hash === '#studio-calculator'",timeout=30000)
            sample=app.get_by_label(re.compile("Search .*real, price-redacted car adverts"))
            sample.wait_for(state="visible",timeout=90000)
            assert hero.count()==1 and release.count()==1, "A banner disappeared after one CTA click"
            top=app.locator("#studio-calculator").evaluate(
                "el=>Math.round(el.getBoundingClientRect().top)")
            report["pricing"].update({"single_cta_click":True,"sample_visible":True,
                 "both_banners_preserved":True,"anchor_viewport_top":top})
            assert top<844, "One click did not bring the pricing section into view"
            safe_shot(page,"after-one-cta.png")
            submit=app.get_by_role("button",name=re.compile("Calculate seller Ask.*wholesale Bid"))
            submit.wait_for(state="visible",timeout=90000)
            if submit.is_disabled():
                raise AssertionError("Public pricing submit is disabled")
            submit.scroll_into_view_if_needed()
            submit.click(timeout=30000)
            report["pricing"]["submit_clicks"]=1
            # The visible acknowledgement or completed result must follow ONE event.
            messages=app.get_by_text(re.compile(
                r"One click received|Calculation complete|Your car . estimated prices|"
                r"Ask engine failed|Seller Ask couldn't finish|Wholesale Bid couldn't finish|"
                r"Pricing runtime is unavailable",re.I))
            messages.first.wait_for(state="visible",timeout=240000)
            report["pricing"]["one_click_reached_server"]=True
            report["pricing"]["first_response"]=messages.first.inner_text()[:240]
            safe_shot(page,"after-one-submit.png")
            page.wait_for_timeout(1000)
            body=app.locator("body").inner_text(timeout=10000)
            if "Calculation complete." in body and "One click received." in body:
                failures.append("Finished calculation still shows the processing notice")
            for width,height in ((390,844),(820,1100),(1440,900),(320,800)):
                if not viewport_check(page,app,width,height,report):
                    failures.append("Responsive overflow at %spx"%width)
            axe_scan(app,"pricing-mobile",report)
            page.set_viewport_size({"width":1440,"height":900})
            axe_scan(app,"pricing-desktop",report)
            # The same long-lived session must navigate every other panel without
            # dropping into a hidden no-navigation route or raising a Streamlit error.
            for label in PANELS:
                item={"panel":label}
                try:
                    radio=app.get_by_role("radio",name=label,exact=True)
                    if not radio.count():
                        radio=app.get_by_role("button",name=label,exact=True)
                    radio.wait_for(state="visible",timeout=60000)
                    radio.click(timeout=30000)
                    page.wait_for_function(
                        """name => [...document.querySelectorAll('[role="radio"]')]
                          .some(x=>x.getAttribute('aria-checked')==='true' &&
                                 (x.getAttribute('aria-label')||x.textContent||'').includes(name))""",
                        arg=label,timeout=45000)
                    page.wait_for_timeout(1300)
                    item["selected"]=True
                    item["exception_cards"]=app.locator('[data-testid="stException"]').count()
                    if item["exception_cards"]:
                        failures.append("Streamlit exception on panel "+label)
                    item["horizontal_overflow"]=app.evaluate(
                        "document.documentElement.scrollWidth > innerWidth+2")
                    safe_shot(page,"panel-%02d.png"%(PANELS.index(label)+1))
                except Exception as exc:
                    item["error"]=str(exc)[:350]
                    failures.append("Panel navigation failed: "+label)
                    safe_shot(page,"panel-error-%02d.png"%(PANELS.index(label)+1))
                report["panels"].append(item)
        except Exception as exc:
            report["fatal_error"]=str(exc)[:650]
            failures.append("Pricing priority flow failed: "+str(exc)[:240])
            safe_shot(page,"fatal-state.png")
        finally:
            report["final_url"]=app.url if "app" in locals() else page.url
            report["failures"]=failures
            (OUT/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            context.close()
            browser.close()
    print(json.dumps({"pricing":report["pricing"],"panel_count":len(report["panels"]),
                      "failures":failures,"errors":report["errors"]},indent=2),flush=True)
    if failures:
        raise SystemExit(1)

if __name__=="__main__":
    run()
