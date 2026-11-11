"""Accessible, 36-language release banner for sole Granite Ask and current Bid.

Explains interval censoring as a data/likelihood concept, not a false claim
that the active Bid ResMLP uses interval-censored AFT.
"""
from html import escape
from pathlib import Path
from pricing.granite_aft_i18n import copy
from pricing.i18n import locale
from pricing.granite_aft_ask_live import available,manifest

def banner_html(root=None):
    root=Path(root or Path(__file__).resolve().parents[1])
    if not available(root):
        return ""
    m=manifest(root)
    t=copy(locale())
    metrics=m["validation"]
    scores=(("CVaR95",metrics["CVaR95_absolute_USD"]),
            ("MAE",metrics["MAE_USD"]),("RMSE",metrics["RMSE_USD"]))
    tiles="".join(
        '<div style="flex:1 1 120px;min-width:0;background:#FFFFFF16;'
        'border:1px solid #BDD4E68A;border-radius:12px;padding:11px 15px;">'
        '<span style="display:block;color:#C7E2E7;font-size:.81rem;font-weight:700">'
        +label+'</span><strong style="display:block;color:#FFE9B5;'
        'font-size:clamp(1.22rem,3.2vw,1.9rem);font-variant-numeric:tabular-nums">'
        +f'{float(value):,.0f} USD'+'</strong></div>'
        for label,value in scores)
    title=escape(t["title"])
    intro=escape(t["intro"])
    historical=escape(t["historical"])
    bid_title=escape(t["bid_title"])
    bid_text=escape(t["bid_explanation"])
    cta=escape(t["cta"])
    code=escape(locale(),quote=True)
    return f"""
<section lang="{code}" role="region" aria-labelledby="granite-aft-release-heading"
  style="padding:clamp(19px,3vw,32px);border-radius:19px;overflow:hidden;
    background:linear-gradient(125deg,#091D33,#11426C 54%,#08695E);
    color:#fff;border:2px solid #F9D17D;box-shadow:0 13px 32px #092A483B;
    margin:6px 0 18px">
 <div style="letter-spacing:.11em;font-size:.75rem;font-weight:800;color:#FFE2A2">
 🌍 CAR PRICE STUDIO · GRANITE AFT · 36 LANGUAGES · MADE IN GERMANY 🇩🇪
 </div>
 <h2 id="granite-aft-release-heading" style="color:white!important;line-height:1.12;
   font-size:clamp(1.55rem,3.3vw,2.35rem);font-weight:900;margin:12px 0 10px">
   {title}
 </h2>
 <p style="font-size:1.04rem;line-height:1.55;color:#F0FAFF;margin:0 0 15px">
   {intro}
 </p>
 <div style="display:flex;flex-wrap:wrap;gap:10px" aria-label="2021 historical Ask validation error metrics">
 {tiles}
 </div>
 <p style="color:#D9EAF5;font-size:.86rem;line-height:1.55;margin:13px 0 10px">
 {historical}
 </p>
 <div role="note" style="border-left:4px solid #F9D17D;
   background:#06192B50;border-radius:0 11px 11px 0;
   padding:12px 15px;margin:11px 0 16px;">
  <strong style="display:block;color:#FFE2A2;font-size:.97rem;margin-bottom:5px">
  🔬 {bid_title}</strong>
  <span style="display:block;color:#E5F2F8;font-size:.91rem;line-height:1.57">
   {bid_text}
  </span>
 </div>
 <div style="display:flex;align-items:center;flex-wrap:wrap;gap:15px">
  <a href="#studio-calculator" style="background:#FFE2A2;color:#142C3F;
     padding:11px 16px;border-radius:11px;font-weight:850;
     text-decoration:none;display:inline-block">
     {cta} →</a>
  <a href="https://github.com/botsandbolts/Pricing/actions/runs/{int(m['source_run_id'])}"
     target="_blank" rel="noopener noreferrer"
     style="color:#FCE5B8;font-size:.87rem;text-decoration:underline;text-underline-offset:3px">
      Ray Tune · {int(metrics['rows']):,} validation records</a>
 </div>
</section>
"""
