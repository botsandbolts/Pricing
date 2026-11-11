"""36-language launch banner for independently verified U.S. Bid ResMLP."""
from html import escape
from pathlib import Path
import json

def _new_resmlp_banner():
    """Localized for each of the site's exact 36 locales; only after verified install."""
    from pricing.us_bid_resmlp_live import available, FOLDER
    from pricing.bid_launch_i18n import copy
    from pricing.i18n import locale
    root=Path(__file__).resolve().parents[1]
    if not available(root):
        return None
    m=json.loads((root/FOLDER/"manifest.json").read_text())
    translations=copy(locale())
    score=m["original_validation"]
    original=f"{int(score['rows']):,}"
    metrics=[
      ("CVaR95",float(score["CVaR95_absolute_USD"]),
       "Mean absolute error among the worst 5% of historical validation sales"),
      ("MAE",float(score["MAE_USD"]),"Mean absolute error on original validation"),
      ("RMSE",float(score["RMSE_USD"]),"Root mean squared validation error")]
    tiles="".join(
        f'<div style="flex:1 1 130px;background:#ffffff14;border:1px solid #cddae174;'
        f'border-radius:13px;padding:12px 14px" title="{escape(hint,quote=True)}">'
        f'<div style="font-size:.83rem;color:#dceaf5;font-weight:760">{name}</div>'
        f'<strong style="display:block;font-size:clamp(1.3rem,3.3vw,1.9rem);'
        f'letter-spacing:-.03em;color:#ffe4a0">{value:,.0f} USD</strong></div>'
        for name,value,hint in metrics)
    return f"""
<section role="region" aria-labelledby="us-bid-resmlp-launch" lang="{escape(locale(),quote=True)}"
  style="background:linear-gradient(125deg,#0c1d37 0%,#123b66 52%,#135a55 100%);
         color:#fff;border:2px solid #f6cc7b;border-radius:19px;
         padding:clamp(18px,3vw,30px);margin:5px 0 17px;
         box-shadow:0 14px 32px #071a313d">
  <div style="color:#ffe4a0;font-weight:800;letter-spacing:.07em;
              font-size:.73rem;line-height:1.5">
    {escape(translations["eyebrow"])}
  </div>
  <h2 id="us-bid-resmlp-launch"
      style="color:#fff!important;font-weight:900;line-height:1.12;
             font-size:clamp(1.55rem,3.3vw,2.35rem);margin:11px 0">
    🚀 {escape(translations["title"])}
  </h2>
  <p style="color:#eff7ff;font-size:1rem;line-height:1.55;margin:0 0 15px">
    {escape(translations["description"])}
  </p>
  <div style="display:flex;flex-wrap:wrap;gap:10px" aria-label="Historical validation errors">
    {tiles}
  </div>
  <p style="color:#e0ebf7;font-size:.85rem;line-height:1.48;margin:15px 0 12px">
    {escape(translations["scope"])}
  </p>
  <div style="display:flex;gap:13px;align-items:center;flex-wrap:wrap">
    <a href="#studio-calculator" style="display:inline-block;background:#f6cc7b;color:#11263b;
       border-radius:10px;padding:12px 17px;font-size:.96rem;font-weight:850;
       text-decoration:none">{escape(translations["cta"])} →</a>
    <a href="https://github.com/botsandbolts/Pricing/actions/runs/{int(m['source_workflow_run'])}"
       target="_blank" rel="noopener noreferrer" style="color:#ffe4a0;
       font-size:.86rem;text-underline-offset:3px">Ray Tune · {original} validation records ↗</a>
  </div>
</section>
"""

def banner_html():
    """Only promote live release messaging after the verified model is installed."""
    from pricing.granite_aft_banner import banner_html as granite_banner
    return granite_banner() or _new_resmlp_banner() or ""
