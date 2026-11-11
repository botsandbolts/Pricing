"""Accessible, self-contained launch announcement for the wholesale Bid model.

Presentation only. Intentionally avoids importing research or inference models.
"""
from html import escape
from pathlib import Path
import json


def _live_winner_banner():
    """Render the actually installed loss and honestly labelled measurements."""
    root = Path(__file__).resolve().parents[1]
    from pricing.ray_bid_winner_live import available
    if not available(root):
        return None
    m=json.loads((root/"models/ray-bid-9am-winner/manifest.json").read_text())
    internal=float(m["retrained_internal_CVaR95_USD"])
    external=float(m["original_validation"]["CVaR95_absolute_USD"])
    mae=float(m["original_validation"]["MAE_USD"])
    rmse=float(m["original_validation"]["RMSE_USD"])
    cfg=m["configuration"]
    prior_cvar95=8436.360576611467
    delta_percent=(external/prior_cvar95-1.0)*100.0
    return f"""
<section role="region" aria-labelledby="bid-breakthrough-title"
  style="background:linear-gradient(125deg,#082d35 0%,#114452 58%,#183c35 100%);
         color:#f7faf7;border:2px solid #f8d67a;border-radius:20px;
         padding:clamp(17px,3.6vw,28px);margin:4px 0 17px;
         box-shadow:0 12px 28px rgba(7,34,42,.19)">
  <div style="color:#ffe49a;font-weight:900;letter-spacing:.08em;font-size:.8rem">
    ✅ DEPLOYED · LIVE BID ENGINE · INDEPENDENT · MADE IN GERMANY 🇩🇪 · 9 NOV 2026
  </div>
  <h2 id="bid-breakthrough-title" style="color:#fff8d9;font-size:clamp(1.6rem,4vw,2.5rem);
    font-weight:900;line-height:1.15;margin:13px 0">
    Deployed! Meet the new Ray Tune Bid brain. 🧠🚗🍋🚫
  </h2>
  <p style="color:#f5fffa;font-size:1.05rem;line-height:1.55">
    The <strong>Ray Tune TRAIN-internal CVaR95 winner</strong> is now deployed:
    a newly retrained <strong>non-Lemon DeepFM16</strong> model using
    <strong>CVaR95-weighted Huber loss</strong>, not AFT.
    It estimates historical wholesale auction sale prices from structured vehicle
    facts. No Lemon exclusions, no XGBoost blending, no seller-description encoder.
    The model aims to limit costly tail errors, rather than optimizing only the average.
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
    gap:12px;margin:13px 0">
    <div style="padding:15px;border:1px solid #ffffff61;border-radius:14px;background:#ffffff12">
      <small style="color:#d5e7e1">TRAIN-internal CVaR95 · selection</small>
      <strong style="display:block;color:#ffe49a;font-size:1.7rem">&#36;{internal:,.2f}</strong>
    </div>
    <div style="padding:15px;border:1px solid #ffffff61;border-radius:14px;background:#ffffff12">
      <small style="color:#d5e7e1">Untouched original validation · CVaR95</small>
      <strong style="display:block;color:#ffe49a;font-size:1.7rem">&#36;{external:,.2f}</strong>
    </div>
    <div style="padding:15px;border:1px solid #ffffff61;border-radius:14px;background:#ffffff12">
      <small style="color:#d5e7e1">Untouched original validation · MAE</small>
      <strong style="display:block;color:#ffe49a;font-size:1.7rem">&#36;{mae:,.2f}</strong>
    </div>
  </div>
  <div style="background:#ffffff10;border:1px solid #ffffff4f;
    padding:14px 16px;border-radius:14px;margin:12px 0">
    <strong style="color:#ffe49a">🔬 Inside the Ray Tune winning model</strong>
    <p style="margin:8px 0;color:#f7faf9;line-height:1.54">
      Ten parallel shards compared 14 loss functions across four model families.
      The selected architecture combines <strong>rank {int(cfg["rank"])}
      factorization-machine interactions</strong> with a
      <strong>{escape(cfg["shape"])} GELU neural branch</strong>.
      The objective adds a dollar-denominated penalty for the
      <strong>worst {int(float(cfg["tail_frac"])*100)}% of prediction errors</strong>
      to robust Huber regression on log prices. The tail penalty weight is
      <strong>{float(cfg["tail_weight"]):.2f}</strong>.
    </p>
    <p style="margin:8px 0;color:#d6ede6;font-size:.9rem;line-height:1.55">
      Exact training setup: RAdam optimizer · cosine learning-rate schedule ·
      learning rate {float(cfg["lr"]):.7f} ·
      dropout {float(cfg["dropout"])*100:.0f}% ·
      batch size {int(cfg["batch"]):,} ·
      best saved epoch {int(m["retrained_best_epoch"])}.
      From {int(m["training_split"]["full_train"]):,} original TRAIN cars,
      {int(m["training_split"]["fit"]):,} fitted the weights and
      {int(m["training_split"]["inner"]):,} selected the checkpoint.
      Original validation contains {int(m["original_validation"]["rows"]):,}
      other cars (MAE &#36;{mae:,.2f}, RMSE &#36;{rmse:,.2f}).
    </p>
  </div>
  <p style="color:#d8eeea;font-size:.94rem;line-height:1.55">
    <strong>What does CVaR95 measure?</strong> The mean absolute dollar error
    among the worst 5% of cars. The smaller TRAIN-internal figure is a
    <em>selection</em> metric, not independent accuracy.
    On untouched original validation, this winner's CVaR95 is
    <strong>{delta_percent:.1f}% higher</strong> than the previous BOHB Bid
    release's &#36;{prior_cvar95:,.2f}. An internal win is not proof of
    better generalization. Historical 2014–15 results do not validate 2026 prices.
    The original TEST data were not used to choose this checkpoint.
  </p>
  <div style="padding:12px 15px;border-left:4px solid #f8d67a;background:#071f2780">
    <strong style="color:#ffe49a">🛃 Customs &amp; law enforcement</strong>
    <p style="margin:6px 0;color:#f7faf9">
      Screening estimates can flag unusual declarations and support research;
      they are <strong>not official customs values, proof of wrongdoing, or
      a substitute for investigation and professional appraisal.</strong>
    </p>
  </div>
  <details style="margin-top:12px;color:#dff2eb">
    <summary style="cursor:pointer;color:#ffe49a;font-weight:800">🔬 The actual science</summary>
    <p>
      The original unfiltered, non-Lemon auction dataset contains exact sale
      prices with no verified interval-censoring bounds; none were invented.
      We reserved 10% of TRAIN for internal selection. No Lemon exclusions,
      XGBoost blend or text encoder. AFT and interval likelihood objectives
      were among competing experiments, <em>not</em> the installed objective.
      Robust Huber loss does not automatically correct truncation or
      selection bias. The original TEST set was not used to choose or evaluate
      this retraining.
    </p>
  </details>
  <p style="font-size:.87rem;color:#ffe49a;font-weight:700;margin:15px 0 0">
    🇩🇪 Independent · Open source · No ads · No registration ·
    <a href="https://github.com/botsandbolts/Pricing/actions/runs/42584931682"
       target="_blank" rel="noopener noreferrer" style="color:#fff7cd;text-decoration:underline">
      Retraining and verification receipts ↗</a>
  </p>
  <p style="font-size:.9rem;color:#cceee0;margin:9px 0 0">
    <strong>Less haggle. More signal. Science before sirens.</strong> 🚓
  </p>
</section>
"""



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
    """Show the verified installed winner, or the existing conditional candidate bulletin."""
    breakthrough = _new_resmlp_banner()
    if breakthrough is not None:
        return breakthrough
    from pricing.i18n import locale
    if locale() not in ("en-GB","en-US"):
        return ""
    live = _live_winner_banner()
    if live is not None:
        return live
    run_url = "https://github.com/botsandbolts/Pricing/actions/runs/42570947698"
    return f"""
<section role="region" aria-labelledby="bid-breakthrough-title"
  style="background:linear-gradient(125deg,#082d35 0%,#114452 58%,#183c35 100%);
         color:#f7faf7;border:2px solid #f8d67a;border-radius:20px;
         padding:clamp(17px,3.6vw,28px);margin:4px 0 17px;
         box-shadow:0 12px 28px rgba(7,34,42,.19)">
  <div style="display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:12px">
    <span style="display:inline-block;background:#f8d67a;color:#142d32;
      padding:7px 11px;border-radius:7px;font-size:.76rem;font-weight:900;
      letter-spacing:.09em">🚨 BID REVOLUTION · RELEASE CANDIDATE</span>
    <span style="font-size:.78rem;font-weight:750;color:#dcf9ed">
      CAR PRICE STUDIO · 9 NOVEMBER 2026 · MADE IN GERMANY 🇩🇪
    </span>
  </div>
  <h2 id="bid-breakthrough-title"
      style="color:#fff8d9;font-size:clamp(1.65rem,4.4vw,2.65rem);
             font-weight:950;line-height:1.13;letter-spacing:-.035em;margin:0 0 11px">
    The bid revolution is in the pit lane. 🏁
  </h2>
  <p style="color:#f6fbfb;font-size:clamp(1.02rem,2.4vw,1.18rem);
            font-weight:650;line-height:1.55;margin:0 0 14px">
    Our next-generation <strong style="color:#ffe29a">wholesale auction-value engine</strong>
    is undergoing its final research race. We're aiming to launch a stronger Bid model
    <strong style="color:#ffe29a">in the next few hours</strong>—<em>only if</em>
    independent accuracy, stability and release checks pass.
    Your current Bid calculator stays available in the meantime.
  </p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,175px),1fr));
              gap:10px;margin:15px 0">
    <div style="background:#ffffff13;border:1px solid #ffffff4f;
                border-radius:12px;padding:13px">
      <strong style="display:block;color:#ffe29a;margin-bottom:5px">🧪 AFT likelihood</strong>
      <span style="color:#f0f8f7;line-height:1.46;font-size:.92rem">
        Learns from positive sale prices. Can use genuine price intervals
        when verified lower and upper bounds are available.
      </span>
    </div>
    <div style="background:#ffffff13;border:1px solid #ffffff4f;
                border-radius:12px;padding:13px">
      <strong style="display:block;color:#ffe29a;margin-bottom:5px">🛡️ Robust Huber loss</strong>
      <span style="color:#f0f8f7;line-height:1.46;font-size:.92rem">
        Softens the influence of extreme errors. One outrageous auction
        should not drive the entire market estimate.
      </span>
    </div>
    <div style="background:#ffffff13;border:1px solid #ffffff4f;
                border-radius:12px;padding:13px">
      <strong style="display:block;color:#ffe29a;margin-bottom:5px">🎯 Worst-case focus</strong>
      <span style="color:#f0f8f7;line-height:1.46;font-size:.92rem">
        We select by CVaR95: the average absolute dollar error among
        the worst 5% of predictions, not just the typical miss.
      </span>
    </div>
  </div>
  <div style="padding:13px 16px;border-left:4px solid #f8d67a;
              background:#071f2780;border-radius:0 10px 10px 0;margin:12px 0">
    <strong style="color:#ffe29a">🛃 WHY CUSTOMS &amp; LAW ENFORCEMENT SHOULD CARE</strong>
    <p style="margin:7px 0 0;color:#f7faf9;line-height:1.53;font-size:.95rem">
      Independent wholesale-price benchmarks can help flag unusual vehicle
      declarations, prioritize inspections and support asset-valuation research.
      When an error matters, a few thousand dollars are not pocket change.
      <strong>But a model estimate is not a legal customs value, proof of
      wrongdoing or a substitute for evidence and professional judgment.</strong>
    </p>
  </div>
  <details style="margin:11px 0;font-size:.88rem;color:#ddedea">
    <summary style="cursor:pointer;color:#f8d67a;font-weight:800">
      The scientific fine print · Censoring is not truncation
    </summary>
    <p style="margin:9px 0 0;line-height:1.55">
      We're testing a combined accelerated failure time (AFT) and Huber
      objective alongside other loss functions on the full, unfiltered historical
      auction training cohort. Despite its name, AFT here models <em>log price</em>,
      not time until failure. Current observed auction sale prices are exact
      labels; verified row-level interval-censoring bounds have not been established.
      Truncation and selection bias are separate data problems, and this loss
      does not automatically correct them. Any claim of superior accuracy
      awaits matched held-out evaluation.
    </p>
  </details>
  <p style="margin:12px 0 0;display:flex;flex-wrap:wrap;gap:8px 17px;
            align-items:center;color:#ffe6a6;font-size:.87rem;font-weight:750">
    <span>🇩🇪 Independent · Open source · No ads · No registration</span>
    <a href="{escape(run_url, quote=True)}" target="_blank" rel="noopener noreferrer"
       style="color:#fff9db;text-decoration:underline;text-underline-offset:3px">
      Follow the public model research ↗
    </a>
  </p>
  <p style="font-size:.9rem;color:#d5f2e9;margin:11px 0 0;line-height:1.4">
    <strong>Less haggle. More signal. Science before sirens.</strong> 🚓
  </p>
</section>
"""
