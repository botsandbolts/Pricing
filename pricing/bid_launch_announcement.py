"""Accessible, self-contained launch announcement for the wholesale Bid model.

Presentation only. Intentionally avoids importing research or inference models.
"""
from html import escape


def banner_html():
    """Announce a conditional candidate release without claiming proven scores."""
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
