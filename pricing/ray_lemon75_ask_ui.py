"""Evidence-led 8 November 2026 Lemon75 edition; no unfiltered CVaR figure."""
def render(root):
    import streamlit as st
    from pricing.ray_lemon75_ask_live import manifest
    m=manifest(root)
    if m is None:
        return
    k=m["filtered_validation"]
    st.markdown("""
    <style>
    .ray-lemon-hero{background:radial-gradient(circle at 95% 5%,#c8ed7e77,transparent 30%),
      linear-gradient(122deg,#101e32 5%,#12434a 56%,#345d43 110%);
      border:1px solid #b8e5a1;border-radius:22px;
      padding:clamp(20px,3vw,34px);box-shadow:0 16px 55px #10232a27;
      color:white;margin:14px 0 24px}
    .ray-lemon-hero small{letter-spacing:.13em;color:#d6fbb7;
      text-transform:uppercase;font-weight:800;font-size:.79rem}
    .ray-lemon-hero h2{color:white!important;font-weight:900;
      font-size:clamp(1.7rem,4vw,2.6rem);line-height:1.13;
      letter-spacing:-.045em;margin:.35em 0}
    .ray-lemon-hero p{max-width:810px;color:#e3f1ef;line-height:1.58;font-size:1rem}
    .ray-lemon-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:15px}
    .ray-lemon-pills span{color:white;background:#ffffff17;
      border:1px solid #d8efe877;border-radius:999px;
      padding:6px 11px;font-size:.81rem}
    .ray-lemon-link{display:inline-block;background:#eaf9a9;
      color:#17312c!important;border-radius:11px;margin-top:15px;
      font-weight:800;padding:11px 16px;text-decoration:none}
    </style>""",unsafe_allow_html=True)
    st.markdown("""
    <section class="ray-lemon-hero" aria-label="Lemon75 research release">
    <small>🇩🇪 Car Price Studio · Hamburg · 8 November 2026 · Lemon Edition</small>
    <h2>We squeezed the lemons.<br>Not the numbers. 🍋</h2>
    <p>Meet today's new Ray-tuned DeepCross asking-price challenger:
    it reads the advert, learns car details, and uses a compact ONNX INT8
    neural pricing engine. Independent. Open-source. No registration.
    No ads. Actual comparisons, not magic beans.</p>
    <div class="ray-lemon-pills">
      <span>✅ Parity-checked INT8 neural head</span>
      <span>🍋 75% retained-cohort validation</span>
      <span>🛡 CVaR95: costly-miss focus</span>
      <span>🔬 See the evidence</span>
      <span>🚗 Two distinct prices: Ask and Bid</span>
    </div>
    <a class="ray-lemon-link" href="#studio-calculator">Try my car</a>
    </section>
    """,unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Lemon-validation CVaR95","$"+format(k["CVaR95_absolute_USD"],",.0f"))
    b.metric("Lemon-validation MAE","$"+format(k["MAE_USD"],",.0f"))
    c.metric("Lemon-validation RMSE","$"+format(k["RMSE_USD"],",.0f"))
    st.caption(
        format(int(k["rows"]),",")+" retained historical asking listings; "
        "lower is better. Conditional retrospective validation after "
        "excluding 25% by an independent reference model; not a guaranteed "
        "error for future cars.")
    with st.expander("💼 The business case · fewer expensive pricing surprises"):
        st.markdown(
            "**Ask** estimates a seller's advertised price, not a completed "
            "transaction. **Bid** estimates a separate wholesale auction "
            "price, and the difference is not dealer profit. "
            "CVaR95 is the average absolute dollar error among the **worst "
            "5%** of retained comparison cars. It highlights costly misses. "
            "MAE describes average absolute misses; RMSE penalizes large "
            "errors more heavily.")
    with st.expander("🧪 Lemon: science, sample selection, and limitations"):
        st.markdown(
            "A frozen out-of-fold reference ranked original training cars "
            "by absolute log-price residual. Its hardest 25% were removed, "
            "leaving **110,832** listings for this Lemon training study. "
            "For this report, the same independent reference rule retained "
            "**23,751 of 31,668** historical validation cars. The filtered "
            "selection does use observed historical prices, so these scores "
            "are **optimistic for that selected cohort** and are not an "
            "unbiased estimate of future unknown-car accuracy. The full "
            "original holdout is still verified in the release audit.")
    with st.expander("⚙️ What exactly is ONNX INT8?"):
        st.markdown(
            "The whole DeepCross **pricing network** runs via ONNX Runtime "
            "with 8-bit dynamic quantization for matrix operations. "
            "The exact original checkpoint is preserved for optional "
            "gradient explanations. The shared BGE text encoder runs "
            "as INT8 only if its **separate** tokenization/price/memory "
            "parity gate has passed. Compression does not prove a model "
            "predicts more accurately.")
        q=m["native_vs_int8_parity"]
        st.caption(
            "Neural price-head file size reduced by "
            +format(q["weight_shrink_pct"],".1f")+"%; "
            "95th-percentile relative prediction drift "
            +format(q["native_vs_int8_p95_relative"]*100,".2f")+"%.")
    with st.expander("🔎 Evidence, open source and historical limits"):
        st.markdown(
            "This model predicts **2021 US advertised asking prices**. "
            "Transporting historical prices to the current date is a "
            "scenario, not a verified November 2026 transaction. "
            "The original test set was not used for this promotion. "
            "This is today's strongest measured Lemon-75% neural contender "
            "in the compared studies—not a claim to be world's best. "
            "[Original one-hour Ray study](https://github.com/botsandbolts/"
            "Pricing/actions/runs/42485146978) · "
            "[Open-source repository](https://github.com/botsandbolts/Pricing) · "
            "[CVaR science](https://doi.org/10.21314/JOR.2000.038).")
