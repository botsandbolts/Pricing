"""Memory-bounded explanations of existing price receipts.

No Captum, transformer, extra model copy, SHAP, 64-coalition sweep or heavy
plotting imports. Only a user click triggers ONE native mileage re-prediction.
"""
from math import isfinite


def dollar_bridge(receipt):
    anchor=float(receipt["anchor_price_reference_usd"])
    age=float(receipt["age_factor"])
    market=float(receipt["market_factor"])
    price=float(receipt["price"])
    aged=anchor*age
    rebuilt=aged*market
    if not all(map(isfinite,(anchor,age,market,price,rebuilt))):
        raise ValueError("Nonfinite transport receipt")
    if min(anchor,age,market,price)<=0:
        raise ValueError("Invalid historical factors")
    if abs(rebuilt-price)>max(.50,price*.0001):
        raise ValueError("Historical factors do not reconcile")
    return dict(historical_anchor_usd=anchor,
                vehicle_age_effect_usd=aged-anchor,
                market_index_effect_usd=price-aged,
                selected_date_usd=price)


def single_scenario(predict,original,fields,mileage):
    before=float(original);new_miles=int(mileage)
    if not isfinite(before) or before<=0 or not 0<=new_miles<=2_000_000:
        raise ValueError("Invalid original price or mileage")
    after=float(predict({**fields,"odometer":new_miles}))
    if not isfinite(after) or not 100<=after<=10_000_000:
        raise ValueError("Native price is invalid")
    return dict(historical_original_usd=before,historical_scenario_usd=after,
                difference_usd=after-before,mileage=new_miles)


def render(saved,dated_receipts,get_pipeline):
    """Public-facing zero-extra-inference explanation until clicked."""
    import streamlit as st
    st.markdown("### 🔍 Why this price? · Lightweight edition")
    st.caption("Explanations without another neural network in your browser, "
               "or 64 simulated cars quietly eating the server's RAM.")
    if dated_receipts:
        with st.expander("📅 How the historical price became your dated estimate"):
            for side in ("ask","bid"):
                if side not in dated_receipts:continue
                st.markdown("**"+("Seller Ask" if side=="ask" else "Wholesale Bid")+"**")
                try:r=dollar_bridge(dated_receipts[side])
                except (KeyError,ValueError,TypeError):
                    st.caption("The full time factors could not be reconciled "
                               "on this model receipt. No invented waterfall.");continue
                a,b,c=st.columns(3)
                a.metric("Historical model",f"${r['historical_anchor_usd']:,.0f}")
                b.metric("Vehicle-age change",f"${r['vehicle_age_effect_usd']:+,.0f}")
                c.metric("Market-index change",f"${r['market_index_effect_usd']:+,.0f}")
                st.caption("The three amounts reconcile to "+
                           f"${r['selected_date_usd']:,.0f}. "
                           "This is calculation accounting, not causal SHAP.")
    st.markdown("**What if the SAME model saw a different odometer?**")
    old=int(saved[2]["odometer"])
    choices=sorted(set((max(0,old-10000),old,min(2_000_000,old+10000))))
    mileage=st.select_slider("Comparison mileage (miles)",options=choices,
                             value=old,key="lightweight_price_xai_miles")
    if st.button("🔬 Compare one real model scenario",
                 key="lightweight_price_xai_run",type="primary"):
        try:
            pipeline=get_pipeline()
            result=single_scenario(
                lambda car:pipeline.predict(saved[3],car,explain=False)["price"],
                saved[1]["price"],saved[2],mileage)
            st.metric("Historical asking estimate at that mileage",
                 f"${result['historical_scenario_usd']:,.0f}",
                 f"${result['difference_usd']:+,.0f} vs your actual inputs")
            st.caption("Actual model sensitivity, NOT proven causal depreciation "
                       "or a guaranteed sale price.")
        except (ValueError,TypeError,KeyError,RuntimeError,OSError,MemoryError) as exc:
            st.warning("This optional comparison is unavailable: "+str(exc))
    st.caption("Ask is an advertised price; Bid is a separate wholesale "
               "target. Their gap is not dealer profit. Historical errors "
               "do not establish November 2026 accuracy.")
