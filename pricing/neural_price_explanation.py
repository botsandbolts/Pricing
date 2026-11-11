"""On-demand, model-faithful price explanations for the active U.S. neural engines.

Captum runs only after an explicit Explain click. Keep the frozen feature
pipeline, shared model instance, and the 24-step/one-example memory ceiling.
All attributions describe the HISTORICAL neural estimate, not market-clock
adjustments or a causal effect of editing an advertisement.
"""
from __future__ import annotations

import math
import numpy as np

_STEPS = 24
_LOG_FLOOR = math.log(100.0)
_LOG_CEILING = math.log(1e6)


def _price_from_log(value):
    return float(math.exp(float(np.clip(value, _LOG_FLOOR, _LOG_CEILING))))


def _inputs(root, market, saved):
    import torch

    if market == "ask":
        from pricing.granite_aft_ask_live import CATS, load

        model = load(root)
        if "ask_embedding" not in saved:
            raise ValueError("The original Granite description vector is missing; calculate a new quote.")
        prepared = {"raw": saved["notes"], "embedding": saved["ask_embedding"]}
        numeric, categories, embedding = model.encode_inputs(saved["fields"], prepared)
        network = model.price_network
        text_tensor = torch.from_numpy(np.ascontiguousarray(embedding, dtype=np.float32))
        categorical_names = CATS
        numeric_names = (
            "Model year", "Vehicle age in 2021", "Log mileage",
            "Mileage per year", "Description length", "Description words",
            "Description punctuation", "Expected annual mileage",
            "Mileage relative to expected", "Excess mileage",
        )
    elif market == "bid":
        from pricing.neural_bid_blend_live import CATS, _norm, raw_numeric
        from pricing.us_bid_resmlp_live import available as resmlp_available
        from pricing.us_bid_resmlp_live import load as load_resmlp
        from pricing.ray_bid_winner_live import load as load_ray

        if "bid_fields" not in saved:
            raise ValueError("The historical auction inputs are missing; calculate a new quote.")
        fields = saved["bid_fields"]
        if resmlp_available(root):
            model = load_resmlp(root)
            numeric, categories = model.features(fields)
            network = model.network
        else:
            model = load_ray(root)
            raw = raw_numeric(fields)
            numeric = np.clip(
                (np.where(np.isfinite(raw), raw, model.median) - model.median)
                / model.scale, -8, 8
            ).astype(np.float32).reshape(1, 5)
            categories = np.asarray(
                [model.maps[name].get(_norm(fields.get(name)), 1) for name in CATS],
                dtype=np.int64,
            )[None, :]
            network = model.model
        text_tensor = None
        categorical_names = CATS
        numeric_names = (
            "Model year", "Age at auction", "Log mileage",
            "Mileage relative to age", "Auction grade",
        ) + (
            ("Expected annual mileage", "Mileage relative to expected", "Excess mileage")
            if numeric.shape[1] == 8 else ()
        )
    else:
        raise ValueError("Unrecognized price market")

    if numeric.shape != (1, len(numeric_names)):
        raise ValueError("Numeric feature schema does not match the explanation")
    if categories.shape != (1, len(categorical_names)):
        raise ValueError("Categorical feature schema does not match the explanation")
    num_tensor = torch.from_numpy(np.ascontiguousarray(numeric, dtype=np.float32))
    cat_tensor = torch.from_numpy(np.ascontiguousarray(categories, dtype=np.int64))
    return network, num_tensor, cat_tensor, text_tensor, numeric_names, categorical_names


def calculate(root, market, saved):
    """Explain exactly the neural checkpoint that supplied this saved quote.

    Numeric and (for Ask) Granite embedding Integrated Gradients use a
    standardized-median numeric/zero-vector baseline with actual categories
    held fixed. Categorical effects are separate one-field-at-a-time
    unknown-category counterfactuals, not SHAP values.
    """
    import torch
    from captum.attr import IntegratedGradients

    if market not in saved or "price" not in saved[market]:
        raise ValueError("There is no completed valuation to explain")

    network, num, cat, text, numeric_names, category_names = _inputs(
        root, market, saved
    )

    def forward(x_num, x_text=None, *, categories=cat):
        batch_categories = categories.expand(x_num.shape[0], -1)
        if market == "ask":
            return network(x_num, batch_categories, x_text)[:, 0]
        return network(x_num, batch_categories).reshape(-1)

    network.eval()
    with torch.no_grad():
        actual_log = float(forward(num, text)[0].item())
    historical = float(saved[market]["price"])
    if not math.isclose(_price_from_log(actual_log), historical, rel_tol=2e-4, abs_tol=.5):
        raise ValueError("The explanation model does not reproduce this quote")
    inputs = (num, text) if text is not None else num
    baselines = (torch.zeros_like(num), torch.zeros_like(text)) if text is not None else torch.zeros_like(num)
    ig = IntegratedGradients(forward)
    attributions = ig.attribute(
        inputs=inputs, baselines=baselines, n_steps=_STEPS,
        internal_batch_size=1,
    )
    if text is not None:
        num_contributions = attributions[0][0].detach().cpu().numpy().astype(float)
        text_contribution = float(attributions[1][0].detach().sum().cpu().item())
    else:
        num_contributions = attributions[0].detach().cpu().numpy().astype(float)
        text_contribution = 0.0
    if not np.isfinite(num_contributions).all() or not math.isfinite(text_contribution):
        raise ValueError("The neural attribution contains invalid values")

    if market == "ask":
        grouped = [
            ("Model year and age", float(num_contributions[[0, 1]].sum())),
            ("Mileage and expected usage", float(num_contributions[[2, 3, 7, 8, 9]].sum())),
            ("Description text and wording", float(num_contributions[[4, 5, 6]].sum()) + text_contribution),
        ]
        fact_fields = saved["fields"]
    else:
        mileage_indices = [2, 3] + ([5, 6, 7] if len(num_contributions) == 8 else [])
        grouped = [
            ("Model year and age", float(num_contributions[[0, 1]].sum())),
            ("Mileage and expected usage", float(num_contributions[mileage_indices].sum())),
            ("Auction condition grade", float(num_contributions[4])),
        ]
        fact_fields = saved["bid_fields"]

    # Categorical inputs are integers, so Captum cannot differentiate them.
    # These are separate, directly computed model sensitivities to withholding
    # one known category at a time; they are not additive feature credits.
    categorical_checks = []
    with torch.no_grad():
        for i, name in enumerate(category_names):
            if int(cat[0, i]) == 1:  # already unknown
                continue
            masked = cat.clone()
            masked[0, i] = 1
            masked_log = float(forward(num, text, categories=masked)[0].item())
            difference = historical - _price_from_log(masked_log)
            if math.isfinite(difference):
                categorical_checks.append({
                    "feature": name.replace("_", " ").title(),
                    "value": str(fact_fields.get(name) or "unknown")[:55],
                    "historical_difference_usd": round(difference, 2),
                })
    categorical_checks.sort(key=lambda row: abs(row["historical_difference_usd"]), reverse=True)
    return {
        "market": market, "historical_price": historical,
        "model": saved[market].get("model", "Neural model"),
        "steps": _STEPS,
        "groups": [
            {"feature": name, "signed_log_price_points": round(score * 100.0, 2)}
            for name, score in grouped
        ],
        "numeric": [
            {"feature": name, "signed_log_price_points": round(float(score) * 100.0, 2)}
            for name, score in zip(numeric_names, num_contributions)
        ],
        "category_sensitivity": categorical_checks[:8],
    }


def render(result, adjusted_receipt=None, bid_analogue=None):
    """Clear, non-generative explanation of computed neural signals."""
    import pandas as pd
    import streamlit as st

    market = result["market"]
    st.success("🔍 Neural price explanation computed · 24-step Integrated Gradients")
    st.caption("Model: " + result["model"])
    historical = float(result["historical_price"])
    st.markdown("**How your displayed number is built**")
    if adjusted_receipt:
        today = float(adjusted_receipt["price"])
        st.write(
            f"Historical {market.upper()} neural estimate: **\${historical:,.0f}**. "
            f"Market-date adjustment: **\${today - historical:+,.0f}**. "
            f"Displayed date-scenario price: **\${today:,.0f}**."
        )
        st.caption("The market-clock adjustment is separate from the neural model "
                   "and is not counted as a learned feature contribution.")
    else:
        st.write(f"Historical {market.upper()} model estimate: **\${historical:,.0f}**.")
    if market == "bid" and bid_analogue:
        st.warning("This wholesale quote uses a historical age-matched vehicle proxy. "
                   "The explanation describes that proxy, not a verified sale of this model year.")

    groups = sorted(result["groups"], key=lambda g: abs(g["signed_log_price_points"]), reverse=True)
    messages = []
    for group in groups[:2]:
        score = group["signed_log_price_points"]
        if abs(score) >= 0.5:
            direction = "upward" if score > 0 else "downward"
            messages.append(f"{group['feature'].lower()} pulls the historical model {direction}")
    if messages:
        st.write("**In everyday language:** " + "; ".join(messages) + ".")
    else:
        st.write("**In everyday language:** the measured feature groups exert "
                 "little net directional pull relative to this mathematical reference.")
    st.caption("Positive values push predicted log-price up relative to a "
               "training-median numeric and zero-text mathematical baseline; "
               "negative values pull down. They do not imply causation or a guaranteed sale price.")
    chart = pd.DataFrame(result["groups"]).set_index("feature")
    st.bar_chart(chart, horizontal=True, height=230)
    with st.expander("🔬 Individual neural input contributions"):
        st.dataframe(pd.DataFrame(result["numeric"]), hide_index=True, use_container_width=True)
        st.caption("The description combines wording statistics and the full "
                   "384-dimensional Granite embedding. No unsupported individual-word "
                   "importance claim is made.")
    if result["category_sensitivity"]:
        st.markdown("**What if the model didn't know a particular car fact?**")
        st.dataframe(pd.DataFrame(result["category_sensitivity"]).rename(columns={
            "feature": "Car detail", "value": "Current value",
            "historical_difference_usd": "Historical price difference (USD)",
        }), hide_index=True, use_container_width=True)
        st.caption("Each difference reruns the SAME neural model with just one "
                   "categorical detail marked unknown. These what-if differences "
                   "are not additive attributions and are not buyer behavior.")
    st.caption("Scientific scope: 24 Integrated Gradients steps; batch size 1; "
               "only on request. The historical network and the independently "
               "applied valuation-date adjustment are described separately. "
               "This explanation is model-derived, not generated by a chatbot.")
