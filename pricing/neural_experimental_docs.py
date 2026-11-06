"""Evidence-first business/science documentation for the Neural Lab."""
from pathlib import Path
import streamlit as st

def render(root,report=None):
    st.divider()
    st.markdown("## 05 / Behind the price oracle")
    st.caption("Choose the business explanation or the scientific protocol. Both report limits, not magic.")
    business,science,limits=st.tabs(["💼 Business","🔬 Scientific","⚠️ Limits & privacy"])
    with business:
        st.markdown("### What does the figure mean?")
        st.write("It estimates the **advertised asking price** a seller might list in an advert. "
                 "It is not a verified completed-sale price, wholesale auction estimate, "
                 "trade-in offer, dealer margin, or appraisal certificate.")
        st.markdown("### Why another calculator?")
        st.write("Traditional tree models learn vehicle facts. This challenger also models "
                 "relationships among structured specifications and linguistic signals in "
                 "the car description. Two vehicles with similar age and mileage can be "
                 "described very differently, but descriptions are imperfect evidence.")
        st.markdown("### What changed?")
        st.write("After 36 model variants, DeepCross plus AFT plus MiniLM with 12% "
                 "main dropout won on historical validation error: about 3,047 USD "
                 "validation MAE and 3,262 USD test MAE. The older full-data XGBoost "
                 "reported about 3,378 USD test MAE on the same-sized cohort. "
                 "That XGBoost comparison is aggregate, not row-paired.")
        st.write("A historical average error of 3,262 dollars is **not** a "
                 "plus-or-minus 3,262-dollar prediction interval. A given damaged, unusual "
                 "or rare vehicle could be much harder to estimate.")
        st.markdown("### Use it like a curious buyer, not a bank")
        st.write("Enter exact details, review the description, inspect the May 2021 neural "
                 "anchor and its market-clock adjustment, then test mileage or wording scenarios. "
                 "Date-adjusted prices are extrapolations, not newly evaluated 2026 sales.")
        st.info("Alpha software intentionally favors a verified inference path over speed. "
                "The production calculator is unchanged; wholesale pricing is not included.")
    with science:
        st.markdown("### Data and targets")
        st.write("211,127 historical 2021 Craigslist asking listings; original chronological "
                 "training, validation and test partitions contain 147,777 / 31,668 / 31,682 rows. "
                 "The target is log advertised USD, not a completed transaction.")
        st.markdown("### Input representation")
        st.write("Seven robust-scaled numeric features encode model year, age at 2021, "
                 "log mileage, miles per age-year, and three description-length/punctuation "
                 "statistics. Eleven categorical embeddings encode make, model, body, "
                 "transmission, state, fuel, drive, title, cylinders, paint and condition. "
                 "Unknown categories are handled explicitly.")
        st.write("One text branch uses an unsupervised TF-IDF unigram/bigram projector "
                 "compressed by train-fitted truncated SVD to 24 dimensions. The other uses "
                 "a frozen, pinned all-MiniLM-L6-v2 encoder: 384 text embedding dimensions "
                 "are converted to one log-dollar price signal by three Ridge heads.")
        st.markdown("### Two genuinely out-of-bag prices")
        st.write("Three text price heads are each fitted on one separate third of the "
                 "133,000 network fitting cars. Each fitting car is unseen by exactly "
                 "two heads and receives their mean prediction. The 14,777 training-only "
                 "early-stopper rows, 31,668 validation and 31,682 test rows are outside "
                 "all three heads' fitting data and receive the mean of all three.")
        st.markdown("### DeepCross + AFT")
        st.write("Categorical embeddings, numerical values and both description signals "
                 "enter explicit low-rank cross interactions in parallel with residual "
                 "nonlinear deep layers. The selected 12%-dropout configuration predicts "
                 "a log-price location and positive scale. It was fitted with AdamW, "
                 "regularization, gradient clipping and a cosine learning-rate schedule.")
        st.latex(r"\mathcal L = \frac{1}{N}\sum_i\left[\frac{1}{2}\left(\frac{\ln y_i-\mu_i}{\sigma_i}\right)^2+\ln\sigma_i\right]")
        st.write("Here AFT denotes an exact-observation log-normal likelihood. "
                 "No verified censoring intervals or completed sales exist, so no "
                 "survival censoring bounds are invented.")
        st.markdown("### 36-way selection and release parity")
        st.write("Eighteen architecture/training settings times two text families were "
                 "ranked strictly on validation MAE. The untouched historical test is "
                 "an assessment, not the tuning target. The selected result is a "
                 "single-seed exploratory finding, not yet multi-seed replication.")
        st.write("Production-like serving is enabled only if the packaging workflow "
                 "recovers the archived checkpoint and original fitted preprocessing "
                 "and verifies input tensors and original holdout predictions. "
                 "Mismatches disable the model rather than substituting a guessed price.")
        st.markdown("### The separate market clock")
        st.latex(r"\widehat P_t=\widehat P_{\mathrm{May\,2021}}\times D_{\mathrm{age}}(t)\times I(t)/I(\mathrm{May\,2021})")
        st.write("Historical May 2021 neural prices are multiplied by the existing "
                 "vehicle-age depreciation curve and monthly used-car market index. "
                 "This transport is experimental, with no independently validated "
                 "current-date MAE or individual prediction interval.")
        st.write("Mileage, condition, wording and date what-ifs rerun the model and "
                 "transport. They are model response surfaces, not causal features or SHAP.")
        file=Path(root)/"NEURAL_LAB_MODEL_CARD.md"
        if file.exists():
            st.download_button("Download the complete model card",file.read_text(),
                               "neural-lab-model-card.md","text/markdown")
    with limits:
        st.warning("Asking price is not selling price. No current-year validation "
                   "or guaranteed interval; historical observations and one research "
                   "seed do not make a certified appraisal.")
        st.write("Training descriptions may be inaccurate or strategic. "
                 "Data are historical US listings, and uncommon models may "
                 "perform differently. Model years after 2021 are unsupported.")
        st.write("Preprocessing and weights are local, versioned files. Trusted "
                 "compressed joblib objects contain the text projectors; native "
                 "PyTorch contains the neural checkpoint. No database stores models.")
        st.write("With explicit analytics consent, existing Turso tables receive "
                 "model/version labels, asking-price events, status and duration. "
                 "No seller descriptions, embeddings or VINs are included. "
                 "Analytics errors never stop inference.")
        st.write("The wholesale bid model is intentionally not part of this release.")
