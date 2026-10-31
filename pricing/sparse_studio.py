"""XGBoost-only live pricing release."""
import json
from pathlib import Path
import streamlit as st

def render(root):
    from pricing.analytics import render_privacy
    render_privacy()
    st.markdown('## 🍋 The Sparse Lemon Squeezer')
    st.write('We squeezed the biggest training misses. Your car still gets the full inspection.')
    st.caption('XGBoost asking AFT + auction median models · original holdouts · Bayesian uncertainty')
    pages=st.tabs(['🚗 Price my car + family tree','🧠 How the squeeze works','📊 Evaluation'])
    with pages[0]:
        from pricing.sparse_live import render as live
        live(root)
    with pages[1]:
        st.markdown('### Business English')
        st.write('Each training car is priced by a fresh TF-IDF XGBoost model that did not train on that car. We rank the gap between its recorded and predicted log price, remove the largest 25% of training misses, and train fresh XGBoost models. This may remove errors, but can also remove valid unusual cars. The original validation and test cars stay in the exam.')
        st.write('The asking model reads the advert through sparse TF-IDF plus a frozen MiniLM transformer with a cross-fitted price head. MiniLM uses up to 128 tokens per description; TF-IDF sees the supplied description. The auction model estimates a typical wholesale sale using median regression. Bayesian make and model effects borrow evidence for rare cars and provide historical outcome ranges. Today’s prices additionally use the existing ageing and market-index scenario.')
        st.markdown('### Technical English')
        st.write('Selection: absolute log(observed/predicted), from five vehicle-group-held-out XGBoost folds; ceil(25% of eligible calibration rows) removed once per market. Ask: normal survival:aft, exact [price,price] bounds. Bid: reg:quantileerror, alpha=0.5, log-dollar labels. Both use 300 boosting rounds, max depth 5, two CPU threads and native categorical splits. Asking descriptions use a fold-fitted 2,000-feature sparse TF-IDF vocabulary. No UMAP or dense TF-IDF matrix is loaded. A shared int8 ONNX MiniLM encoder (128 tokens) supplies one predicted_text_price feature. Inside every outer fold, three inner group folds generate training text-price features; the outer-held cars use a training-only Ridge log-price head (alpha=100). The frozen backbone never learns price labels.')
        st.write('Bayesian layer: separate Gaussian make/model-within-make/age residual posteriors for ask and bid, sampled with two blocked Gibbs chains. Validation calibrates 90% historical individual prediction ranges; full test measures coverage. This is not a jointly pooled ask/bid market model, and Gaussian/t/mixture LOO comparison is not implemented.')
        st.warning('Cleaning is adaptive. Post-selection OOF scores are conditional diagnostics, not nested CV of the entire selection pipeline. Full held-out MAE determines whether cleaning helped. Historical interval coverage does not validate today’s extrapolation.')
    with pages[2]:
        for side in ['ask','bid']:
            p=Path(root)/'models/xgboost-sparse-minilm'/side
            base=json.loads((p/'evaluation.json').read_text())
            bayes=json.loads((p/'bayesian/bayesian-evaluation.json').read_text())
            st.markdown('### '+('Seller asking · Craigslist' if side=='ask' else 'Wholesale auction · Kaggle 2014–2015'))
            a,b,c=st.columns(3)
            a.metric('XGBoost test MAE',f"${base['scores']['test']['mae_usd']:,.0f}")
            b.metric('XGBoost + Bayes test MAE',f"${bayes['point_scores']['mae_usd']:,.0f}")
            c.metric('90% historical range coverage',f"{bayes['test_coverage']:.1%}")
            st.caption(f"{bayes['test_rows']:,} original test cars. Prices and MAE shown in dollars; model residuals use log dollars.")
            st.json({'cleaning':json.loads((p/'cleaning.json').read_text()),'base':base,'bayesian':bayes},expanded=False)
