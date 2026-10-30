"""Plain-language and technical explanations of the two requested defaults."""
import json
from pathlib import Path
import streamlit as st


def render_methods(root):
    from pricing.iterative_release import completed
    from pricing.residual_ui import bid_folder,model_root
    iterative=completed(root)
    with st.expander('Why asking AFT, auction median regression, and a separate cleaning model?'):
        simple,technical=st.tabs(['Explain it without a statistics degree','Technical · losses, censoring and selection'])
        with simple:
            st.write('**Ask — MiniLM + CatBoost normal AFT:** MiniLM reads the advert and proposes one text-price signal. AFT combines it with vehicle facts under a log-normal price model. It is a useful positive-price model, with a likelihood that can also accommodate real censoring bounds if those bounds are actually observed. An asking price is still a seller’s wish, not proof of a sale.')
            st.write('**Bid — CatBoost median, quantile 0.5:** predicts the middle of comparable reported sale prices on a log scale. Big proportional mistakes matter, while a billionaire’s unicorn does not get unlimited voting power. Unlike normal AFT, this loss does not assume Gaussian log-price errors. Both are reasonable modelling choices; neither is universally best.')
            from pricing.iterative_release import completed
            if completed(root):st.write('**Five pit stops before final training:** a tabular log-median CatBoost cleaner is refitted five times. Each stage removes the largest absolute log residuals: ceil(5% of remaining cars) within each original phase. Training residuals are cross-fitted; validation/test predictions come from training-only cleaners. Both markets then receive fresh AFT and quantile final fits, with MiniLM stacking retained for asking prices. Around 22.6% are removed overall; filtered held-out scores are optimistic.')
            else:st.write('**A separate mechanic does the cleaning:** a first log-median CatBoost produces out-of-fold training residuals and training-only predictions for validation/test. We remove the worst 25% within each phase, then train a fresh median model on retained cars. The asking model and its saved weights are unchanged.')
            st.warning('Removing difficult validation/test cars improves the exam score partly by shrinking the exam. These filtered scores are optimistic. Large residuals can be valid rare cars, not bad records. Original unfiltered-test diagnostics remain visible.')
            st.write('**Sold-only blind spot:** the current auction source contains sales, not the complete population of sold and unsold listings. IMR is therefore disabled. We do not dress Craigslist adverts up as failed auctions. When a genuine listing population is supplied, IMR is a cleaner-only diagnostic; it never enters the final bid price model.')
        with technical:
            st.write('Normal AFT models log(P)=f(X)+sigma*epsilon with epsilon normal. Under this assumption exp(f(X)) is the conditional median; it is not the arithmetic mean. The current ask fit uses exact positive bounds [P,P]. No interval-censoring mechanism has been verified, so no censoring claim is made for these datasets.')
            st.latex(r'\log P=f(X)+\sigma\varepsilon,\quad\varepsilon\sim N(0,1)')
            st.write('Bid cleaner and final estimator both use CatBoost Quantile:alpha=0.5 with log dollar targets. Prices retain the existing 2015-reference CPI normalization before fitting and date transport afterward. The pinball loss at 0.5 is half the absolute log error. Monotonic exponentiation maps a log-price median to a price median; this does not target mean dollars or minimize squared dollar error.')
            st.latex(r'\rho_{0.5}(u)=\tfrac12|u|,\quad u=\log P-\widehat{\log P},\quad\widehat P=\exp(f(X))')
            if iterative:st.write('Five rounds each remove ceil(5% of remaining cars) per original phase. Training residuals use three-fold cross-fitting; held-out predictions use training-only fits. No other outlier flags are unioned. A fresh assessment fit uses retained training; deployment separately refits all retained phases. This adaptive selection is not nested cross-validation.')
            else:st.write('Original training cleaner predictions use three-fold cross-fitting. Validation/test cleaner fits never use their price targets. Subsequently using those targets to rank and remove ceil(0.25*n_phase) cars still selects the evaluation sample. Ties are broken by row_id. This new bid cut does not union the old OutlierTree or Isolation Forest flags. Assessment fits retained training only; a separate final deployment refit uses all retained phases.')
            st.latex(r'\Pr(S=1\mid Z)=\Phi(Z\gamma),\qquad\lambda_{sold}(Z)=\frac{\phi(Z\gamma)}{\Phi(Z\gamma)}')
            st.write('Optional selection fitting uses Probit on same-market sold AND observed unsold listings, using pre-sale predictors only. Its fit excludes validation/test phases and the cleaner’s OOF-held sold cars/VINs. The normal IMR formula follows the Probit index; plugging a logistic index into this formula is not equivalent. A price column is never a selection-model feature. Lambda is supplied only to the residual-cleaning CatBoost and omitted from the final median estimator.')
            st.write('Adding lambda to a flexible quantile cleaner is not the classical Heckman two-step mean regression and does not by itself identify population prices. Population coverage, selection assumptions and credible exclusion restrictions still matter. With only S=1, the selection equation is unidentified; the implementation rejects one-class inputs.')
            file=bid_folder(root)/'manifest.json'
            if file.exists():st.json(json.loads(file.read_text()).get('selection_imr',{}))
            current=bid_folder(root)/'evaluation.json'
            previous=(model_root(root)/'bid-aft/evaluation.json') if iterative else Path(root)/'models/minilm-catboost-aft/bid/evaluation.json'
            if current.exists() and previous.exists():
                import pandas as pd
                records=[]
                for name,path in [('Requested default · log quantile 0.5',current),('Same-cleaning normal AFT' if iterative else 'Previous · normal AFT',previous)]:
                    r=json.loads(path.read_text());m=r['assessment']['original_unfiltered_test']
                    records.append({'Model':name,'Original test cars':m['rows'],'MAE USD':m['mae_usd'],'RMSE USD':m['rmse_usd'],'Spearman':m['spearman']})
                st.subheader('Measured trade-off on the same original auction test')
                st.dataframe(pd.DataFrame(records),hide_index=True)
                st.caption('The default follows the requested cleaning and loss choices. The table reports actual full-test errors; selected-cohort scores do not establish better generalization. This historical test has informed development.')
            st.link_button('CatBoost objectives: Quantile and SurvivalAft','https://catboost.ai/docs/en/concepts/loss-functions-regression')
            st.link_button('Stata: Probit selection and inverse Mills ratio','https://www.stata.com/support/faqs/statistics/inverse-mills-ratio/')
