"""Plain-language and technical explanations of the independent defaults."""
import json
from pathlib import Path
import streamlit as st


def render_methods(root):
    with st.expander('Why asking AFT, auction Huber regression, and a separate cleaning model?'):
        simple,technical=st.tabs(['Explain it without a statistics degree','Technical · losses, censoring and selection'])
        with simple:
            st.write('**Ask — MiniLM + CatBoost normal AFT, unchanged:** MiniLM reads the advert and proposes one text-price signal. AFT combines it with vehicle facts under a log-normal price model. The asking price is a seller’s wish, not proof of a sale. Real censoring bounds can be modelled when actually observed.')
            st.write('**Bid — CatBoost Huber on log prices, new default:** small proportional errors get a smooth squared-error penalty; large ones get a linear penalty. An ordinary bidding wobble gets attention; a typo does not get the microphone for the whole evening. Huber limits extreme residual influence, but cannot tell a rare genuine car from a bad record. It is the requested robust compromise, not a proven universal winner.')
            st.write('**Delta = 0.2:** the transition is about +22% or −18% relative to the predicted price. Delta is measured in natural-log price units, not dollars. This value was fixed before this experiment; it was not chosen by looking at test scores. The previous quantile 0.5 option still predicts a median using absolute log error; Huber predicts a different robust centre.')
            st.write('**Two separate mechanics:** a first Huber CatBoost creates three-fold out-of-fold training residuals and training-only validation/test predictions. Remove the largest 25% of absolute log residuals in EACH phase, then train a fresh Huber CatBoost. These are newly calculated Huber cuts; the old quantile masks are not reused. Ask stays exactly as installed.')
            st.warning('Removing difficult validation/test cars improves the exam partly by shrinking the exam. Filtered scores are optimistic. Large residuals may be valid unusual cars. Original unfiltered-test diagnostics remain visible, and that historical test is already reused.')
            st.write('**Sold-only blind spot:** IMR remains disabled because the auction source has reported sales only. A genuine same-market sold AND observed unsold listing population is required. We do not invent failed auctions. If supplied, Probit and its inverse Mills ratio feed the cleaner only, never the final Huber estimator.')
        with technical:
            st.write('Ask normal AFT models log(P)=f(X)+sigma*epsilon with epsilon normal. Under this assumption exp(f(X)) is a conditional median, not arithmetic mean dollars. Current ask bounds are exact [P,P]. No interval-censoring mechanism is verified for these datasets; no new censoring claim is made.')
            st.latex(r'\log P=f(X)+\sigma\varepsilon,\quad\varepsilon\sim N(0,1)')
            st.write('Both new bid stages use CatBoost Huber:delta=0.2 with natural-log reference-dollar targets. Existing lagged CPI normalization and date transport remain. Each fitting model centres log targets using its own fitting rows only. Inference restores the saved offset exactly once, including the native TreeSHAP base value. No log1p/expm1 or double logging is used.')
            st.latex(r'L_\delta(u)=\begin{cases}\tfrac12u^2&|u|\leq\delta\\\delta(|u|-\tfrac12\delta)&|u|>\delta\end{cases},\quad u=\log P-\widehat{\log P}')
            st.write('Huber estimates a robust location in log space, characterized by zero expected clipped residual. Exponentiating is neither generally a conditional median nor an unbiased arithmetic mean. Quantile alpha=0.5 instead minimizes half absolute log error and targets a conditional median. Huber limits response-residual influence, not leverage or selection bias. Fixed 25% trimming is a separate user-requested operation, not part of Huber itself.')
            st.write('Cleaning: three-fold cross-fit within original training; held validation/test prices never fit the cleaner. Those prices DO select retained evaluation rows afterward. Remove exactly ceil(0.25*n_phase), breaking absolute-log-residual ties by row_id. No old OutlierTree/Isolation Forest union. Assessment fits retained training; retained validation chooses tree count. A separate deployment refit uses all retained phases.')
            st.latex(r'\Pr(S=1\mid Z)=\Phi(Z\gamma),\qquad\lambda_{sold}(Z)=\phi(Z\gamma)/\Phi(Z\gamma)')
            st.write('Optional Probit fits pre-sale features on the full genuine listing population’s training partition, excluding held sold IDs/VINs during cleaner cross-fitting. Prices never enter that equation. Normal IMR uses the Probit index; plugging a logistic index into phi/Phi is not the same correction. Lambda enters only the cleaning Huber model. With sold-only S=1 the selection equation is unidentified and the implementation rejects the fit. Adding lambda to a nonlinear Huber cleaner is not classical Heckman two-step mean regression and does not identify population prices without coverage, distribution assumptions and credible exclusion restrictions.')
            from pricing.residual_ui import bid_folder
            active=bid_folder(root)
            if (active/'manifest.json').exists():st.json(json.loads((active/'manifest.json').read_text()).get('selection_imr',{}))
            import pandas as pd
            records=[]
            for name,path in [('Requested default · log Huber',Path(root)/'models/bid-huber-cleaned'),('Previous · log quantile 0.5',Path(root)/'models/bid-quantile-cleaned'),('Previous · normal AFT',Path(root)/'models/minilm-catboost-aft/bid')]:
                if not (path/'evaluation.json').exists():continue
                r=json.loads((path/'evaluation.json').read_text());m=r['assessment']['original_unfiltered_test']
                records.append({'Model':name,'Original test cars':m['rows'],'MAE USD':m['mae_usd'],'RMSE USD':m['rmse_usd'],'Spearman':m['spearman']})
            if records:
                st.subheader('Measured trade-off on the same original auction test')
                st.dataframe(pd.DataFrame(records),hide_index=True)
                st.caption('Default follows the requested loss, not an assumption that it wins. Compare these original-test figures. Filtered tests contain different easier cars; their MAEs are not a paired accuracy comparison. This historical benchmark has informed earlier choices; a fresh external unfiltered test is needed.')
            st.link_button('CatBoost objectives: Huber, Quantile and SurvivalAft','https://catboost.ai/docs/en/concepts/loss-functions-regression')
            st.link_button('Stata: Probit selection and inverse Mills ratio','https://www.stata.com/support/faqs/statistics/inverse-mills-ratio/')
