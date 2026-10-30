"""On-demand phrase explanations of the complete served asking model."""
import time

import numpy as np
import pandas as pd
import streamlit as st

from pricing.text_attribution import (DETAILS, signature, explain_text,
                                      highlight_html, contribution_chart)


def render_text_attribution(result, fields, notes, pipeline):
    st.divider()
    st.subheader('Which words moved my estimate?')
    st.write('Your advert, under the microscope. See which phrases steer the asking-price model up or down—no magic resale spell required.')
    if getattr(pipeline, 'text_model', None) is None:
        st.info('This selected model uses vehicle facts only. Choose a MiniLM model to explore description effects. The wholesale auction model has no seller descriptions.')
        return
    # Match the deployed MiniLM cleaning and character window exactly. Explain
    # cleaned text, so deleting currency words cannot bypass leakage prevention.
    from scripts.common import clean
    text = clean(notes)[:8000]
    if not text:
        st.info('Add a seller description and estimate the car first. A blank advert has very little to say.')
        return
    detail = st.radio('Phrase analysis detail', list(DETAILS), horizontal=True,
                      key='phrase_shap_detail', help='Quick: up to 8 groups and 4 sampled orders. Closer: up to 12 groups and 8 orders. Short descriptions use exact group Shapley values.')
    st.caption('Asking estimate only · same car facts throughout · compared with a blank description. Runs only when you press the button; long adverts take longer on CPU.')
    date = result.get('valuation_date')
    if date:
        st.info('Phrase effects explain the date-adjusted asking estimate for '+date+'. Vehicle ageing and the market clock are included in every prediction.')
    key = signature(pipeline.manifest, fields, text, detail,
                    getattr(pipeline,'explanation_identity',id(pipeline)), result['price'])
    if st.button('🔎 Reveal my advert’s price drivers', key='build_phrase_shap', type='primary'):
        st.session_state.pop('_phrase_shap_result', None)
        bar = st.progress(0., text='Reading phrase combinations…')
        started = time.monotonic()
        try:
            def predict(candidate):
                return pipeline.predict(candidate, dict(fields), explain=False)['price']
            def progress(done, total):
                bar.progress(done/total, text=f'Checking phrase combinations · {done}/{total}')
            limit, pairs = DETAILS[detail]
            payload = explain_text(predict, text, limit=limit, pairs=pairs, progress=progress)
            if not np.isclose(payload['prediction'], result['price'], rtol=1e-7, atol=.01):
                raise ValueError('The explanation does not match the submitted asking estimate. Please estimate the car again.')
            payload['seconds'] = time.monotonic() - started
            st.session_state['_phrase_shap_result'] = {'signature': key, 'payload': payload}
        except Exception:
            st.warning('The phrase analysis could not be completed. Your price estimate is still available. Try estimating the car again or use Quick look.')
            return
        finally:
            bar.empty()
    saved = st.session_state.get('_phrase_shap_result')
    if not saved or saved['signature'] != key:
        st.info('Press Reveal to investigate this submitted advert. Changing the car, description, model or detail setting requires a fresh analysis.')
        return
    p = saved['payload']
    a, b, c = st.columns(3)
    a.metric('Same car · blank advert', f"${p['base']:,.0f}")
    b.metric('With your description', f"${p['prediction']:,.0f}")
    c.metric('Description’s net effect', f"{p['difference']:+,.0f} USD")
    st.caption('Blank advert means an actual MiniLM prediction for empty text—not a zero text-price feature or an average car.')
    highlights, dollars, technical = st.tabs(['🖍 Your advert, highlighted', '💵 Phrase effects in dollars', '🧠 Method & limitations'])
    with highlights:
        st.write('**Rust red: raises the estimate · Blue: lowers it · Grey: near zero.** The signed dollar chart gives the same information without relying on colour.')
        st.markdown(highlight_html(p), unsafe_allow_html=True)
        st.caption('Highlights show contiguous word groups, not individual-word certainty. Hover for dollar contributions; on mobile, use the chart or the full phrase table.')
        st.caption('This is the text the price model reads after its existing cleanup. URLs, HTML and explicit price amounts are removed or replaced; the model reads up to 8,000 cleaned characters.')
    with dollars:
        st.write('The strongest phrase effects come first. Together they add to the difference between the two asking estimates above.')
        show_all = st.checkbox('Show every phrase separately', key='phrase_shap_show_all')
        st.plotly_chart(contribution_chart(p, show_all), width='stretch', key='phrase_shap_chart')
        with st.expander('Read every phrase and its signed dollar effect'):
            rows = [{'Phrase': group.strip(), 'Allocated effect USD': value}
                    for group, value in zip(p['groups'], p['values'])]
            st.dataframe(pd.DataFrame(rows).sort_values('Allocated effect USD', key=abs, ascending=False),
                         hide_index=True, width='stretch')
    with technical:
        st.write('**Direct grouped Shapley attribution in served dollars.** Missing phrase groups are deleted. Every combination reruns the same MiniLM → text-price feature → final asking model, with structured car inputs fixed. When the receipt is date-adjusted, the same ageing and market adjustment is applied to each combination. No TF-IDF surrogate is trained. Inverse target transforms and prediction clipping remain inside the prediction function.')
        st.write('Exact group Shapley values for at most four groups; otherwise deterministic random permutations paired with their reverse. The quick budget is up to 30 unique coalition predictions; closer look up to 90. Groups prefer sentence boundaries, splitting long chunks and merging short neighbours to fit the budget; unpunctuated text uses balanced word chunks. These are not discovered semantic concepts. Each path telescopes from blank text to the submitted estimate.')
        st.write({'Groups': len(p['groups']), 'Orders': p['orders'], 'Model evaluations': p['evaluations'],
                  'Exact for these groups': p['exact'], 'Reconstruction error USD': p['error'],
                  'Runtime seconds': round(p['seconds'], 2), 'Seed': p['seed']})
        st.dataframe(pd.DataFrame({'Phrase': [g.strip() for g in p['groups']],
            'Effect USD': p['values'], 'Sampled-order SE USD': p['se']}), hide_index=True, width='stretch')
        st.caption('A small reconstruction error verifies arithmetic, not attribution accuracy. Sampled-order SE is a rough Monte Carlo diagnostic from a small number of antithetic pairs; it is not a car-price confidence interval. Changing grouping, masking or baseline changes the explanation.')
        st.caption('Removing words can break grammar or negation and create unfamiliar model inputs. Contributions can reflect make, model or mileage mentioned in the advert, not just condition. Shared credit is context dependent. Positive wording does not repair a transmission or prove a causal price premium. Historical asking estimates are not completed-sale values; this panel does not explain auction prices.')
    st.info('Model reaction, not market magic: “immaculate” can move an estimate. It cannot remove a dent.')
