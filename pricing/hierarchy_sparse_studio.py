"""One XGBoost pricing experience, with distinct asking and auction targets."""
import json
from pathlib import Path
import streamlit as st
def reports(root):
    return {side:json.loads((Path(root)/'models/xgboost-hierarchy'/side/'evaluation.json').read_text()) for side in ['ask','bid']}
def scorecard(root,*,details=False):
    st.markdown('### Historical XGBoost accuracy')
    for col,(side,r) in zip(st.columns(2),reports(root).items()):
        with col:
            st.markdown('#### '+('Seller asking · Craigslist' if side=='ask' else 'Wholesale auction'))
            test=r['scores']['test']
            st.metric('Test MAE · average dollar error',f"${test['mae_usd']:,.0f}")
            st.caption(f"{test['n']:,} original held-out test cars. Lower MAE is better.")
            if details:
                st.json({'objective':r['objective'],'test':test,'fold_audit':r.get('fold_audit',[])})
    st.caption('Historical errors do not establish today’s individual-car accuracy. No calibrated current-price interval is claimed.')
def render(root):
    from pricing.analytics import render_privacy
    render_privacy()
    st.markdown('<div id="studio-calculator"></div>',unsafe_allow_html=True)
    choices=['🚗 Price my car','📊 Accuracy','🧠 How it works']
    if st.session_state.get('xgboost_pricing_view') not in choices:st.session_state['xgboost_pricing_view']=choices[0]
    panel=st.radio('Pricing view',choices,horizontal=True,key='xgboost_pricing_view')
    if panel=='📊 Accuracy':scorecard(root,details=True);return
    if panel=='🧠 How it works':explain(root);return
    from pricing.studio_design import market_guide
    market_guide()
    st.caption('One pricing experience · XGBoost + MiniLM for asking prices; XGBoost for wholesale auctions.')
    from pricing.hierarchy_sparse_live import render as live
    live(root,bayesian=False)
def explain(root=Path('.')):
    st.markdown('### Business English')
    st.write('XGBoost estimates two different market prices: what a seller might advertise and a typical wholesale auction sale. The asking model also reads the description through MiniLM. Both use the retained Lemon Squeeze training cohort. No Bayesian correction is applied.')
    st.write('The largest 25% of training log-price misses were removed once. Five-fold cross-validation then prices every retained training car with a model that excluded its vehicle group. Validation and test cars remain unchanged. Cleaning can exclude legitimate unusual cars, so it does not guarantee improved accuracy.')
    st.markdown('### Technical English')
    st.write('Ask: survival:aft with a normal log-price distribution and exact positive [price, price] bounds; measured censoring intervals are unavailable. Bid: reg:quantileerror, alpha=0.5, trained on log dollars. Five vehicle-group folds operate within the retained 75% of the original eligible training cohort: approximately 80% fit / 20% predict per fold. A final fit on the retained training cohort supplies live predictions. Validation and test labels never train these trees.')
    st.write('Both markets use 300 boosting rounds, depth 5 and native categorical features. Ask receives one MiniLM description-price scalar: an average of two cross-fitted dollar predictions, with text-head fitting nested inside each outer fold. No TF-IDF or UMAP is used in this pricing model.')
    st.caption('Dollar outputs are transported to the selected date using the existing age/mileage curves and market index. These are scenarios, not verified current sale prices. Ask and auction populations differ; their gap is not dealer profit.')
