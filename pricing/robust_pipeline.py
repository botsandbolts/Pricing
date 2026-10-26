"""Pricing-compatible adapter for the promoted regularized ensemble."""
from pathlib import Path
import numpy as np
import pandas as pd
from marketbridge.robust import RobustAsking
from marketbridge.structural import StructuralTransfer

class RobustPricePipeline:
    def __init__(self,path):
        path=Path(path);self.ensemble=RobustAsking(path,StructuralTransfer(path.parent/'structural-transfer'))
        self.manifest=self.ensemble.manifest;self.schema=self.ensemble.schema
        self.text_model=None;self.tokenizer=None
    def predict(self,text,fields,explain=False):
        reference={**fields,'as_of':'2021-05-01'}
        value=float(self.ensemble.predict_frame(pd.DataFrame([reference]))[0]);radius=self.manifest['band80_log_radius']
        result={'price':value,'predicted_text_price':None,'ensemble':True,'target_kind':'asking_price','reference_date':'2021-05-01',
            'range':[float(np.expm1(np.clip(np.log1p(value)-radius,0,np.log1p(1e7)))),float(np.expm1(np.clip(np.log1p(value)+radius,0,np.log1p(1e7))))]}
        if explain:result['explanation']={'method':'Per-member TreeSHAP available; dollar-averaged ensemble predictions do not have a shared additive log-price decomposition.','ensemble_members':len(self.ensemble.members)}
        return result

def render_ensemble_explanation(result,fields,notes,pipeline):
    import streamlit as st
    from marketbridge.robust_explanations import explain_member
    ensemble=pipeline.ensemble if hasattr(pipeline,'ensemble') else pipeline._ensemble.ensemble
    simple,technical,whatif=st.tabs(['Explain the committee','Under the bonnet · ensemble','What if?'])
    with simple:
        st.subheader('Nine committees, one asking-price estimate')
        st.write('The larger model averages nine regularized LightGBM predictions. Each committee uses structured car facts plus age-adjusted auction signals. They agree on a price without letting one enthusiastic tree sell you a castle with wheels.')
        st.write('This is a historical 2021 asking estimate. Free text is not used in this trained ensemble, and an auction-equivalent input is not a verified retail-sale target.')
        st.caption('The residual band is a historical validation guide, not an individual or future-market coverage guarantee.')
    with technical:
        st.write('Total trees:',sum(m['trees'] for m,_ in ensemble.members))
        st.dataframe(pd.DataFrame(ensemble.manifest['members'])[['file','weight','seed','target','trees']],hide_index=True)
        index=st.selectbox('Choose a member to explain',list(range(len(ensemble.members))),key='pricing_explain_ensemble',format_func=lambda i:ensemble.members[i][0]['file'])
        e=explain_member(ensemble,pd.DataFrame([{**fields,'as_of':'2021-05-01'}]),index)
        st.write('Member price:',f"${e['member_price_usd']:,.0f}");st.write('Member target scale:',e['target'])
        st.dataframe(pd.DataFrame(e['features']).sort_values('contribution',key=abs,ascending=False),hide_index=True)
        st.caption(e['method']);st.caption(f"Native additivity error: {e['additivity_error']:.3g}. Member log contributions are not summed into fictitious ensemble dollar effects.")
    with whatif:
        mileage=st.number_input('Ensemble mileage scenario',0,2_000_000,int(fields['odometer']),step=10000)
        if st.button('Recalculate ensemble mileage scenario'):
            p=pipeline.predict(notes,{**fields,'odometer':mileage});st.metric('Scenario asking price',f"${p['price']:,.0f}",f"{p['price']-result['price']:+,.0f} USD")
            st.caption('This is a model sensitivity comparison, not a causal depreciation estimate.')
