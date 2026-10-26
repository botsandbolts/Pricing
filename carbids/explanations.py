"""Human-readable presentation of native LightGBM attributions."""
import numpy as np
import pandas as pd
import streamlit as st

LABELS={'year':'Model year','odometer':'Mileage','manufacturer':'Make','model':'Model / trim',
 'type':'Body type','title_status':'Title status','paint_color':'Paint colour',
 'predicted_text_price':'Description-price signal'}
def label(name):return LABELS.get(name,name.replace('_',' ').title())

def render_explanation(p,fields,notes,pipeline):
    e=p['explanation'];rows=sorted(e['features'],key=lambda r:abs(r['contribution']),reverse=True)
    simple,technical,whatif=st.tabs(['Explain it like I buy cars','Under the bonnet · technical','What if?'])
    with simple:
        st.subheader('Why this price? The model shows its working.')
        st.write('Think of LightGBM as a committee of small decision trees. Each new tree corrects earlier mistakes. No tree has ever kicked a tyre, but together they have read a lot of listings.')
        up=[label(r['feature']) for r in rows if r['contribution']>1e-8][:3]
        down=[label(r['feature']) for r in rows if r['contribution']< -1e-8][:3]
        if up:st.write('**Pulling this estimate up:** '+', '.join(up)+'.')
        if down:st.write('**Pulling this estimate down:** '+', '.join(down)+'.')
        st.caption('Compared with the model’s reference prediction. These are shared-credit explanations of this car’s prediction, not universal price rules or proof of causation.')
        table=pd.DataFrame([{'Detail':label(r['feature']),'Contribution':r['contribution']} for r in rows])
        st.bar_chart(table.set_index('Detail'),horizontal=True)
        unit='log(1 + price)' if e['target']=='log1p' else 'dollars'
        st.caption(f'Right of zero raises the fitted score; left lowers it. Bar units: {unit}. Zero means no allocated contribution for this prediction.')
        if e['target']=='log1p':st.info('The model works on a log-price scale. The bars add on that scale; they are not separate dollar discounts. Mileage and model year often share information, so their credit can move between them.')
        st.write('**The description branch:** '+('MiniLM reads your notes and produces one text-price signal. LightGBM explains that signal as a single feature; this chart does not explain individual words.' if p.get('predicted_text_price') is not None else 'This baseline uses structured fields only. Your prose cannot sweet-talk it into a higher price.'))
        st.write('**The error band:** a historical auction error guide, not an 80% guarantee for this particular car. Rust is still allowed to surprise us.')
    with technical:
        st.subheader('Native TreeSHAP attribution')
        st.write(f"{e['method']}. {e['trees']:,} trees. Contributions are rescaled from the standardized training target.")
        st.latex(r'z=b+\sum_j\phi_j')
        st.write(f"Reference b = {e['base']:.6f}; sum of contributions = {sum(r['contribution'] for r in rows):.6f}; fitted score z = {e['raw_prediction']:.6f}.")
        if e['target']=='log1p':st.latex(r'\widehat{price}=\exp(\operatorname{clip}(z,0,\log(1+10^7)))-1')
        else:st.latex(r'\widehat{price}=\operatorname{clip}(z,0,10^7)')
        st.caption(f"Additivity error: {e['additivity_error']:.3g}. Prediction clipped: {e['clipped']}. The reference is not the mean dollar price; exponentiating an expected log price is a different quantity.")
        data=[]
        for r in rows:
            item={'Feature':r['feature'],'Encoded input':r['value'],'Signed contribution':r['contribution'],'Global gain %':r['gain_pct']}
            if e['target']=='log1p':item['Factor on (price + 1)']=float(np.exp(np.clip(r['contribution'],-700,700)))
            data.append(item)
        st.dataframe(pd.DataFrame(data),hide_index=True,width='stretch')
        st.caption('Categorical inputs are integer codes; unknown categories become missing. Gain % summarizes total training split gains across the model: it is not a local attribution or an accuracy score. Correlated features and tree structure affect credit allocation. The product of the displayed factors maps the reference price+1 to the un-clipped prediction+1.')
        st.write('**Training:** description regression learns standardized log1p targets. Three-fold OOF predictions keep each row’s own target out of its text predictor. Isolation Forest auto acts on training OOF text prices only. LightGBM tries raw/log targets and Huber thresholds; validation MAE selects settings, then the chronological test is evaluated before deployment refitting.')
        st.write('**Limits:** TreeSHAP explains the fitted LightGBM function, including the scalar text-price feature when active. It does not explain MiniLM token behaviour, establish causal effects, remove leakage automatically, or certify a fair transaction price.')
        st.link_button('LightGBM prediction / contribution documentation','https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict')
    with whatif:
        st.subheader('A small thought experiment. No time machine required.')
        st.write('Change one detail while keeping the rest of this submitted car fixed. The result is a fresh model prediction, not a causal claim.')
        mileage=st.number_input('Try a different mileage',min_value=0,max_value=2000000,value=int(fields['odometer']),step=10000,key='explain_mileage_'+str(fields['odometer']))
        if st.button('Recalculate this mileage scenario'):
            scenario=pipeline.predict(notes,{**fields,'odometer':mileage})
            delta=scenario['price']-p['price']
            st.metric('Scenario auction sale price',f"${scenario['price']:,.0f}",f"{delta:+,.0f} USD versus submitted car")
            st.caption('Make, model, year and all other inputs stayed fixed. The text-price signal is unchanged because the description is unchanged. Unusual combinations may be outside the training distribution.')
