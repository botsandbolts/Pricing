"""Visual explanation studio: lay explanations and honest technical scales."""
import hashlib,json,io
import numpy as np
import pandas as pd
import streamlit as st
from pricing.visual_explanations import load_reference,clean_frame,analyse,label
from pricing import explanation_plots as plots


def render_visual_explanations(result,fields,notes,pipeline):
    st.divider();st.subheader('The price detective · visual explanations')
    st.write('The trees have been questioned. Here are their statements—with graphs, not a police sketch of your Corolla.')
    with st.expander('Reference cars and calculation settings'):
        reference,provenance=load_reference()
        st.caption('Default: 30 historical listing fixtures, evenly spaced through the original eligible cohort. This small gallery is not a representative survey of all cars. No price labels are used.')
        upload=st.file_uploader('Optional reference cars · CSV, up to 1,000 rows and 2 MB',type=['csv'],key='shap_reference_csv',
            help='Use year, odometer, manufacturer, model and other vehicle fields. Price, descriptions, VINs and other extra columns are ignored.')
        if upload is not None:
            try:
                if upload.size>2_000_000:raise ValueError('Use a CSV smaller than 2 MB')
                raw=pd.read_csv(io.BytesIO(upload.getvalue()))
                if len(raw)>1000:raise ValueError('Use at most 1,000 rows')
                if not {'year','odometer'}.issubset(raw.columns):raise ValueError('CSV needs year and odometer columns')
                reference=clean_frame(raw)
                provenance={'source':'User-uploaded predictor-only reference CSV','rows':len(reference),
                            'sampling':'User-selected sample; not automatically representative'}
            except Exception as exc:st.error(f'Reference CSV: {exc}');return
        ensemble=getattr(pipeline,'ensemble',None)
        if ensemble is None and getattr(pipeline,'_ensemble',None) is not None:ensemble=pipeline._ensemble.ensemble
        count=len(ensemble.members) if ensemble is not None else 1
        member=st.selectbox('Member for exact TreeSHAP plots',list(range(count)),
            format_func=lambda i:f'Member {i+1}' if ensemble is not None else 'Selected tree model',key='shap_graph_member')
        speed=st.select_slider('Dollar explanation detail',options=['Quick','Balanced','Thorough'],value='Balanced',key='shap_graph_detail')
        pairs={'Quick':4,'Balanced':8,'Thorough':16}[speed]
        st.caption('Dollar waterfall: approximate grouped permutation SHAP of the whole served model. Other SHAP charts: exact native contributions for the selected member, on its own target scale. More detail samples more feature orders; it does not make the model more accurate.')
    manifest=pipeline.manifest
    signature=hashlib.sha256((json.dumps(manifest,sort_keys=True)+json.dumps(fields,sort_keys=True,default=str)+notes+
        reference.to_json()+str(member)+str(pairs)+str(id(pipeline))).encode()).hexdigest()
    key='pricing_visual_explanation'
    if st.button('Build all explanation graphs',key='build_pricing_graphs',type='primary'):
        try:
            with st.spinner('Interviewing the trees and checking the arithmetic…'):
                payload=analyse(pipeline,fields,result.get('predicted_text_price'),reference,member,pairs)
                if not np.isclose(payload['local']['prediction'],result['price'],rtol=1e-9,atol=1e-4):
                    raise ValueError('Explanation does not match the submitted prediction')
            st.session_state[key]={'signature':signature,'payload':payload,'provenance':provenance}
        except Exception as exc:
            st.session_state.pop(key,None);st.error(f'Could not build the explanation: {exc}');return
    saved=st.session_state.get(key)
    if not saved or saved['signature']!=signature:
        st.info('Press “Build all explanation graphs” after estimating a car. Changing the car, model, reference gallery or calculation settings requires a fresh explanation. No graph-shaped guesswork.');return
    a=saved['payload'];local=a['local'];native=a['native']
    st.caption(f"{len(a['reference'])} reference cars · member {member+1} · native plot units: {a['unit']}. Reference choice changes the question being explained.")
    if a['conditional_text']:
        st.info('The submitted MiniLM description-price signal is held fixed across reference and scenario cars. These graphs explain vehicle facts and the scalar text signal. The separate “Which words moved my estimate?” panel above reruns the complete asking model with phrase combinations; these graphs do not explain words.')
    waterfall,swarm,effects,compare,heat,technical=st.tabs(['Why this price?','Across reference cars','Feature effects','Compare cars','Year × mileage','Technical receipt'])
    with waterfall:
        st.write('Start at the average prediction for eight reference cars. Each step allocates credit to a group of car facts until we reach your estimate. Red raises it; blue lowers it. Your paintwork has a vote, not a casting vote.')
        st.plotly_chart(plots.waterfall(local),width='stretch',key='shap_dollar_waterfall')
        st.caption('Units are dollars for the complete model, including all ensemble members. Make/model/body stay together, and CarBids, age and mileage-derived features are recalculated after each change. Contributions describe model behaviour, not fair-value adjustments or causal effects.')
        st.text(f"Reference prediction ${local['base']:,.0f} → submitted estimate ${local['prediction']:,.0f}.")
    with swarm:
        st.write('Each dot is one reference car. A wide horizontal spread means this feature often changes the selected member’s score. The chart is a crowd portrait, not a universal league table of car facts.')
        st.plotly_chart(plots.beeswarm(native),width='stretch',key='shap_beeswarm')
        st.caption('Features are ranked by mean absolute native SHAP within this gallery. Red/blue means high/low numeric feature values, scaled within each feature. Categories are grey; their codes are not a numerical ranking. Missing numeric values have no meaningful colour. This is one member, not the dollar-averaged ensemble.')
    with effects:
        st.write('Dots show how a feature’s contribution varies across the gallery. Colour can reveal a second feature’s relationship. It can suggest an interaction, but it does not prove one—let alone prove that changing a car’s birthday changes its engine.')
        numeric=[name for name in native['names'] if pd.api.types.is_numeric_dtype(native['x'][name])]
        default=native['names'].index('odometer') if 'odometer' in native['names'] else 0
        feature=st.selectbox('Feature to inspect',native['names'],index=default,format_func=label,key='shap_dependence_feature')
        colours=['None']+numeric
        default_colour=colours.index('vehicle_age_years') if 'vehicle_age_years' in colours else 0
        colour=st.selectbox('Colour by numeric feature',colours,index=default_colour,format_func=lambda x:'No second feature' if x=='None' else label(x),key='shap_dependence_colour')
        st.plotly_chart(plots.dependence(native,feature,None if colour=='None' else colour),width='stretch',key='shap_dependence')
        st.write('Below, the model answers a different question: what would it predict for this submitted car at other mileages? All derived inputs are recalculated. This curve uses the complete model and dollar prices.')
        st.plotly_chart(plots.mileage_curve(a['ice'],fields.get('odometer'),result['price']),width='stretch',key='shap_ice')
        st.caption('This individual conditional expectation curve is a sensitivity exercise, not a measured depreciation rate. Some combinations may be unrealistic or outside training support.')
    with compare:
        st.write('Follow your car’s line and compare it with up to four reference cars nearest in year and mileage. Similar prices can have different explanations. The committee may agree on the bill while arguing about the receipt.')
        st.plotly_chart(plots.decision(a['decision'],a['comparison']),width='stretch',key='shap_decision')
        st.caption('Each line adds native member SHAP contributions from a shared reference score. Feature order is for display; it is not the order trees execute. “Other features” preserves exact reconstruction. Neighbours use year and mileage, not price labels; they are illustrative, not matched comparable sales.')
    with heat:
        st.write('A map of the model’s asking estimates when year and mileage vary together. Brighter means a higher prediction—not that we found buried treasure under the bonnet.')
        st.plotly_chart(plots.interaction_grid(a['heatmap']),width='stretch',key='shap_interaction_grid')
        st.caption('This is a two-feature sensitivity surface, not a SHAP interaction decomposition. Make/model and other submitted facts stay fixed. The historical reference date remains May 2021; changing year can cross model generations, so read unusual combinations cautiously.')
    with technical:
        st.write('**Two explanation contracts.** The USD graph explains the served prediction function using grouped interventional permutation Shapley estimates. Native charts explain the selected tree model on its fitted output scale: log(price) for AFT, or the recorded target scale for LightGBM.')
        st.latex(r'\widehat p(x)=\sum_m w_m\,T_m^{-1}(f_m(x));\qquad \widehat p(x)\approx b_{\$}+\sum_g\phi_{g,\$}')
        st.write('The dollar approximation samples feature orders, not the final summation: every path telescopes from its reference prediction to the submitted prediction. Additivity checks can pass even when individual Monte Carlo allocations remain noisy.')
        st.write({'USD reconstruction error':local['error'],'Native member reconstruction error':native['error'],
            'Sampled feature orders':local['orders'],'Fixed background cars':local['background_rows'],
            'Coalition prediction rows':local['evaluated_rows'],'Selected member':native['member'],
            'Member weight':native['weight'],'Native units':native['unit'],'Seed':local['seed']})
        receipt=pd.DataFrame({'Group':local['names'],'Contribution USD':local['values'],'Monte Carlo SE USD':local['se']})
        st.dataframe(receipt,hide_index=True,width='stretch')
        st.caption('SE measures variation across independent antithetic permutation pairs, conditional on this fixed background. It is not a confidence interval for the car price, background representativeness, or causal effect. Correlated groups can create implausible hybrids; grouping make/model/body reduces, but does not eliminate, that problem.')
        st.write('Exponentiating individual log contributions does not create additive dollar effects. Averaging log-SHAP across members would explain a different function. No such shortcut is used here.')
        st.json(saved['provenance'],expanded=False)
        export={'method':local['method'],'prediction_usd':local['prediction'],'reference_usd':local['base'],
            'groups':receipt.to_dict('records'),'sampled_orders':local['orders'],'background_rows':local['background_rows'],
            'reconstruction_error_usd':local['error'],'native_units':native['unit'],'member':native['member'],
            'reference_provenance':saved['provenance'],'scope':'Historical 2021 asking model; not causal or current-sale valuation'}
        st.download_button('Download explanation receipt',json.dumps(export,indent=2),'pricing-explanation.json','application/json',key='shap_graph_receipt')
