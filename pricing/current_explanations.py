"""Bounded explanation studio for the actual current XGBoost/date/Bayesian function."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from marketbridge.time_engine import age_on,today
from pricing.hierarchy_sparse_live import load
from pricing.bayesian_hierarchy import predict as posterior_predict
from pricing.forest_time import adjust
from pricing.visual_explanations import load_reference,permutation_shap,GROUPS
from pricing import explanation_plots as plots

class CurrentView:
    def __init__(self,root,side,date,notes,bayesian,transport):
        self.side=side;self.date=date;self.notes=notes;self.bayesian=bayesian;self.transport=transport
        self.model,self.schema,self.posterior,self.draws,self.projector,_=load(str(root),side)
    def matrix(self,cars):
        from scipy.sparse import csr_matrix,hstack
        import xgboost as xgb
        cars=cars.copy().reset_index(drop=True)
        age=np.array([age_on(y,'2021-05-01') for y in cars.year]) if self.side=='ask' else np.full(len(cars),5.)
        x=pd.DataFrame({'age':age,'log_miles':np.log1p(cars.odometer.astype(float)),
                        'auction_grade':pd.to_numeric(cars.get('auction_grade',pd.Series(np.nan,index=cars.index)),errors='coerce')})
        display=x.copy();types=['q']*3
        for c,levels in self.schema.items():
            values=cars.get(c,pd.Series('unknown',index=cars.index)).fillna('unknown').astype(str).str.strip().str.lower()
            codes=pd.Categorical(values,categories=levels).codes.astype(float);codes[codes<0]=np.nan
            x[c]=codes;display[c]=values;types.append('c')
        values=x.to_numpy(dtype=np.float32);n,k=values.shape
        data=csr_matrix((values.ravel(),np.tile(np.arange(k),n),np.arange(0,(n+1)*k,k)),shape=(n,k))
        if self.projector is not None:
            t=self.projector.transform([self.notes]*len(cars));data=hstack([data,t],format='csr');types+=['q']*t.shape[1]
            if t.shape[1]==1:display['MiniLM description price']=np.asarray(t.toarray()).ravel()
            else:display['Description (all text dimensions)']=self.notes[:100] or '(blank)'
        return xgb.DMatrix(data,feature_names=self.model.feature_names,feature_types=types,enable_categorical=True,nthread=2),display,age
    def native(self,cars):
        matrix,display,_=self.matrix(cars)
        raw=self.model.predict(matrix,pred_contribs=True)
        margin=self.model.predict(matrix,output_margin=True)
        values=raw[:,:-1]
        if self.projector is not None:
            k=len(display.columns)-1
            values=np.column_stack([values[:,:k],values[:,k:].sum(axis=1)])
        error=float(np.max(abs(raw[:,-1]+values.sum(axis=1)-margin)))
        if error>1e-4:raise ValueError('Native contributions do not reconstruct the current model.')
        return dict(x=display,values=values,base=raw[:,-1],prediction=margin,error=error,names=list(display),unit='log USD',member='Current XGBoost',weight=1.)
    def predict(self,cars):
        cars=cars.copy().reset_index(drop=True)
        matrix,_,age=self.matrix(cars)
        raw=self.model.predict(matrix)
        base=raw if self.side=='ask' else np.exp(raw)
        cases=cars.copy();cases['age']=age;cases['base_price']=base
        if self.bayesian:
            corrected=posterior_predict(self.posterior,self.draws,cases).corrected_price.to_numpy()
        else:corrected=base
        prices=[]
        for i,car in cars.iterrows():
            row={**car.to_dict(),'age':age[i],'base_price':base[i],'corrected_price':corrected[i],
                 'source_date':'2021-05-01' if self.side=='ask' else f'{int(car.year)+5}-01-01'}
            prices.append(adjust(self.transport,row,self.side,self.date,car.odometer,as_of=today())['price'])
        return np.array(prices)

def analyse(root,fields,notes,date,bayesian,side,receipt):
    from pricing.current_value import current_transport
    transport,_=current_transport(root)
    view=CurrentView(root,side,date,notes,bayesian,transport)
    reference,provenance=load_reference()
    low=max(1992,pd.Timestamp(date).year-29);high=2021 if side=='ask' or receipt.get('analogue') else 2015
    reference=reference.loc[reference.year.between(low,high)&reference.odometer.notna()].copy()
    reference['auction_grade']=fields.get('auction_grade')
    if len(reference)<2:raise ValueError('Too few supported reference cars for this date and market.')
    reference=reference.iloc[np.linspace(0,len(reference)-1,min(16,len(reference)),dtype=int)].reset_index(drop=True)
    query=pd.DataFrame([fields]);groups={**GROUPS,'Auction grade':('auction_grade',)}
    bg=reference.iloc[np.linspace(0,len(reference)-1,min(3,len(reference)),dtype=int)]
    local=permutation_shap(view.predict,query,bg,pairs=2,groups=groups)
    if not np.isclose(local['prediction'],receipt['price'],rtol=1e-5,atol=.05):raise ValueError('Explanation does not match your displayed price. Estimate again.')
    native=view.native(reference)
    distance=((reference.year-float(fields['year']))/10)**2+((np.log1p(reference.odometer)-np.log1p(fields['odometer']))/2)**2
    peers=reference.iloc[np.argsort(distance.to_numpy())[:4]]
    comparison=pd.concat([query,peers],ignore_index=True)
    decision=view.native(comparison)
    miles=np.unique(np.r_[np.linspace(0,max(150000,fields['odometer']*1.3),15),fields['odometer']])
    scenarios=pd.concat([query]*len(miles),ignore_index=True);scenarios['odometer']=miles
    ice={'mileage':miles,'price':view.predict(scenarios)}
    years=np.arange(max(low,int(fields['year'])-3),min(high,int(fields['year'])+3)+1)
    gm=np.linspace(0,max(150000,fields['odometer']*1.3),7)
    grid=pd.concat([query]*(len(years)*len(gm)),ignore_index=True)
    grid['year']=np.repeat(years,len(gm));grid['odometer']=np.tile(gm,len(years))
    heat={'years':years,'mileage':gm,'price':view.predict(grid).reshape(len(years),len(gm))}
    return dict(local=local,native=native,decision=decision,comparison=comparison,ice=ice,heatmap=heat,provenance=provenance,reference=reference)

def render(root,fields,notes,date,results,bayesian):
    st.markdown('### 🔎 The full visual explanation studio')
    st.caption('Current installed models · both markets · graphs built only on request. The dollar waterfall and sensitivity plots use your selected date and pricing mode; native TreeSHAP explains the underlying historical XGBoost log-price score.')
    choices=[s for s,r in results.items() if 'error' not in r]
    if not choices:return
    side=st.radio('Market to explain',choices,format_func=lambda s:'Seller asking' if s=='ask' else 'Wholesale auction',horizontal=True,key='current_shap_market')
    identity=json.dumps([fields,notes,str(date),bayesian,side,results[side]['price']],sort_keys=True,default=str)
    if st.button('Build all explanation graphs',type='primary',key='current_all_graphs'):
        st.session_state.pop('_current_graphs',None)
        try:
            with st.spinner('Checking the current model, reference cars and explanation arithmetic…'):
                data=analyse(root,fields,notes,date,bayesian,side,results[side])
            st.session_state['_current_graphs']={'identity':identity,'data':data}
        except (ValueError,KeyError,RuntimeError,MemoryError) as exc:
            st.warning('Graphs unavailable: '+str(exc))
    saved=st.session_state.get('_current_graphs',{})
    if saved.get('identity')!=identity:
        st.info('Build the suite after estimating. Changing the car, date, market or pricing mode invalidates the graphs.');return
    data=saved['data'];native=data['native']
    st.caption(f"{len(data['reference'])} supported historical reference cars, selected from a small fixture gallery—not a population sample. The submitted description and optional auction grade are held fixed. Reference cars are illustrative inputs, not current comparable sales.")
    tabs=st.tabs(['Why this price?','Across reference cars','Feature effects','Compare cars','Year × mileage','Technical receipt'])
    with tabs[0]:
        fig=plots.waterfall(data['local']);fig.update_layout(title='Your displayed price · grouped dollar SHAP');fig.update_xaxes(title='Date-adjusted model price / contribution (USD)')
        st.plotly_chart(fig,width='stretch')
        st.caption('Grouped interventional permutation Shapley estimates of the complete selected dollar function, including date transport and Bayesian correction when selected. Four sampled orders, three background cars at most. Exact total reconstruction does not imply precise individual allocations or causal effects.')
    with tabs[1]:
        st.plotly_chart(plots.beeswarm(native),width='stretch')
        import plotly.express as px
        importance=pd.DataFrame({'Feature':native['names'],'Mean absolute SHAP (log USD)':np.abs(native['values']).mean(axis=0)}).sort_values('Mean absolute SHAP (log USD)').tail(12)
        st.plotly_chart(px.bar(importance,x='Mean absolute SHAP (log USD)',y='Feature',orientation='h',title='Feature importance · this reference gallery only'),width='stretch')
        st.caption('Native TreeSHAP in log dollars for this XGBoost model. Grey categories have no numerical ranking. All text dimensions are summed into a description group; description text stays fixed across the reference gallery. This chart excludes the Bayesian and date layers.')
    with tabs[2]:
        feature=st.selectbox('Feature to inspect',native['names'],key='current_dependence')
        colours=['None']+[c for c in native['x'] if pd.api.types.is_numeric_dtype(native['x'][c])]
        colour=st.selectbox('Colour by numeric feature',colours,key='current_dependence_colour')
        st.plotly_chart(plots.dependence(native,feature,None if colour=='None' else colour),width='stretch')
        fig=plots.mileage_curve(data['ice'],fields['odometer'],results[side]['price']);fig.update_yaxes(title='Selected-date scenario price (USD)');st.plotly_chart(fig,width='stretch')
        st.caption('Mileage curve recomputes the complete selected function. This is individual conditional sensitivity, not causal depreciation, a time forecast or population partial dependence. The description remains fixed—even mileage mentioned in its text.')
    with tabs[3]:
        st.plotly_chart(plots.decision(data['decision'],data['comparison']),width='stretch')
        st.caption('Cumulative native log-price contributions compare your car with four gallery neighbours selected by year and mileage. Date adjustment and Bayesian corrections are excluded from these native paths. Neighbours are not matched sales.')
    with tabs[4]:
        st.plotly_chart(plots.interaction_grid(data['heatmap']),width='stretch')
        st.caption('Complete selected-date scenario price, with make/model and description fixed. This is a sensitivity surface, not SHAP interaction values. Model-year substitutions can imply unobserved generations and unrealistic vehicles.')
    with tabs[5]:
        st.write({'Dollar reconstruction error':data['local']['error'],'Native log reconstruction error':native['error'],'Sampled orders':data['local']['orders'],'Background cars':data['local']['background_rows'],'Valuation date':str(date),'Bayesian layer included in dollar plots':bayesian})
        st.dataframe(pd.DataFrame({'Group':data['local']['names'],'Contribution USD':data['local']['values'],'Sampled-order SE USD':data['local']['se']}),hide_index=True)
        st.caption('The SE is conditional Monte Carlo variability from two antithetic pairs, not a car-price confidence interval. Correlated inputs may create implausible hybrids. Exponentiating individual log SHAP terms does not produce additive dollar SHAP; dollar allocations above come from a separate complete-function calculation.')
        st.json(data['provenance'],expanded=False)
        export={'date':str(date),'market':side,'bayesian':bayesian,'price':data['local']['prediction'],'reference':data['local']['base'],'groups':data['local']['names'],'contributions_usd':data['local']['values'].tolist(),'native_error_log':native['error'],'dollar_error':data['local']['error']}
        st.download_button('Download explanation receipt',json.dumps(export,indent=2),'current-explanation.json','application/json')
