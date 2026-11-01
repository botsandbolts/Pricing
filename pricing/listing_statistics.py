"""Offline-only controlled regional and description-signal experiments."""
import hashlib
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

CATEGORIES=['manufacturer','make_model','state','type','fuel','transmission','drive','title_status','retail_condition','cylinders','paint_color','month']

def prepare(rows):
    d=pd.DataFrame(rows).copy()
    for name in ('price','year','odometer'):d[name]=pd.to_numeric(d.get(name),errors='coerce')
    d['date']=pd.to_datetime(d.get('date'),utc=True,errors='coerce')
    d=d.loc[d.price.between(1,10000000)&d.year.between(1900,2021)&d.odometer.between(0,2000000)&d.date.notna()].copy()
    if d.empty:raise ValueError('No valid historical price records')
    condition=d.get('condition',pd.Series('unknown',index=d.index)).fillna('unknown').astype(str).str.lower()
    d['retail_condition']=condition.where(pd.to_numeric(condition,errors='coerce').isna(),'unknown')
    for name in CATEGORIES:
        if name not in d:d[name]='unknown'
        d[name]=d[name].fillna('unknown').astype(str).str.lower().str.strip().replace({'':'unknown','nan':'unknown'})
    d['make_model']=d.manufacturer+'|'+d.model.fillna('unknown').astype(str).str.lower().str.strip()
    d['month']=d.date.dt.strftime('%Y-%m')
    age=d.date.dt.year-d.year
    miles=np.log1p(d.odometer)
    d['age']=age;d['age_squared']=age**2;d['log_miles']=miles;d['log_miles_squared']=miles**2;d['age_mileage']=age*miles
    d['day']=(d.date-pd.Timestamp('2010-01-01',tz='UTC')).dt.total_seconds()/86400
    grade=pd.to_numeric(d.get('auction_grade',d.get('condition',pd.Series(np.nan,index=d.index))),errors='coerce')
    grade=grade.where(grade<=5,grade/10).where(grade.between(1,50))
    d['grade_missing']=grade.isna().astype(float);d['grade']=grade.fillna(0)
    d['text']=d.get('text',pd.Series('',index=d.index)).fillna('').astype(str)
    d['group']=d.get('vehicle_group',d['row_id']).fillna(d['row_id']).astype(str)
    return d.reset_index(drop=True)

NUMERIC=['age','age_squared','log_miles','log_miles_squared','age_mileage','day','grade_missing','grade']
class AuditRegressor:
    def fit(self,d,embeddings=None):
        self.text=embeddings is not None
        self.encoder=OneHotEncoder(handle_unknown='ignore',min_frequency=10,sparse_output=True,dtype=np.float64)
        x=self.encoder.fit_transform(d[CATEGORIES]);self.scaler=StandardScaler()
        numeric=self.scaler.fit_transform(d[NUMERIC])
        blocks=[x,csr_matrix(numeric)]
        if self.text:blocks.append(csr_matrix(embeddings))
        self.model=Ridge(alpha=10.,solver='lsqr',tol=1e-5,max_iter=1000).fit(hstack(blocks,format='csr'),np.log(d.price))
        self.names=list(self.encoder.get_feature_names_out())+NUMERIC+([f'MiniLM_{i}' for i in range(np.asarray(embeddings).shape[1])] if self.text else [])
        return self
    def predict(self,d,embeddings=None):
        blocks=[self.encoder.transform(d[CATEGORIES]),csr_matrix(self.scaler.transform(d[NUMERIC]))]
        if self.text:
            if embeddings is None:raise ValueError('Description embeddings are required')
            blocks.append(csr_matrix(embeddings))
        return np.exp(np.clip(self.model.predict(hstack(blocks,format='csr')),0,np.log(1e7)))
    def coefficients(self):return dict(zip(self.names,self.model.coef_.tolist()))

def paired_bootstrap(delta,groups,draws=300):
    table=pd.DataFrame({'delta':delta,'group':groups}).groupby('group').agg(total=('delta','sum'),n=('delta','size'))
    rng=np.random.default_rng(42);total=table.total.to_numpy();n=table.n.to_numpy();values=[]
    for _ in range(draws):
        selected=rng.integers(0,len(table),len(table));values.append(total[selected].sum()/n[selected].sum())
    return [float(x) for x in np.quantile(values,[.025,.975])]

def experiment(rows,side,bootstrap=40,embedding_encoder=None,encoder_name=None):
    d=prepare(rows);part={p:d.loc[d.phase.eq(p)].copy() for p in ('train','validation','test')}
    if any(len(x)<50 for x in part.values()):raise ValueError('Need at least 50 valid records in each original partition')
    if any(set(part[a].group)&set(part[b].group) for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Vehicle leakage across historical partitions')
    tr,va,te=(part[x] for x in ('train','validation','test'))
    base=AuditRegressor().fit(tr);pred=base.predict(te);residual=np.log(te.price.to_numpy()/pred)
    validation_residual=np.log(va.price.to_numpy()/base.predict(va));radius=float(np.quantile(np.abs(validation_residual),min(1,np.ceil((len(va)+1)*.9)/len(va)),method='higher'))
    report={'market':side,'rows':{p:len(v) for p,v in part.items()},'basis':'Historical advertised asking dollars' if side=='ask' else 'Historical reported auction dollars',
            'method':'Ridge log-price audit baseline; sparse categorical controls, nonlinear age/mileage terms, grade and calendar controls. Separate from production XGBoost/Bayesian models.',
            'test_mae_usd':float(np.mean(np.abs(te.price-pred))),'historical_interval_coverage':float(np.mean(np.abs(residual)<=radius)),
            'interval_calibration_rows':len(va),'nominal_coverage':.9,'limits':'Exploratory selected historical cohort, not representative current prices. These source test records may have been examined by earlier project experiments.'}
    # Regional associations: fixed categories and regularization, row/vehicle bootstrap.
    coef=base.coefficients();counts=tr.state.value_counts();states=[s for s,n in counts.items() if n>=100 and s not in ('unknown','nan')]
    def centered(coefficients):
        raw={s:coefficients.get('state_'+s,0.) for s in states}
        center=sum(raw[s]*counts[s] for s in states)/sum(counts[s] for s in states) if states else 0.
        return {s:raw[s]-center for s in states}
    effects=centered(coef);boot={s:[] for s in states};rng=np.random.default_rng(1729)
    # Reuse the training-only encoding to keep the meaning of each state column fixed.
    matrix=hstack([base.encoder.transform(tr[CATEGORIES]),csr_matrix(base.scaler.transform(tr[NUMERIC]))],format='csr');target=np.log(tr.price.to_numpy())
    for _ in range(bootstrap if states else 0):
        ids=rng.integers(0,len(tr),len(tr));fit=Ridge(alpha=10.,solver='lsqr',tol=1e-4,max_iter=500).fit(matrix[ids],target[ids]);values=centered(dict(zip(base.names,fit.coef_)))
        for s in states:boot[s].append(values[s])
    report['regions']=[{'state':s,'training_vehicles':int(counts[s]),'adjusted_association_pct':float(100*np.expm1(effects[s])),
                        'bootstrap_90_pct':[float(100*np.expm1(x)) for x in np.quantile(boot[s],[.05,.95])] if boot[s] else None} for s in states]
    report['regional_method']='State coefficients relative to the training-count-weighted mean of states with >=100 training vehicles; controls as above; 90% percentile bootstrap, fixed encoding and penalty. Associations, not transport arbitrage or causal location effects; overlap is not established.'
    if side=='ask':
        if embedding_encoder is None:raise ValueError('A real frozen description encoder is required for the asking experiment')
        embeddings=np.asarray(embedding_encoder(d.text.tolist()),dtype=np.float32)
        if embeddings.ndim!=2 or len(embeddings)!=len(d) or not np.isfinite(embeddings).all():raise ValueError('Invalid frozen embeddings')
        augmented=AuditRegressor().fit(tr,embeddings=embeddings[tr.index]);text_pred=augmented.predict(te,embeddings=embeddings[te.index])
        improvement=np.abs(te.price.to_numpy()-pred)-np.abs(te.price.to_numpy()-text_pred)
        report['description_experiment']={'base_mae_usd':report['test_mae_usd'],'with_minilm_mae_usd':float(np.mean(np.abs(te.price-text_pred))),
            'mae_saved_usd':float(improvement.mean()),'paired_vehicle_bootstrap_95_usd':paired_bootstrap(improvement,te.group.to_numpy()),
            'encoder':encoder_name or 'Provided frozen description encoder','embedding_dimensions':embeddings.shape[1],
            'interpretation':'Frozen MiniLM description vectors are added to structured vehicle controls in a separate Ridge log-price experiment. No hand-written keyword flags or verified-condition labels are used. Paired dollar MAE is measured on the same unchanged historical test vehicles; it does not identify a causal premium for wording.'}
    # Fresh OOF diagnostics for the audit baseline. Never delete records or alter production weights.
    oof=np.full(len(tr),np.nan);groups=tr.group.to_numpy()
    for fit,held in GroupKFold(3).split(tr,groups=groups):oof[held]=AuditRegressor().fit(tr.iloc[fit]).predict(tr.iloc[held])
    errors=np.abs(np.log(tr.price.to_numpy()/oof));threshold=float(np.quantile(errors,.75))
    flagged=errors>=threshold
    report['outlier_review']={'method':'Fresh three-fold vehicle-group-held-out absolute log residuals from the audit baseline; largest quartile flagged for review only. No deletion or retraining of production models.',
        'training_rows':len(tr),'flagged_rows':int(flagged.sum()),'threshold_absolute_log_residual':threshold,
        'diagnostics':[{'make':make,'training_vehicles':len(group),'largest_residual_share_pct':float(100*flagged[group.index].mean())} for make,group in tr.reset_index(drop=True).groupby('manufacturer') if len(group)>=100],
        'limits':'Large residuals can be legitimate unusual cars, omitted features or bad labels. Flags are not verified data errors.'}
    return report
