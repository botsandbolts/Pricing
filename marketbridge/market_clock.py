"""Release-aware monthly market data and short-horizon forecast experiments."""
import io,json,re,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

SERIES='CUUR0000SETA02'
PAGE=f'https://fred.stlouisfed.org/series/{SERIES}'
CSV=f'https://fred.stlouisfed.org/graph/?g=1&id={SERIES}'
DATA=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES}&cosd=2013-01-01'


def fetch_snapshot():
    r=requests.get(DATA,timeout=(8,25));r.raise_for_status();d=pd.read_csv(io.StringIO(r.text))
    if d.shape[1]!=2:raise ValueError('Unexpected FRED CSV schema')
    d.columns=['date','value'];d['date']=pd.to_datetime(d.date);d['value']=pd.to_numeric(d.value,errors='coerce');d=d.dropna()
    d['month']=d.date.dt.to_period('M').astype(str);d=d[d.month.ge('2013-01')][['month','value']]
    if len(d)<100 or d.month.duplicated().any() or (d.value<=0).any():raise ValueError('Invalid market index')
    periods=pd.PeriodIndex(d.month,freq='M')
    if not np.all(np.diff(periods.asi8)==1):raise ValueError('Market index contains missing months')
    page=requests.get(PAGE,timeout=(8,25));page.raise_for_status()
    match=re.search(r'"dateModified"\s*:\s*"([^"]+)"',page.text)
    if not match:raise ValueError('No publication metadata; keep the audited snapshot')
    released=pd.Timestamp(match[1]).date();today=datetime.now(timezone.utc).date()
    if released>today or periods[-1]>pd.Period(today,freq='M'):raise ValueError('Future market observation')
    return d,{'series':SERIES,'source':PAGE,'csv':DATA,'retrieved_at':datetime.now(timezone.utc).isoformat(),'latest_release_date':str(released),'latest_observation':str(periods[-1]),'vintage':'current vintage; historical release replay uses conservative two-month lag, not ALFRED vintages'}


def forecast(values,h,method):
    y=np.log(np.asarray(values,float));h=int(h)
    if not 1<=h<=12:raise ValueError('Forecast horizon must be 1–12 months')
    if method=='last_value':z=np.repeat(y[-1],h)
    elif method=='seasonal_naive':z=np.array([y[-12+(i%12)] for i in range(h)])
    elif method=='damped_holt':
        # Damped additive Holt on log levels, implemented with SciPy only.
        # Each information cutoff estimates its own smoothing parameters.
        initial_trend=float(np.median(np.diff(y[:min(13,len(y))])))
        def states(params,score=False):
            alpha,beta,phi=params;level=float(y[0]);trend=initial_trend;sse=0.
            for observed in y[1:]:
                prediction=level+phi*trend;sse+=(observed-prediction)**2
                previous=level;level=alpha*observed+(1-alpha)*prediction
                trend=beta*(level-previous)+(1-beta)*phi*trend
            return sse if score else (level,trend)
        fitted=minimize(lambda p:states(p,True),[.6,.1,.95],method='L-BFGS-B',bounds=[(.001,.999),(.001,.999),(.8,.9999)],options={'maxiter':150,'ftol':1e-12})
        if not np.isfinite(fitted.fun):raise ValueError('Damped Holt fit failed')
        level,trend=states(fitted.x);phi=fitted.x[2]
        z=level+np.cumsum(phi**np.arange(1,h+1))*trend
    elif method=='ar_1_12':
        xx=np.array([[1.,y[i-1],y[i-12]] for i in range(12,len(y))]);b=np.linalg.lstsq(xx,y[12:],rcond=None)[0]
        history=list(y);z=[]
        for _ in range(h):v=float(b@[1.,history[-1],history[-12]]);history.append(v);z.append(v)
        z=np.array(z)
    else:raise ValueError('Unknown forecast method')
    if not np.isfinite(z).all():raise ValueError('Nonfinite forecast')
    return np.exp(z)


def backtest(data):
    d=data.reset_index(drop=True);methods=['last_value','seasonal_naive','damped_holt','ar_1_12'];records=[]
    for origin in range(60,len(d)-12,3):
        for method in methods:
            p=forecast(d.value.iloc[:origin+1],12,method)
            for h in [1,3,6,12]:
                observed=float(d.value.iloc[origin+h]);pred=float(p[h-1])
                records.append({'origin':d.month.iloc[origin],'target_month':d.month.iloc[origin+h],'horizon':h,'method':method,'observed':observed,'predicted':pred,'abs_log_error':abs(np.log(observed/pred)),'squared_log_error':np.log(observed/pred)**2,'stage':'selection' if d.month.iloc[origin]<'2023-01' else 'locked_test'})
    frame=pd.DataFrame(records);summary=[]
    for (stage,method,h),group in frame.groupby(['stage','method','horizon']):summary.append({'stage':stage,'method':method,'horizon':int(h),'forecasts':len(group),'mean_abs_log_error':float(group.abs_log_error.mean()),'rms_log_error':float(np.sqrt(group.squared_log_error.mean()))})
    # The live use is 1–3 months; select ONLY on pre-2023 origins, then report locked tests.
    select=frame[(frame.stage=='selection')&frame.horizon.isin([1,3])].groupby('method').abs_log_error.mean()
    selected=str(select.idxmin());errors={}
    for h in [1,3,6,12]:
        e=frame[(frame.stage=='selection')&(frame.method==selected)&(frame.horizon==h)].abs_log_error
        errors[str(h)]=float(e.quantile(.9))
    return {'selected':selected,'selection_metric':'mean absolute log index error at horizons 1 and 3, origins before 2023 only','summary':summary,'sensitivity_abs_log_error_90':errors,'warning':'Current-vintage rolling-origin index backtest, not a real-time vintage or per-car price backtest. These empirical errors describe market-index forecast sensitivity only.'},frame


class MarketClock:
    def __init__(self,folder,live=None):
        self.folder=Path(folder);self.data=pd.read_csv(self.folder/'market-index.csv')
        self.meta=json.loads((self.folder/'market-index-meta.json').read_text());self.report=json.loads((self.folder/'market-backtest.json').read_text())
        self.live_status='saved_snapshot'
        if live is not None:
            d,meta=live
            if d.month.iloc[-1]>=self.data.month.iloc[-1]:self.data=d;self.meta=meta;self.live_status='refreshed_public_index'
        self.data=self.data.set_index('month').value.astype(float)
    def value(self,target_date,as_of=None):
        target=pd.Timestamp(target_date);known=pd.Timestamp(as_of if as_of is not None else target_date)
        last=pd.Period(self.meta['latest_observation'],freq='M');released=pd.Timestamp(self.meta['latest_release_date'])
        cutoff=last if known.normalize()>=released else pd.Period(known,freq='M')-2
        cutoff=min(cutoff,pd.Period(known,freq='M'));history=self.data.loc[self.data.index<=str(cutoff)]
        if history.empty:raise ValueError('No published market index available at this date')
        month=pd.Period(target,freq='M');end=pd.Period(history.index[-1],freq='M');h=month.ordinal-end.ordinal
        if h<=0:
            if str(month) not in history:raise ValueError('Market observation unavailable')
            return float(history.loc[str(month)]),{'kind':'observed_monthly_index','month':str(month),'last_known_month':str(end),'horizon':0,'release_rule':'latest-release metadata or conservative historical two-month lag'}
        if h>6:raise ValueError('Market forecast beyond six months is unsupported; refresh data or choose a nearer date')
        pred=float(forecast(history.to_numpy(),h,self.report['selected'])[-1])
        return pred,{'kind':'forecast_monthly_index','month':str(month),'last_known_month':str(end),'horizon':h,'method':self.report['selected'],'release_rule':'latest-release metadata or conservative historical two-month lag'}
    def reference(self,date):
        # Retrospective historical market of the asking anchor. This is known today;
        # it is not a claim that the monthly index was already published that day.
        month=str(pd.Period(pd.Timestamp(date),freq='M'))
        if month not in self.data:raise ValueError('Reference index missing')
        return float(self.data.loc[month]),month
