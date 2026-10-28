"""Date-aware scenarios anchored to the existing asking and auction estimators."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import numpy as np
import pandas as pd
from marketbridge.depreciation import DepreciationCurve
from marketbridge.market_clock import MarketClock


def today():return datetime.now(ZoneInfo('Europe/Berlin')).date()
def age_on(year,date):
    d=pd.Timestamp(date);return d.year+(d.dayofyear-1)/365.25-int(year)


class TimeTransport:
    def __init__(self,root,clock=None):
        self.root=Path(root);folder=self.root/'models/time-transport'
        self.manifest=json.loads((folder/'manifest.json').read_text());self.clock=clock or MarketClock(folder)
        self.curves={s:DepreciationCurve.load(folder/f'{s}-curve.json') for s in ['ask','bid']}
    def validate(self,fields,date):
        try:
            raw_year=float(fields['year']);year=int(raw_year);miles=float(fields['odometer'])
            if raw_year!=year or not np.isfinite(raw_year):raise ValueError('Invalid model year')
        except (KeyError,TypeError,ValueError,OverflowError) as exc:raise ValueError('A valid model year and mileage are required for time transport') from exc
        age=age_on(year,date)
        if year>2021:raise ValueError('Model years after 2021 are unsupported by the asking data. No current price is invented for a new generation.')
        if not 0<=age<=30:raise ValueError('This depreciation scenario supports vehicle ages 0–30 years; collector and pre-production cases need another model.')
        if not np.isfinite(miles) or not 0<=miles<=2_000_000:raise ValueError('Mileage must be between 0 and 2,000,000 miles')
        return year,age,miles
    def estimate(self,pipeline,text,fields,date,side='ask',allow_newer_bid=False,as_of=None,text_price=None):
        year,age,miles=self.validate(fields,date)
        if side=='bid' and year>2015 and not allow_newer_bid:raise ValueError('Auction training ends with model year 2015. Enable the explicit generation-analogue assumption for 2016–2021 cars.')
        ref='2021-05-01' if side=='ask' else '2015-01-01'
        anchor_age=age_on(year,ref) if side=='ask' else 5.
        anchor_fields={**fields,'as_of':ref}
        if side=='bid':
            if 'year' in pipeline.manifest.get('feature_names',[]):raise ValueError('Time transport requires an age-based bid predictor, not a raw model-year feature')
            # The bid tree uses age, not raw year/calendar features. Ask it about an
            # age-five comparable in its saved reference-dollar unit, not a future year.
            anchor_fields.update(as_of=f'{year+5}-01-01',condition=None)
        if text_price is not None and getattr(pipeline,'aft',None) is not None:
            base=float(pipeline.aft.predict_frame(pd.DataFrame([anchor_fields]),text_price)[0])
        else:base=float(pipeline.predict(text if side=='ask' else '',anchor_fields)['price'])
        shape=pd.DataFrame([{**fields,'age':age,'odometer':miles}])
        ratio=float(self.curves[side].ratio(shape,anchor_age,age)[0])
        level,market=self.clock.value(date,as_of=as_of or date)
        if side=='ask':reference,reference_month=self.clock.reference(ref)
        else:reference=float(self.clock.data.loc['2015-01']);reference_month='2015-01'
        factor=level/reference;price=base*ratio*factor
        if not np.isfinite(price) or not 0<price<=1e7:raise ValueError('Transport produced an unsupported price')
        make=str(fields.get('manufacturer','unknown')).lower().strip();model=make+'::'+str(fields.get('model','unknown')).lower().strip()
        known=model in self.curves[side].levels['model'];make_known=make in self.curves[side].levels['manufacturer']
        return {'side':side,'date':str(pd.Timestamp(date).date()),'price':price,'anchor_price_reference_usd':base,'anchor_age':anchor_age,'age':age,'age_factor':ratio,'market_factor':factor,'market':market,'reference_index_month':reference_month,'model_known_to_curve':known,'make_known_to_curve':make_known,'generation_analogue':bool(side=='bid' and year>2015),'target_coverage_validated':False,'status':'experimental_time_transport','curve':self.curves[side].kind,'warning':'Cross-sectional depreciation + aggregate retail market index transport; individual current prices and whole-price intervals are unvalidated. Ask and Bid remain different economic targets.'}
    def trajectory(self,pipelines,text,fields,months,annual_miles=0.,allow_newer_bid=False,as_of=None):
        known=pd.Timestamp(as_of or today());records=[];miles=float(fields['odometer']);text_price=None
        ask=pipelines.get('ask')
        if ask is not None and getattr(ask,'text_model',None) is not None:
            from pricing.text import predict_text
            text_price=float(predict_text(ask.text_model,ask.tokenizer,[text],ask.device)[0])
        for d in months:
            delta=(pd.Timestamp(d)-known).days/365.25
            mileage=max(0.,miles+float(annual_miles)*delta)
            scenario={**fields,'odometer':mileage}
            for side,pipeline in pipelines.items():
                try:
                    r=self.estimate(pipeline,text,scenario,d,side,allow_newer_bid,as_of=known,text_price=text_price if side=='ask' else None)
                    records.append({'Date':pd.Timestamp(d),'Market':'Ask · advertised' if side=='ask' else 'Bid · wholesale','Value USD':r['price'],'Mileage':mileage,'Market index':r['market']['kind'],'Reconstructed':pd.Timestamp(d)<known,'Age':r['age']})
                except ValueError:continue
        return pd.DataFrame(records)
