"""Transport actual historical reference prices to a declared valuation date."""
import numpy as np
import pandas as pd
from marketbridge.time_engine import age_on,today

def adjust_reference(car,side,target,transport,target_mileage=None):
    if side not in ('ask','bid'):raise ValueError('Unknown market')
    price=car.get('price');source=car.get('observed_date') or car.get('sale_date')
    if price is None or source is None:raise ValueError('Original price and observation date are not installed for this record yet.')
    price=float(price)
    if not np.isfinite(price) or price<=0:raise ValueError('Invalid historical price')
    mileage=car.get('odometer')
    if mileage is None:raise ValueError('Original mileage is unavailable; a date-adjusted price would require another assumption.')
    mileage=float(mileage);to_miles=mileage if target_mileage is None else float(target_mileage)
    transport.validate({**car,'odometer':to_miles},target)
    source_age=max(0.,age_on(car['year'],source));target_age=age_on(car['year'],target)
    if source_age>30:raise ValueError('This historical vehicle is outside the fitted depreciation-age range.')
    before=pd.DataFrame([{**car,'age':source_age,'odometer':mileage}])
    factor=float(transport.curves[side].ratio(before,source_age,target_age,from_miles=mileage,to_miles=to_miles)[0])
    current,market=transport.clock.value(target,as_of=today())
    historical,reference_month=transport.clock.reference(source)
    market_factor=float(current/historical);adjusted=price*factor*market_factor
    if not np.isfinite(adjusted) or not 0<adjusted<=1e7:raise ValueError('Date adjustment is outside the supported numerical range.')
    make=str(car.get('manufacturer') or '').lower();model=make+'::'+str(car.get('model') or '').lower()
    return {'original_price_usd':price,'original_date':str(pd.Timestamp(source).date()),'price_usd':adjusted,'valuation_date':str(pd.Timestamp(target).date()),
            'age_mileage_factor':factor,'market_factor':market_factor,'reference_index_month':str(reference_month),'market':market,'comparison_mileage':to_miles,
            'pooled_model_curve':model not in transport.curves[side].levels['model'],'scope':'Historical price transported with cross-sectional depreciation and an aggregate retail used-car CPI index, also used as a wholesale proxy. Not a live offer or validated current-car appraisal.'}

def feature_prices(rows,side,target,transport,target_mileage=None):
    """Vectorized same-date observed-price scenarios for price-aware retrieval."""
    prices=np.array([r.get('price') if r.get('price') is not None else np.nan for r in rows],dtype=float)
    miles=np.array([r.get('odometer') if r.get('odometer') is not None else np.nan for r in rows],dtype=float)
    years=np.array([r['year'] for r in rows],dtype=float)
    sources=pd.to_datetime([r.get('observed_date') or r.get('sale_date') for r in rows],utc=True,errors='coerce',format='mixed')
    target_timestamp=pd.Timestamp(target)
    to_age=target_timestamp.year+(target_timestamp.dayofyear-1)/365.25-years
    from_age=np.maximum(0,sources.year+(sources.dayofyear-1)/365.25-years)
    valid=np.isfinite(prices)&(prices>0)&np.isfinite(miles)&sources.notna()&(to_age>=0)&(to_age<=30)&(from_age<=30)
    out=np.full(len(rows),np.nan)
    ids=np.flatnonzero(valid)
    if not len(ids):return out
    frame=pd.DataFrame([rows[i] for i in ids]);frame['age']=from_age[ids];frame['odometer']=miles[ids]
    factors=transport.curves[side].ratio(frame,from_age[ids],to_age[ids],from_miles=miles[ids],to_miles=miles[ids] if target_mileage is None else float(target_mileage))
    current,_=transport.clock.value(target,as_of=today())
    months=sources.strftime('%Y-%m');lookup={month:transport.clock.reference(month+'-01')[0] for month in set(months[ids])}
    reference=np.array([lookup[months[i]] for i in ids])
    values=prices[ids]*factors*current/reference
    accepted=np.isfinite(values)&(values>0)&(values<=1e7)
    out[ids[accepted]]=values[accepted]
    return out
