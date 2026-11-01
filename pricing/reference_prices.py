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
