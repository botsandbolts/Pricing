from datetime import date
import numpy as np
import pytest
from pricing.reference_prices import adjust_reference
class Curve:
    levels={'model':['ford::focus']}
    def ratio(self,frame,from_age,to_age,from_miles,to_miles):
        return np.atleast_1d(np.exp(-.05*(to_age-from_age)-.000001*(to_miles-from_miles)))
class Clock:
    def value(self,target,as_of):return 150.,{'last_known_month':'2026-09','horizon':2,'method':'damped_holt','kind':'forecast_monthly_index'}
    def reference(self,source):return 100.,'2021-05'
class Transport:
    curves={'ask':Curve(),'bid':Curve()};clock=Clock()
    def validate(self,fields,target):
        if target.year-fields['year']>30:raise ValueError('Unsupported age')
def test_actual_price_source_date_and_mileage_transport():
    car={'year':2015,'manufacturer':'ford','model':'focus','odometer':50000,'price':10000,'observed_date':'2021-05-01'}
    r=adjust_reference(car,'ask',date(2026,11,1),Transport(),95000)
    assert r['original_price_usd']==10000 and r['original_date']=='2021-05-01'
    assert r['market_factor']==1.5 and r['comparison_mileage']==95000
    assert r['price_usd']==pytest.approx(10000*r['age_mileage_factor']*1.5)
    assert car['odometer']==50000
    with pytest.raises(ValueError):adjust_reference(dict(car,price=None),'ask',date(2026,11,1),Transport())
    with pytest.raises(ValueError):adjust_reference(dict(car,year=1980),'ask',date(2026,11,1),Transport())


def test_vectorized_reference_prices_match_individual_adjustments():
    from pricing.reference_prices import feature_prices
    cars=[{'year':2015,'manufacturer':'ford','model':'focus','odometer':50000,'price':10000,'observed_date':'2021-05-01'},
          {'year':2014,'manufacturer':'toyota','model':'corolla','odometer':70000,'price':12000,'observed_date':'2021-05-01'},
          {'year':2014,'manufacturer':'toyota','model':'corolla','odometer':70000,'price':None,'observed_date':'2021-05-01'}]
    values=feature_prices(cars,'ask',date(2026,11,1),Transport(),95000)
    assert values[:2]==pytest.approx([adjust_reference(c,'ask',date(2026,11,1),Transport(),95000)['price_usd'] for c in cars[:2]])
    assert np.isnan(values[2])
