from pathlib import Path
import numpy as np
import pytest
from pricing.market_review import parse_inventory,duplicate_pairs,export_csv,review_estimate,batch_review
ROOT=Path(__file__).resolve().parents[1]

def csv_rows(extra=''):
    return parse_inventory(('manufacturer,model,year,odometer,price,description,vin,id\nToyota,Corolla,2015,95000,8000,Full service history,1HGCM82633A004352,a\nToyota,Corolla,2015,97000,9000,New tires,1HGCM82633A004352,b\n'+extra).encode())

def test_duplicate_vin_conflicts_and_exports():
    rows=csv_rows();pairs=duplicate_pairs(rows)
    assert len(pairs)==1 and 'odometer' in pairs[0]['different_fields'] and 'price' in pairs[0]['different_fields']
    assert '1HGCM82633A004352' not in export_csv(pairs).decode()
    assert "'=cmd" in export_csv([{'id':'=cmd','x':12}]).decode()
    assert parse_inventory(b'manufacturer,model,year,odometer\nToyota,Camry,nan,-1')[0]['errors']
    with pytest.raises(ValueError):parse_inventory(b'bad,columns\n1,2')
    with pytest.raises(ValueError):parse_inventory(b'manufacturer,model,year,odometer\n'+b'Ford,Focus,2015,2000\n'*101)

def test_template_text_alone_not_duplicate():
    rows=csv_rows();rows[1]['vin_hash']='';rows[0]['vin_hash']=''
    rows[0]['description']=rows[1]['description']='This is an excellent well maintained vehicle with a full service history and service records from our local dealership.'
    assert duplicate_pairs(rows)
    rows[1]['fields']['model']='Camry'
    assert not duplicate_pairs(rows)

def test_review_labels_and_batch_failure_isolation():
    good={'price':10000,'interval_low':8000,'interval_high':13000,'model_support':30,'unknown_categories':[]}
    assert review_estimate(good,5000,[])['priority']=='Price worth investigating'
    assert review_estimate(dict(good,model_support=0),5000,[])['priority']=='Check evidence first'
    rows=csv_rows()
    calls=[]
    def evaluate(fields,date,allow):
        calls.append(fields)
        if len(calls)==1:raise ValueError('Bad car')
        return {'ask':good,'bid':{'price':6000}}
    result=batch_review(rows,evaluate,'2026-11-01',True)
    assert len(result)==2 and len(calls)==2
    assert any(r['priority']=='Unavailable' for r in result)
    assert any(r.get('auction_estimate_usd')==6000 for r in result)

def test_statistics_experiment_and_vehicle_leakage():
    from pricing.listing_statistics import experiment
    rng=np.random.default_rng(42);rows=[]
    for i in range(600):
        year=2000+i%16;miles=20000+i*200
        rows.append(dict(row_id=str(i),vehicle_group=str(i),phase='train' if i<400 else 'validation' if i<500 else 'test',
                         manufacturer='ford',model='focus',state='ca' if i%2 else 'ny',type='sedan',year=year,odometer=miles,
                         price=float(np.exp(10-.04*(2015-year)-miles/300000+.05*(i%2)+rng.normal(0,.1))),date='2015-01-01',text='Full service history' if i%3 else 'Needs work'))
    r=experiment(rows,'ask',bootstrap=3)
    assert len(r['regions'])==2
    assert r['test_mae_usd']>0 and 0<=r['historical_interval_coverage']<=1
    assert len(r['description_experiment']['flags'])==6
    assert r['outlier_review']['flagged_rows']>=100
    rows[-1]['vehicle_group']='1'
    with pytest.raises(ValueError,match='leakage'):experiment(rows,'ask',bootstrap=1)

def test_presets_fill_editable_form_and_all_tools_render():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_string("from pathlib import Path\nfrom pricing.listing_inspector import render\nrender(Path('.'))").run(timeout=30)
    assert not at.exception
    presets=next(x for x in at.selectbox if '1,000' in x.label)
    assert len(presets.options)==1000
    presets.select('Ford f-150').run(timeout=30)
    assert not at.exception
    assert next(x for x in at.text_input if x.label=='Make').value=='Ford'
    assert next(x for x in at.text_input if x.label=='Model / trim').value=='f-150'
    assert len(at.tabs)==9
