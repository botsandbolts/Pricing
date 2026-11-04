from pricing.advert_text import scrub
def test_money_removed_vehicle_numbers_retained():
    text='2015 Toyota 95000 miles. Asking price: $12,500. 2.0 engine'
    result=scrub(text)
    assert '12,500' not in result
    assert '2015' in result and '95000' in result and '2.0 engine' in result
def test_plain_amount_and_usd():
    assert '12000' not in scrub('price 12000')
    assert '14000' not in scrub('14000 USD')
def test_no_money_unchanged():
    assert scrub('2019 model 4 cylinders')=='2019 model 4 cylinders'
