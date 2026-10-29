import base64, io, json
from urllib.parse import urlparse,parse_qs
from datetime import date
import pytest
from PIL import Image,ImageDraw
from pricing.deal_tools import *

CAR=dict(year=2015,manufacturer='toyota',model='Corolla SE',odometer=95000,state='ca')

def test_roundtrip_shares_facts_only_and_three_cars():
    private={**CAR,'description':'secret','ip':'1.2.3.4','visitor_id':'secret','token':'secret','price':12000}
    url=share_link([private]*3)
    token=parse_qs(urlparse(url).query)['garage'][0]
    assert read_share(token)==[CAR]*3
    raw=base64.urlsafe_b64decode(token+'='*(-len(token)%4)).decode()
    assert 'secret' not in raw and 'price' not in raw
    for bad in ['!', 'a'*6001,base64.urlsafe_b64encode(b'{"v":9,"cars":[]}').decode()]:
        with pytest.raises(ValueError):read_share(bad)
    assert safe_fields({**CAR,'year':float('nan'),'odometer':float('inf'),'model':'person@example.com'})==dict(manufacturer='toyota',state='ca')

def test_social_urls_and_captions_are_encoded_and_historical():
    url=share_link([CAR]);text=caption(CAR,12000)
    links=social_links(url,text)
    assert parse_qs(urlparse(links['WhatsApp']).query)['text']==[text+'\n'+url]
    assert parse_qs(urlparse(links['LinkedIn']).query)['url']==[url]
    assert 'May 2021' in text and 'current sale guarantee' in text
    with pytest.raises(ValueError):social_links('https://evil.example',text)

def test_parser_redacts_and_converts_without_inventing_fields():
    result=parse_advert('2015 Toyota Corolla SE | 160,934 km | $13,500\nCall 202-555-1234 or seller@example.com')
    assert result['fields']['odometer']==100000
    assert result['fields']['manufacturer']=='toyota'
    assert result['seller_price']==13500
    assert 'seller@example.com' not in result['text'] and '202-555-1234' not in result['text']
    assert parse_advert('nice wheels')['fields']=={}

def test_cards_are_real_png_in_both_formats(tmp_path):
    for portrait,size in [(False,(1200,630)),(True,(1080,1350))]:
        png=card_png(CAR,12996.17,share_link([CAR]),portrait)
        assert Image.open(io.BytesIO(png)).size==size
        (tmp_path/('portrait.png' if portrait else 'landscape.png')).write_bytes(png)
    with pytest.raises(ValueError):card_png(CAR,float('nan'),APP_URL)

def test_real_ocr_and_historical_reference():
    image=Image.new('RGB',(1000,250),'white');d=ImageDraw.Draw(image)
    d.text((30,30),'2015 Toyota Corolla\n95,000 miles\nPrice $13,500',font=font(38),fill='black')
    out=io.BytesIO();image.save(out,format='PNG')
    result=parse_advert(ocr_image(out.getvalue()))
    assert result['fields']['year']==2015 and result['seller_price']==13500
    data=reference_cases();assert len(data['cases'])==4000
    assert 'CC0' in data['source_license']
    assert daily_case(data['cases'],date(2026,10,29))==daily_case(data['cases'],date(2026,10,29))
    assert all(not c['settlement_verified'] and not {'vin','seller'}&c.keys() for c in data['cases'])
    matches=comparables(CAR,data['cases']);assert matches
    assert all(c['manufacturer']=='toyota' and c['year_difference']<=3 for c in matches)
