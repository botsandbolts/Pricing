"""Optional inference API. Start: uvicorn carbids.api:app --host 0.0.0.0 --port 8000."""
import os
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from carbids.inference import PricePipeline

app=FastAPI(title='CarBids sale-price API',version='1.0',description='Public provider-reported auction sale prices, not verified settlements. USD; excludes unspecified fees.')
class Valuation(BaseModel):
    year:int=Field(ge=1886,le=2027)
    manufacturer:str=Field(max_length=100)
    model:str=Field(max_length=160)
    odometer:float=Field(ge=0,le=2000000)
    condition:str='unknown'
    source:str='wholesale_2015'
    transmission:str='unknown'
    type:str='unknown'
    state:str='unknown'
    fuel:str='unknown'
    drive:str='unknown'
    paint_color:str='unknown'
    cylinders:str='unknown'
    title_status:str='unknown'
    text:str=Field(default='',max_length=8000)
    as_of:str|None=None
    explain:bool=True
@lru_cache
def get_model():return PricePipeline(Path(os.environ.get('CARBIDS_MODEL_DIR','models/tabular')))
@app.get('/health')
def health():
    try:m=get_model();return {'status':'ready','target_kind':m.manifest['target_kind'],'currency':'USD','settlement_verified':False,'sources':m.manifest['sources']}
    except (FileNotFoundError,ValueError):return {'status':'model_unavailable'}
@app.post('/predict')
def predict(request:Valuation):
    try:
        model=get_model()
        if request.source not in model.manifest['sources']:raise HTTPException(422,'Source is outside the fitted model domain')
        if model.text_model is not None and len(request.text.strip())<80:raise HTTPException(422,'A meaningful pre-sale description is required')
        row=request.model_dump();text=row.pop('text');explain=row.pop('explain')
        return {'target_kind':'platform_reported_sale_price','currency':'USD','settlement_verified':False,**model.predict(text,row,explain=explain)}
    except HTTPException:raise
    except (FileNotFoundError,ValueError) as e:raise HTTPException(503,str(e)) from e
