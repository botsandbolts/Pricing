"""Shared cleaning and dollar-price metrics. No sparse text features."""
import re
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
MONEY=re.compile(r'(?i)(?:[$€£]\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:usd|dollars?|euros?|per month|/mo|monthly)\b|\b(?:price|asking|payment|down payment)\s*[:=]?\s*\$?\d[\d,.]*)')
def clean(text,lower=False):
 t=re.sub(r'<[^>]+>',' ',str(text));t=re.sub(r'https?://\S+|\b\S+@\S+\b',' ',t);t=re.sub(r'\s+',' ',MONEY.sub(' price_removed ',t)).strip();return t.lower() if lower else t
def metrics(y,p):
 p=np.maximum(0,p);return {'mae_usd':float(mean_absolute_error(y,p)),'rmse_usd':float(np.sqrt(mean_squared_error(y,p))),'median_ape_pct':float(np.median(abs(y-p)/y)*100),'rmsle':float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2))),'r2':float(r2_score(y,p))}

