"""Training-fitted category vocabularies; unknown values become missing."""
import numpy as np
import pandas as pd

NUMERIC = ['year', 'odometer']
CATEGORICAL = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status',
               'transmission', 'drive', 'type', 'paint_color', 'state', 'source']


def normal(series):
    return series.astype('string').str.lower().str.strip().replace('', pd.NA)


def fit_schema(frame):
    return {c: sorted(normal(frame[c]).dropna().unique().tolist()) for c in CATEGORICAL}


def features(frame, schema, text_prices=None):
    x = pd.DataFrame(index=frame.index)
    from carbids.census import FEATURES
    optional=[c for c in FEATURES if c in frame.columns]
    for c in NUMERIC+optional:
        x[c] = pd.to_numeric(frame[c], errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in CATEGORICAL:
        x[c] = pd.Categorical(normal(frame[c]), categories=schema[c])
    if text_prices is not None:
        prices = np.asarray(text_prices, dtype=float)
        if len(prices) != len(frame) or not np.isfinite(prices).all() or (prices < 0).any():
            raise ValueError('One finite nonnegative out-of-fold/inference text price required per row')
        x['predicted_text_price'] = prices
    return x
