"""Experimental make/model month-to-month *relative* ASK price movement.

Data: genuine listing timestamps. Produces research-only small-group Huber
hedonic residual contrasts, NEVER a validated make/model historical index.
No false past extrapolation, no model-specific Holt from two snapshots.

Usage:
  python scripts/build_europe_model_pulse.py \
      --output models/europe-ask/model_pulse.json

Stage 1 needs robust multi-month repeated listings; with only Sep/Oct 2026
Germany we publish a clearly labelled one-step descriptive pilot.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_europe_hedonic_fisher import (
    DEFAULT_SOURCE, clean, design, month_coordinate,
)

# These thresholds prevent a "trend" based on a dozen exotic vehicles.
MIN_EARLY = 60
MIN_LATE = 120
MIN_GROUPS = 3
MAX_TRAIN_EARLY = 27000
MAX_TRAIN_LATE = 70000
MAX_NUM_GROUPS = 80
RELEASE_MONTHS = 6  # Requires at least six actual consecutive monthly periods.


def _trimmed_mean(x, trim=.1):
    arr = np.sort(np.asarray(x, dtype=float))
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 5:
        return float("nan")
    k = int(n * trim)
    return float(np.mean(arr[k:n-k]))


def _robust_se(x):
    """Conservative residual mean uncertainty, log-price units."""
    v = np.asarray(x, dtype=float)
    if len(v) < 5:
        return float("inf")
    center = np.median(v)
    robust_sigma = 1.4826 * np.median(np.abs(v-center))
    return max(0.04, float(robust_sigma))/math.sqrt(len(v))


def _stratified_sample(df, count, *, seed):
    if len(df) <= count:
        return df
    return df.sample(n=count,random_state=seed)


def fit_relative_pulse(data, *, early="2026-09", late="2026-10",
                       min_early=MIN_EARLY, min_late=MIN_LATE):
    """Returns descriptive Huber-adjusted make/model contrasts vs DE listings.

    Fits one pooled, *time-blind* sparse SGD Huber log-ASK model. It contains
    registration age, mileage, power, source portal, fuel, gearbox, body and
    make/model fixed effects. Residual difference is inspected by cohort; we
    do not give this two-month contrast a predictive time-series interpretation.
    """
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import StandardScaler

    if month_coordinate(late) - month_coordinate(early) != 1:
        raise ValueError("Research pairs must be genuine adjacent months.")
    source = data[data.seller_country.eq("DE") & data.month.isin((early, late))].copy()
    period_counts=source.groupby(["month","vehicle_group"]).size().unstack(0, fill_value=0)
    if early not in period_counts or late not in period_counts:
        return None
    eligible=period_counts[(period_counts[early]>=min_early)
                           & (period_counts[late]>=min_late)]
    if len(eligible)<MIN_GROUPS:
        return None
    early_data=source[source.month.eq(early)]
    late_data=source[source.month.eq(late)]
    # Keep both periods in the same common-support, comparable vehicle basket.
    common_sites=set(early_data.source_site)&set(late_data.source_site)
    common_groups=set(eligible.index)
    early_data=early_data[early_data.source_site.isin(common_sites)
                          &early_data.vehicle_group.isin(common_groups)]
    late_data=late_data[late_data.source_site.isin(common_sites)
                        &late_data.vehicle_group.isin(common_groups)]
    # Do not allow one overwhelmingly larger snapshot to dictate the fit.
    train=pd.concat([
        _stratified_sample(early_data,MAX_TRAIN_EARLY,seed=1626),
        _stratified_sample(late_data,MAX_TRAIN_LATE,seed=1627),
    ],ignore_index=True)
    if len(train)<1500:
        return None
    vectorizer=DictVectorizer(sparse=True)
    x=vectorizer.fit_transform(design(train))
    x=StandardScaler(with_mean=False).fit_transform(x)
    log_y=np.log(train.price_eur.to_numpy(float))
    center=float(np.median(log_y))
    model=SGDRegressor(
        loss="huber",epsilon=0.20,penalty="l2",alpha=0.0002,
        learning_rate="adaptive",eta0=0.008,
        max_iter=350,tol=1e-4,n_iter_no_change=12,
        average=True,random_state=20261110,
    )
    model.fit(x,log_y-center)
    if model.n_iter_>=350:
        return None
    predicted=model.predict(x)+center
    if not np.all(np.isfinite(predicted)):
        return None
    if np.median(np.abs(log_y-predicted))>0.40:
        return None
    train["residual"]=log_y-predicted

    # Compute a common reference on a balanced basket of eligible models.
    # Thus numbers are relative to German matched-model listings in our sample,
    # NOT already measured market growth or official Eurostat price movement.
    country_early=_trimmed_mean(train.loc[train.month.eq(early),"residual"])
    country_late=_trimmed_mean(train.loc[train.month.eq(late),"residual"])
    reference_delta=country_late-country_early
    records=[]
    for group, part in train.groupby("vehicle_group"):
        before=part.loc[part.month.eq(early),"residual"].to_numpy(float)
        after=part.loc[part.month.eq(late),"residual"].to_numpy(float)
        n_early=len(before)
        n_late=len(after)
        if min(n_early/max(min_early,1),n_late/max(min_late,1))<1:
            continue
        raw=_trimmed_mean(after)-_trimmed_mean(before)-reference_delta
        if not math.isfinite(raw):
            continue
        err=math.hypot(_robust_se(after),_robust_se(before))
        # Empirical-Bayes shrinkage toward the national/common-basket trend.
        # Rare/noisy model slices automatically collapse towards zero.
        prior_sd=0.06
        shrink=prior_sd**2/(prior_sd**2+err**2)
        delta=float(raw*shrink)
        post_se=math.sqrt((err**2*prior_sd**2)/(err**2+prior_sd**2))
        if not all(math.isfinite(z) for z in (delta,post_se)):
            continue
        # Clip unreasonable descriptive effects; do not amplify large outliers.
        if abs(delta)>0.15:
            continue
        make,model_name=group.split(" · ",1)
        records.append({
            "make":make,"model":model_name,
            "n_early":int(n_early),"n_late":int(n_late),
            "relative_log_change":round(delta,6),
            "relative_pct":round(100*(math.exp(delta)-1),2),
            "interval_low_pct":round(100*(math.exp(delta-1.96*post_se)-1),2),
            "interval_high_pct":round(100*(math.exp(delta+1.96*post_se)-1),2),
            "shrink_weight":round(shrink,3),
            "raw_relative_log_change":round(float(raw),6),
        })
    records=sorted(records,key=lambda r:(-min(r["n_early"],r["n_late"]),r["make"],r["model"]))
    return {"country":"DE","early_month":early,"late_month":late,
            "reference":"Other matched German make/model listing groups (not Eurostat)",
            "reference_delta_log":round(float(reference_delta),6),
            "n_training":int(len(train)), "models":records[:MAX_NUM_GROUPS],
            "model_count":len(records),
            "converged_epochs":int(model.n_iter_),
            "description":"Exploratory relative month shift among observed advertisements, "
                          "adjusted for age, mileage, power, fuel, gearbox, portal "
                          "and body; empirical-Bayes shrinkage. NOT a time-series "
                          "index, not validated for time adjustment."}


def build(source=DEFAULT_SOURCE):
    d=clean(source)
    periods=(d[d.seller_country.eq("DE")].groupby("month").size()
             .sort_index())
    report={
        "schema_version":1,"status":"research_only",
        "source":"KTHESAT/european-used-car-market-2m-listings",
        "method":"Pooled time-blind sparse SGDRegressor Huber on log(EUR ASK); "
                 "relative make/model residual contrasts and hierarchical shrinkage",
        "minimum_observed_months_to_enable_forecasting":RELEASE_MONTHS,
        "country_coverage":{"DE":{str(m):int(n) for m,n in periods.items()}},
        "research_pairs":[],"message":"No validated make/model monthly series is available; "
                                   "Eurostat national HICP remains the production date clock.",
    }
    keys=sorted(periods.index)
    if len(keys)>=2:
        for first,second in zip(keys,keys[1:]):
            if month_coordinate(second)-month_coordinate(first)!=1:
                continue
            pilot=fit_relative_pulse(d,early=first,late=second)
            if pilot and pilot["models"]:
                report["research_pairs"].append(pilot)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default=DEFAULT_SOURCE)
    parser.add_argument("--output",default="models/europe-ask/model_pulse.json")
    opts=parser.parse_args()
    result=build(opts.source)
    path=Path(opts.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],
                      "models":[{"country":g["country"],"n":len(g["models"])}
                                for g in result["research_pairs"]],
                      "output":str(path)},sort_keys=True))


if __name__=="__main__":
    main()
