"""Auditable *hypothetical* age/mileage continuation for stale EU adverts.

Age from registration *year* is approximate. Future mileage is UNOBSERVED.
Estimate a conditional annual-km pace from similarly aged, fuel-matched
observations; shrink sparse make/model cohorts to make/fuel, country/fuel,
and a transparent global assumption. All amounts are scenario inputs only.

No mechanically "depreciated" current price is fabricated from unknown actual
vehicle condition, mileage or sale status; older ASK labels cannot be passed
off as current model-training targets.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd
import pyarrow as pa

TARGET = "2026-11"
MIN_GROUP_ROWS = 15
GLOBAL_KM_PER_YEAR = 12000.0
MAX_MILEAGE_KM = 2_000_000.0
MAX_ANNUAL_KM = 50_000.0

AGE_COL = "hypothetical_vehicle_age_nov2026_years"
KM_COL = "hypothetical_nov2026_mileage_km"
KM_LOW = "hypothetical_nov2026_mileage_low_km"
KM_HIGH = "hypothetical_nov2026_mileage_high_km"
KM_STATUS = "hypothetical_mileage_projection_method"

NEW_FIELDS = (
    pa.field(AGE_COL, pa.float64()),
    pa.field(KM_COL, pa.float64()),
    pa.field(KM_LOW, pa.float64()),
    pa.field(KM_HIGH, pa.float64()),
    pa.field(KM_STATUS, pa.string()),
)


def _column(batch, name, default=None):
    if name in batch.schema.names:
        return batch.column(name).to_pandas()
    return pd.Series([default]*batch.num_rows)


def _normalize(value):
    return str(value).lower().strip() if pd.notna(value) else "unknown"


def _age_bucket(age):
    return int(max(0, min(5, int(age)//3)))


def _date_age_and_pace(batch):
    # Registration year is not identical to precise registration date.
    date = pd.to_datetime(_column(batch, "last_verified_at"),
                          utc=True, errors="coerce", format="mixed")
    registration = pd.to_numeric(_column(batch, "reg_year"),errors="coerce")
    km = pd.to_numeric(_column(batch, "mileage_km"),errors="coerce")
    observed_age = (date.dt.year + (date.dt.month-.5)/12 - registration).to_numpy(float)
    distance = km.to_numpy(float)
    return date, observed_age, distance


class UsagePriors:
    """Streaming shrinkage: model/fuel/age -> make/fuel -> country/fuel."""

    def __init__(self):
        self.sufficient = defaultdict(lambda: [0,0.0])
        self.observed_records = 0
        self.cohorts = 0

    def observe(self, batch):
        date, age, km = _date_age_and_pace(batch)
        pace = km/np.maximum(age, 1.0)
        valid=(np.isfinite(age) & (age>=2) & (age<=40)
               & np.isfinite(km) & (km>=1000) & (km<=800000)
               & (pace>=1500) & (pace<=50000)
               & date.notna().to_numpy(bool)
               & (date.dt.year>=2024).to_numpy(bool))
        if not valid.any():
            return
        country=_column(batch,"seller_country").astype("string").str.upper()
        make=_column(batch,"make").astype("string").str.lower()
        model=_column(batch,"model").astype("string").str.lower()
        fuel=_column(batch,"fuel").astype("string").str.lower()
        frame=pd.DataFrame({
            "country":country.to_numpy(dtype=object)[valid],
            "make":make.to_numpy(dtype=object)[valid],
            "model":model.to_numpy(dtype=object)[valid],
            "fuel":fuel.to_numpy(dtype=object)[valid],
            "bucket":np.minimum(np.floor(age[valid]/3),5).astype(int),
            "log_annual_km":np.log(pace[valid]),
        }).fillna("unknown")
        self.observed_records+=len(frame)
        configurations=[
            (0,["country","make","model","fuel","bucket"]),
            (1,["country","make","fuel"]),
            (2,["country","fuel"]),
        ]
        for level, keys in configurations:
            grouped=frame.groupby(keys,dropna=False,observed=True)["log_annual_km"].agg(
                ["size","sum"])
            for group,row in grouped.iterrows():
                if not isinstance(group,tuple):
                    group=(group,)
                key=(level,)+tuple(str(v) for v in group)
                slot=self.sufficient[key]
                slot[0]+=int(row["size"])
                slot[1]+=float(row["sum"])
        global_slot=self.sufficient[(3,)]
        global_slot[0]+=len(frame)
        global_slot[1]+=float(frame["log_annual_km"].sum())

    def _lookup(self,record):
        c,make,model,fuel,bucket=record
        levels=[
            ((0,c,make,model,fuel,str(bucket)),100),
            ((1,c,make,fuel),130),
            ((2,c,fuel),200),
            ((3,),400),
        ]
        prior=math.log(GLOBAL_KM_PER_YEAR)
        selected_method="global_12000km_assumption"
        for key,prior_count in reversed(levels):
            count, sum_log=self.sufficient.get(key,(0,0))
            if count:
                weight=count/(count+prior_count)
                prior=weight*sum_log/count+(1-weight)*prior
                selected_method={3:"global_observed_usage",
                                 2:"country_fuel_observed_usage",
                                 1:"make_fuel_observed_usage",
                                 0:"make_model_fuel_age_observed_usage"}[key[0]]
        return float(np.clip(math.exp(prior),2500,45000)),selected_method

    def project(self,batch,*,target=TARGET):
        target_year, target_month=map(int,target.split("-"))
        target_coordinate=target_year*12+target_month-1
        last_verified, old_age, observed_km=_date_age_and_pace(batch)
        registration=pd.to_numeric(_column(batch,"reg_year"),errors="coerce").to_numpy(float)
        verified_months=(last_verified.dt.year*12+last_verified.dt.month-1).to_numpy(float)
        months=(target_coordinate-verified_months)
        age=(target_year+(target_month-.5)/12.0-registration)
        valid=(np.isfinite(age) & (age>=0) & (age<=90)
               & np.isfinite(observed_km) & (observed_km>=0)
               & (observed_km<=MAX_MILEAGE_KM) & np.isfinite(months)
               & (months>=0))
        low=np.full(batch.num_rows,np.nan)
        mid=low.copy()
        high=low.copy()
        status=np.full(batch.num_rows,"missing_age_mileage_or_date",dtype=object)
        eligible=np.where(valid)[0]
        if len(eligible):
            country=_column(batch,"seller_country").fillna("unknown").astype(str).str.upper()
            make=_column(batch,"make").fillna("unknown").astype(str).str.lower()
            model=_column(batch,"model").fillna("unknown").astype(str).str.lower()
            fuel=_column(batch,"fuel").fillna("unknown").astype(str).str.lower()
            records=[(country.iloc[i],make.iloc[i],model.iloc[i],fuel.iloc[i],
                      _age_bucket(old_age[i] if np.isfinite(old_age[i]) else 0))
                     for i in eligible]
            cache={}
            for i,record in zip(eligible,records):
                if months[i]==0:
                    low[i]=mid[i]=high[i]=observed_km[i]
                    status[i]="observed_mileage_not_projected"
                    continue
                if record not in cache:
                    cache[record]=self._lookup(record)
                rate,source=cache[record]
                elapsed_years=months[i]/12.0
                low[i]=min(MAX_MILEAGE_KM,observed_km[i]+rate*.6*elapsed_years)
                mid[i]=min(MAX_MILEAGE_KM,observed_km[i]+rate*elapsed_years)
                high[i]=min(MAX_MILEAGE_KM,observed_km[i]+rate*1.6*elapsed_years)
                status[i]=source+"_hypothetical"
        out=batch.append_column(
            AGE_COL,pa.array(age,type=pa.float64(),mask=~valid))
        for key,values in ((KM_COL,mid),(KM_LOW,low),(KM_HIGH,high)):
            out=out.append_column(
                key,pa.array(np.round(values,1),type=pa.float64(),
                             mask=~np.isfinite(values)))
        out=out.append_column(KM_STATUS,pa.array(status,type=pa.string()))
        return out
