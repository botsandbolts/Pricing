"""Fresh November-2026 European neural research cohort, leakage-safe.

Input is the separately produced full European Parquet adjusted-data artifact.
Output is original observed ASK prices with confirmed latest verification
dates, never synthetic HICP/Holt targets, never hypothetical future mileage.

last_verified_at is the observational date used to construct age and
date-aware tabular inputs. Only verified <=2-month listings are included.
Source price_eur and observed mileage_km remain immutable model targets/inputs.
ISO2 country names are normalized to avoid duplicate country embeddings.
The original 80/10/10 fingerprint holdout and its sealed test are recreated.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.europe_price_ray import INPUT_COLUMNS, prepare as prepare_holdout

ALIASES={
    "DE":("de","deu","ger","germany","deutschland","allemagne","alemania"),
    "FR":("fr","fra","france","frankreich"),
    "NL":("nl","nld","netherlands","nederland","holland","the netherlands"),
    "BE":("be","bel","belgium","belgien","belgië","belgique"),
    "AT":("at","aut","austria","österreich","oesterreich"),
    "ES":("es","esp","spain","españa","spanien"),
    "IT":("it","ita","italy","italia","italien"),
    "PL":("pl","pol","poland","polska","polen"),
}
COUNTRY_MAP={alias:country for country,aliases in ALIASES.items() for alias in aliases}
ALLOWED=set(ALIASES)
TARGET="2026-11"
MAX_AGE_MONTHS=2


def normalize_country(series):
    raw=series.astype("string").str.strip().str.casefold()
    return raw.map(COUNTRY_MAP).astype("string")


def make_recent_cohort(frame,target=TARGET):
    required={"price_eur","seller_country","reg_year","last_verified_at",
              "mileage_km","eligible_as_current_price_training_label"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing source/audit fields: "+str(sorted(required-set(frame))))
    frame=frame.copy()
    original=len(frame)
    countries=normalize_country(frame["seller_country"])
    country_before=Counter(frame["seller_country"].astype("string").fillna("").str.lower())
    dates=pd.to_datetime(frame["last_verified_at"],utc=True,errors="coerce",format="mixed")
    observed_month=dates.dt.year*12+dates.dt.month-1
    target_year,target_m=map(int,target.split("-"))
    target_month=target_year*12+target_m-1
    gap=target_month-observed_month
    age=dates.dt.year-pd.to_numeric(frame["reg_year"],errors="coerce")
    price=pd.to_numeric(frame["price_eur"],errors="coerce")
    km=pd.to_numeric(frame["mileage_km"],errors="coerce")
    fresh=frame["eligible_as_current_price_training_label"].fillna(False).astype(bool)
    verified=(gap.between(0,MAX_AGE_MONTHS)
              & dates.notna()
              & age.between(0,90)
              & km.between(0,2_000_000)
              & price.between(100,1_000_000)
              & fresh
              & countries.isin(ALLOWED))
    selected=frame.loc[verified].copy()
    selected["seller_country"]=countries.loc[verified].str.lower().to_numpy(dtype=object)
    # Old first_seen date is not the best observed price timestamp.
    # Encoding reads first_seen_at; supply the most recent actual verification.
    selected["first_seen_at"]=dates.loc[verified].dt.strftime("%Y-%m-%d").to_numpy()
    # New price, HICP factor, hypothetical mileage and price-adjustment status
    # are strictly excluded from features/labels to avoid calculated-label leakage.
    selected=selected[[key for key in INPUT_COLUMNS if key in selected.columns]]
    counts=Counter(selected["seller_country"])
    if len(selected)<1000:
        raise ValueError("Fewer than 1000 eligible fresh observed listing labels.")
    return selected,{
        "source_rows":int(original),"included_rows":int(len(selected)),
        "excluded_rows":int(original-len(selected)),
        "label":"price_eur (original observed ASK, NOT forecast-adjusted)",
        "time_feature_from":"last_verified_at",
        "source_country_labels":dict(country_before.most_common()),
        "standard_country_counts":dict(counts.most_common()),
        "min_source_date":str(dates.loc[verified].min().date()),
        "max_source_date":str(dates.loc[verified].max().date()),
        "no_derived_hicp_or_mileage_labels":True,
        "freshness_cutoff_months":MAX_AGE_MONTHS,
        "iso2_alias_map":ALIASES,
    }


def run(source,output,*,target=TARGET):
    source=Path(source)
    if not source.is_file():
        raise FileNotFoundError(source)
    reader=pq.ParquetFile(source)
    mandatory={"eligible_as_current_price_training_label",
               "current_depreciated_price_eur","last_verified_at"}
    if not mandatory.issubset(reader.schema_arrow.names):
        raise ValueError("Input must be the verified derived European Parquet artifact")
    columns=[col for col in INPUT_COLUMNS if col in reader.schema_arrow.names]
    columns+=["last_verified_at","eligible_as_current_price_training_label"]
    frame=pq.read_table(source,columns=list(dict.fromkeys(columns))).to_pandas()
    chosen,meta=make_recent_cohort(frame,target=target)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cohort=output/"fresh_observed_ask_input.parquet"
    pq.write_table(pa.Table.from_pandas(chosen,preserve_index=False),
                   cohort,compression="zstd",compression_level=3)
    result=prepare_holdout(cohort,output)
    manifest_path=output/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    if manifest["eligible_rows"]!=len(chosen):
        raise AssertionError("Fresh observed dataset lost rows during preparation")
    if manifest["counts"]["test"]<100:
        raise AssertionError("Test split is not adequately populated")
    manifest.update({
        "freshness_and_country_audit":meta,
        "historical_model_labels_excluded":True,
        "synthetic_hicp_prices_used_as_training_labels":False,
        "hypothetical_mileage_used_as_training_input":False,
        "training_field_policy":"Use original ASK, observed mileage, normalized "
          "ISO2 country, and age at last_verified_at. No HICP-derived targets "
          "or hypothetical November odometers; drop stale verified listings.",
        "source_derived_date":"2026-11",
    })
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    (output/"freshness_audit.json").write_text(json.dumps(meta,indent=2)+"\n")
    cohort.unlink(missing_ok=True)
    print(json.dumps({"source_rows":meta["source_rows"],
                      "fresh_rows":len(chosen),
                      "countries":meta["standard_country_counts"],
                      "counts":manifest["counts"],
                      "target":manifest["target"]},sort_keys=True),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run(args.source,args.output)


if __name__=="__main__":
    main()
