"""Offline European ASK hedonic Fisher builder.

Example:
  pip install pandas numpy scikit-learn pyarrow
  python scripts/build_europe_hedonic_fisher.py --source path/to/listings.parquet

If genuine multi-month coverage is insufficient, emits an *unavailable* manifest.
Never promotes a cross-sectional dataset to a time series or simulates monthly prices.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = ("https://huggingface.co/datasets/KTHESAT/"
                  "european-used-car-market-2m-listings/resolve/main/"
                  "data/train-00000-of-00001.parquet")
SUPPORTED_COUNTRIES = frozenset({"DE", "FR", "NL", "BE", "AT", "CH", "ES", "IT", "PL"})
COLUMNS = ["id", "source_site", "seller_country", "make", "model", "price_eur",
           "mileage_km", "reg_year", "power_hp", "first_seen_at", "fuel",
           "transmission", "body_type"]


def clean(source):
    d = pd.read_parquet(source, columns=COLUMNS) if isinstance(source, (str, Path)) else source.copy()
    absent = set(COLUMNS) - set(d)
    if absent:
        raise ValueError("Required fields missing: " + ", ".join(sorted(absent)))
    d = d[COLUMNS].copy()
    first = pd.to_datetime(d["first_seen_at"], utc=True, errors="coerce")
    d["month"] = first.dt.strftime("%Y-%m")
    d["year"] = first.dt.year
    for field in ("price_eur", "mileage_km", "reg_year", "power_hp"):
        d[field] = pd.to_numeric(d[field], errors="coerce")
    valid = (d["month"].notna() & d["price_eur"].between(500, 500000)
             & d["mileage_km"].between(0, 800000)
             & d["reg_year"].between(1950, 2026)
             & (d["reg_year"] <= d["year"])
             & d["power_hp"].between(20, 1200))
    d = d.loc[valid].copy()
    for key in ("seller_country", "make", "model", "source_site",
                "fuel", "transmission", "body_type"):
        d[key] = d[key].fillna("unknown").astype(str).str.strip().str.lower()
    d["seller_country"] = d["seller_country"].str.upper()
    d = d[d["seller_country"].isin(SUPPORTED_COUNTRIES)]
    d = d.drop_duplicates(["seller_country", "month", "id"], keep="first")
    d["vehicle_group"] = d["make"] + " · " + d["model"]
    d["age"] = d["year"] - d["reg_year"]
    d["log_mileage"] = np.log1p(d["mileage_km"])
    return d


def month_coordinate(month):
    y, m = map(int, month.split("-"))
    return y * 12 + m - 1


def design(rows):
    records = []
    for row in rows.itertuples(index=False):
        age = float(row.age)
        records.append({
            "car=" + str(row.vehicle_group): 1.0,
            "portal=" + str(row.source_site): 1.0,
            "fuel=" + str(row.fuel): 1.0,
            "gear=" + str(row.transmission): 1.0,
            "body=" + str(row.body_type): 1.0,
            "age": age / 20.0,
            "age_sq": (age / 20.0) ** 2,
            "log_km": float(row.log_mileage) / 12.0,
            "hp": float(row.power_hp) / 300.0,
        })
    return records


def pair_index(first, second, *, min_overlap=800, sample_per_month=2500):
    """Paired Huber(log-price) models on the *same* common quality baskets."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import HuberRegressor

    a = first.copy()
    b = second.copy()
    group_a = set(a.vehicle_group.value_counts().loc[lambda v: v >= 4].index)
    group_b = set(b.vehicle_group.value_counts().loc[lambda v: v >= 4].index)
    support = group_a & group_b
    portals = set(a.source_site) & set(b.source_site)
    a = a[a.vehicle_group.isin(support) & a.source_site.isin(portals)]
    b = b[b.vehicle_group.isin(support) & b.source_site.isin(portals)]
    if min(len(a), len(b)) < min_overlap or len(support) < 10:
        return None
    a = a.sample(min(sample_per_month, len(a)), random_state=20261110)
    b = b.sample(min(sample_per_month, len(b)), random_state=20261110)
    x = DictVectorizer(sparse=True).fit_transform(design(pd.concat([a, b])))
    n = len(a)
    ya = np.log(a.price_eur.to_numpy(float))
    yb = np.log(b.price_eur.to_numpy(float))
    kwargs = dict(alpha=0.02, epsilon=1.35, max_iter=175, tol=1e-4)
    model_a = HuberRegressor(**kwargs).fit(x[:n], ya)
    model_b = HuberRegressor(**kwargs).fit(x[n:], yb)
    if model_a.n_iter_ >= kwargs["max_iter"] or model_b.n_iter_ >= kwargs["max_iter"]:
        return None  # nonconverged models must never release a production index
    # Quality-adjusted arithmetic price ratios, held over identical price baskets.
    pa0 = np.exp(np.clip(model_a.predict(x[:n]), 4.0, 14.0))
    pa1 = np.exp(np.clip(model_b.predict(x[:n]), 4.0, 14.0))
    pb0 = np.exp(np.clip(model_a.predict(x[n:]), 4.0, 14.0))
    pb1 = np.exp(np.clip(model_b.predict(x[n:]), 4.0, 14.0))
    l = float(pa1.sum() / pa0.sum())
    p = float(pb1.sum() / pb0.sum())
    f = math.sqrt(l * p)
    if not all(math.isfinite(v) and 0.7 < v < 1.4 for v in (l, p, f)):
        return None
    return {"fisher": f, "laspeyres": l, "paasche": p,
            "n": min(len(a), len(b)), "overlap": len(support),
            "raw_median": float(b.price_eur.median() / a.price_eur.median())}


def largest_chain(rows):
    """Only link consecutive months; never connect disjoint time ranges."""
    best, current = [], []
    for row in rows:
        if current and month_coordinate(row["month"]) - month_coordinate(current[-1]["month"]) != 1:
            if len(current) > len(best):
                best = current
            current = []
        current.append(row)
    if len(current) > len(best):
        best = current
    return best


def build(df, *, min_month_rows=1200, min_months=6, min_overlap=800):
    data = clean(df)
    coverage = data.groupby(["seller_country", "month"]).size().rename("n").reset_index()
    report = {
        "schema_version": 1, "status": "unavailable",
        "method": "Adjacent-month common-support Huber regression on log(EUR ASK); "
                  "Fisher=sqrt(Laspeyres*Paasche); log interpolation; damped-log Holt forecast",
        "source": "KTHESAT/european-used-car-market-2m-listings",
        "coverage": [{"country": r.seller_country, "month": r.month, "n": int(r.n)}
                     for r in coverage.itertuples(index=False)],
        "countries": {}, "pilot_countries": {}, "message": "Not enough genuine consecutive observations "
                                 "to estimate a reliable monthly Fisher index.",
    }
    for country, all_periods in coverage.groupby("seller_country"):
        available = sorted(all_periods.loc[all_periods.n >= min_month_rows, "month"].tolist())
        if len(available) < 2:
            continue
        country_data = data[data.seller_country == country]
        # Evaluate each *adjacent* real month; gaps deliberately break chains.
        groups = {m: country_data[country_data.month == m] for m in available}
        chains = []
        chain = []
        for m in available:
            current = groups[m]
            if not chain:
                chain = [{"month": m, "index": 100.0, "n": len(current),
                          "overlap": 0, "raw": 100.0, "laspeyres": 100.0, "paasche": 100.0}]
                continue
            previous_month = chain[-1]["month"]
            if month_coordinate(m) != month_coordinate(previous_month) + 1:
                chains.append(chain)
                chain = [{"month": m, "index": 100.0, "n": len(current),
                          "overlap": 0, "raw": 100.0, "laspeyres": 100.0, "paasche": 100.0}]
                continue
            result = pair_index(groups[previous_month], current, min_overlap=min_overlap)
            if result is None:
                chains.append(chain)
                chain = [{"month": m, "index": 100.0, "n": len(current),
                          "overlap": 0, "raw": 100.0, "laspeyres": 100.0, "paasche": 100.0}]
                continue
            chain.append({"month": m,
                          "index": chain[-1]["index"] * result["fisher"],
                          "n": len(current), "overlap": result["overlap"],
                          "raw": result["raw_median"],
                          "laspeyres": result["laspeyres"],
                          "paasche": result["paasche"]})
        if chain:
            chains.append(chain)
        winner = max(chains, key=len, default=[])
        if len(winner) >= min_months:
            report["countries"][country] = {
                "months": winner,
                "start": winner[0]["month"], "end": winner[-1]["month"],
                "sample_count": int(sum(r["n"] for r in winner))}
        elif len(winner) >= 2:
            # An empirical two-to-five-month pilot is useful for research but
            # NEVER eligible for date-adjusted production quotes or Holt.
            report["pilot_countries"][country] = {
                "months": winner, "status": "experimental",
                "reason": "Fewer than six consecutive observed months",
                "sample_count": int(sum(r["n"] for r in winner))}
    if report["countries"]:
        report["status"] = "validated"
        report["message"] = ("Published observed months only. Holt beyond sample "
                             "is a model forecast, never actual market evidence.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="models/europe-ask/hedonic_fisher.json")
    parser.add_argument("--min-months", type=int, default=6)
    args = parser.parse_args()
    output = build(args.source, min_months=args.min_months)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": output["status"], "countries": list(output["countries"]),
                      "coverage_months": len(output["coverage"]),
                      "output": str(destination)}))


if __name__ == "__main__":
    main()
