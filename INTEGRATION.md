# Structural transfer: CarBids → Pricing

The first tab in **both apps** is now **Pricing + CarBids**. Each repository bundles the upstream structural model, the original Pricing baseline, the experimental downstream asking model and identical shared source. No second server, login, remote inference API or GPU is required for inference.

## What the prices mean

- `asking_price_usd`: the unchanged original model's April–May 2021 advertised asking estimate.
- `transported_auction_price_usd`: an auction-equivalent scenario evaluated at the supplied date and age, with actual mileage and an official used-car CPI anchor. This is **not a calibrated retail sale price**.
- `asking_with_auction_structure_usd`: the experimental 2021 asking model using transferred auction features. Outside its historical target dates, the displayed asking comparison is explicitly anchored to May 2021; it does not pretend to be a current asking price.
- The optional retail premium slider is a **user assumption**. It is not a learned transaction-price correction. At 0%, it simply reproduces the auction-equivalent estimate.

The original historical model tabs are retained. The shared comparison tab is the default. Original single-car screens also show the structural companion.

## The age correction

Age is computed from valuation/sale date minus model year, including the fraction of the calendar year. Production date is unavailable: this is a model-year-age proxy, clipped at zero for next-model-year vehicles. A 2014 model evaluated in May 2015 is about 1.3 years old; in May 2022 it is about 8.3. **Raw model year is not an upstream predictor.**

We fit log(price / index ratio), not raw 2015 nominal price. The reference index month is January 2015. Three alternating backfitting steps combine robust, clipped-residual empirical partial pooling (global → make → make/model) with Huber LightGBM. Sparse model offsets shrink toward the make/global hierarchy. The shape learner receives age, log1p mileage, numeric auction grade, coarse body class, transmission and state. Age/mileage effects are constrained non-increasing; grade is non-decreasing. Typical mileage by age is a separate descriptive feature, not another unconstrained route around the monotonicity restriction.

Model names match exactly after case/whitespace normalization. Unknown models fall back to make/global effects; no fuzzy trim match is invented. Auction grades are optional actual grades. Seller adjectives are never mapped to grades by guesswork. Generation, MSRP and equipment-quality changes are not available in these data and are **not** claimed as learned inputs.

The displayed decomposition changes a reference age-3/36,000-mile/grade-3 vehicle first to actual age, then actual mileage, then actual or missing grade, then the target index level. Make/model, segment, transmission and state stay fixed. Components reconstruct the final estimate exactly, but their order-dependent interpretation is not causal attribution.

## Market time: actual BLS data, explicit transport assumption

The bundle includes **165 monthly official BLS observations, January 2013–September 2026**, series `CUUR0000SETA02`, Consumer Price Index for All Urban Consumers: Used Cars and Trucks, not seasonally adjusted. The source API responses were retrieved on October 26, 2026. Source URLs and file hash are in `models/structural-transfer/used-car-index.meta.json`.

Every valuation uses the observation **two calendar months before** its date. Missing months cause an unavailable status; future months are never filled from later data. This conservative lag avoids unreleased current-month information, but the stored data are a current-vintage snapshot, not proof of original vintage availability. Historical backtests are retrospective, not a fully vintage-correct online replay.

The aggregate **retail** used-car CPI is used as an assumed **wholesale** market anchor. It is not a segment-specific wholesale index. BLS's tracked used vehicles are primarily ages two to seven; transfer to much older cars is a stronger assumption. Segment/time interactions, collector appreciation, structural EV changes, different condition mixes and geographic selection are not identified by this anchor. CPI is applied once, after constant-market valuation.

References:
- https://www.bls.gov/cpi/factsheets/used-cars-and-trucks.htm
- https://fred.stlouisfed.org/series/CUUR0000SETA02
- https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SETA02

## Identification and leakage

Age = period − model year. Short auction coverage cannot separately identify unrestricted age, period and cohort/generation effects. The specification anchors period externally, omits raw model year, and assumes cross-sectional age relationships transfer across comparable generations. Constraints make a coherent scenario, not proof that those relationships are causal depreciation trajectories. Newer generations, sparse/unknown models, unsupported ages and mileage are flagged. No calibrated target-domain prediction interval is claimed.

Before fitting the upstream model, **110 source VINs** found in the full Pricing description cohort were excluded. VINs were extracted only as pre-existing identity covariates, never as predictors; unobserved VINs cannot establish full vehicle disjointness. All upstream auction labels predate the downstream 2021 target outcomes. No Pricing price is used to fit upstream structure. Downstream category vocabularies are fitted on the training partition; validation selects settings; test is evaluated before deployment refit. No target text, price, bid, reserve, MMR or post-sale result enters the upstream feature builder.

This is independent older-source transfer, **not a claimed same-dataset OOF text model**. Existing neural description training remains separate. For same-period upstream training or strict VIN-disjoint applications, build appropriate grouped/time OOF predictions; `carbids.bridge` retains its conservative VIN-index guard. Selection into wholesale sales versus retail listings remains a transport limitation.

## Actual training and assessment

Structural upstream deployment was refitted on **550,181** eligible auctions after exclusions. Chronological source test MAE is **$2,299.80**, RMSE **$4,288.16**. These scores describe the source assessment model before full-row refit, not 2021/2022 sale accuracy. Its constraints intentionally differ from the original historical CarBids baseline. The source validation-residual sensitivity band is labeled **source residual sensitivity**, never a transported 80% coverage guarantee.

Both downstream experiments use identical 2021 partitions: **148,290 training / 31,777 validation / 31,777 test** records, then refit the experimental deployment on all **211,844** eligible records. Each compares 500/1,000 Huber trees and selects using validation dollar MAE.

| Asking model | Test MAE | Test RMSE |
|---|---:|---:|
| Original tabular specification | $3,328.66 | $6,839.35 |
| With auction structural features | $3,338.12 | $6,807.22 |

The transfer variant slightly worsens MAE and improves RMSE. It is exposed as an experiment; the original asking model remains the default. The paired day-cluster bootstrap MAE-difference range is approximately **−$29.37 to +$13.35**, but the test set spans only **three day clusters**, so that interval provides weak evidence and is not a robust generalization guarantee. Do not claim an established MAE improvement.

The new downstream features are age, constant-market auction value, transported auction value, partially pooled model-value factor, observed model count, excess mileage relative to age, local one-year age slope, and CPI multiplier. The fitted LightGBM can combine these with the existing tabular fields. Native TreeSHAP in the GUI explains that asking model, with correlated-feature and non-causal caveats.

## Every Pricing car

The full cohort was processed: **211,844 rows retained**, **211,127 transported auction equivalents**, and **717 explicit unavailable rows** for missing/invalid year or other invalid vehicle fields. Missing dates also prevent transport. All rows still receive the historical asking calculation; missing categories use learned missing-value routes. No fake year is invented to force a sale-equivalent price.

The CLI supports CSV or JSONL, processing in chunks:

```bash
python -m marketbridge.engine --input training-cache/full-cohort.jsonl --output reports/pricing-with-carbids.csv
```

Required for structural transport: `year`, `as_of` or `date`, and sensible mileage if supplied. Recommended: `manufacturer`, `model`, `odometer`, `type`, `transmission`, `state`. Aliases: `make`, `mileage_miles`, `body_type`. Optional: `auction_condition`, `auction_model`, `row_id`. CLI output contains row IDs and predictions/support flags, without seller descriptions, VINs or contact information. Raw input/cohort and the large per-row CSV are not committed; the audit summary is.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m marketbridge.index --output /tmp/used-car-index.csv --end-year 2026
python -m marketbridge.structural --cohort data/cohort.jsonl --index /tmp/used-car-index.csv --output models/structural-transfer --exclude-pricing training-cache/full-cohort.jsonl
python -m marketbridge.asking --cohort training-cache/full-cohort.jsonl --structural models/structural-transfer --output models/transfer-asking
python -m streamlit run app.py
```

The auction cohort and Pricing cohort come from their original audited preparation workflows; provide their actual paths when running these commands. Refreshed index values or training cohorts change model IDs. The experimental asking bundle refuses a mismatched upstream version. Use the frozen bundled snapshot for the published assessment; refreshing the snapshot is not a free accuracy upgrade.

## What is deliberately not fitted

There is no matched contemporary retail sale dataset here. Therefore no learned retail-channel premium, negotiation discount, retail-sale error metric or target-sale conformal coverage is claimed. The GUI makes this boundary explicit. Census context remains available in CarBids, but was not used to manufacture unavailable historical retail labels or future context.
