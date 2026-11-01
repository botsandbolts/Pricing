# Listing decision tools — business guide

## Three independent SVDs

Ask uses asking vehicles and optional MiniLM text similarity. Bid uses auction vehicles and an imputed MiniLM description-price proxy. Hybrid uses both markets' vehicle attributes only, including a separate price/text-free imputer. Ask and Bid each return five neighbours; Hybrid returns five of each, mixed and sorted by the same cosine similarity. Ask rows are purple and auction rows blue.

Adjusted advert/auction prices assist Ask/Bid missing-feature imputation, then are dropped before SVD. The only price-derived scalar SVD feature is the auction description-price proxy. Actual auction descriptions do not exist and are not invented. Detailed neighbour records show source and adjusted prices, year/mileage/specification differences and same-market price differences from the queried car. No FM or LambdaRank is used. See COMPARABLE_INSPECTOR.md for the complete feature and missing-data contract.

## What this workspace is for

Listing Inspector helps a buyer prepare questions and a dealer organize appraisal work. It complements the existing valuation calculator. It does not certify a vehicle, authenticate a seller, determine realized profit or supply live market listings.

## Choose a specific car

Search 1,000 shuffled individual historical asking and auction vehicle records by year, car, mileage or state. Selecting one fills its actual available vehicle details and description. This is not a popularity list or a list of generic make/model combinations. Edit the fields and select Price my car & find neighbours. Some auction examples contain no description; it stays empty rather than borrowing another seller’s text.

## Comparable cars

Asking adverts and auction sales are separate historical reference sets. A single form prices your own car with the unchanged landing-page XGBoost/Bayesian models before finding neighbours; adjusted prices assist Ask/Bid imputation, then are removed before SVD. FAISS returns five nearest cars by cosine similarity of normalized SVD vectors, with optional MiniLM description similarity. There are no brand/model compatibility exclusions or second-stage ranking. Each comparison shows actual age, mileage and specification differences; any retrieval-only imputation is identified. Descriptions contain seller claims. Auction prices retain their original dates; an old auction amount is not today's offer. Matching methods and candidate-pool sizes are disclosed in the interface.

## Purchase sandbox

Enter assumed realized resale proceeds and acquisition, repair, transport, fee and holding costs. The result is a contribution scenario before overhead and tax. Changing an assumption changes the arithmetic, not the underlying evidence. The difference between asking and auction estimates is never automatically treated as dealer profit.

## Deal review and appraisal passport

Explicitly evaluate the inspected car to obtain existing Bayesian asking and auction estimates. An asking price below the date-adjusted historical range is a reason to investigate. Wide ranges, rare models, unfamiliar categories, generation extrapolation may instead require more information or specialist review.

The downloadable passport records the vehicle, valuation date, pricing release, estimates and support. Historical test MAE and coverage are assessment-population statistics, not a measured error for this car. Transported current-date intervals have not been validated against current transactions. There is no calibrated probability of a bargain.

## Duplicate check

Upload up to 100 CSV rows. A supplied VIN match is strong identity evidence; near-identical text requires matching make/model/year and close mileage. Review differences in price and mileage against listing dates. These are review leads, not accusations of fraud. VINs are hashed for matching and excluded from the results.

## Regional Price Lens

The offline experiment controls for vehicle and calendar attributes and reports historical state associations for sufficiently supported states. It shows exploratory bootstrap intervals. Conditional associations do not establish causation, current local demand or profitable geographic arbitrage.

## Dealer review queue

Upload up to 25 cars to evaluate them sequentially using the installed models. Asking discrepancies are assessed against asking estimates. Both market values are returned separately. Invalid rows and unsupported cars are isolated so they do not stop the entire batch. Download the sorted queue for human review; it is not a predicted-profit ranking.

## Information premium and Lemon inspection

A separate research baseline asks whether frozen MiniLM description vectors improve historical held-out price prediction beyond structured vehicle facts. Its dollar MAE may improve or worsen; both outcomes are shown. This measures predictive information, not money created by editing an advert. No keyword flag engine is used.

Fresh grouped cross-validation flags the largest research-training log residuals for inspection. It keeps all observations. Legitimate unusual cars can have large residuals. This experiment does not change the production Lemon Squeezer cut or any serving model.

## Data and privacy

Original historical sources: Craigslist Used Cars / Tom / Zenodo 10457828 (CC BY 4.0) and Used Car Auction Prices / tunguz / Kaggle (CC0). Compact statistical reports contain aggregate results rather than raw uploaded inventory. Uploads and results are processed in the current Streamlit session; these tools add no inventory database or third-party transmission. Opt-in GA events use finite feature/action labels and do not include prices, VINs or description text.


## Historical prices adjusted to the valuation date

Comparison cards retain the original observed asking/auction price and observation date, then apply the existing depreciation age/mileage ratio and the ratio of the aggregate used-car market index at the valuation and observation dates. The default valuation date follows today. Optional mileage normalization puts each reference at the entered car’s mileage; turning it off holds each reference’s stated mileage fixed. Neither assumption tracks the physical vehicle’s subsequent use or condition. National retail used-car CPI is also a wholesale proxy; when observations lag, the existing damped market forecast is disclosed. In Ask/Bid only, adjusted prices and the query estimate assist missing-feature imputation and are then dropped before SVD. This indirect conditioning prevents treating neighbours as independent validation. Unsupported age ranges or missing source prices/dates/mileage yield an explicit unavailable adjustment, not a guessed current price. These scenarios do not have validated current-vehicle accuracy or prediction coverage.


## Price-assisted completion, price-free SVD

Ask/Bid use the queried car’s selected-market model estimate as auxiliary imputation context where supported. Reference and query context share the saved imputer's dated price basis. The actual price column is dropped before SVD; only Bid retains the separate imputed MiniLM description-price proxy. Hybrid excludes every price/text signal. Displayed historical prices are independently adjusted to the selected date. These are scenarios, not verified current offers or an independent valuation check.

## Missing auction description-price feature

Auction records contain no seller description. Their description-price feature starts as NaN, not zero or invented text. A shared sklearn IterativeImputer with small native-categorical LightGBM regressors learns from observed asking-description prices, vehicle attributes and reference-2021 price context. The asking signal is the mean of the two installed MiniLM price-head dollar predictions. Known asking signals are preserved; missing asking and all auction signals are imputed. The inferred quantity is an asking-style statistical proxy, not an observed auction description or auction offer. It is used only in Bid retrieval, never Ask, Hybrid or production valuation.

The reconstruction audit masks asking signals and compares dollar error with a median baseline. It does not validate transfer to auction descriptions, which do not exist. Models and schedules are stored in hashed compressed JSON with LightGBM text trees. Hybrid uses a separate vehicle-only imputer, excluding price and text even during missing-value completion.

