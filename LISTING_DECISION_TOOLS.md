# Listing decision tools — business guide

## What this workspace is for

Listing Inspector helps a buyer prepare questions and a dealer organize appraisal work. It complements the existing valuation calculator. It does not certify a vehicle, authenticate a seller, determine realized profit or supply live market listings.

## Choose a specific car

Search 1,000 individual historical vehicle records by year, car, mileage or state. Selecting one fills its actual available vehicle details and description. This is not a popularity list or a list of generic make/model combinations. Edit the fields and select Find comparable cars. Some auction examples contain no description; it stays empty rather than borrowing another seller’s text.

## Comparable cars

Asking adverts and auction sales are separate historical reference sets. FAISS returns five nearest cars by cosine similarity of normalized SVD vectors, with optional MiniLM description similarity. There are no brand/model compatibility exclusions or second-stage ranking. Each comparison shows actual age, mileage and specification differences; any retrieval-only imputation is identified. Descriptions contain seller claims. Auction prices retain their original dates; an old auction amount is not today's offer. Matching methods and candidate-pool sizes are disclosed in the interface.

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
