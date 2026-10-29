# Deal Detective release

Open **Deal Detective** in the studio, or use **Share my price** after calculating. The calculator’s selected asking model and existing scientific panels remain available; no weights or training filters changed in this release.

## Sharing without accidentally publishing your backstory

- Download a real PNG: landscape 1200×630 or portrait 1080×1350. Cards include the estimate’s May 2021 market reference and a QR code. They never describe a historical estimate as today’s guaranteed selling price.
- Native **Share link** / **Share image** work where the browser, operating system and hosting permissions allow. They require a click and open the device chooser. Cancellation stays cancellation; there is no claim a post was published.
- WhatsApp, LinkedIn, Facebook, Reddit and email buttons prepare a share. LinkedIn/Facebook accept the link; paste the caption and attach the PNG when desired. Instagram uses the native image menu or a downloaded portrait image. No social account credentials are requested and no automatic posting occurs.
- Playful, professional and challenge captions have copy buttons plus selectable text fallbacks. Clipboard and native sharing can be unavailable inside embedded/mobile browsers.
- Vehicle facts are included in a public link only after explicit opt-in. Preview the allowlisted fields first. Seller descriptions, estimates, receipt keys, visitor IDs and analytics location are excluded. Anyone with a link can decode its vehicle facts; encoding is not encryption.
- Facts-only recipients load and recalculate inputs using their installed model. Without the original seller description, their result can differ. Shared links are not signed valuation certificates. QR codes go to the home page unless vehicle sharing is selected.
- The price-receipt button opens the Share panel directly. Shared vehicle URLs open Deal Detective, with an explicit load-into-calculator action.

## Useful before and after the screenshot

Paste an advert or upload PNG/JPEG for local Tesseract OCR. No external AI service receives it. Year, make/model, miles and a USD seller price are proposed, not trusted: edit before loading. Kilometres are converted when detected. Regex contact redaction is best effort; review your text. Images/drafts stay in session memory and are not visitor analytics. Upload limits: 8 MB / 16 megapixels; OCR timeout: 15 seconds. `packages.txt` installs the OCR engine and fonts on Streamlit Cloud.

Park up to three estimates in a session garage. Compare Ask/Bid on the same May 2021 reference date, export a shortlist without seller text, or opt into a facts-only comparison link. Auction-era comparison is restricted to model years through 2015. A spread is not profit. Running-cost calculations use visible editable assumptions; they are not forecasts. Buyer briefing/checklist questions are not findings about the car. The time panel remains the place for current-date scenarios and its limitations.

## Evidence and the price challenge

`data/deal-reference.json.gz` contains 4,000 deterministic, identifier-free historical wholesale records from [Used Car Auction Prices](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices). Official Kaggle API metadata reports **CC0: Public Domain**, checked 2026-10-29. No VIN or seller name is redistributed. Rebuild with:

```sh
python scripts/build_deal_reference.py /path/to/source.zip
```

The builder records the archive hash and selection rules: plausible 1980–2015 model years, 0–300k miles, $500–200k, latest sale per source VIN, up to eight rows per make/model, hashed down to 4,000. This is not a representative sample. The evidence panel ranks same-make/model matches by year and mileage, never by price proximity. These are provider-reported 2014–2015 wholesale sales, not contemporary retail comparables or independently verified settlements. Missing close matches stay missing.

The daily/practice challenge reveals a reported sale after a guess and compares the installed Bid model where supported. First guess per case counts in a private session scoreboard. This gallery can overlap training data and is accessible in the repository. It is entertainment and explanatory evidence, not a secure contest or fresh validation benchmark.

## Private outcome feedback

Completed-sale feedback is optional and requires two explicit confirmations separate from analytics consent. All rows start **unverified**; no automatic retraining occurs. Vehicle/sale facts and an optional **already redacted** receipt image go into private Turso `sale_outcomes` / `sale_evidence` tables, not the public gallery. No analytics visitor ID is linked. Images are limited to 4 MB / 16 MP, resized and re-encoded to JPEG without EXIF. This does not redact visible personal information: contributors must remove it themselves.

An atomic conditional Hrana batch saves parent and evidence together, rolling back after failure. Duplicate normalized receipt hashes are rejected; this is a duplicate check, not a fraud detector. A private receipt provides a random deletion key; only its SHA-256 hash is stored. Deletion checks the key, removes evidence first and does not reveal whether an unmatched ID exists. Both tables retain records for at most 90 days when purge runs; opportunistic visits and the separately configured daily retention job perform cleanup. A period with neither does not guarantee wall-clock deletion. Exports made by the operator need separate deletion.

Private operator tooling uses environment credentials, never browser UI or command-line tokens:

```sh
python -m scripts.review_sale_feedback pending
python -m scripts.review_sale_feedback evidence SUBMISSION_ID /private/review.jpg
python -m scripts.review_sale_feedback review SUBMISSION_ID confirmed --reason 'Reviewed redacted receipt, market and transaction date.'
```

`confirmed` means operator-reviewed, not independent guaranteed settlement. Review is bounded to an unverified row; record a non-personal rationale. The tool never triggers training. Feedback remains excluded until a separately designed, leakage-safe data curation process. Do not use self-selected feedback to claim improved MAE.

## Release validation

Automated checks cover share-link allowlisting, malformed payloads, encoding, actual Tesseract extraction, both image dimensions, provenance, import/review callbacks, shortlist comparison, feedback validation/rollback/deletion/retention, and native-share/copy cancellation/fallback behavior. CI runs these alongside the existing analytics privacy checks. Manual visual review checks card typography and mobile controls. Browser-native sharing and platform preview caching remain device/service dependent; direct URL composers do not force a social preview image. PNG upload is the reliable visual path.
