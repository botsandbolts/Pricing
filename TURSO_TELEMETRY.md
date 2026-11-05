# First-party Turso functionality diagnostics

Car Price Studio uses its **existing remote Turso/libSQL database**, not a second
analytics database. The Streamlit backend is the only place that can access the
Turso token; a consented browser component sends bounded event identifiers to
Streamlit, never to Turso directly.

## Tables

| Table | Purpose | Stored information |
| --- | --- | --- |
| `visitors` (existing) | Opt-in pseudonymous visitor | Random ID, first/last visit, coarse device/browser and approximate location |
| `analytics_sessions` (new) | One browser session | Random session ID, full country name and ISO country code, region, city, provenance and precision, browser language(s), IANA timezone, browser, OS family, device and viewport bucket |
| `feature_events` (new) | UI activity across panels | Allowed panel and subpanel slugs, action/event category, elapsed panel seconds (max 1800), server timestamp |
| `feature_runs` (new) | Server-executed feature outcomes | Feature, model family, success/unavailable/error, input/result counts and bounded elapsed milliseconds |
| `feature_errors` (new) | Bounded debugging codes | Feature, fixed error code, severity and timestamp, **not** exception messages or stacks |
| `pricing_queries` (existing) | Ask/Bid estimates | Price, model, valuation date, approved vehicle attributes; seller text excluded |
| `hierarchical_estimations` (existing) | Research comparisons | Controlled historic held-out comparison fields |
| `sale_outcomes`, `sale_evidence` (existing) | Explicit buyer feedback | Separately submitted outcome/evidence with deletion receipt; not mixed into visitor telemetry |

New schema creation is additive and happens when configured, authorized Turso
storage is first initialized. Existing table rows are preserved.

## Coverage

Browser-side events are sent only after optional consent. The allowlist covers
calculator, listing inspection, Drive Chemistry, two-market comparisons,
time scenarios, science/explanations, model/evaluation labs, Deal Detective,
Outlier lab and Merchant lens. Existing finite event names cover panel views,
section toggles, feature actions, setting-change **categories**, downloads,
sharing and active-panel time. They **never include actual setting/input values**.

Server-run records are currently instrumented for the Drive Chemistry ranking
and Listing Inspector pricing. Existing pricing and hierarchy writes remain in
their own purpose-built tables. This is not a claim that every deeper feature
operation has a dedicated server-run hook yet.

## Geography: most accurate truthful result, not a fictional GPS fix

1. After opting in, the browser asks GeoJS to estimate country, region and city
   from its internet connection; these are **IP estimates** and can be wrong.
2. The optional local MaxMind GeoLite2 / DB-IP city database can use **only a
   configured, proxy-overwritten trusted client-IP header**. A browser-provided
   forwarding header must never be trusted.
3. A visitor can correct the estimated city in the privacy dialog; this is
   tagged `self_reported_city` instead of `ip_approximate`.
4. `country_name` is derived by the browser's built-in `Intl.DisplayNames`
   using the returned ISO country code. Browser language and timezone are
   stored separately as UI diagnostics, **not used to infer city**.

No device GPS permission is requested. Device GPS would require an explicit
additional browser permission and safe reverse geocoding. No provider can
guarantee city accuracy from IP alone. Raw IPs, geocoordinates, text
descriptions, full user-agent strings and high-entropy fingerprints are never
written into these Turso tables.

## Consent and retention

The opt-in version is `2026-11-05-v4` so previously accepted preferences
must be renewed after a deployment. Refusal stops optional first-party
analytics, third-party GeoJS lookup and Google Analytics. A preference cookie
remembers accept/reject; the random visitor cookie is created **only after
acceptance**. Corrected city is session-scoped in the browser.

These records have a 90-day retention target. The best-effort purge runs at
most hourly on a consented visit; it is **not a guaranteed scheduled deletion
job on an idle app**. Add a separate scheduled purge if a guaranteed deletion
deadline is required. Disconnecting analytics or losing Turso credentials
prevents purges until service is restored.

Some necessary operational hosting logs may exist outside this Turso database.
Do not describe all functionality analytics as "strictly necessary" just
because they are useful for debugging. Optional visitor IDs, session analytics
and geo lookup continue to require a choice; none of this is advertising
profiling. Existing Google Analytics uses separate opt-in, with ad signals
disabled, under the current combined privacy choice.

## Configuration

Set in the Streamlit application's private secrets (never in GitHub):

```toml
ANALYTICS_ENABLED = "true"
TURSO_DATABASE_URL = "libsql://YOUR-DATABASE.turso.io"
TURSO_AUTH_TOKEN = "YOUR_PRIVATE_TOKEN"
ANALYTICS_CONTROLLER_NAME = "Your operator name"
ANALYTICS_CONTROLLER_CONTACT = "Your contact address"
ANALYTICS_GEO_MODE = "geojs"
```

For optional local IP geolocation, use the existing
`ANALYTICS_TRUSTED_IP_HEADER`, `ANALYTICS_GEO_MMDB`,
`ANALYTICS_GEO_PROVIDER` configuration and a legally usable updated local
MMDB. Never infer that `X-Forwarded-For` is trusted without the hosting
platform documenting that it overwrites the header.

## Private reporting queries

Do not expose raw tables to the public Streamlit UI. Aggregate privately:

```sql
SELECT country_name, city, geo_precision, COUNT(*) AS sessions
FROM analytics_sessions
GROUP BY country_name, city, geo_precision
ORDER BY sessions DESC;

SELECT feature_name, event_type, COUNT(*) AS actions
FROM feature_events
GROUP BY feature_name, event_type
ORDER BY actions DESC;

SELECT feature_name, model_family, status, COUNT(*) AS runs,
       ROUND(AVG(duration_ms),0) AS avg_ms
FROM feature_runs
GROUP BY feature_name, model_family, status;

SELECT feature_name, error_code, COUNT(*) AS occurrences
FROM feature_errors
GROUP BY feature_name, error_code;
```

Small result groups may need suppression before publishing. A visitor/session
ID can be personal data even if randomly generated: keep these reports private,
apply access controls and honor withdrawal/deletion requests.

## Testing

Run locally with an in-memory SQLite simulator of the Turso schema:

```bash
pytest -q tests/test_analytics.py tests/test_feature_telemetry.py
```

A passing SQLite test demonstrates schema correctness and consent logic. It
does **not** prove that your production Turso token, GeoJS, Streamlit Cloud
deployment, or JavaScript component is operational.
