# Private Turso analytics

The pricing models and their weights are unchanged. Analytics is optional and
disabled until configured. No visitor/query dashboard or public export route exists.
The privacy notice is public; the records are private.

## Deploy and activate

1. Create a free Turso database in your own account (choose an EU location where
   available). Obtain its database URL and a database-scoped token. Keep overages
   disabled. Review the provider agreement and processing terms for your use.
2. Set the keys in `.streamlit/secrets.example.toml` in **Streamlit Community Cloud
   app settings → Secrets**, using real operator name/contact. Set
   `ANALYTICS_ENABLED = true` only when these are complete. The example is not a
   live connection. No credential belongs in Git or browser JavaScript.
3. Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` as private GitHub Actions secrets
   too, to enable the daily retention workflow. The app also purges at most hourly
   during consented traffic. Scheduled jobs can be delayed or disabled by GitHub;
   monitor their outcome. Exports/backups need their own retention/deletion policy.
4. Verify acceptance creates a visitor row; a price creates linked Ask/Bid rows;
   rejection creates neither. Confirm reload does not duplicate a price. Check
   withdrawal, local storage restrictions, database errors and your actual proxy
   headers before treating tracking coverage or geography as complete.
5. Administrator health check: with credentials in environment, run
   `python -m scripts.analytics_admin check`. It prints counts only.

No authenticated Turso account or database token was available during development;
live database provisioning and end-to-end writes require that account connection.

## Consent and data

Before acceptance: no analytics ID cookie, device categorization, IP lookup or
database writes. The necessary `pricing_consent` preference cookie records only
choice/version/time for 90 days. Closing the dialog leaves analytics off. Accept
and Reject have equal prominence; pricing works without acceptance.

After acceptance: `pricing_visitor` is a random UUID cookie with 90-day lifetime,
and `pricing_session` is a random UUID in sessionStorage. The JavaScript component
passes them through Streamlit's component channel, never directly to Turso. Only
Chrome/Firefox/Safari/Edge/Other and mobile/desktop categories are retained, not the
raw User-Agent. No canvas, font, audio, hardware fingerprint, GPS or cross-site ID.
These are **pseudonymous records**, not guaranteed anonymous data.

Visitors: UUID, first/last seen UTC, approximate country/city/region and provider,
browser/device category, consent version/time. Pricing queries: linked visitor and
session UUIDs, query UUID/time, same approximate location, Ask/Bid side, price USD,
model/source, valuation date, allowlisted vehicle fields. Seller descriptions,
text-price signals, IP addresses, URLs/referrers, VINs and contact fields are excluded.
Free make/model fields are length limited and obvious contacts/URLs are rejected;
no free-text filter can guarantee that all volunteered personal details are caught.

Only successful price estimates are recorded. Main calculator, its auction
companion, default two-market calculator, model lab and current-value scenarios
are instrumented. Repeated rerenders of the same scenario are deduplicated.
Graph points, uploads, downloads, SHAP calculations, errors and older experimental
batch/structural forms are not tracked. Visitors who decline, block cookies, or
leave before the component loads are not counted. Do not call this all visitors.

Withdraw via **Privacy & cookies → Reject analytics**: future writes stop and
analytics IDs are removed. Existing records remain under the 90-day policy; visitors
can quote their visible ID to request deletion. Admin deletion:
`python -m scripts.analytics_admin delete-visitor --visitor-id UUID`.
This deletes child queries before the visitor. It is not a public deletion API.

## Free approximate country and city

The default `ANALYTICS_GEO_MODE = "geojs"` uses GeoJS from the browser **after
acceptance**, so Streamlit does not need to expose a client IP. This is a free
HTTPS/CORS service using MaxMind GeoLite data. The popup explicitly discloses
that GeoJS and its network receive the visitor's IP through the lookup connection.
Only country code, city and region are passed back to Streamlit and saved in
**both tables**; IP, coordinates, ASN, timezone and other response fields are
discarded. A consented per-tab cache lasts up to one hour; requests time out at
three seconds and fail to unknown geography. No server-IP substitution, API key,
remote script execution or fallback chain across additional providers is used.
Client-provided locations are unverified analytics hints and must never determine
access, identity or a security decision. City data are never displayed in the GUI.
Sources: https://www.geojs.io/docs/v1/endpoints/geo/ and
https://www.geojs.io/privacy/ . Availability and free terms can change.

For local-only lookup set `ANALYTICS_GEO_MODE = "local"`:

`pricing/geo.py` supports local **DB-IP City Lite** or **MaxMind GeoLite2 City**
MMDB files. Both provide approximate IP geography; city may be wrong, especially
for VPNs and mobile networks. They do not locate a household or person.

- DB-IP: https://db-ip.com/db/download/ip-to-city-lite — monthly free download,
  CC BY 4.0, attribution link included in the privacy notice. Review license before
  downloading/redistributing. No database is bundled or silently downloaded.
- MaxMind: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data/ — free
  GeoLite2 City option with account/license requirements. Supply a legitimately
  downloaded file and keep it current according to its license.

Set `ANALYTICS_GEO_MMDB`, `ANALYTICS_GEO_PROVIDER` and
`ANALYTICS_TRUSTED_IP_HEADER`. **A trusted client IP must actually be provided by
your host.** Streamlit Community Cloud may not expose one suitable for this use;
the code deliberately does not invent one or geolocate the server's public IP.
If absent, city/region stay NULL. A separately configured trusted country header
can supply country only. A local lookup means no visitor IP is sent to DB-IP or
MaxMind; only coarse results go to Turso. Lat/lon, postal codes and ASN are discarded.
No IP-keyed lookup cache or IP logging is added. The host still processes the
connection and may have independent operational logs.

## Reliability, security and storage

Turso SQL-over-HTTP uses bound parameters, HTTPS, server-only authentication,
short request timeouts, no redirects and a 60-second failure cooldown. Each event
has an idempotent primary key. Schema creation is idempotent. Analytics failures
do not affect price estimates. Delivery is best effort, with no disk buffer or
background retry queue storing visitor information. DB authentication is never
passed to the JavaScript component. Use a separate database for analytics and
rotate its credential independently of all pricing resources.

Parquet is an export, not the event store. Install `pyarrow` for admin exports and
run `python -m scripts.analytics_admin export --output private-analytics`.
The two linked Parquet files are paginated at 1,000 rows, Zstd-compressed,
mode 0600, excluded from Git, and never served by the app. Exports are not a
transactional snapshot and normalize fields to strings for stable nullable chunks;
cast prices/dates when analysing. No automated export is published as a CI artifact.

Tests use local in-memory SQLite, mocked HTTP and a JavaScript consent harness. They validate schema/linkage,
consent, parameter binding, retention, deduplication, invalid URLs, geography
fallback and failure behavior. They are not proof of an authenticated live Turso
write or a legal compliance certification. Operator notices, processor terms and
lawful deployment configuration remain deployment responsibilities.

Protocol: https://docs.turso.tech/sdk/http/quickstart
Component: https://docs.streamlit.io/develop/concepts/custom-components/components-v2/communicate
