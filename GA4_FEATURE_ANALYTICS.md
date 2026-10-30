# Car Price Studio: feature analytics

Google Analytics measurement ID: `G-YT533T36L8` (existing property).

## What the release collects

Only after accepting the revised analytics notice:

| Event | Meaning | Parameters |
|---|---|---|
| panel_view | Main panel or supported nested tab opened | panel_name, subpanel_name |
| panel_engagement | Seconds with the panel visible, flushed on navigation, hiding and every 30 seconds | panel_name, subpanel_name, duration_seconds |
| estimate_requested | A recognized estimate button clicked | action_name |
| estimate_completed | Valid positive estimate produced in the active panel | market_side, feature_name |
| explanation_requested | Build explanation graphs clicked | action_name |
| explanation_opened | An expandable section opened/clicked | action_name |
| scenario_requested | Build timeline clicked | action_name |
| setting_changed | A recognized field committed a change; never its value | action_name |
| model_selected | Recognized model-family option clicked | model_family |
| share_opened | Share panel requested | action_name |
| share | Recognized social-link click | method, content_type |
| download_requested | Recognized download button clicked | action_name |
| import_requested / import_applied | Advert import interaction | action_name |
| shortlist_action | Add, remove, inspect, compare or timeline | action_name |
| game_action | Reveal guess or next practice car | action_name |
| feedback_action | Feedback submission/deletion requested | action_name |
| navigation_action | Recognized navigation shortcut clicked | action_name |

Events also carry current main/subpanel names. No seller descriptions, input values,
vehicle identifiers, prices, URL queries, link URLs, filenames or Turso visitor IDs
are included in these custom events. Fixed label maps produce bounded categories.
Changing a field counts only the field category. It does not record keystrokes.

`requested` means interaction, not success. A share click does not prove a post
was published; a download click does not prove a file was saved. Feedback actions
do not establish that the write/deletion succeeded. Visible-panel time includes
idle visible time, not verified attention; background tabs are excluded. A person
closing the browser can lose the last partial interval. Ad blockers, consent,
network failures and abrupt exits cause undercounting. There is no guarantee of
complete delivery or identification of a visitor/company.

## GA interface setup (property owner/editor)

1. Open **Reports → Realtime**, accept analytics in the studio and navigate its
   panels. Look for `panel_view`, `estimate_requested`, `estimate_completed`, etc.
2. In **Admin → Data display → Custom definitions**, add event-scoped dimensions:
   `panel_name`, `subpanel_name`, `action_name`, `feature_name`, `market_side`,
   `model_family`. `method` and `content_type` can use existing GA dimensions where
   available; avoid duplicate definitions.
3. Add custom metric `duration_seconds`, scope Event, unit Seconds.
4. Build **Explore → Free form**: rows Panel name, Subpanel name, Event name;
   values Event count, Total users and the custom duration metric. Filter engagement
   to `panel_engagement` when summing visible time. Use Funnel exploration for
   requested → completed → share; filter the relevant feature/panel.
5. New custom dimensions may take up to 48 hours to populate reporting. They do not
   reconstruct old activity. Realtime can inspect incoming event parameters sooner.

GA administration is separate from a source deployment: this release does not claim
that property custom definitions were automatically registered. No GA admin
credentials or Measurement Protocol secret are embedded in the repository.

## Implementation and verification

- `consent/notice.js`: one global delegated listener set, finite maps, consent
  gating, visible-panel heartbeat and nested tabs; no third-party recorder.
- `ga_events.py` + `consent/events.js`: consent-gated, bounded completion queue;
  stable request IDs deduplicate rerenders without transmitting those IDs. Passive
  computations from hidden Streamlit panels are excluded. Best-effort transport.
- Consent version v3 re-asks existing visitors for the newly described interactions.
  Withdrawal stops new custom events. The separate Turso price logging is unchanged.
- Tests: `node tests/ga_interactions.cjs` and
  `python -m pytest tests/test_ga_events.py tests/test_analytics.py -q`.

GA itself can collect standard automatic/enhanced events according to property
settings; this document describes the additional explicit app events. Review
Enhanced Measurement settings separately if enabling outbound URL/form tracking.


## 30 October 2026 correction (tracking v4)

All eight main panels now map to stable names, including `outlier_lab`. Each actual main/nested tab transition emits `panel_view` plus a virtual GA `page_view`: title `Car Price Studio | evaluation | none`, location `/panels/evaluation`. These are analytics-only paths, not real app routes. This makes panels visible in Pages and screens and page paths without first creating custom definitions. The initial generic site page view is still recorded once. Re-clicking the current tab and Streamlit rerenders do not count as navigation. Mouse, touch click activation, keyboard and selected-tab mutations are supported.

Named expanders emit `section_toggled` with bounded `action_name` and string `expanded=true/false`; this records toggle intent, not reading. New phrase-driver, outlier-download, date-clock, generation-analogue, radio and sharing controls are mapped. Change events contain only the field category, never entered text or option text. Duplicate click/change callbacks within 500 ms are suppressed. Unknown labels are ignored rather than uploaded.

Visible-panel engagement also sets GA's `engagement_time_msec`. The last queued safe event is available on the app document's `data-pricing-ga-last-event` attribute for DOM testing. This is a browser queue diagnostic, not proof of Google's receipt. No raw input is put in that attribute. The tag status displays tracking v4 after the Google script loads.

### Where the owner sees this

- Reports → Engagement → Pages and screens: choose Page title and screen class or Page path and screen class. Look for `Car Price Studio | …` or `/panels/…`.
- Reports → Realtime: event names and incoming event parameters.
- Admin → Custom definitions: Event-scoped `panel_name`, `subpanel_name`, `action_name`, `market_side`, `feature_name`, `model_family`, `expanded`, `tracking_version`. Create only missing definitions. Optional custom metric `duration_seconds`, unit seconds.
- Explore → Free form: rows `panel_name`, `subpanel_name`, `action_name`, Event name; values Event count and Total users. Filter `section_toggled` / `expanded=true` for opens. Use `estimate_requested` vs `estimate_completed` for start/success, not generic clicks alone.

Custom definitions require access to the GA property and were not registered by this code deployment. Previously missed activity cannot be recovered. Consented and unblocked visitors only; GA is not a recording of every person's session. A loaded tag or queued event does not certify ingestion or report availability.
