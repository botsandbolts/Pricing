"""Read-only GLiNER view of the seller's description."""
from html import escape
from threading import Lock

import streamlit as st


MODEL = "fastino/gliner2.5-base-v1"
SCHEMA = {
    "vehicle": [
        "make::str", "model::str", "trim::str", "year::str", "mileage::str",
        "engine::str", "transmission::str", "fuel::str", "drivetrain::str",
        "body::str", "colour::str", "equipment", "ownership_history",
        "title_status::str", "service_history", "accident_history::str",
    ],
    "repair": ["component::str", "action::str", "time::str", "mileage::str", "evidence::str"],
    "fault": ["component::str", "condition::str", "negation_or_uncertainty::str", "evidence::str"],
}
_LOCK = Lock()
_STATE = "_seller_description_scan"


def extract(description):
    """Load on demand and serialize inference across sessions to limit peak memory."""
    from gliner2 import AutoExtractor

    with _LOCK:
        model = AutoExtractor.from_pretrained(MODEL, map_location="cpu")
        try:
            return model.extract_json(description, SCHEMA)
        finally:
            del model


def _records(data, section):
    records = data.get(section, [])
    if isinstance(records, dict):
        return [records]
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _source_values(record, key, source):
    values = record.get(key, [])
    if not isinstance(values, list):
        values = [values]
    grounded = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("text")
        if isinstance(value, str):
            value = value.strip()
            if value and value.casefold() in source and value not in grounded:
                grounded.append(value)
    return grounded


def sections(data, description):
    """Order source-backed claims for reading. Empty groups remain unknown."""
    source = description.casefold()
    vehicle = _records(data, "vehicle")

    def vehicle_fields(fields):
        rows = []
        for label, key in fields:
            values = list(dict.fromkeys(
                value for record in vehicle for value in _source_values(record, key, source)
            ))
            if values:
                rows.append((label, " · ".join(values)))
        return [rows] if rows else []

    def event_fields(section, fields):
        events = []
        for record in _records(data, section):
            rows = [(label, " · ".join(values)) for label, key in fields
                    if (values := _source_values(record, key, source))]
            if rows and rows not in events:
                events.append(rows)
        return events

    maintenance = vehicle_fields([("Service history", "service_history")])
    maintenance += event_fields("repair", [
        ("Component", "component"), ("Work mentioned", "action"),
        ("When", "time"), ("Mileage at service", "mileage"), ("Seller’s wording", "evidence"),
    ])
    return [
        ("Vehicle details", vehicle_fields([
            ("Make", "make"), ("Model", "model"), ("Trim", "trim"), ("Year", "year"),
            ("Mileage", "mileage"), ("Body", "body"), ("Paint", "colour"),
            ("Engine", "engine"), ("Fuel", "fuel"), ("Transmission", "transmission"),
            ("Drivetrain", "drivetrain"),
        ])),
        ("Title & history", vehicle_fields([
            ("Title", "title_status"), ("Accident history", "accident_history"),
            ("Ownership", "ownership_history"),
        ])),
        ("Condition claims", event_fields("fault", [
            ("Component", "component"), ("Condition", "condition"),
            ("Context", "negation_or_uncertainty"), ("Seller’s wording", "evidence"),
        ])),
        ("Maintenance & repairs", maintenance),
        ("Equipment", vehicle_fields([("Equipment mentioned", "equipment")])),
    ]


def _cards(groups):
    style = """
    <style>
    .advert-scan {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%, 16rem),1fr));gap:.8rem}
    .advert-scan article {border:1px solid #48788b;border-top:3px solid #51e7d1;
      border-radius:1rem;padding:1rem;background:linear-gradient(145deg,#102338,#17283f);
      color:#f0f9ff;min-width:0;overflow-wrap:anywhere}
    .advert-scan h4 {color:#a9ffec;margin:0 0 .55rem;padding:0;font-size:1.04rem}
    .advert-scan dl {margin:.45rem 0;padding:.5rem .65rem;border-left:2px solid #5598af;background:#0e2033}
    .advert-scan dt {color:#b2d6eb;font-size:.86rem;font-weight:650}
    .advert-scan dd {margin:0 0 .35rem;color:#f0f9ff}
    .advert-scan p {margin:0;color:#c5d7e6}
    </style>
    """
    cards = []
    for title, events in groups:
        rows = "".join(
            "<dl>" + "".join(f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>"
                             for label, value in event) + "</dl>"
            for event in events
        )
        cards.append(f"<article><h4>{escape(title)}</h4>{rows or '<p>No details extracted</p>'}</article>")
    return style + '<div class="advert-scan">' + "".join(cards) + "</div>"


def render(description, requested):
    if requested:
        st.session_state.pop(_STATE, None)
        if not description.strip():
            st.info("Add a seller description to scan it.")
            return
        with st.spinner("Reading the seller’s description…"):
            try:
                result = extract(description)
                groups = sections(result, description)
            except Exception:
                st.warning("The description scan couldn’t finish. Try Scan description again.")
                return
        st.session_state[_STATE] = (description, groups)

    saved = st.session_state.get(_STATE)
    if saved is None or saved[0] != description:
        return
    groups = saved[1]
    populated = sum(bool(events) for _, events in groups)
    st.subheader("Description scan")
    st.caption(
        f"{populated} of {len(groups)} categories with extracted details · "
        "Unverified seller claims from the last scanned description. Unmentioned details are unknown."
    )
    st.markdown(_cards(groups), unsafe_allow_html=True)
