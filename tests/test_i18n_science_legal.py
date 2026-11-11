"""The 36 supported language variants must have legal, science and consent copy.

The consent choice values and data fields must remain stable English tokens;
only displayed text changes with locale.
"""
from pricing.i18n import LOCALES
from pricing.i18n_science_legal import CATALOG, KEYS as LEGAL_KEYS
from pricing.i18n_research import SCIENCE, KEYS as RESEARCH_KEYS
from pricing.i18n_consent import LABELS, KEYS as MODAL_KEYS
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_every_supported_language_has_complete_information():
    assert len(LOCALES)==36
    assert set(CATALOG)==set(SCIENCE)==set(LABELS)==set(LOCALES)
    for code in LOCALES:
        for messages, keys in ((CATALOG[code],LEGAL_KEYS),
                               (SCIENCE[code],RESEARCH_KEYS),
                               (LABELS[code],MODAL_KEYS)):
            assert set(messages)==set(keys), code
            assert all(v.strip() for v in messages.values()), code


def test_dutch_law_and_science_copy_not_english():
    assert "Privacy en cookies" == CATALOG["nl"]["privacy_title"]
    assert "Weiger statistieken" == LABELS["nl"]["reject"]
    assert "MAE" in SCIENCE["nl"]["evaluation"]
    assert "CVaR95" in SCIENCE["nl"]["evaluation"]
    assert "90" in CATALOG["nl"]["retention"]


def test_reference_privacy_policy_remains_intact():
    policy=(ROOT/"pricing/legal_privacy.md").read_text()
    imprint=(ROOT/"pricing/legal_imprint.md").read_text()
    assert "GDPR Article 6(1)(a)" in policy
    assert "GDPR Article 6(1)(f)" in policy
    assert "90 days" in policy
    assert "Hamburg" in imprint
    assert "amitsteinberg74@gmail.com" in policy


def test_privacy_choices_not_translated_in_browser_storage():
    script=(ROOT/"pricing/consent/notice.js").read_text()
    assert "for (const choice of ['accepted','rejected'])" in script
    assert "setStateValue('consent',payload)" in script
    assert "data.labels" in script
    assert "node.textContent = t(key)" in script
    assert "consentChanged()" in script


def test_localized_law_uses_full_original_as_secondary_source():
    script=(ROOT/"pricing/legal_pages.py").read_text()
    assert "with st.expander" in script
    assert "_FILES[kind].read_text(encoding=\"utf-8\")" in script
    assert "kind == \"privacy\"" in script
