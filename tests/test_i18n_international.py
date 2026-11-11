"""International language completeness, scripts and inference-safety tests."""
import re

import pytest
from pricing.i18n import (
    LOCALES, CATALOGS, PANEL_CATALOGS, COUNTRY_CATALOGS,
    browser_locale, localized_panel_for, localized_country_for,
    format_eur, VOICE,
)
from pricing.i18n_extra_core import EXTRA_LOCALES
from pricing.i18n_story import STORIES, A11Y
from pricing.i18n_fields import FIELDS
from pricing.i18n_enums import ENUMS, option_label_for
from pricing.i18n_research import SCIENCE
from pricing.i18n_science_legal import CATALOG as LEGAL
from pricing.i18n_consent import LABELS as CONSENT

REQUIRED = {
    "uk", "ru", "tr", "nb", "sr-Latn", "sr-Cyrl", "sq", "bs", "mk", "is", "ca"
}

def test_each_non_eu_variant_has_all_product_surfaces():
    assert set(EXTRA_LOCALES)==REQUIRED
    assert len(LOCALES)==36
    for code in REQUIRED:
        assert code in LOCALES
        for catalog in (CATALOGS,PANEL_CATALOGS,COUNTRY_CATALOGS,STORIES,
                        FIELDS,ENUMS,A11Y,SCIENCE,LEGAL,CONSENT):
            assert code in catalog, (code, catalog)
            assert all(value.strip() for value in catalog[code].values()), code
        assert all(VOICE[code])
        # A new UI language never creates or unlocks additional price markets.
        assert set(COUNTRY_CATALOGS[code])=={"de","fr","nl","be","at","es","it","pl"}


@pytest.mark.parametrize("raw,expected",[
    ("sr","sr-Cyrl"), ("sr-RS","sr-Cyrl"),
    ("sr-Latn","sr-Latn"), ("sr-Latn-RS","sr-Latn"),
    ("sr-Cyrl-BA","sr-Cyrl"), ("no-NO","nb"), ("nb-NO","nb"),
    ("uk-UA","uk"), ("ru-RU","ru"), ("tr-TR","tr"),
    ("sq-AL","sq"), ("bs-BA","bs"), ("mk-MK","mk"),
    ("is-IS","is"), ("ca-ES","ca"),
])
def test_bcp47_browser_language(raw, expected):
    assert browser_locale(raw)==expected


def test_distinct_serbian_scripts_preserve_canonical_values():
    assert re.search(r"[\u0400-\u04ff]", CATALOGS["sr-Cyrl"]["market_title"])
    assert not re.search(r"[\u0400-\u04ff]", CATALOGS["sr-Latn"]["market_title"])
    assert CATALOGS["sr-Cyrl"]["market_title"]!=CATALOGS["sr-Latn"]["market_title"]
    assert option_label_for("sr-Cyrl","diesel")!=option_label_for("sr-Latn","diesel")
    assert localized_panel_for("sr-Cyrl","🔬 The science").startswith("🔬 ")
    assert localized_panel_for("sr-Latn","🔬 The science").startswith("🔬 ")


def test_terminology_and_consents_are_local_not_english():
    assert "Çerez" in LEGAL["tr"]["privacy_title"] or "çerez" in LEGAL["tr"]["privacy_title"]
    assert CONSENT["tr"]["accept"] != CONSENT["tr"]["reject"]
    assert CONSENT["uk"]["accept"] != CONSENT["uk"]["reject"]
    assert "CVaR95" in SCIENCE["ru"]["evaluation"]
    assert "CVaR95" in SCIENCE["ca"]["evaluation"]
    assert "90" in LEGAL["nb"]["retention"]
    assert "90" in LEGAL["sr-Cyrl"]["retention"]
    assert "English" not in STORIES["ru"]["us_desc"]
    assert STORIES["ca"]["us_desc"]!=STORIES["es"]["us_desc"]
    assert VOICE["bs"]!=VOICE["sr-Latn"]


def test_no_translated_country_or_enum_enters_models():
    # Formatter changes UI only; caller retains the original enum/country code.
    assert localized_country_for("tr","de")=="Almanya"
    assert localized_country_for("uk","de")=="Німеччина"
    assert option_label_for("tr","petrol")=="Benzin"
    assert option_label_for("uk","petrol")=="Бензин"
    assert option_label_for("tr","__missing__")=="—"


def test_euro_display_is_not_exchange_conversion(monkeypatch):
    import pricing.i18n as i18n
    monkeypatch.setattr(i18n, "locale", lambda: "tr")
    assert format_eur(12345.6)=="12.346 €"
    monkeypatch.setattr(i18n, "locale", lambda: "uk")
    assert format_eur(12345.6)=="12\u202f346 €"
    monkeypatch.setattr(i18n, "locale", lambda: "sr-Cyrl")
    assert format_eur(12345.6)=="12.346 €"
