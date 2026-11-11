"""Static regression tests: all EU languages, English variants and safe fallbacks."""
from pricing.i18n import LOCALES, CATALOGS, KEYS, VOICE, PANEL_CATALOGS, _PANEL_KEYS, COUNTRY_CATALOGS, _COUNTRY_KEYS, browser_locale, format_eur

OFFICIAL_EU = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "ga", "it", "lv", "lt", "mt", "pl", "pt", "ro",
    "sk", "sl", "es", "sv",
}

def test_all_24_official_eu_languages_plus_english_variants():
    assert {k.split("-")[0] for k in LOCALES} == OFFICIAL_EU
    assert set(LOCALES) == set(CATALOGS) == set(VOICE) == set(PANEL_CATALOGS) == set(COUNTRY_CATALOGS)
    assert "en-GB" in LOCALES and "en-US" in LOCALES
    assert len(LOCALES) == 25

def test_core_catalog_has_no_missing_or_blank_translations():
    for code, messages in CATALOGS.items():
        assert set(messages) == set(KEYS), code
        assert all(v.strip() for v in messages.values()), code
        assert all(v.strip() for v in VOICE[code]), code
        assert set(PANEL_CATALOGS[code]) == set(_PANEL_KEYS), code
        assert all(v.strip() for v in PANEL_CATALOGS[code].values()), code
        assert set(COUNTRY_CATALOGS[code]) == set(_COUNTRY_KEYS), code

def test_browser_locale_selection():
    cases = {
        "pl-PL": "pl", "es-ES": "es", "fr-FR": "fr",
        "pt-BR": "pt", "nl-NL": "nl", "ga-IE": "ga",
        "en-US": "en-US", "en-GB": "en-GB", "en-IE": "en-GB",
        "de-AT": "de", "EN_us": "en-US", "zz-ZZ": "en-GB",
        None: "en-GB", "": "en-GB",
    }
    for value, expected in cases.items():
        assert browser_locale(value) == expected, value

def test_editorial_tone_is_not_identical_across_markets():
    assert VOICE["pl"] != VOICE["es"]
    assert VOICE["en-US"] != VOICE["en-GB"]
    assert "Szwagier" in VOICE["pl"][0]
    assert "cuñado" in VOICE["es"][0]


def test_euro_formatting_does_not_convert_values(monkeypatch):
    import pricing.i18n as i18n
    monkeypatch.setattr(i18n, "locale", lambda: "en-GB")
    assert format_eur(12345.6) == "€12,346"
    monkeypatch.setattr(i18n, "locale", lambda: "de")
    assert format_eur(12345.6) == "12.346 €"
    monkeypatch.setattr(i18n, "locale", lambda: "fr")
    assert format_eur(12345.6) == "12\u202f346 €"
