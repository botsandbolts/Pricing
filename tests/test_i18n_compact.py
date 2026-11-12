"""Regression: public quote, date and market controls have 36 explicit translations."""
import pytest
from pricing.i18n import LOCALES, CATALOGS
from pricing.i18n_price_short import COPY as PRICE, short
from pricing.i18n_clock import COPY as CLOCK, clock_text
from pricing.i18n_story import STORIES as STORY


def test_36_locale_short_copy_is_complete():
    assert len(LOCALES) == 36
    for catalog in (CATALOGS, PRICE, CLOCK, STORY):
        assert set(catalog) == set(LOCALES)
    for language in LOCALES:
        assert all(v.strip() for v in PRICE[language].values())
        assert all(v.strip() for v in CLOCK[language].values())


@pytest.mark.parametrize("code,ask,explain,today", [
    ("en-GB", "Seller asking", "Explain this price", "Today"),
    ("de", "Angebotspreis", "Preis erklären", "Heute"),
    ("fr", "Prix demandé", "Expliquer ce prix", "Aujourd’hui"),
    ("ru", "Цена объявления", "Объяснить цену", "Сегодня"),
])
def test_localized_primary_price_controls(code,ask,explain,today,monkeypatch):
    from pricing import i18n
    monkeypatch.setattr(i18n.st, "session_state", {"cps_locale":code})
    assert short("ask") == ask
    assert short("explain") == explain
    assert clock_text("today") == today


def test_public_form_preserves_canonical_numeric_inputs():
    from pathlib import Path
    code=(Path(__file__).resolve().parents[1]/"pricing/current_value.py").read_text()
    assert '("today","history","monthend")' in code
    assert 'format_func=clock_text' in code
    assert 'key=prefix+\'_allow_analogue_v2\'' in code
