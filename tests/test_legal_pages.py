"""Ensure legal pages stay accessible and readable with no ML dependencies."""
from pathlib import Path
import unittest

from scripts.audit_legal_readability import audit_markdown, young_reader_targets

ROOT = Path(__file__).resolve().parents[1]


class LegalPagesTests(unittest.TestCase):
    def test_plain_english_reading_targets(self):
        notice = (ROOT / "pricing" / "legal_privacy.md").read_text(encoding="utf-8")
        score = audit_markdown(notice)
        self.assertGreater(score["word_count"], 700)
        self.assertTrue(all(young_reader_targets(score).values()), score)

    def test_material_disclosures(self):
        notice = (ROOT / "pricing" / "legal_privacy.md").read_text(encoding="utf-8")
        for term in ("Amit Steinberg", "amitsteinberg74@gmail.com", "Hamburg",
                     "Streamlit", "Turso", "GeoJS", "Google Analytics",
                     "90 days", "consent", "GDPR", "outside the European",
                     "complain", "delete", "sale report"):
            self.assertIn(term, notice)

    def test_footer_has_two_real_page_links(self):
        source = (ROOT / "pricing" / "studio_design.py").read_text(encoding="utf-8")
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('?legal=imprint', source)
        self.assertIn('?legal=privacy', source)
        self.assertIn("render_legal_page", app)
        self.assertIn("st.stop()", app)


if __name__ == "__main__":
    unittest.main()
