"""Standalone legal pages. This route does not load car-pricing models."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_FILES = {
    "imprint": _ROOT / "legal_imprint.md",
    "privacy": _ROOT / "legal_privacy.md",
}


def render_legal_page(kind: str) -> None:
    import streamlit as st
    from pricing.studio_design import page_intro

    if kind not in _FILES:
        st.error("That legal page was not found.")
        return

    if kind == "imprint":
        page_intro("Legal information", "Imprint / Legal Notice",
                   "Independent research. A real person. A clear point of contact.", "🚗")
    else:
        page_intro("Your data, your choice", "Privacy Policy",
                   "The small print, with the small words. You stay in charge.", "💚")

    st.markdown(_FILES[kind].read_text(encoding="utf-8"))

    if kind == "privacy":
        with st.expander("How easy is this notice to read? · Open reading check"):
            from scripts.audit_legal_readability import audit_markdown
            check = audit_markdown(_FILES["privacy"].read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Reading grade (approx.)", f"{check['flesch_kincaid_grade']:.1f}")
            b.metric("Flesch Reading Ease (approx.)", f"{check['flesch_reading_ease']:.0f} / 100")
            c.metric("Coleman–Liau grade", f"{check['coleman_liau_grade']:.1f}")
            st.caption(
                f"Average sentence length: {check['mean_words_per_sentence']:.1f} words. "
                f"Sentences longer than 20 words: {check['long_sentence_share']:.0%}. "
                "Target: grade 6 or easier (about age 12). These scores are rough. "
                "The full check also uses textstat and wordfreq to spot hard words. "
                "They run in GitHub tests, not on the live site. "
                "No computer score can prove that every child will understand this page."
            )
            st.markdown(
                "[See the age-12 reading checks and results]"
                "(https://github.com/botsandbolts/Pricing/actions/workflows/legal-pages-qa.yml)"
                " · [View the test code]"
                "(https://github.com/botsandbolts/Pricing/blob/main/scripts/audit_legal_readability.py)"
            )

    st.markdown("[← Return to the price calculator](./)")
