"""English privacy policy reading audit: short sentences + modern word-frequency NLP.

The web app uses only cheap built-in checks. CI uses textstat (multiple validated
formulas, including Dale-Chall) and wordfreq (corpus word familiarity).
Neither test replaces a real comprehension test with 12-year-old readers.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
_VOWEL = re.compile(r"[aeiouy]+")
_SENTENCE = re.compile(r"[.!?]+(?=\s|$)")


def plain_text(markdown: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", markdown)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+", "", text)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-*]\s+", "", text)
    text = re.sub(r"[\x60*_]", "", text)
    # A list marker with a digit is not part of the sentence being measured.
    text = re.sub(r"(?m)^\s*\d+[.]\s+", "", text)
    return text


def estimate_syllables(word: str) -> int:
    word = re.sub("[^a-z]", "", word.lower())
    if len(word) <= 3:
        return 1
    groups = len(_VOWEL.findall(word))
    if re.search(r"[^aeiou]e$", word) and not word.endswith(("le", "ye")):
        groups -= 1
    if re.search("ia|io|ua", word):
        groups += 1
    return max(1, groups)


def audit_markdown(markdown: str) -> dict[str, float | int]:
    """Cheap check safe for the memory-constrained Streamlit page."""
    text = plain_text(markdown)
    words = _WORD.findall(text)
    sentences = max(1, len(_SENTENCE.findall(text)))
    count = len(words)
    if not count:
        raise ValueError("The notice contains no English words.")

    syllables = sum(estimate_syllables(w) for w in words)
    letters = sum(len(re.sub("[^A-Za-z]", "", w)) for w in words)
    per_sentence = count / sentences
    per_word = syllables / count
    fragments = re.split(r"[.!?]+(?=\s|$)", text)
    lengths = [len(_WORD.findall(s)) for s in fragments if len(_WORD.findall(s)) > 3]

    return {
        "word_count": count,
        "sentence_count": sentences,
        "mean_words_per_sentence": round(per_sentence, 2),
        "flesch_reading_ease": round(206.835 - 1.015 * per_sentence - 84.6 * per_word, 2),
        "flesch_kincaid_grade": round(0.39 * per_sentence + 11.8 * per_word - 15.59, 2),
        "coleman_liau_grade": round(0.0588 * (letters * 100 / count)
                                    - 0.296 * (sentences * 100 / count) - 15.8, 2),
        "long_sentence_share": round(
            sum(1 for n in lengths if n > 20) / max(1, len(lengths)), 4),
        "longest_sentence_words": max(lengths, default=0),
    }


def young_reader_targets(result: dict) -> dict[str, bool]:
    """Grade 6 (~age 11–12) targets, NOT a legal or scientific guarantee."""
    return {
        "flesch_kincaid_at_most_grade_6": result["flesch_kincaid_grade"] <= 6,
        "flesch_ease_at_least_75": result["flesch_reading_ease"] >= 75,
        "coleman_liau_at_most_grade_6_5": result["coleman_liau_grade"] <= 6.5,
        "mean_sentence_at_most_12_words": result["mean_words_per_sentence"] <= 12,
        "sentences_over_20_words_at_most_5_percent": result["long_sentence_share"] <= 0.05,
    }


def advanced_audit(markdown: str) -> dict:
    """CI-only audit. Call with --advanced; never import word lists in the app."""
    try:
        import textstat
        from wordfreq import zipf_frequency
    except ImportError as error:
        raise RuntimeError(
            "Install CI-only tools: python -m pip install textstat wordfreq"
        ) from error
    text = plain_text(markdown)
    tokens = [w.lower() for w in _WORD.findall(text)]
    # A corpus-based check sees hard words that a syllable formula can miss.
    long_tokens = [w for w in tokens if len(w) >= 4]
    uncommon = [w for w in long_tokens if zipf_frequency(w, "en") < 3.5]
    rare_unique = sorted(set(uncommon), key=lambda w: (zipf_frequency(w, "en"), w))

    # textstat uses its own word and syllable methods.
    return {
        "package": "textstat + wordfreq",
        "textstat_flesch_ease": textstat.flesch_reading_ease(text),
        "textstat_flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "textstat_coleman_liau_grade": textstat.coleman_liau_index(text),
        "textstat_automated_readability_grade": textstat.automated_readability_index(text),
        # Raw legacy dictionary score, including brand names and legal terms.
        # Report it even if it disagrees with other measures.
        "textstat_dale_chall_score": textstat.dale_chall_readability_score(text),
        "textstat_grade_consensus": textstat.text_standard(text, float_output=True),
        "dale_chall_note": "This fixed older word list counts modern brands and legal terms as hard. Its full score is shown but is not an age-12 pass test.",
        "textstat_difficult_word_count": textstat.difficult_words(text),
        "wordfreq_uncommon_token_share": round(len(uncommon) / max(1, len(long_tokens)), 4),
        "wordfreq_uncommon_word_examples": rare_unique[:25],
    }


def advanced_targets(result: dict) -> dict[str, bool]:
    """Empirical QA guardrails, not a claim about an individual child's skill."""
    return {
        "textstat_flesch_grade_6_or_less": result["textstat_flesch_kincaid_grade"] <= 6,
        "textstat_reading_ease_75_or_more": result["textstat_flesch_ease"] >= 75,
        "textstat_coleman_liau_7_or_less": result["textstat_coleman_liau_grade"] <= 7,
        "textstat_ari_7_or_less": result["textstat_automated_readability_grade"] <= 7,
        "textstat_grade_consensus_6_or_less": result["textstat_grade_consensus"] <= 6,
        "wordfreq_uncommon_share_below_10_percent": result["wordfreq_uncommon_token_share"] <= .10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[1] /
                        "pricing" / "legal_privacy.md")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--advanced", action="store_true",
                        help="Use textstat and wordfreq in CI, not the live application.")
    args = parser.parse_args()
    notice = args.path.read_text(encoding="utf-8")
    metrics = audit_markdown(notice)
    targets = young_reader_targets(metrics)
    result = {"file": str(args.path), "age_goal": 12,
              "metrics": metrics, "engineering_targets": targets}
    if args.advanced:
        advanced = advanced_audit(notice)
        checks = advanced_targets(advanced)
        result["advanced_metrics"] = advanced
        result["advanced_targets"] = checks
        targets = {**targets, **checks}
    result["all_targets_met"] = all(targets.values())
    print(json.dumps(result, indent=2))
    if args.strict and not result["all_targets_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
