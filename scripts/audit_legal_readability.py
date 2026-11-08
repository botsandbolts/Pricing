"""Reproducible, dependency-free readability check for English legal notices.

Flesch Reading Ease and Flesch–Kincaid use estimated syllables; they are
approximations. Coleman–Liau instead uses letters and sentence length.
No reading metric can prove actual comprehension or GDPR compliance.
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
    """Engineering guardrails, not official legal thresholds or certification."""
    return {
        "flesch_kincaid_at_most_7_5": result["flesch_kincaid_grade"] <= 7.5,
        "flesch_ease_at_least_65": result["flesch_reading_ease"] >= 65,
        "coleman_liau_at_most_8": result["coleman_liau_grade"] <= 8,
        "mean_sentence_at_most_16_words": result["mean_words_per_sentence"] <= 16,
        "long_sentences_at_most_12_percent": result["long_sentence_share"] <= 0.12,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?",
        default=Path(__file__).resolve().parents[1] / "pricing" / "legal_privacy.md"
    )
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any internal young-reader target fails.")
    args = parser.parse_args()
    metrics = audit_markdown(args.path.read_text(encoding="utf-8"))
    targets = young_reader_targets(metrics)
    print(json.dumps({"file": str(args.path), "metrics": metrics,
                      "engineering_targets": targets, "all_targets_met": all(targets.values())},
                     indent=2))
    if args.strict and not all(targets.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
