#!/usr/bin/env python3
"""
audit_token_coverage.py — catch query-key terms that fragment before they ship.

A term is USABLE on the device if EITHER of these tokenizes to a single token:
  - its bare form      ("Bulbasaur")  -> whole-name special token (presplit match)
  - its space-prefixed form (" Potion") -> BPE-learned in-context token
If BOTH forms fragment, questions that hinge on that term will retrieve garbage
on the ESP32 (the model never learned an atomic embedding for it), and the
firmware's vocab-aware casing pass has no whole token to map onto.

Terms are auto-extracted from the training corpus: every Capitalized word
appearing >= --min-count times (these are the entity names / items / places
users actually ask about), plus everything in --special-tokens if given.

Usage:
  # audit a trained model against its corpus
  python audit_token_coverage.py --model ../out_kanto_pokemon_master \\
      --corpus training_data/pokemon_kanto.txt

  # exit non-zero on any fragmenting term (CI/pre-flight gate)
  python audit_token_coverage.py --model DIR --corpus FILE --strict

The trainers run this same audit automatically right after the tokenizer is
built (before training starts), so a bad vocab fails in seconds, not hours.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path


def extract_terms(corpus_text, min_count=5):
    """Capitalized words (3+ chars) appearing min_count+ times, by frequency."""
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", corpus_text)
    counts = Counter(words)
    return [(w, c) for w, c in counts.most_common() if c >= min_count]


def audit(tokenize, terms, label="token coverage"):
    """tokenize: callable(str) -> list. terms: [(word, count)].
    Returns list of (word, count, bare_n, space_n) for fragmenting terms."""
    bad = []
    for w, c in terms:
        bare_n = len(tokenize(w))
        space_n = len(tokenize(" " + w))
        if bare_n != 1 and space_n != 1:
            bad.append((w, c, bare_n, space_n))
    print("=" * 64)
    print(f"TOKEN COVERAGE AUDIT ({label}): {len(terms)} terms checked")
    if not bad:
        print("  CLEAN — every term is a single token (bare or in-context).")
    else:
        print(f"  !! {len(bad)} term(s) FRAGMENT in both forms — questions about")
        print(f"  !! these will retrieve garbage on-device:")
        for w, c, bn, sn in bad:
            print(f"  !!   {w:<16} corpus x{c:<5} bare={bn} tok, ' {w}'={sn} tok")
        print("  !! Fix: add to --special-tokens (if a query key) or add more")
        print("  !! training text using the term so BPE learns it whole.")
    print("=" * 64)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="trained model dir (tokenizer.json)")
    ap.add_argument("--corpus", type=Path, required=True, help="training text to extract terms from")
    ap.add_argument("--special-tokens", type=Path, default=None, help="also audit every term in this file")
    ap.add_argument("--min-count", type=int, default=5, help="min corpus occurrences (default 5)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any term fragments")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = args.corpus.read_text(encoding="utf-8", errors="ignore")
    terms = extract_terms(text, args.min_count)
    if args.special_tokens and args.special_tokens.is_file():
        seen = {w for w, _ in terms}
        for line in args.special_tokens.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line not in seen:
                terms.append((line, 0))

    bad = audit(tok.tokenize, terms, label=str(args.model))
    sys.exit(1 if (bad and args.strict) else 0)


if __name__ == "__main__":
    main()
