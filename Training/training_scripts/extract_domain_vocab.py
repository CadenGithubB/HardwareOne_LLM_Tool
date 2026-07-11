#!/usr/bin/env python3
"""
extract_domain_vocab.py — build a domain allow-list for the on-device refusal gate.

The ESP32 LLM can refuse prompts that fall outside a model's topic: at generation
time it splits the prompt into words and, if NONE of them appear in the model's
embedded "domain vocab", it emits a fixed refusal instead of generating. This
script derives that vocab straight from the training corpus, so the list always
matches what the model was actually taught.

The output is one lowercase word per line. Feed it to the converter's
"Domain words" input; it gets baked into model.bin next to the description/icon.

CRITICAL: the on-device matcher splits on non-alphanumeric characters and folds
A-Z to lowercase (whole-word, case-insensitive). This script tokenizes the SAME
way, so a corpus word and its on-device match are byte-identical. Multi-word or
punctuated names ("Mr. Mime", "Farfetch'd") are split into their component words
exactly as the device would see them.

Pipeline:
  1. tokenize the corpus (lowercase, split on [^a-z0-9], drop all-digit tokens)
  2. drop stopwords (function words like the/is/was/a/or — the common-word concern)
  3. keep words seen >= --min-count times and >= --min-len chars
  4. force-include every word from --special-tokens (names are domain by definition)
  5. flag (or with --drop-common, remove) words that are also common English —
     the "gray zone" like Pokemon TYPES (water/fire/normal) that are real domain
     terms AND everyday words. Flagged-and-kept by default so a legit typed
     question ("what fire types beat grass") is never wrongly refused.

Usage:
  python extract_domain_vocab.py corpus.txt [corpus2.txt ...] \
      --special-tokens special_tokens.txt \
      --out domain_vocab.txt
  # review the printed report, then hand domain_vocab.txt to the converter.
"""
import argparse
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
WORDLIST_DIR = os.path.join(HERE, "wordlists")

# Matches the device's isalnum word boundary after lowercasing.
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase + split exactly like the on-device gate (isalnum words)."""
    return _WORD_RE.findall(text.lower())


def load_wordset(path):
    """One entry per line; '#' comments and blank lines ignored; lowercased."""
    words = set()
    if not path or not os.path.exists(path):
        return words
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.lower())
    return words


def load_special_words(path, min_len):
    """Force-include list: each name tokenized the same way the device would."""
    out = set()
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in tokenize(line):
                if len(tok) >= min_len and not tok.isdigit():
                    out.add(tok)
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract an on-device domain allow-list from a training corpus.")
    ap.add_argument("corpus", nargs="+", help="training corpus .txt file(s) (Q:/A:/prose)")
    ap.add_argument("--special-tokens", help="whole-word names to force-include (e.g. the model's --special-tokens file)")
    ap.add_argument("--stopwords", default=os.path.join(WORDLIST_DIR, "stopwords_en.txt"),
                    help="function words to always exclude (default: bundled English stoplist)")
    ap.add_argument("--common", default=os.path.join(WORDLIST_DIR, "common_en.txt"),
                    help="everyday English words used to FLAG the domain-vs-common gray zone (default: bundled)")
    ap.add_argument("--min-count", type=int, default=2, help="minimum corpus occurrences to keep a word (default: 2)")
    ap.add_argument("--min-len", type=int, default=3, help="minimum word length to keep (default: 3)")
    ap.add_argument("--max-words", type=int, default=600, help="cap on total words emitted (default: 600)")
    ap.add_argument("--drop-common", action="store_true",
                    help="REMOVE gray-zone common words (stricter gate; may false-refuse questions phrased with only common words). Default keeps + flags them.")
    ap.add_argument("--out", default="domain_vocab.txt", help="output word list (default: domain_vocab.txt)")
    args = ap.parse_args()

    stopwords = load_wordset(args.stopwords)
    common = load_wordset(args.common)
    special = load_special_words(args.special_tokens, args.min_len)

    if not stopwords:
        print(f"[warn] no stopwords loaded from {args.stopwords} — common function words may leak into the list!",
              file=sys.stderr)

    # 1-2-3: count corpus words, drop stopwords / short / all-digit.
    counts = Counter()
    total_tokens = 0
    for path in args.corpus:
        if not os.path.exists(path):
            print(f"[error] corpus not found: {path}", file=sys.stderr)
            return 2
        with open(path, encoding="utf-8") as fh:
            for tok in tokenize(fh.read()):
                total_tokens += 1
                if tok.isdigit() or len(tok) < args.min_len or tok in stopwords:
                    continue
                counts[tok] += 1

    kept = {w for w, c in counts.items() if c >= args.min_count}

    # 4: force-include special-token words (bypass min-count; still lowercase/min-len).
    kept |= special

    # 5: gray zone — words that are both domain-frequent and everyday English.
    flagged = sorted(w for w in kept if w in common)
    if args.drop_common:
        kept -= set(flagged)

    # Rank by corpus frequency (special-only words get a small nominal weight so
    # they survive the cap), then apply the max-words cap.
    ranked = sorted(kept, key=lambda w: (-counts.get(w, 1), w))
    if len(ranked) > args.max_words:
        dropped_by_cap = ranked[args.max_words:]
        ranked = ranked[: args.max_words]
    else:
        dropped_by_cap = []

    out_words = sorted(ranked)  # alphabetical for a human-readable file
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# Domain allow-list for the on-device refusal gate.\n")
        fh.write("# One lowercase word per line. Feed to the converter's 'Domain words' input.\n")
        fh.write(f"# {len(out_words)} words from {', '.join(os.path.basename(p) for p in args.corpus)}\n")
        for w in out_words:
            fh.write(w + "\n")

    # ---- Review report (the eyeball checkpoint) ----
    n_special_only = sum(1 for w in out_words if w in special and counts.get(w, 0) < args.min_count)
    top = sorted(((counts.get(w, 0), w) for w in out_words), reverse=True)[:25]
    print("=" * 64)
    print(f"DOMAIN VOCAB: {len(out_words)} words -> {args.out}")
    print(f"  corpus tokens scanned: {total_tokens:,}   unique kept: {len(out_words)}")
    print(f"  force-included names (below min-count): {n_special_only}")
    print(f"  stopwords excluded: {len(stopwords)}  (is/was/the/a/or... never enter the list)")
    if dropped_by_cap:
        print(f"  dropped by --max-words cap ({args.max_words}): {len(dropped_by_cap)} least-frequent words")
    print("-" * 64)
    print("TOP 25 by corpus frequency (should look on-topic):")
    for c, w in top:
        print(f"    {c:6d}  {w}")
    print("-" * 64)
    if flagged:
        verb = "REMOVED (--drop-common)" if args.drop_common else "KEPT — REVIEW THESE"
        print(f"GRAY ZONE — domain words that are also everyday English [{verb}]:")
        print("  (e.g. Pokemon TYPES. Kept => 'I like water' may slip through the gate,")
        print("   but a typed 'what fire types...' is never wrongly refused. Remove any")
        print("   you consider purely generic, or re-run with --drop-common for a stricter gate.)")
        print("   " + ", ".join(flagged))
    else:
        print("GRAY ZONE: none flagged (no kept word appears in the common-English list).")
    print("=" * 64)
    print("Sanity check: confirm no function words (is/was/the/a/or/what/who) are in the")
    print(f"list above, then hand {args.out} to the converter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
