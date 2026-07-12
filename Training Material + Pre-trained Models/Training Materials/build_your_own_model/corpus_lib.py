"""
corpus_lib.py — reusable machinery for building a tiny-LLM training corpus.

You do NOT need to edit this file. Your topic generator (see TEMPLATE.py)
imports it and calls Corpus() + qa_variants(...) etc. This holds the invariant
plumbing so every topic gets the same battle-tested behaviour:

  * ONE answer per question (first-write-wins) — a tiny model trained on the
    same question with two different answers learns to blend/hedge, so later
    conflicting duplicates are dropped and counted.
  * Whole-word special-token export for your entity names.
  * Deterministic shuffle + write, with a --out/--tokens-out/--seed CLI.
"""
import argparse
import random
from pathlib import Path


def list_join(items):
    """Human list joining: 'A' / 'A and B' / 'A, B, and C'."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return items[0] + " and " + items[1]
    return ", ".join(items[:-1]) + ", and " + items[-1]


class Corpus:
    """Collects Q&A pairs and prose blocks, then writes a training file."""

    def __init__(self):
        self.blocks = []          # each block is a list of text lines
        self._q_answer = {}        # question -> its (first) answer
        self.conflicts_dropped = 0

    def qa(self, question, answer):
        """Add one Q&A pair. Enforces one answer per question (first wins)."""
        q, a = question.strip(), answer.strip()
        if not q or not a:
            return
        prev = self._q_answer.get(q)
        if prev is not None:
            if prev != a:
                self.conflicts_dropped += 1
            return
        self._q_answer[q] = a
        self.blocks.append([f"Q: {q}", f"A: {a}"])

    def qa_variants(self, questions, answer):
        """Emit the SAME answer under many phrasings — this is how the model
        learns a fact no matter how it's asked. Provide DISTINCT wordings, not
        lowercase/punctuation duplicates of one phrasing (those just bloat the
        corpus; casing is handled at inference)."""
        for q in questions:
            self.qa(q, answer)

    def prose(self, text):
        """A standalone passage with no question — pure language exposure.
        Keep passages SHORT and factually dense; tiny models memorize tight
        text far better than long flowery prose, and short passages drift less."""
        t = text.strip()
        if t:
            self.blocks.append([t])

    def write(self, path, seed=1234):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        rng.shuffle(self.blocks)
        lines = []
        for b in self.blocks:
            lines.extend(b)
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        qa = sum(1 for b in self.blocks if len(b) == 2)
        return len(self.blocks), qa


def write_special_tokens(names, path):
    """Write one whole-word token per line (your entity names) so the tokenizer
    keeps each name intact instead of splitting it into fragments the model
    can't bind to. Pass the file to the trainer with --special-tokens."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, uniq = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    header = (
        "# Whole-word tokens — keep each of these intact in the tokenizer so a\n"
        "# name can't be garbled into partial fragments the model can't use.\n"
        "# Pass to the trainer with:  --special-tokens <this file>\n"
        "# One token per line; blank lines and # comments are ignored.\n\n"
    )
    path.write_text(header + "\n".join(uniq) + "\n", encoding="utf-8")
    return len(uniq)


def run(build, default_out="training_data/corpus.txt",
        default_tokens="training_data/special_tokens.txt"):
    """CLI entry point. `build(corpus)` fills the corpus and RETURNS the list of
    entity names to keep as whole-word tokens. Handles --out/--tokens-out/--seed
    and prints a summary."""
    ap = argparse.ArgumentParser(description="Generate a tiny-LLM training corpus")
    ap.add_argument("--out", type=Path, default=Path(default_out))
    ap.add_argument("--tokens-out", type=Path, default=Path(default_tokens))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    c = Corpus()
    names = build(c) or []
    n, qa = c.write(args.out, seed=args.seed)
    print(f"Wrote {args.out}")
    print(f"  blocks: {n}  (Q&A: {qa}, prose: {n - qa})")
    print(f"  conflicting-answer duplicates dropped: {c.conflicts_dropped}")
    if names:
        k = write_special_tokens(names, args.tokens_out)
        print(f"Wrote {args.tokens_out}  ({k} whole-word tokens)")
