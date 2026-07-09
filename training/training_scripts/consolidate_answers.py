#!/usr/bin/env python3
"""Reduce ANSWER DRIFT in a hand-written Q&A corpus.

Answer drift = the same fact written several slightly-different ways. A tiny
model can't memorize a wobbling target, so it averages the variants into mush
("confidently muddled"). The cure is canonicalization: keep every QUESTION
(question diversity helps), but make every paraphrase point at ONE byte-identical
answer so the reinforcement lands on a single string.

This tool is deliberately conservative. It splits the work into two buckets:

  1. SAFE auto-merge (applied with --apply): answers that are identical EXCEPT
     for a leading command verb (Type/Run/Use). e.g.
         "Run wifiadd ... openwifi to connect."
         "Use wifiadd ... openwifi to connect."
     both become
         "Type wifiadd ... openwifi to connect."
     The remainder is byte-identical, so meaning is provably preserved.

  2. REVIEW clusters (reported, NEVER auto-applied): near-duplicate answers that
     differ by CONTENT (an added trailing clause, "current", a different intent).
     Merging these needs human judgment — some carry extra facts that should
     become their own Q&A pair rather than be dropped. The tool proposes a
     canonical for each cluster; you decide.

It also flags Yes/No POLARITY CONTRADICTIONS (the same statement attached to both
"Yes." and "No."), which are bugs, not drift.

Canonicalization happens at the LINE level: only "A: ..." lines are rewritten.
Q:, Do:, prose lines and block boundaries are left untouched, so multi-turn and
prose blocks survive intact.

Usage:
    # Report only (no files written): what would merge, what needs review
    python training_scripts/consolidate_answers.py [FILE]

    # Apply ONLY the safe merges to a new file + write the review report
    python training_scripts/consolidate_answers.py [FILE] \
        --apply training_data/hardwareone_rich.consolidated.txt \
        --review training_data/consolidation_review.txt

Defaults to hardwareone_rich.txt in the sibling training_data/ directory.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

VERBS = ("Type", "Run", "Use")          # interchangeable command-entry verbs
CANONICAL_VERB = "Type"
REVIEW_THRESHOLD = 0.85                  # word-overlap to land in a review cluster


def word_overlap(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def norm_ws(s: str) -> str:
    """Collapse internal whitespace and strip ends — never changes wording."""
    return re.sub(r"\s+", " ", s).strip()


def verb_key(ans: str) -> "tuple[bool, str]":
    """Return (had_verb, remainder).

    If the answer starts with an interchangeable command verb, the remainder is
    the text after it; answers sharing a remainder mean the same thing. If not,
    the 'remainder' is the whole normalized answer (so it only groups with an
    exact match).
    """
    a = norm_ws(ans)
    for v in VERBS:
        if a.startswith(v + " "):
            return True, a[len(v) + 1:]
    return False, a


def collect_answers(lines: list[str]) -> "dict[str, int]":
    counts: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if s.startswith("A: "):
            ans = s[3:]
            counts[ans] = counts.get(ans, 0) + 1
    return counts


def build_safe_merge(counts: "dict[str, int]") -> "dict[str, str]":
    """Map each drifted answer -> canonical, for verb-only differences.

    Groups answers by their post-verb remainder. A group with >1 distinct answer
    differs only by the leading verb (or whitespace), so it is safe to collapse.
    Canonical preference: an existing 'Type ...' form, else the most frequent,
    tie-broken by shortest.
    """
    groups: dict[str, list[str]] = {}
    for ans in counts:
        had_verb, rem = verb_key(ans)
        # Only verb-led answers are eligible to merge across verbs. Non-verb
        # answers key on their full normalized text (exact-dup whitespace merge).
        key = ("V:" + rem) if had_verb else ("X:" + rem)
        groups.setdefault(key, []).append(ans)

    mapping: dict[str, str] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue  # nothing to merge

        def score(a: str) -> tuple:
            starts_type = norm_ws(a).startswith(CANONICAL_VERB + " ")
            return (starts_type, counts[a], -len(a))  # prefer Type, then frequent, then short

        canonical = max(members, key=score)
        # If the chosen canonical is verb-led, rewrite its verb to the canonical one.
        had_verb, rem = verb_key(canonical)
        canonical_text = f"{CANONICAL_VERB} {rem}" if had_verb else norm_ws(canonical)
        for m in members:
            if m != canonical_text:
                mapping[m] = canonical_text
    return mapping


def cluster_review(distinct: list[str], counts: "dict[str, int]",
                   threshold: float) -> list[list[str]]:
    """Connected-component clusters of near-duplicate answers (content diffs)."""
    parent = {a: a for a in distinct}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i, a in enumerate(distinct):
        for b in distinct[i + 1:]:
            if a != b and word_overlap(a, b) >= threshold:
                union(a, b)

    comps: dict[str, list[str]] = {}
    for a in distinct:
        comps.setdefault(find(a), []).append(a)
    clusters = [sorted(m, key=lambda x: (-counts[x], len(x))) for m in comps.values() if len(m) > 1]
    clusters.sort(key=lambda c: -sum(counts[x] for x in c))
    return clusters


def split_lead(ans: str) -> "tuple[str, str]":
    """Split a leading 'Yes.'/'No.' polarity word from the body.

    The lead is question-dependent and correct ("Do I need a phone?" -> "No."),
    so canonicalization must preserve it and only converge the body.
    """
    a = norm_ws(ans)
    m = re.match(r"^(Yes|No)\.\s+(.*)$", a)
    if m:
        return m.group(1) + ".", m.group(2)
    return "", a


def build_trim_map(stage1_counts: "dict[str, int]") -> "tuple[dict[str, str], dict[str, str]]":
    """Trailing-clause trimming on lead-stripped bodies.

    If body B == body X + appended sentence(s) (X is a sentence-complete prefix
    that is itself an attested standalone body), map B's answer down to X. This
    converges "core + extra clause" onto the short core WITHOUT ever merging two
    differently-structured answers (a definition never starts with a how-to), so
    intent is preserved. Returns (answer_map, reason_map).
    """
    # Aggregate counts by lead-stripped body.
    body_count: dict[str, int] = {}
    bodies_by_answer: dict[str, tuple[str, str]] = {}
    for ans, c in stage1_counts.items():
        lead, body = split_lead(ans)
        bodies_by_answer[ans] = (lead, body)
        body_count[body] = body_count.get(body, 0) + c

    bodies = list(body_count)

    # For each body, find the shortest attested body it strictly extends.
    trim: dict[str, str] = {}
    for B in bodies:
        best = None
        for X in bodies:
            if X == B or len(X) >= len(B) or len(X.split()) < 3:
                continue
            if X.endswith(".") and B.startswith(X) and B[len(X)] == " ":
                if best is None or len(X) < len(best):
                    best = X
        if best is not None:
            trim[B] = best

    def resolve(b: str) -> str:
        seen = set()
        while b in trim and b not in seen:
            seen.add(b)
            b = trim[b]
        return b

    answer_map: dict[str, str] = {}
    reason_map: dict[str, str] = {}
    for ans, (lead, body) in bodies_by_answer.items():
        core = resolve(body)
        if core != body:
            new_ans = f"{lead} {core}" if lead else core
            if new_ans != ans:
                answer_map[ans] = new_ans
                reason_map[ans] = "trim trailing clause"
    return answer_map, reason_map


def find_divergent_bodies(stage1_counts: "dict[str, int]",
                          threshold: float) -> "list[list[str]]":
    """Flag (do NOT change) near-duplicate bodies that diverge in content.

    These share most of their words but neither is a prefix of the other (e.g.
    ESP-NOW: '...no infrastructure.' vs '...up to 200 metres.'). Choosing which
    fact to keep is an editorial call, so they are reported, not merged.
    """
    body_count: dict[str, int] = {}
    for ans, c in stage1_counts.items():
        _, body = split_lead(ans)
        body_count[body] = body_count.get(body, 0) + c
    bodies = sorted(body_count)

    # union-find clusters
    parent = {b: b for b in bodies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            if word_overlap(a, b) >= threshold and not (b.startswith(a) or a.startswith(b)):
                parent[find(a)] = find(b)

    comps: dict[str, list[str]] = {}
    for b in bodies:
        comps.setdefault(find(b), []).append(b)
    clusters = [sorted(m, key=lambda x: (-body_count[x], len(x)))
                for m in comps.values() if len(m) > 1]
    clusters.sort(key=lambda c: -sum(body_count[x] for x in c))
    return clusters


def find_polarity_contradictions(counts: "dict[str, int]") -> list[tuple[str, str]]:
    """Answers where the same body is attached to both Yes. and No."."""
    bodies: dict[str, dict[str, str]] = {}  # body -> {polarity: full_answer}
    for ans in counts:
        m = re.match(r"^(Yes|No)\.\s+(.*)$", ans)
        if m:
            pol, body = m.group(1), norm_ws(m.group(2))
            bodies.setdefault(body, {})[pol] = ans
    out = []
    for body, pols in bodies.items():
        if "Yes" in pols and "No" in pols:
            out.append((pols["Yes"], pols["No"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Reduce answer drift in a Q&A corpus.")
    default = Path(__file__).resolve().parent.parent / "training_data" / "hardwareone_rich.txt"
    p.add_argument("file", nargs="?", default=str(default), help="corpus .txt")
    p.add_argument("--apply", metavar="OUT", default=None,
                   help="write corpus with merges applied to OUT (new file)")
    p.add_argument("--changelog", metavar="OUT", default=None,
                   help="write a before->after changelog of every answer change to OUT")
    p.add_argument("--review", metavar="OUT", default=None,
                   help="write the divergent-body review report to OUT")
    p.add_argument("--aggressive", action="store_true",
                   help="also trim trailing clauses (converge 'core + extra' onto the short core)")
    p.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD,
                   help=f"divergent-body cluster word-overlap threshold (default {REVIEW_THRESHOLD})")
    args = p.parse_args()

    path = Path(args.file)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    counts = collect_answers(lines)
    n_lines = sum(counts.values())
    n_distinct_before = len(counts)

    # Stage 1 — safe verb-only merges.
    safe_map = build_safe_merge(counts)

    # Effective answer counts after stage 1 (the input to the trim pass).
    stage1_counts: dict[str, int] = {}
    for ans, c in counts.items():
        s1 = safe_map.get(ans, ans)
        stage1_counts[s1] = stage1_counts.get(s1, 0) + c

    # Stage 2 — trailing-clause trim (only with --aggressive).
    trim_map, trim_reason = (build_trim_map(stage1_counts) if args.aggressive else ({}, {}))

    # Compose stage1 ∘ stage2 into a single original -> final map, with reasons.
    final_map: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for ans in counts:
        s1 = safe_map.get(ans, ans)
        final = trim_map.get(s1, s1)
        if final != ans:
            final_map[ans] = final
            reasons[ans] = "verb→Type" if (ans in safe_map and s1 == final) else (
                "verb→Type + trim" if ans in safe_map else trim_reason.get(s1, "trim trailing clause"))

    lines_touched = sum(counts[k] for k in final_map)
    distinct_final = {final_map.get(a, a) for a in counts}

    # Divergent-body clusters — flagged, never auto-changed.
    divergent = find_divergent_bodies(stage1_counts, args.threshold)
    contradictions = find_polarity_contradictions(counts)

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 64)
    print(f"ANSWER DRIFT REPORT — {path.name}")
    print("=" * 64)
    print(f"  A: lines total            : {n_lines}")
    print(f"  distinct answers (before) : {n_distinct_before}")
    print(f"  safe verb-only merges     : {len(safe_map)} variants")
    if args.aggressive:
        print(f"  trailing-clause trims     : {len(trim_map)} bodies")
    print(f"  A: lines rewritten        : {lines_touched}")
    print(f"  distinct answers (after)  : {len(distinct_final)}")
    print(f"  divergent-body clusters   : {len(divergent)} (FLAGGED, not changed)")
    print(f"  Yes/No contradictions     : {len(contradictions)} (flagged)")
    print("=" * 64)

    # ── Changelog ────────────────────────────────────────────────────────────
    changelog: list[str] = ["ANSWER CHANGELOG — every before→after rewrite. Veto any line.\n\n"]
    by_reason: dict[str, list[str]] = {}
    for ans, final in sorted(final_map.items()):
        by_reason.setdefault(reasons[ans], []).append(ans)
    for reason, items in by_reason.items():
        changelog.append(f"### {reason}  ({len(items)} changes)\n")
        for ans in items:
            changelog.append(f"  - {ans}\n  + {final_map[ans]}\n")
        changelog.append("\n")
    if args.changelog:
        Path(args.changelog).write_text("".join(changelog), encoding="utf-8")
        print(f"Changelog written: {args.changelog}")

    # ── Divergent-body review (your editorial pick) ───────────────────────────
    review_lines = ["DIVERGENT BODIES — same fact, content differs; neither is a prefix.\n"]
    review_lines.append("Auto-merging these risks conflating intents, so pick ONE per cluster.\n\n")
    for i, c in enumerate(divergent, 1):
        review_lines.append(f"### Cluster {i}\n")
        for b in c:
            review_lines.append(f"    {b}\n")
        review_lines.append("\n")
    if contradictions:
        review_lines.append("=" * 60 + "\nYES/NO SAME-BODY (verify each is correct for its question):\n\n")
        for yes, no in contradictions:
            review_lines.append(f"  YES: {yes}\n  NO : {no}\n\n")
    if args.review:
        Path(args.review).write_text("".join(review_lines), encoding="utf-8")
        print(f"Review report written: {args.review}")
    else:
        print("\n--- divergent-body clusters (top 6; pass --review to dump all) ---")
        for i, c in enumerate(divergent[:6], 1):
            print(f"\n  Cluster {i}:")
            for b in c:
                print(f"    {b[:104]}")

    # ── Apply to a new file ────────────────────────────────────────────────────
    if args.apply:
        out: list[str] = []
        for line in lines:
            s = line.rstrip("\n")
            if s.strip().startswith("A: "):
                ans = s.strip()[3:]
                if ans in final_map:
                    out.append(f"A: {final_map[ans]}\n")
                    continue
            out.append(line if line.endswith("\n") else line + "\n")
        Path(args.apply).write_text("".join(out), encoding="utf-8")
        print(f"\nConsolidated corpus written: {args.apply}")
        print(f"  ({lines_touched} A: lines rewritten; original untouched)")


if __name__ == "__main__":
    main()
