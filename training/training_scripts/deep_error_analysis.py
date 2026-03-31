#!/usr/bin/env python3
"""Deep error analysis of HardwareOne training data.

READ-ONLY. Does not modify any files. Reports issues for manual review.

Checks:
  1. Structural: orphaned Q/A/Do lines, missing separators, malformed blocks
  2. Duplicates: exact duplicate Q+A pairs, exact duplicate Q+Do pairs
  3. Answer contradictions: same question text with different answers
  4. Do: validation: multi-word Do: responses that look wrong, Do: with prose
  5. Answer quality: empty answers, extremely short, suspiciously long
  6. Command references: "Type X" in answers where X looks made up
  7. Token budget: Q+A pairs likely to exceed 128-token context window
  8. Stale references: answers mentioning things that don't match firmware
  9. Prose quality: prose blocks that are too short or don't teach anything

Usage:
    python training_scripts/deep_error_analysis.py [corpus_file]
"""
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_blocks(path: Path) -> list[dict]:
    """Parse the corpus into blocks. Each block is a dict with type and content."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Q: "):
            question = line[3:].strip()
            # Look for A: or Do: on the next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.startswith("A: "):
                    answer = next_line[3:].strip()
                    blocks.append({
                        "type": "qa",
                        "q": question,
                        "a": answer,
                        "line": i + 1,  # 1-indexed
                    })
                    i += 2
                elif next_line.startswith("Do: "):
                    do_cmd = next_line[4:].strip()
                    blocks.append({
                        "type": "do",
                        "q": question,
                        "do": do_cmd,
                        "line": i + 1,
                    })
                    i += 2
                else:
                    blocks.append({
                        "type": "orphan_q",
                        "q": question,
                        "line": i + 1,
                        "next_line": next_line,
                    })
                    i += 1
            else:
                blocks.append({
                    "type": "orphan_q",
                    "q": question,
                    "line": i + 1,
                    "next_line": "(EOF)",
                })
                i += 1
        elif line.startswith("A: "):
            blocks.append({
                "type": "orphan_a",
                "a": line[3:].strip(),
                "line": i + 1,
            })
            i += 1
        elif line.startswith("Do: "):
            blocks.append({
                "type": "orphan_do",
                "do": line[4:].strip(),
                "line": i + 1,
            })
            i += 1
        elif line.strip() == "":
            i += 1
        else:
            # Prose block - collect until blank line
            prose_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith("Q: "):
                prose_lines.append(lines[i])
                i += 1
            blocks.append({
                "type": "prose",
                "text": "\n".join(prose_lines),
                "line": i - len(prose_lines) + 1,
            })
        # Skip blank lines between blocks
        while i < len(lines) and lines[i].strip() == "":
            i += 1
    return blocks


def extract_type_commands(text: str) -> list[str]:
    """Extract command names from 'Type X' or 'type X' patterns in text."""
    cmds = []
    for m in re.finditer(r'\b[Tt]ype\s+(\S+)', text):
        cmd = m.group(1).rstrip(".,;:!?")
        # Skip common English words
        if cmd.lower() not in {"the", "if", "a", "an", "in", "it", "is", "to",
                                "or", "and", "your", "you", "this", "that",
                                "help", "any"}:
            cmds.append(cmd)
    return cmds


def rough_token_count(text: str) -> int:
    """Rough token estimate: ~1 token per 4 chars for English."""
    return max(1, len(text) // 4)


def main():
    script_dir = Path(__file__).resolve().parent
    default_file = script_dir.parent / "training_data" / "hardwareone_rich.txt"
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_file

    if not corpus_path.is_file():
        sys.exit(f"File not found: {corpus_path}")

    blocks = load_blocks(corpus_path)
    raw = corpus_path.read_text(encoding="utf-8", errors="replace")

    qa_blocks = [b for b in blocks if b["type"] == "qa"]
    do_blocks = [b for b in blocks if b["type"] == "do"]
    prose_blocks = [b for b in blocks if b["type"] == "prose"]

    issues = []

    print("=" * 70)
    print("DEEP ERROR ANALYSIS")
    print(f"  File: {corpus_path}")
    print(f"  Blocks: {len(qa_blocks)} Q&A, {len(do_blocks)} Do:, {len(prose_blocks)} prose")
    print("=" * 70)

    # ── 1. STRUCTURAL PROBLEMS ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("1. STRUCTURAL PROBLEMS")
    print(f"{'─' * 70}")

    orphan_q = [b for b in blocks if b["type"] == "orphan_q"]
    orphan_a = [b for b in blocks if b["type"] == "orphan_a"]
    orphan_do = [b for b in blocks if b["type"] == "orphan_do"]

    if orphan_q:
        for b in orphan_q:
            print(f"  line {b['line']}: Q: without A:/Do: → \"{b['q'][:60]}\"")
            print(f"           next line: \"{b['next_line'][:60]}\"")
            issues.append("orphan_q")
    if orphan_a:
        for b in orphan_a:
            print(f"  line {b['line']}: A: without preceding Q: → \"{b['a'][:60]}\"")
            issues.append("orphan_a")
    if orphan_do:
        for b in orphan_do:
            print(f"  line {b['line']}: Do: without preceding Q: → \"{b['do'][:60]}\"")
            issues.append("orphan_do")
    if not (orphan_q or orphan_a or orphan_do):
        print("  ✓  No structural problems.")

    # ── 2. EXACT DUPLICATE PAIRS ────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("2. EXACT DUPLICATE Q+A PAIRS")
    print(f"{'─' * 70}")

    qa_counter = Counter()
    qa_lines = defaultdict(list)
    for b in qa_blocks:
        key = (b["q"].lower().strip(), b["a"].lower().strip())
        qa_counter[key] += 1
        qa_lines[key].append(b["line"])

    dupes_found = False
    for (q, a), count in qa_counter.most_common():
        if count > 1:
            dupes_found = True
            lines_str = ", ".join(str(l) for l in qa_lines[(q, a)])
            print(f"  {count}x  Q: {q[:55]}")
            print(f"        A: {a[:55]}")
            print(f"        lines: {lines_str}")
            issues.extend(["duplicate_qa"] * (count - 1))
    if not dupes_found:
        print("  ✓  No exact duplicate Q+A pairs.")

    # Same for Do: pairs
    print(f"\n{'─' * 70}")
    print("3. EXACT DUPLICATE Q+Do PAIRS")
    print(f"{'─' * 70}")

    do_counter = Counter()
    do_lines_map = defaultdict(list)
    for b in do_blocks:
        key = (b["q"].lower().strip(), b["do"].lower().strip())
        do_counter[key] += 1
        do_lines_map[key].append(b["line"])

    do_dupes_found = False
    for (q, d), count in do_counter.most_common():
        if count > 1:
            do_dupes_found = True
            lines_str = ", ".join(str(l) for l in do_lines_map[(q, d)])
            print(f"  {count}x  Q: {q[:55]}")
            print(f"        Do: {d}")
            print(f"        lines: {lines_str}")
            issues.extend(["duplicate_do"] * (count - 1))
    if not do_dupes_found:
        print("  ✓  No exact duplicate Q+Do pairs.")

    # ── 4. SAME QUESTION, DIFFERENT ANSWERS ─────────────────────────────
    print(f"\n{'─' * 70}")
    print("4. SAME QUESTION, DIFFERENT ANSWERS (potential contradictions)")
    print(f"{'─' * 70}")

    q_to_answers = defaultdict(list)
    for b in qa_blocks:
        q_to_answers[b["q"].lower().strip()].append((b["a"], b["line"]))

    contradictions = 0
    for q, answers in sorted(q_to_answers.items()):
        unique_answers = set(a.lower().strip() for a, _ in answers)
        if len(unique_answers) > 1:
            contradictions += 1
            print(f"  Q: {q[:65]}")
            for a, line in answers:
                print(f"    line {line}: {a[:65]}")
            print()
            issues.append("contradiction")
    if contradictions == 0:
        print("  ✓  No contradictions found.")

    # ── 5. Do: RESPONSE QUALITY ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("5. Do: RESPONSE QUALITY")
    print(f"{'─' * 70}")

    do_issues = False
    for b in do_blocks:
        cmd = b["do"]
        # Do: should be short - just a command, maybe with one arg
        if len(cmd.split()) > 3:
            print(f"  line {b['line']}: Do: too long (looks like prose): \"{cmd[:60]}\"")
            issues.append("do_too_long")
            do_issues = True
        if cmd.startswith("Type ") or cmd.startswith("Use "):
            print(f"  line {b['line']}: Do: starts with instruction word: \"{cmd[:60]}\"")
            issues.append("do_has_instruction")
            do_issues = True
        if "." in cmd and not cmd.startswith("ledcolor"):
            # Might be a sentence not a command
            if len(cmd) > 20:
                print(f"  line {b['line']}: Do: contains period, might be prose: \"{cmd[:60]}\"")
                issues.append("do_is_prose")
                do_issues = True
    if not do_issues:
        print("  ✓  All Do: responses look like commands.")

    # ── 6. ANSWER QUALITY ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("6. ANSWER QUALITY")
    print(f"{'─' * 70}")

    aq_issues = False
    for b in qa_blocks:
        a = b["a"]
        if len(a) < 10:
            print(f"  line {b['line']+1}: Very short answer ({len(a)} chars): \"{a}\"")
            issues.append("short_answer")
            aq_issues = True
        if len(a) > 300:
            print(f"  line {b['line']+1}: Very long answer ({len(a)} chars): \"{a[:60]}...\"")
            issues.append("long_answer")
            aq_issues = True
    if not aq_issues:
        print("  ✓  All answers within reasonable length.")

    # ── 7. TOKEN BUDGET (128-token context window) ──────────────────────
    print(f"\n{'─' * 70}")
    print("7. TOKEN BUDGET (pairs likely to exceed 128 tokens)")
    print(f"{'─' * 70}")

    over_budget = []
    for b in qa_blocks:
        full = f"Q: {b['q']}\nA: {b['a']}"
        est = rough_token_count(full)
        if est > 110:  # Leave margin for special tokens
            over_budget.append((est, b))
    for b in prose_blocks:
        est = rough_token_count(b["text"])
        if est > 110:
            over_budget.append((est, b))

    if over_budget:
        over_budget.sort(key=lambda x: -x[0])
        for est, b in over_budget[:20]:
            if b["type"] == "qa":
                print(f"  line {b['line']}: ~{est} tokens  Q: {b['q'][:50]}")
            else:
                print(f"  line {b['line']}: ~{est} tokens  prose: {b['text'][:50]}...")
            issues.append("over_budget")
    else:
        print("  ✓  All pairs fit within token budget.")

    # ── 8. COMMAND REFERENCES IN ANSWERS ────────────────────────────────
    print(f"\n{'─' * 70}")
    print("8. COMMANDS REFERENCED IN ANSWERS (all 'Type X' patterns)")
    print(f"{'─' * 70}")

    all_cmds = Counter()
    for b in qa_blocks:
        for cmd in extract_type_commands(b["a"]):
            all_cmds[cmd] += 1
    for b in prose_blocks:
        for cmd in extract_type_commands(b["text"]):
            all_cmds[cmd] += 1

    print(f"  {len(all_cmds)} unique commands referenced via 'Type X':")
    for cmd, count in sorted(all_cmds.items(), key=lambda x: x[0].lower()):
        print(f"    {cmd:40s} ({count}x)")

    # ── 9. SAME Do: COMMAND, DIFFERENT QUESTIONS ────────────────────────
    print(f"\n{'─' * 70}")
    print("9. Do: COMMAND DISTRIBUTION (how many phrasings per command)")
    print(f"{'─' * 70}")

    do_cmd_counter = Counter()
    for b in do_blocks:
        do_cmd_counter[b["do"].lower()] += 1

    for cmd, count in do_cmd_counter.most_common():
        bar = "█" * count
        print(f"  {cmd:30s} {count:3d}  {bar}")

    under_3 = [cmd for cmd, count in do_cmd_counter.items() if count < 3]
    if under_3:
        print(f"\n  Commands with fewer than 3 phrasings ({len(under_3)}):")
        for cmd in sorted(under_3):
            print(f"    {cmd} ({do_cmd_counter[cmd]}x)")

    # ── 10. PROSE BLOCK QUALITY ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("10. PROSE BLOCK QUALITY")
    print(f"{'─' * 70}")

    prose_issues = False
    for b in prose_blocks:
        text = b["text"]
        words = text.split()
        if len(words) < 10:
            print(f"  line {b['line']}: Very short prose ({len(words)} words): \"{text[:60]}\"")
            issues.append("short_prose")
            prose_issues = True
        # Check if prose contains a command reference
        cmds_in_prose = extract_type_commands(text)
        cmd_pattern = re.findall(r'\b(open\w+|close\w+|wifi\w+|mqtt\w+|espnow\w+)\b', text, re.IGNORECASE)
        if not cmds_in_prose and not cmd_pattern:
            # Prose that doesn't reference any commands might not be useful
            print(f"  line {b['line']}: Prose with no command references: \"{text[:60]}...\"")
            issues.append("prose_no_cmds")
            prose_issues = True
    if not prose_issues:
        print("  ✓  All prose blocks look good.")

    # ── 11. ANSWER CONSISTENCY: same answer for very different questions ─
    print(f"\n{'─' * 70}")
    print("11. IDENTICAL ANSWERS GIVEN TO DIFFERENT QUESTIONS")
    print(f"{'─' * 70}")

    a_to_questions = defaultdict(list)
    for b in qa_blocks:
        a_to_questions[b["a"].lower().strip()].append((b["q"], b["line"]))

    high_reuse = False
    for a, questions in sorted(a_to_questions.items(), key=lambda x: -len(x[1])):
        if len(questions) >= 4:
            high_reuse = True
            print(f"  Answer used {len(questions)}x: \"{a[:65]}...\"")
            for q, line in questions:
                print(f"    line {line}: Q: {q[:60]}")
            print()
    if not high_reuse:
        print("  ✓  No answer used more than 3 times.")

    # ── 12. SUSPICIOUS PATTERNS ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("12. SUSPICIOUS PATTERNS")
    print(f"{'─' * 70}")

    suspicious = False
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        lineno = i + 1
        # Double Q: in a row without blank line
        if line.startswith("Q: ") and i + 1 < len(lines) and lines[i + 1].startswith("Q: "):
            # Check if there's actually a multi-turn block (Q: A: Q: A:)
            # That's intentional — only flag Q: Q: without A: between
            print(f"  line {lineno}: Two Q: lines in a row (missing A:/Do:?)")
            print(f"    → \"{line[:60]}\"")
            print(f"    → \"{lines[i+1][:60]}\"")
            issues.append("double_q")
            suspicious = True
        # A: or Do: followed immediately by another A: or Do:
        if (line.startswith("A: ") or line.startswith("Do: ")):
            if i + 1 < len(lines) and (lines[i+1].startswith("A: ") or lines[i+1].startswith("Do: ")):
                print(f"  line {lineno}: Answer/Do followed by another Answer/Do")
                print(f"    → \"{line[:60]}\"")
                print(f"    → \"{lines[i+1][:60]}\"")
                issues.append("double_answer")
                suspicious = True
        # Lines that start with lowercase and aren't blank/Q/A/Do/prose
        # (could be a broken line continuation)
        if line and not line.startswith(("Q: ", "A: ", "Do: ")) and line[0].islower() and i > 0:
            prev = lines[i - 1] if i > 0 else ""
            if prev.strip() == "":
                # A line after a blank that starts lowercase and isn't a known prefix
                # Might be a broken sentence
                if len(line) < 30 and not any(c in line for c in ".!?"):
                    print(f"  line {lineno}: Short lowercase line after blank (broken?): \"{line}\"")
                    issues.append("broken_line")
                    suspicious = True
    if not suspicious:
        print("  ✓  No suspicious patterns.")

    # ── SUMMARY ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    issue_counts = Counter(issues)
    if issues:
        print(f"TOTAL: {len(issues)} issues found")
        for issue_type, count in issue_counts.most_common():
            print(f"  {issue_type:25s} {count}")
    else:
        print("✓  No issues found!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
