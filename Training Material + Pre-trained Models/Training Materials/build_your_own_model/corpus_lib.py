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
import json
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


# ── Guided-input menu (menu_manifest.json) ────────────────────────────────
# A generator can also emit menu_manifest.json so its model ships a guided
# "pick a question" menu (spec: LLM Guided-Input Menu, section 2). Each group
# has question TEMPLATES (a phrasing with a single "{}" entity slot, or no slot
# for a canned question) plus an ENTITY roster; the converter bakes them into
# the .bin. The composed template + entity string must be a BYTE-EXACT corpus
# line — so template phrasings come from your first ask-phrasing per attribute
# and entities keep their corpus casing. This MenuBuilder is mirrored in
# training_scripts/menu_manifest.py for the standalone generators; keep the two
# in sync with the spec's caps below.
MENU_VERSION = 1
MENU_SLOT = "{}"
MENU_MAX_GROUPS = 8
MENU_MAX_TEMPLATES_PER_GROUP = 64
MENU_MAX_ENTITIES_PER_GROUP = 1024
MENU_MAX_NAME_BYTES = 32
MENU_MAX_Q_BYTES = 120
MENU_MAX_ENTITY_BYTES = 48


def _menu_bytelen(s):
    return len(str(s).encode("utf-8"))


class MenuGroup:
    """One menu group: question templates + a shared entity roster."""

    def __init__(self, name):
        self.name = name
        self.templates = []   # list of {"q": str, "label": str(optional)}
        self.entities = []    # list of str

    def template(self, q, label=None):
        """Register one question archetype. `q` is a corpus-exact phrasing with
        at most one `{}` slot; `label` is an optional short (<=20 char) display
        form for small screens. Returns self for chaining."""
        if q.count(MENU_SLOT) > 1:
            raise ValueError(f"menu template has more than one '{{}}' slot: {q!r}")
        item = {"q": q}
        if label:
            item["label"] = label
        self.templates.append(item)
        return self

    def entity(self, name):
        self.entities.append(str(name))
        return self

    def add_entities(self, names):
        for n in names:
            self.entity(n)
        return self


class MenuBuilder:
    """Collects groups and serializes menu_manifest.json (spec section 2)."""

    def __init__(self):
        self.groups = []

    def menu_group(self, name):
        g = MenuGroup(name)
        self.groups.append(g)
        return g

    def is_empty(self):
        return not self.groups

    def validate(self):
        errs = []
        if len(self.groups) > MENU_MAX_GROUPS:
            errs.append(f"{len(self.groups)} groups > cap {MENU_MAX_GROUPS}")
        for g in self.groups:
            if _menu_bytelen(g.name) > MENU_MAX_NAME_BYTES:
                errs.append(f"group name {g.name!r} > {MENU_MAX_NAME_BYTES} bytes")
            if len(g.templates) > MENU_MAX_TEMPLATES_PER_GROUP:
                errs.append(f"group {g.name!r}: {len(g.templates)} templates "
                            f"> cap {MENU_MAX_TEMPLATES_PER_GROUP}")
            if len(g.entities) > MENU_MAX_ENTITIES_PER_GROUP:
                errs.append(f"group {g.name!r}: {len(g.entities)} entities "
                            f"> cap {MENU_MAX_ENTITIES_PER_GROUP}")
            for t in g.templates:
                if t["q"].count(MENU_SLOT) > 1:
                    errs.append(f"group {g.name!r}: template {t['q']!r} has >1 slot")
                if _menu_bytelen(t["q"]) > MENU_MAX_Q_BYTES:
                    errs.append(f"group {g.name!r}: q {t['q']!r} > {MENU_MAX_Q_BYTES} bytes")
            for e in g.entities:
                if _menu_bytelen(e) > MENU_MAX_ENTITY_BYTES:
                    errs.append(f"group {g.name!r}: entity {e!r} > {MENU_MAX_ENTITY_BYTES} bytes")
        if errs:
            raise ValueError("menu manifest invalid:\n  " + "\n  ".join(errs))

    def to_dict(self):
        return {
            "menu_version": MENU_VERSION,
            "groups": [
                {"name": g.name, "templates": g.templates, "entities": g.entities}
                for g in self.groups
            ],
        }

    def write_menu(self, out_path):
        """Validate and write menu_manifest.json. Returns (groups, templates,
        entities) counts."""
        self.validate()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        n_t = sum(len(g.templates) for g in self.groups)
        n_e = sum(len(g.entities) for g in self.groups)
        return len(self.groups), n_t, n_e


def run(build, default_out="training_data/corpus.txt",
        default_tokens="training_data/special_tokens.txt", menu=None):
    """CLI entry point. `build(corpus)` fills the corpus and RETURNS the list of
    entity names to keep as whole-word tokens. Handles --out/--tokens-out/--seed
    and prints a summary.

    Optional `menu(builder)` populates a MenuBuilder for the guided-input menu;
    when given, run() also writes menu_manifest.json next to --out (or to
    --menu-out). `--menu-only` writes JUST the manifest and leaves the corpus and
    tokens untouched — safe to re-run against an already-trained corpus."""
    ap = argparse.ArgumentParser(description="Generate a tiny-LLM training corpus")
    ap.add_argument("--out", type=Path, default=Path(default_out))
    ap.add_argument("--tokens-out", type=Path, default=Path(default_tokens))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--menu-out", type=Path, default=None,
                    help="Where to write menu_manifest.json (default: next to --out).")
    ap.add_argument("--menu-only", action="store_true",
                    help="Only (re)write the guided-input menu manifest; leave the corpus untouched.")
    args = ap.parse_args()

    c = Corpus()
    names = build(c) or []
    if not args.menu_only:
        n, qa = c.write(args.out, seed=args.seed)
        print(f"Wrote {args.out}")
        print(f"  blocks: {n}  (Q&A: {qa}, prose: {n - qa})")
        print(f"  conflicting-answer duplicates dropped: {c.conflicts_dropped}")
        if names:
            k = write_special_tokens(names, args.tokens_out)
            print(f"Wrote {args.tokens_out}  ({k} whole-word tokens)")
    if menu is not None:
        mb = MenuBuilder()
        menu(mb)
        menu_out = args.menu_out or (args.out.parent / "menu_manifest.json")
        ng, nt, ne = mb.write_menu(menu_out)
        print(f"Wrote {menu_out}  (menu: {ng} groups / {nt} templates / {ne} entities)")
