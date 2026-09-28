"""
corpus_lib.py — reusable machinery for building a tiny-LLM training corpus.

You do NOT need to edit this file. Your topic generator (see TEMPLATE.py)
imports it and calls Corpus() + qa_variants(...) etc. This holds the invariant
plumbing so every topic gets the same battle-tested behaviour:

  * ONE answer per question (first-write-wins) — a tiny model trained on the
    same question with two different answers learns to blend/hedge, so later
    conflicting duplicates are dropped, and every one is listed so you can fix it.
  * Checks that fail loudly instead of quietly shipping a worse model: answer
    and prose length against the device limits, one-line blocks, and the
    guided menu against the corpus and the converter's caps.
  * facts.json loading; entries without a "source" are listed in
    UNVERIFIED.md for a person to check before training.
  * Whole-word special-token export for your entity names.
  * test_prompts.txt for the trainer's --qa-test-prompts, and val.txt
    (held-out phrasings, never trained) for its --val-text.
  * Answer helpers: render() (template or function), with_article() ("an
    actor"), count_phrase() ("no moons" / "one moon"), list_some() ("A, B,
    and 5 more").
  * Deterministic shuffle + write, with a --out/--tokens-out/--seed/--strict CLI.
"""
import argparse
import inspect
import json
import random
import re
import sys
from collections import Counter
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


def list_some(items, limit=6):
    """list_join for answers that must stay short: the first `limit` items,
    then "and N more" ('A, B, C, and 5 more'). Lead such an answer with the
    full count."""
    items = list(items)
    if len(items) <= limit:
        return list_join(items)
    return list_join(items[:limit] + [f"{len(items) - limit} more"])


def count_phrase(n, singular, plural=None):
    """'no moons' / 'one moon' / '2 moons', for answers that state a count.
    Pass `plural` when adding "s" is wrong ("person" -> "people")."""
    plural = plural or singular + "s"
    if n == 0:
        return f"no {plural}"
    if n == 1:
        return f"one {singular}"
    return f"{n} {plural}"


# Letters whose spoken name starts with a vowel sound ("an MRI", "an R&B hit"),
# and word starts where the sound doesn't match the first letter.
_VOWEL_SOUND_LETTERS = set("AEFHILMNORSX")
_AN_STARTS = ("hour", "honest", "honor", "honour", "heir")
_A_STARTS = ("eu", "ewe", "one", "once", "unic", "unif", "unio", "uniq", "unit",
             "univ", "use", "usu", "uten", "uti")


def article(phrase):
    """'a' or 'an' for `phrase`, by sound rather than letter: "an hour", "a
    unicorn", and initialisms read letter by letter ("an R&B singer", "an NBA
    player"). A heuristic: write the article yourself for acronyms read as
    words ("a NASA mission") and other odd cases."""
    words = phrase.split()
    word = words[0] if words else ""
    if word[:1].isupper() and not any(ch.islower() for ch in word):
        return "an" if word[0] in _VOWEL_SOUND_LETTERS else "a"
    low = word.lower()
    if low.startswith(_AN_STARTS):
        return "an"
    if low.startswith(_A_STARTS):
        return "a"
    return "an" if low[:1] in ("a", "e", "i", "o", "u") else "a"


def with_article(phrase):
    """The phrase with 'a' or 'an' in front: 'an actor', 'a rocky planet'."""
    return f"{article(phrase)} {phrase}"


def render(answer, **fields):
    """Fill in an answer: a str.format template ("{name} is {a_value}.") or a
    function that returns the text. A function gets only the fields it names
    as parameters (e.g. lambda name, value: ...) — use one when the wording
    depends on the value: plurals, zero, yes/no."""
    if not callable(answer):
        return answer.format(**fields)
    params = inspect.signature(answer).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return answer(**fields)
    return answer(**{k: v for k, v in fields.items() if k in params})


# ── Length budgets ────────────────────────────────────────────────────────
# The firmware stops generating after 2 sentences (Training/INSTRUCTIONS.txt),
# and a whole Q&A pair has to fit the model's short context, so answers stay
# within ~30 words (the guidance INSTRUCTIONS.txt gives). Prose is trained in
# 128-token blocks and risks being cut past ~55 words (prose_analysis.py).
MAX_ANSWER_SENTENCES = 2
MAX_ANSWER_WORDS = 30
MAX_PROSE_WORDS = 55

_WORD = re.compile(r"\S*\w\S*")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")


def count_words(text):
    """Whitespace-separated words, not counting bare punctuation like dashes."""
    return len(_WORD.findall(text))


def count_sentences(text):
    """Runs of '.', '!' or '?' followed by a space or the end of the text, so an
    abbreviation such as "Mr." counts as a sentence end too."""
    return len(_SENTENCE_END.findall(text))


def _clip(text, n=70):
    text = str(text)
    return text if len(text) <= n else text[:n - 1] + "…"


def _one_line(text, what):
    if "\n" in text or "\r" in text:
        raise ValueError(f"line break in {what}: {_clip(text)!r} — keep each question, "
                         f"answer and passage on one line (a blank line would split it "
                         f"into separate training blocks)")


class Corpus:
    """Collects Q&A pairs and prose blocks, then writes a training file."""

    def __init__(self):
        self.blocks = []          # each block is a list of text lines
        self._q_answer = {}        # question -> its (first) answer
        self.facts = []            # one entry per qa/qa_variants call that trained something
        self.heldout = []          # (question, answer, category): val.txt only, never trained
        self.conflicts = []        # (question, kept answer, dropped answer)

    @property
    def conflicts_dropped(self):
        return len(self.conflicts)

    def _add(self, q, a):
        """Add one Q&A pair unless the question already has an answer (first
        write wins; a different later answer is recorded as a conflict)."""
        _one_line(q, "question")
        _one_line(a, f"the answer to {q!r}")
        prev = self._q_answer.get(q)
        if prev is not None:
            if prev != a:
                self.conflicts.append((q, prev, a))
            return
        self._q_answer[q] = a
        self.blocks.append([f"Q: {q}", f"A: {a}"])

    def qa(self, question, answer, category=None, computed=False):
        """Add one Q&A pair. Enforces one answer per question (first wins)."""
        self.qa_variants([question], answer, category=category, computed=computed)

    def qa_variants(self, questions, answer, category=None, computed=False, heldout=()):
        """Emit the SAME answer under many phrasings — this is how the model
        learns a fact no matter how it's asked. Provide DISTINCT wordings, not
        lowercase/punctuation duplicates of one phrasing (those just bloat the
        corpus; casing is handled at inference).

        category  groups related facts ("identity", "attribute:moons", ...) so
                  test_prompts.txt samples every kind of question.
        computed  marks an answer worked out in Python (a comparison, superlative
                  or count); UNVERIFIED.md lists these for spot-checking.
        heldout   extra phrasings that are NOT trained. They go to val.txt (for
                  the trainer's --val-text) and test_prompts.txt, to show how the
                  model copes with wordings it never saw."""
        a = answer.strip()
        qs = [q.strip() for q in questions if q.strip()]
        if not a or not qs:
            return
        for q in qs:
            self._add(q, a)
        trained = [q for q in qs if self._q_answer.get(q) == a]
        if not trained:
            return  # every phrasing conflicted; the conflicts are reported
        category = category or "other"
        self.facts.append({"category": category, "questions": trained,
                           "answer": a, "computed": computed})
        for q in heldout:
            q = q.strip()
            if q:
                _one_line(q, "held-out question")
                self.heldout.append((q, a, category))

    def prose(self, text):
        """A standalone passage with no question — pure language exposure.
        Keep passages SHORT (within ~55 words), factually dense, and on one
        line; tiny models memorize tight text far better than long flowery
        prose, and short passages drift less."""
        t = text.strip()
        if t:
            _one_line(t, "prose passage")
            self.blocks.append([t])

    def valid_heldout(self):
        """The held-out pairs that really are held out, plus the problems with
        the rest: a phrasing that is also trained isn't held out, and one
        phrasing can't have two answers."""
        seen, pairs, problems = {}, [], []
        for q, a, category in self.heldout:
            if q in self._q_answer:
                problems.append(f"{q!r} is also a training question")
                continue
            prev = seen.get(q)
            if prev is not None:
                if prev != a:
                    problems.append(f"{q!r} has two held-out answers: "
                                    f"{_clip(prev, 40)!r} / {_clip(a, 40)!r}")
                continue
            seen[q] = a
            pairs.append((q, a, category))
        return pairs, problems

    def write(self, path, seed=1234):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        rng.shuffle(self.blocks)
        _write_blocks(path, self.blocks)
        qa = sum(1 for b in self.blocks if len(b) == 2)
        return len(self.blocks), qa


def _write_blocks(path, blocks):
    lines = []
    for b in blocks:
        lines.extend(b)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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


# ── Test prompts + held-out set ───────────────────────────────────────────
def pick_test_prompts(corpus, heldout_pairs, seed, trained_limit=24, heldout_limit=8):
    """A small, stable sample for the trainer's post-training test: the first
    trained phrasing of a fact, taken round-robin across categories so every
    kind of question gets tried, plus some held-out phrasings."""
    rng = random.Random(seed)

    def round_robin(by_category, limit):
        for items in by_category.values():
            rng.shuffle(items)
        picked = []
        while len(picked) < limit and any(by_category.values()):
            for items in by_category.values():
                if items and len(picked) < limit:
                    picked.append(items.pop())
        return picked

    trained, seen = {}, set()
    for f in corpus.facts:
        if f["answer"] not in seen:
            seen.add(f["answer"])
            trained.setdefault(f["category"], []).append(f["questions"][0])
    held = {}
    for q, _a, category in heldout_pairs:
        held.setdefault(category, []).append(q)
    return round_robin(trained, trained_limit), round_robin(held, heldout_limit)


def write_test_prompts(trained, heldout, path):
    lines = ["# Post-training test for the trainer's --qa-test-prompts (it uses the lines",
             "# starting with \"Q:\"). TEMPLATE.py rewrites this file on every run.",
             "",
             "# Trained phrasings: the model should answer these exactly."]
    lines += [f"Q: {q}" for q in trained]
    if heldout:
        lines += ["",
                  "# Held-out phrasings (in val.txt, never trained): how it copes with new wordings."]
        lines += [f"Q: {q}" for q in heldout]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── facts.json ────────────────────────────────────────────────────────────
FACT_LISTS = ("entities", "relationships", "lore")


def load_facts(path):
    """Read facts.json and check its shape, exiting with a readable message if
    it is missing or malformed:

      {"topic": str,
       "entities":      [{"name": str, <attribute>: value, ..., "source": str}],
       "relationships": [{"from": str, "rel": str, "to": str, "source": str}],   (optional)
       "lore":          [{"questions": [str], "passage": str, "source": str}]}   (optional)

    "source" is the URL or citation a fact came from, or "unverified"."""
    path = Path(path)
    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"{path}: not found")
    except json.JSONDecodeError as e:
        sys.exit(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}")
    if not isinstance(facts, dict):
        sys.exit(f'{path}: must be a JSON object with "topic" and "entities"')

    problems = []
    if not isinstance(facts.get("topic"), str) or not facts["topic"].strip():
        problems.append('"topic" must be a non-empty string')
    entities = facts.get("entities")
    if not isinstance(entities, list) or not entities:
        problems.append('"entities" must be a non-empty list')
        entities = []
    for key in ("relationships", "lore"):
        facts.setdefault(key, [])
        if not isinstance(facts[key], list):
            problems.append(f'"{key}" must be a list')
            facts[key] = []

    def text(v):
        return isinstance(v, str) and v.strip()

    seen = set()
    for i, e in enumerate(entities):
        name = e.get("name") if isinstance(e, dict) else None
        if not text(name):
            problems.append(f'entities[{i}] needs a non-empty "name"')
        elif name in seen:
            problems.append(f"entity {name!r} appears twice")
        else:
            seen.add(name)
    for i, r in enumerate(facts["relationships"]):
        if not (isinstance(r, dict) and all(text(r.get(k)) for k in ("from", "rel", "to"))):
            problems.append(f'relationships[{i}] needs "from", "rel" and "to" strings')
    for i, lore in enumerate(facts["lore"]):
        qs = lore.get("questions") if isinstance(lore, dict) else None
        if not (isinstance(qs, list) and qs and all(text(q) for q in qs)
                and text(lore.get("passage"))):
            problems.append(f'lore[{i}] needs a non-empty "questions" list and a "passage"')
    if problems:
        sys.exit(f"{path} is malformed:\n  - " + "\n  - ".join(problems))
    return facts


def unverified_entries(facts):
    """facts.json entries without a usable "source" (missing, empty, or marked
    "unverified"), as (list name, entry) pairs."""
    out = []
    for key in FACT_LISTS:
        for entry in facts.get(key, []):
            src = entry.get("source")
            if not isinstance(src, str) or not src.strip() \
                    or src.strip().lower().startswith("unverified"):
                out.append((key, entry))
    return out


def _describe_entry(key, entry):
    note = entry.get("source") or "no source"
    if key == "entities":
        attrs = "; ".join(f"{k} = {v}" for k, v in entry.items()
                          if k not in ("name", "desc", "source"))
        out = [f"- **{entry['name']}** ({note}): {attrs}"]
        if entry.get("desc"):
            out.append(f"  - desc: {entry['desc']}")
        return out
    if key == "relationships":
        return [f"- **{entry['from']} → {entry['rel']} → {entry['to']}** ({note})"]
    return [f"- **Lore: {entry['questions'][0]}** ({note}) {entry['passage']}"]


def write_unverified(facts, corpus, path, per_group=20):
    """UNVERIFIED.md: the facts.json entries without a source, and the answers
    worked out in Python, for a person to check before training. Returns
    (entries without a source, computed answers)."""
    missing = unverified_entries(facts)
    total = sum(len(facts.get(k, [])) for k in FACT_LISTS)
    lines = ["# Unverified facts — check before training", "",
             "TEMPLATE.py rewrites this file on every run. A tiny model repeats whatever",
             "it was trained on with full confidence, so check what's listed here against",
             "a reliable reference, fix facts.json, and re-run.", "",
             f"## Facts without a source ({len(missing)} of {total})", ""]
    if missing:
        lines += ["These facts.json entries have no `source` (or are marked \"unverified\"),",
                  "so nothing but memory backs them. Check each one, then add a `source`.", ""]
        for key, entry in missing:
            lines += _describe_entry(key, entry)
    else:
        lines.append("Every entry in facts.json cites a source.")

    groups, seen = {}, set()
    for f in corpus.facts:
        if f["computed"] and f["answer"] not in seen:
            seen.add(f["answer"])
            groups.setdefault(f["category"], []).append(f)
    n_computed = len(seen)
    lines += ["", f"## Computed answers to spot-check ({n_computed})", ""]
    if groups:
        lines += ["Worked out in Python, so they're only as right as the facts and the code",
                  "that derives them. Check a few from each group.", ""]
        for category, items in groups.items():
            lines.append(f"### {category}")
            for f in items[:per_group]:
                lines.append(f"- {f['questions'][0]} → {f['answer']}")
            if len(items) > per_group:
                lines.append(f"- … and {len(items) - per_group} more in corpus.txt")
            lines.append("")
    else:
        lines.append("None: no qa_variants(..., computed=True) calls.")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return len(missing), n_computed


# ── Guided-input menu (menu_manifest.json) ────────────────────────────────
# A generator can also emit menu_manifest.json so its model ships a guided
# "pick a question" menu (spec: LLM Guided-Input Menu, section 2). Each group
# has question TEMPLATES (a phrasing with a single "{}" entity slot, or no slot
# for a canned question) plus an ENTITY roster; the converter bakes them into
# the .bin. The composed template + entity string must be a BYTE-EXACT corpus
# line — so template phrasings come from your ask-phrasings, entities keep
# their corpus casing, and run() checks every composition against the corpus.
# This MenuBuilder is mirrored in training_scripts/menu_manifest.py for the
# standalone generators; keep the caps, encoded_size() and check_against() in
# sync between the two.
MENU_VERSION = 1
MENU_SLOT = "{}"
MENU_MAX_GROUPS = 8
MENU_MAX_TEMPLATES_PER_GROUP = 64
MENU_MAX_ENTITIES_PER_GROUP = 1024
MENU_MAX_NAME_BYTES = 32
MENU_MAX_Q_BYTES = 120
MENU_MAX_ENTITY_BYTES = 48
MENU_MAX_BYTES = 32768   # the whole encoded MENU section (the converter's menuBytes cap)


def _menu_bytelen(s):
    return len(str(s).encode("utf-8"))


def clip_bytes(text, max_bytes):
    """Trim text to at most max_bytes of UTF-8 without splitting a character.
    Menu caps are in bytes, so a plain [:n] slice can overshoot on non-ASCII."""
    raw = str(text).encode("utf-8")
    if len(raw) <= max_bytes:
        return str(text)
    return raw[:max_bytes].decode("utf-8", errors="ignore").rstrip()


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

    def encoded_size(self):
        """Bytes of the MENU section the converter encodes (index.html,
        encodeMenuSection): a 4-byte header; per group a flags byte, the
        length-prefixed name and two u16 counts; then each template (its "{}"
        slot becomes one byte) and each entity, length-prefixed."""
        n = 4
        for g in self.groups:
            n += 1 + 1 + _menu_bytelen(g.name) + 2 + 2
            for t in g.templates:
                n += 1 + _menu_bytelen(t["q"]) - (1 if MENU_SLOT in t["q"] else 0)
            n += sum(1 + _menu_bytelen(e) for e in g.entities)
        return n

    def problems(self):
        """Every cap the converter enforces, as a list of messages."""
        errs = []
        if not self.groups:
            errs.append("no groups (the converter rejects an empty menu)")
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
        size = self.encoded_size()
        if size > MENU_MAX_BYTES:
            errs.append(f"encoded menu is {size} bytes > cap {MENU_MAX_BYTES} "
                        f"(use fewer or shorter entities)")
        return errs

    def validate(self):
        errs = self.problems()
        if errs:
            raise ValueError("menu manifest invalid:\n  " + "\n  ".join(errs))

    def composed(self):
        """Every question the device menu can show, as (group name, question)."""
        for g in self.groups:
            for t in g.templates:
                if MENU_SLOT in t["q"]:
                    for e in g.entities:
                        yield g.name, t["q"].replace(MENU_SLOT, e, 1)
                else:
                    yield g.name, t["q"]

    def check_against(self, questions, conflicted=()):
        """Compose every question the device menu can show and return a problem
        for each one that isn't trained verbatim: not in `questions` (the
        corpus questions), or in `conflicted` (questions given two different
        answers — which one the model learned depends on emission order)."""
        problems = []
        for g in self.groups:
            for t in g.templates:
                if MENU_SLOT in t["q"] and not g.entities:
                    problems.append(f"template {t['q']!r} has a {{}} slot but group "
                                    f"{g.name!r} has no entities")
        for group, q in self.composed():
            if q not in questions:
                problems.append(f"{q!r} is not a question in the corpus (group {group!r})")
            elif q in conflicted:
                problems.append(f"{q!r} has conflicting answers in the corpus (group {group!r})")
        return problems

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


# ── Checks + CLI ──────────────────────────────────────────────────────────
class Checks:
    """What run() found before writing anything. An error always fails the
    run, a warning fails it only with --strict, a note never does."""

    def __init__(self):
        self.results = []   # (level, title, items)

    def add(self, level, title, items=()):
        self.results.append((level, title, list(items)))

    def expect(self, items, level, problem, ok):
        """Record `problem` at `level` with its items, or `ok` if there are none."""
        if items:
            self.add(level, f"{problem}: {len(items)}", items)
        else:
            self.add("ok", ok)

    def has(self, level):
        return any(lvl == level for lvl, _title, _items in self.results)

    def print(self, limit=10):
        print("Checks:")
        for level, title, items in self.results:
            print(f"  {level if level == 'ok' else level.upper():<8}{title}")
            for item in items[:limit]:
                print(f"            - {item}")
            if len(items) > limit:
                print(f"            ... and {len(items) - limit} more")


def split_words(tokens, text):
    """Words in `text` that a special token would cut apart, as a Counter of
    (token, word). The tokenizer matches special tokens anywhere, even inside
    a longer word (leftmost-longest), so a name like "Mars" turns "Marshall"
    into "Mars" + "hall". A plural "s" right after a token ("rocky planets")
    is fine."""
    occurrences = []
    for t in {t for t in tokens if t}:
        i = text.find(t)
        while i != -1:
            occurrences.append((i, i + len(t)))
            i = text.find(t, i + 1)
    occurrences.sort(key=lambda o: (o[0], o[0] - o[1]))   # leftmost, then longest
    hits, taken_to = Counter(), 0
    for s, e in occurrences:
        if s < taken_to:
            continue   # inside an earlier or longer match, as the tokenizer does
        taken_to = e
        plural_s = text[e:e + 1] == "s" and not text[e + 1:e + 2].isalnum()
        glued_after = text[e:e + 1].isalnum() and not plural_s
        if (s > 0 and text[s - 1].isalnum()) or glued_after:
            ws, we = s, e
            while ws > 0 and text[ws - 1].isalnum():
                ws -= 1
            while we < len(text) and text[we].isalnum():
                we += 1
            hits[(text[s:e], text[ws:we])] += 1
    return hits


def check_corpus(c, checks, heldout_problems, tokens=()):
    checks.expect([f"{q!r}: kept {_clip(kept, 45)!r}, dropped {_clip(dropped, 45)!r}"
                   for q, kept, dropped in c.conflicts],
                  "warning", "conflicting answers (first kept, later dropped)",
                  "no conflicting answers")

    first_question = {}
    for q, a in c._q_answer.items():
        first_question.setdefault(a, q)
    long_answers = []
    for a, q in first_question.items():
        s, w = count_sentences(a), count_words(a)
        if s > MAX_ANSWER_SENTENCES or w > MAX_ANSWER_WORDS:
            long_answers.append(f"{w} words, {s} sentence(s): {_clip(a)!r} (for {q!r})")
    limits = f"{MAX_ANSWER_SENTENCES} sentences / {MAX_ANSWER_WORDS} words"
    checks.expect(long_answers, "warning", f"answers over {limits}",
                  f"all answers within {limits}")

    long_prose = [f"{count_words(b[0])} words: {_clip(b[0])!r}"
                  for b in c.blocks if len(b) == 1 and count_words(b[0]) > MAX_PROSE_WORDS]
    checks.expect(long_prose, "warning", f"prose passages over {MAX_PROSE_WORDS} words",
                  f"all prose within {MAX_PROSE_WORDS} words")

    if c.heldout:
        checks.expect(heldout_problems, "warning",
                      "held-out phrasings left out of val.txt",
                      "held-out phrasings are all distinct from the training questions")

    if tokens:
        text = "\n".join(line for b in c.blocks for line in b)
        splits = split_words(tokens, text)
        checks.expect([f"{tok!r} splits {word!r} ({n} times) — also keep {word!r} whole, "
                       f"or drop {tok!r} from the whole-word tokens"
                       for (tok, word), n in splits.most_common()],
                      "warning", "whole-word tokens that would split a longer word",
                      "no whole-word token splits a longer word")


def print_stats(c, names):
    answers = Counter(c._q_answer.values())
    prose = sum(1 for b in c.blocks if len(b) == 1)
    print("Corpus:")
    print(f"  Q&A pairs: {len(c._q_answer)}   prose passages: {prose}   "
          f"whole-word tokens: {len({n for n in names if n})}")
    if answers:
        per = sorted(answers.values())
        singles = sum(1 for n in per if n == 1)
        print(f"  facts (distinct answers): {len(per)}   phrasings per fact: "
              f"min {per[0]}, median {per[len(per) // 2]}, max {per[-1]}"
              + (f"   ({singles} with only one)" if singles else ""))
        longest = max(answers, key=count_words)
        print(f"  longest answer: {count_words(longest)} words, "
              f"{count_sentences(longest)} sentence(s): {_clip(longest)!r}")


def run(build, default_out="training_data/corpus.txt",
        default_tokens="training_data/special_tokens.txt", menu=None,
        facts=None, verify=None):
    """CLI entry point. `build(corpus)` fills the corpus and RETURNS the list of
    entity names to keep as whole-word tokens.

    Optional `facts` (from load_facts) are checked first with `verify(facts)`,
    which returns a list of problems. Optional `menu(builder)` populates a
    MenuBuilder for the guided-input menu and may return notes about anything
    it left out.

    Every run checks the result and writes NOTHING if a check fails — errors
    always, warnings too with --strict. Otherwise it writes the corpus, the
    special tokens, test_prompts.txt, val.txt (when there are held-out
    phrasings), UNVERIFIED.md (when facts are given) and menu_manifest.json,
    all next to --out. `--menu-only` writes JUST the manifest (the corpus and
    tokens are left untouched) — safe to re-run against an already-trained
    corpus."""
    ap = argparse.ArgumentParser(description="Generate a tiny-LLM training corpus")
    ap.add_argument("--out", type=Path, default=Path(default_out))
    ap.add_argument("--tokens-out", type=Path, default=Path(default_tokens))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--menu-out", type=Path, default=None,
                    help="Where to write menu_manifest.json (default: next to --out).")
    ap.add_argument("--menu-only", action="store_true",
                    help="Only (re)write the guided-input menu manifest; leave the corpus untouched.")
    ap.add_argument("--strict", action="store_true",
                    help="Fail (exit 1, write nothing) on warnings too, not only on errors.")
    args = ap.parse_args()
    # The checks echo corpus text; a console that can't encode a character
    # (e.g. redirected output on Windows) must not crash the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    checks = Checks()
    if facts is not None and verify is not None:
        problems = list(verify(facts) or [])
        checks.expect(problems, "error", "verify() problems in facts.json",
                      "verify() found no problems in facts.json")
        if problems:
            checks.print()
            sys.exit("Nothing built or written: fix facts.json first.")

    c = Corpus()
    names = build(c) or []
    if not c._q_answer:
        checks.add("error", "build() added no Q&A pairs")
    heldout, heldout_problems = c.valid_heldout()
    check_corpus(c, checks, heldout_problems, names)

    mb = None
    if menu is not None:
        mb = MenuBuilder()
        notes = menu(mb) or []
        if mb.is_empty():
            checks.add("note", "the menu function added no groups, so no menu is written")
        else:
            problems = mb.problems() + mb.check_against(
                set(c._q_answer), {q for q, _kept, _dropped in c.conflicts})
            checks.expect(problems, "error", "guided-menu problems",
                          f"guided menu: all {sum(1 for _ in mb.composed())} questions are "
                          f"trained verbatim, and it fits the converter's caps")
        for note in notes:
            checks.add("note", note)

    if facts is not None:
        n_missing = len(unverified_entries(facts))
        if n_missing:
            checks.add("note", f"facts.json entries without a source: {n_missing} "
                               f"(listed in UNVERIFIED.md)")
        else:
            checks.add("ok", "every facts.json entry cites a source")

    print_stats(c, names)
    checks.print()
    if checks.has("error") or (args.strict and checks.has("warning")):
        what = "errors and warnings (--strict)" if args.strict else "errors"
        sys.exit(f"Nothing written: fix the {what} above.")

    out_dir = args.out.parent
    if not args.menu_only:
        n, qa = c.write(args.out, seed=args.seed)
        print(f"Wrote {args.out}  (Q&A: {qa}, prose: {n - qa})")
        if names:
            k = write_special_tokens(names, args.tokens_out)
            print(f"Wrote {args.tokens_out}  ({k} whole-word tokens)")
        trained, held = pick_test_prompts(c, heldout, args.seed)
        prompts_out = out_dir / "test_prompts.txt"
        write_test_prompts(trained, held, prompts_out)
        print(f"Wrote {prompts_out}  (trained: {len(trained)}, held-out: {len(held)})")
        if heldout:
            val_out = out_dir / "val.txt"
            _write_blocks(val_out, [[f"Q: {q}", f"A: {a}"] for q, a, _category in heldout])
            print(f"Wrote {val_out}  (held-out Q&A, never trained: {len(heldout)})")
        if facts is not None:
            review_out = out_dir / "UNVERIFIED.md"
            n_missing, n_computed = write_unverified(facts, c, review_out)
            print(f"Wrote {review_out}  (without a source: {n_missing}, "
                  f"computed answers to spot-check: {n_computed})")
    if mb is not None and not mb.is_empty():
        menu_out = args.menu_out or (out_dir / "menu_manifest.json")
        ng, nt, ne = mb.write_menu(menu_out)
        print(f"Wrote {menu_out}  (menu: {ng} groups / {nt} templates / {ne} entities)")
