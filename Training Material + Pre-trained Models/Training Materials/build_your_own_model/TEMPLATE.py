"""
TEMPLATE.py — build a training corpus for a tiny on-device LLM on ANY topic.

WHAT THIS IS
  A fill-in-the-blanks generator. The FACTS live in facts.json, each with a
  "source"; this file says how questions about them are PHRASED and turns the
  facts into Q&A. It ships with a tiny 4-planet example so it RUNS as-is and
  demonstrates every question pattern a good tiny-model corpus uses. Replace
  the facts and phrasings with your topic and you have a real generator.

HOW TO USE
  1. Give this whole folder + your TOPIC to an AI. See BUILD_YOUR_OWN_MODEL.md
     for the exact prompt — it explains every pattern and the rules to follow.
  2. The AI replaces facts.json with facts about your topic (a source for
     each), fills in the "FILL IN" sections below, and computes any derived
     facts in Python (see part 6).
  3. Run it until it passes:
       python TEMPLATE.py --strict
  4. Check what training_data/UNVERIFIED.md lists.
  5. Train (see BUILD_YOUR_OWN_MODEL.md, "STEP 3: train").

WHY IT LOOKS LIKE THIS
  A ~6M-parameter on-device model is a MEMORIZER, not a reasoner. So: (a) it
  answers a fact only if it saw that fact — hence many phrasings per fact and
  reverse/aggregate lookups written out explicitly; (b) it can't DERIVE answers
  live — hence anything that needs a rule (comparisons, "what beats what") is
  COMPUTED HERE in Python and baked in as flat facts; (c) it repeats whatever
  it was trained on with full confidence — hence every fact names its source.
"""
from pathlib import Path

from corpus_lib import MENU_MAX_NAME_BYTES, clip_bytes, list_join, load_facts, run

# ============================================================================
# ===== FILL IN (1/6): TOPIC + ENTITIES — in facts.json ======================
# ============================================================================
# facts.json holds every FACT the model will learn:
#   "topic"          a short human name for the domain (the menu's group name)
#   "entities"       the things your model should know. Each has a "name" (kept
#                    WHOLE in the tokenizer — see the return of build), a set of
#                    short ATTRIBUTES (keep attribute names identical across
#                    entities), a one-sentence "desc", and a "source".
#   "relationships"  entity -> entity links (part 4)
#   "lore"           background passages (part 5)
# "source" is the URL or citation a fact came from. If it came from memory,
# write "unverified" — never invent a source. Unsourced entries are listed in
# training_data/UNVERIFIED.md for a person to check before training.
FACTS = load_facts(Path(__file__).with_name("facts.json"))
TOPIC = FACTS["topic"]
ENTITIES = FACTS["entities"]
RELATIONSHIPS = FACTS["relationships"]
LORE = FACTS["lore"]


# ============================================================================
# ===== FILL IN (2/6): ATTRIBUTE QUESTIONS ===================================
# ============================================================================
# For each attribute, how users ASK about it and how the answer READS.
# {name} is filled with the entity name; {value} with its attribute value.
# Give 3-6 DISTINCT phrasings per attribute (the way real people would ask).
# The first phrasing that uses only {name} becomes the guided-menu question.
# Optional "heldout": a phrasing or two that are NOT trained. They go to
# val.txt and test_prompts.txt to show how the model copes with new wordings.
ATTRIBUTE_QUESTIONS = {
    # NOTE: don't reuse "What is {name}?" here — it's already the identity
    # question (answered with the description), and one question can only have
    # one answer. Keep attribute phrasings specific to the attribute.
    "kind": {
        "ask": ["What kind of object is {name}?", "Is {name} a planet?",
                "What type of world is {name}?"],
        "heldout": ["What sort of object is {name}?"],
        "answer": "{name} is a {value}.",
    },
    "order": {
        "ask": ["What position is {name} from the Sun?", "How far out is {name}?",
                "Which planet number is {name}?", "Where is {name} in order from the Sun?"],
        "heldout": ["What number planet is {name}?"],
        "answer": "{name} is planet number {value} from the Sun.",
    },
    "moons": {
        "ask": ["How many moons does {name} have?", "Does {name} have moons?",
                "What is {name}'s moon count?"],
        "heldout": ["How many moons orbit {name}?"],
        "answer": "{name} has {value} moon(s).",
    },
}


# ============================================================================
# ===== FILL IN (3/6): REVERSE / AGGREGATE LOOKUPS ===========================
# ============================================================================
# "Which entities have attribute = value?" A tiny model CANNOT derive these
# from the per-entity facts above — it must memorize the list. Lead the answer
# with a COUNT so a cut-short answer is still useful.
REVERSE_LOOKUPS = {
    "kind": {
        "ask": ["List all {value}s.", "Which planets are {value}s?",
                "Name the {value}s.", "How many {value}s are there?"],
        "heldout": ["What are the {value}s?"],
        # {count}, {value}, {list} are filled in automatically.
        "answer": "There are {count} {value}s: {list}.",
    },
}


# ============================================================================
# ===== FILL IN (4/6): RELATIONSHIPS (entity -> entity) ======================
# ============================================================================
# Links between entities (evolves-into, orbits, is-parent-of, comes-after...).
# The links are facts, so they live in facts.json "relationships" as rows like
#   {"from": "Earth", "rel": "beyond", "to": "Mars", "source": "..."}
# Here you say how each kind of link ("rel") is asked and answered; {from}
# and {to} are filled in. For that example row:
#   "beyond": {"ask": ["What is beyond {from}?"], "answer": "Beyond {from} lies {to}."}
# The planet example has none: it derives neighbour links from `order` in build().
RELATIONSHIP_QUESTIONS = {}


# ============================================================================
# ===== FILL IN (5/6): LORE PASSAGES — in facts.json =========================
# ============================================================================
# Background prose lives in facts.json "lore" as
#   {"questions": ["Tell me about X.", ...], "passage": "...", "source": "..."}
# Each passage is trained BOTH as bare prose AND as the answer to its
# questions, so open-ended questions land on the canonical text instead of the
# model free-associating. Keep them SHORT: an answer must stay within 2
# sentences and ~30 words (the device stops generating after 2 sentences).


# ============================================================================
# ===== VERIFY: sanity-check the facts before anything is built ==============
# ============================================================================
# Return a list of problems (strings); any problem stops the run before it
# builds anything. Check what a machine can: required fields, value ranges,
# duplicates — and, best of all, agreement with an authoritative dataset you
# downloaded (Training/training_scripts/verify_pokemon_data.py does this
# against PokeAPI). It doesn't replace a person reading UNVERIFIED.md.
def verify(facts):
    problems = []
    orders = {}
    for e in facts["entities"]:
        name = e["name"]
        # build() uses these for every entity. Other attributes may be missing
        # on some entities: their questions are skipped for those entities and
        # left off the guided menu.
        missing = [k for k in ("desc", "moons", "diameter_km") if k not in e]
        if missing:
            problems.append(f"{name}: missing {', '.join(missing)}")
            continue
        if name not in e["desc"]:
            problems.append(f"{name}: desc doesn't mention {name} — is it another entity's?")
        if not (isinstance(e["moons"], int) and e["moons"] >= 0):
            problems.append(f"{name}: moons must be a whole number >= 0, got {e['moons']!r}")
        if not (isinstance(e["diameter_km"], (int, float)) and e["diameter_km"] > 0):
            problems.append(f"{name}: diameter_km must be a positive number, "
                            f"got {e['diameter_km']!r}")
        if "order" in e:
            if e["order"] in orders:
                problems.append(f"{name} and {orders[e['order']]} share order {e['order']}")
            orders[e["order"]] = name
    return problems


# ============================================================================
# ===== EMIT: turns the data above into Q&A (usually no need to edit) ========
# ============================================================================
# category= groups facts for test_prompts.txt; computed=True marks answers
# worked out here in Python, which UNVERIFIED.md lists for spot-checking.
def build(c):
    by_name = {e["name"]: e for e in ENTITIES}

    for e in ENTITIES:
        name = e["name"]

        # (a) Identity / description — "What is X?" / "Tell me about X."
        c.qa_variants([f"Tell me about {name}.", f"What is {name}?",
                       f"Describe {name}.", f"Give me facts about {name}."],
                      e["desc"], category="identity",
                      heldout=[f"What do you know about {name}?"])

        # (b) Per-attribute lookups.
        for attr, spec in ATTRIBUTE_QUESTIONS.items():
            if attr not in e:
                continue
            val = e[attr]
            asks = [q.format(name=name, value=val) for q in spec["ask"]]
            held = [q.format(name=name, value=val) for q in spec.get("heldout", [])]
            c.qa_variants(asks, spec["answer"].format(name=name, value=val),
                          category=f"attribute:{attr}", heldout=held)

    # (c) Reverse / aggregate lookups.
    for attr, spec in REVERSE_LOOKUPS.items():
        buckets = {}
        for e in ENTITIES:
            if attr in e:
                buckets.setdefault(e[attr], []).append(e["name"])
        for value, names in buckets.items():
            asks = [q.format(value=value) for q in spec["ask"]]
            held = [q.format(value=value) for q in spec.get("heldout", [])]
            ans = spec["answer"].format(count=len(names), value=value,
                                        list=list_join(names))
            c.qa_variants(asks, ans, category=f"reverse:{attr}", computed=True,
                          heldout=held)

    # (d) Explicit relationships: links from facts.json, phrasings from part 4.
    for r in RELATIONSHIPS:
        spec = RELATIONSHIP_QUESTIONS.get(r["rel"])
        if spec is None:
            raise KeyError(f"facts.json relationship rel {r['rel']!r} has no entry "
                           f"in RELATIONSHIP_QUESTIONS")
        asks = [q.format(**r) for q in spec["ask"]]
        held = [q.format(**r) for q in spec.get("heldout", [])]
        c.qa_variants(asks, spec["answer"].format(**r),
                      category=f"relationship:{r['rel']}", heldout=held)

    # (e) Derived relationship example: order-based neighbours (computed).
    ordered = sorted((e for e in ENTITIES if "order" in e), key=lambda e: e["order"])
    for i, e in enumerate(ordered):
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]["name"]
            c.qa_variants([f"What planet comes after {e['name']}?",
                           f"What is right after {e['name']}?"],
                          f"After {e['name']} comes {nxt}.",
                          category="neighbour", computed=True)
        if i > 0:
            prv = ordered[i - 1]["name"]
            c.qa_variants([f"What planet comes before {e['name']}?",
                           f"What is right before {e['name']}?"],
                          f"Before {e['name']} comes {prv}.",
                          category="neighbour", computed=True)

    # ========================================================================
    # ===== FILL IN (6/6): PRECOMPUTED REASONING =============================
    # ========================================================================
    # Anything that needs a RULE applied to the data — comparisons, superlatives,
    # "what beats what". The tiny model can't compute these live, so compute them
    # HERE (correctly, in Python) and bake the answers in, with computed=True.
    # ALWAYS double-check a few by hand: a wrong computed fact becomes a
    # confidently-wrong model. UNVERIFIED.md lists them all for that check.
    #
    # Superlative: which entity has the most of an attribute.
    most_moons = max(ENTITIES, key=lambda e: e["moons"])
    c.qa_variants(["Which planet has the most moons?", "What planet has the most moons?"],
                  f"{most_moons['name']} has the most moons of the inner planets, "
                  f"with {most_moons['moons']}.",
                  category="superlative", computed=True)
    biggest = max(ENTITIES, key=lambda e: e["diameter_km"])
    c.qa_variants(["Which inner planet is the biggest?", "What is the largest inner planet?"],
                  f"{biggest['name']} is the largest inner planet.",
                  category="superlative", computed=True)

    # Pairwise comparison — pick a FEW meaningful pairs, NOT all N*N (that
    # explodes). Here, adjacent planets by size.
    for a, b in [("Earth", "Mars"), ("Venus", "Mercury")]:
        ea, eb = by_name[a], by_name[b]
        bigger = a if ea["diameter_km"] >= eb["diameter_km"] else b
        c.qa_variants([f"Which is bigger, {a} or {b}?", f"Is {a} bigger than {b}?"],
                      f"{bigger} is the bigger of the two.",
                      category="comparison", computed=True)

    # Lore passages (bare prose + bridged questions).
    for lore in LORE:
        c.prose(lore["passage"])
        c.qa_variants(lore["questions"], lore["passage"], category="lore")

    # Return the entity names to keep whole in the tokenizer.
    return [e["name"] for e in ENTITIES]


# ============================================================================
# ===== GUIDED MENU: derived from the data above (usually no need to edit) ===
# ============================================================================
# Emits menu_manifest.json so your model ships a "pick a question" menu on the
# device (spec: LLM Guided-Input Menu). It reuses the SAME data as build(), so
# filling in facts.json / ATTRIBUTE_QUESTIONS is all you need — the menu updates
# itself. Rule: a template's "{}" slot, once filled with an entity, must be a
# BYTE-EXACT line the model trained on. run() checks every composition against
# the corpus and fails the run if one isn't trained.
def _menu_template(asks, slot):
    """The menu question for a phrasing list: its FIRST phrasing whose only
    placeholder is {slot}, rewritten to the menu's single "{}" slot. None if
    there's no such phrasing (the device has one slot, so "Is {name} a
    {value}?" can't be a menu question)."""
    ph = "{" + slot + "}"
    for q in asks:
        if q.count(ph) == 1 and "{" not in q.replace(ph, ""):
            return q.replace(ph, "{}")
    return None


def build_menu(m):
    """Fill the menu builder; returns notes about anything left off the menu."""
    notes = []

    # (1) One entity group: identity + one question per attribute that EVERY
    # entity has. The group's entities share its templates, so a question
    # some entities can't answer would offer untrained questions.
    g = m.menu_group(clip_bytes(TOPIC, MENU_MAX_NAME_BYTES))
    g.template("Tell me about {}.", label="About {}")
    for attr, spec in ATTRIBUTE_QUESTIONS.items():
        lacking = [e["name"] for e in ENTITIES if attr not in e]
        tpl = _menu_template(spec["ask"], "name")
        if lacking:
            notes.append(f"{attr!r} questions left off the menu: not every entity has "
                         f"{attr!r} (e.g. {list_join(lacking[:3])})")
        elif tpl is None:
            notes.append(f"{attr!r} questions left off the menu: no phrasing uses "
                         f"only {{name}}")
        else:
            lbl = f"{attr} of {{}}"
            g.template(tpl, label=lbl if len(lbl) <= 20 else None)
    g.add_entities(e["name"] for e in ENTITIES)

    # (2) A reverse-lookup group per aggregate axis; entities are the values.
    for attr, spec in REVERSE_LOOKUPS.items():
        values = sorted({e[attr] for e in ENTITIES if attr in e})
        tpl = _menu_template(spec["ask"], "value")
        if values and tpl is None:
            notes.append(f"'By {attr}' left off the menu: no phrasing uses only {{value}}")
        elif values:
            m.menu_group(clip_bytes(f"By {attr}", MENU_MAX_NAME_BYTES)).template(tpl) \
                .add_entities(values)

    # (3) A "General" group of slotless canned questions from the lore.
    if LORE:
        gen = m.menu_group("General")
        for lore in LORE:
            gen.template(lore["questions"][0])
    return notes


if __name__ == "__main__":
    run(build, menu=build_menu, facts=FACTS, verify=verify)
