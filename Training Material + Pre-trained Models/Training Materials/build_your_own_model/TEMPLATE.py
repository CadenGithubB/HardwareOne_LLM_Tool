"""
TEMPLATE.py — build a training corpus for a tiny on-device LLM on ANY topic.

WHAT THIS IS
  A fill-in-the-blanks generator. It ships with a tiny 4-planet example so it
  RUNS as-is and demonstrates every question pattern a good tiny-model corpus
  uses. Replace the data with your topic and you have a real generator.

HOW TO USE
  1. Give this whole folder + your TOPIC to an AI. See BUILD_YOUR_OWN_MODEL.md
     for the exact prompt — it explains every pattern and the rules to follow.
  2. The AI replaces the data in the "FILL IN" sections with facts about your
     topic (and, for any derived facts, computes them in Python — see part 5).
  3. Run it:
       python TEMPLATE.py --out training_data/corpus.txt \
                          --tokens-out training_data/special_tokens.txt
  4. Train (see BUILD_YOUR_OWN_MODEL.md, "STEP 3: train").

WHY IT LOOKS LIKE THIS
  A ~6M-parameter on-device model is a MEMORIZER, not a reasoner. So: (a) it
  answers a fact only if it saw that fact — hence many phrasings per fact and
  reverse/aggregate lookups written out explicitly; (b) it can't DERIVE answers
  live — hence anything that needs a rule (comparisons, "what beats what") is
  COMPUTED HERE in Python and baked in as flat facts.
"""
from corpus_lib import list_join, run

# ============================================================================
# ===== FILL IN (1/6): TOPIC + ENTITIES ======================================
# ============================================================================
# TOPIC: a short human name for the domain, used in lore phrasings.
TOPIC = "the Solar System"

# ENTITIES: the things your model should know. Each has:
#   - name: the canonical name, kept WHOLE in the tokenizer (see return of build)
#   - a set of ATTRIBUTES (short, consistent values)
#   - desc: one clean sentence describing it
# Keep attribute names identical across entities.
ENTITIES = [
    {"name": "Mercury", "order": 1, "kind": "rocky planet", "moons": 0, "diameter_km": 4879,
     "desc": "Mercury is the first planet from the Sun and the smallest, a rocky world with no moons."},
    {"name": "Venus", "order": 2, "kind": "rocky planet", "moons": 0, "diameter_km": 12104,
     "desc": "Venus is the second planet from the Sun, a rocky world with a thick, hot atmosphere and no moons."},
    {"name": "Earth", "order": 3, "kind": "rocky planet", "moons": 1, "diameter_km": 12742,
     "desc": "Earth is the third planet from the Sun and the only known world with life. It has one moon."},
    {"name": "Mars", "order": 4, "kind": "rocky planet", "moons": 2, "diameter_km": 6779,
     "desc": "Mars is the fourth planet from the Sun, a cold rocky world with two small moons."},
]


# ============================================================================
# ===== FILL IN (2/6): ATTRIBUTE QUESTIONS ===================================
# ============================================================================
# For each attribute, how users ASK about it and how the answer READS.
# {name} is filled with the entity name; {value} with its attribute value.
# Give 3-6 DISTINCT phrasings per attribute (the way real people would ask).
ATTRIBUTE_QUESTIONS = {
    # NOTE: don't reuse "What is {name}?" here — it's already the identity
    # question (answered with the description), and one question can only have
    # one answer. Keep attribute phrasings specific to the attribute.
    "kind": {
        "ask": ["What kind of object is {name}?", "Is {name} a planet?",
                "What type of world is {name}?"],
        "answer": "{name} is a {value}.",
    },
    "order": {
        "ask": ["What position is {name} from the Sun?", "How far out is {name}?",
                "Which planet number is {name}?", "Where is {name} in order from the Sun?"],
        "answer": "{name} is planet number {value} from the Sun.",
    },
    "moons": {
        "ask": ["How many moons does {name} have?", "Does {name} have moons?",
                "What is {name}'s moon count?"],
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
        # {count}, {value}, {list} are filled in automatically.
        "answer": "There are {count} {value}s: {list}.",
    },
}


# ============================================================================
# ===== FILL IN (4/6): RELATIONSHIPS (entity -> entity) ======================
# ============================================================================
# Links between entities (evolves-into, orbits, is-parent-of, comes-after...).
# Here we derive neighbour relationships from the `order` attribute in build().
# For explicit links, add rows like:
#   {"from": "Earth", "to": "Mars", "ask": ["What is beyond {from}?"],
#    "answer": "Beyond {from} lies {to}."}
RELATIONSHIPS = []


# ============================================================================
# ===== FILL IN (5/6): LORE PASSAGES =========================================
# ============================================================================
# Background prose, each bridged with "tell me about X" questions. The passage
# is trained BOTH as bare prose AND as the answer, so open-ended questions land
# on the canonical text instead of the model free-associating. Keep them SHORT.
LORE = [
    (["Tell me about the Solar System.", "What is the Solar System?",
      "Describe the Solar System."],
     "The Solar System is the Sun and everything orbiting it. The four inner "
     "planets — Mercury, Venus, Earth, and Mars — are small rocky worlds, "
     "while the outer planets are giant balls of gas and ice."),
]


# ============================================================================
# ===== EMIT: turns the data above into Q&A (usually no need to edit) ========
# ============================================================================
def build(c):
    by_name = {e["name"]: e for e in ENTITIES}

    for e in ENTITIES:
        name = e["name"]

        # (a) Identity / description — "What is X?" / "Tell me about X."
        c.qa_variants([f"Tell me about {name}.", f"What is {name}?",
                       f"Describe {name}.", f"Give me facts about {name}."],
                      e["desc"])

        # (b) Per-attribute lookups.
        for attr, spec in ATTRIBUTE_QUESTIONS.items():
            if attr not in e:
                continue
            val = e[attr]
            asks = [q.format(name=name, value=val) for q in spec["ask"]]
            c.qa_variants(asks, spec["answer"].format(name=name, value=val))

    # (c) Reverse / aggregate lookups.
    for attr, spec in REVERSE_LOOKUPS.items():
        buckets = {}
        for e in ENTITIES:
            if attr in e:
                buckets.setdefault(e[attr], []).append(e["name"])
        for value, names in buckets.items():
            asks = [q.format(value=value) for q in spec["ask"]]
            ans = spec["answer"].format(count=len(names), value=value,
                                        list=list_join(names))
            c.qa_variants(asks, ans)

    # (d) Explicit relationships.
    for r in RELATIONSHIPS:
        asks = [q.format(**r) for q in r["ask"]]
        c.qa_variants(asks, r["answer"].format(**r))

    # (e) Derived relationship example: order-based neighbours (computed).
    ordered = sorted((e for e in ENTITIES if "order" in e), key=lambda e: e["order"])
    for i, e in enumerate(ordered):
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]["name"]
            c.qa_variants([f"What planet comes after {e['name']}?",
                           f"What is right after {e['name']}?"],
                          f"After {e['name']} comes {nxt}.")
        if i > 0:
            prv = ordered[i - 1]["name"]
            c.qa_variants([f"What planet comes before {e['name']}?",
                           f"What is right before {e['name']}?"],
                          f"Before {e['name']} comes {prv}.")

    # ========================================================================
    # ===== FILL IN (6/6): PRECOMPUTED REASONING =============================
    # ========================================================================
    # Anything that needs a RULE applied to the data — comparisons, superlatives,
    # "what beats what". The tiny model can't compute these live, so compute them
    # HERE (correctly, in Python) and bake the answers in. ALWAYS double-check a
    # few by hand: a wrong computed fact becomes a confidently-wrong model.
    #
    # Superlative: which entity has the most of an attribute.
    most_moons = max(ENTITIES, key=lambda e: e["moons"])
    c.qa_variants(["Which planet has the most moons?", "What planet has the most moons?"],
                  f"{most_moons['name']} has the most moons of the inner planets, with {most_moons['moons']}.")
    biggest = max(ENTITIES, key=lambda e: e["diameter_km"])
    c.qa_variants(["Which inner planet is the biggest?", "What is the largest inner planet?"],
                  f"{biggest['name']} is the largest inner planet.")

    # Pairwise comparison — pick a FEW meaningful pairs, NOT all N*N (that
    # explodes). Here, adjacent planets by size.
    for a, b in [("Earth", "Mars"), ("Venus", "Mercury")]:
        ea, eb = by_name[a], by_name[b]
        bigger = a if ea["diameter_km"] >= eb["diameter_km"] else b
        c.qa_variants([f"Which is bigger, {a} or {b}?", f"Is {a} bigger than {b}?"],
                      f"{bigger} is the bigger of the two.")

    # Lore passages (bare prose + bridged questions).
    for questions, passage in LORE:
        c.prose(passage)
        c.qa_variants(questions, passage)

    # Return the entity names to keep whole in the tokenizer.
    return [e["name"] for e in ENTITIES]


if __name__ == "__main__":
    run(build)
