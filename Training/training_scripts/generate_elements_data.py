#!/usr/bin/env python3
"""Generate a periodic-table (118 elements) training corpus.

Q:/A: pairs in the same format as the other datasets, with short answers that
fit the device's ~45-token context. Reads training_data/elements.json (built
from the Bowserinator Periodic-Table-JSON dataset, filtered to elements 1-118,
families normalized, room-temp phase only asserted for naturally-occurring
elements <=94 where it is well established).

All answers are derived from elements.json or explicit GENERAL/CAPABILITIES
tables in this file — no scraped uses, discovery dates, or hand-written facts
per element.

Usage:
    python training_scripts/generate_elements_data.py
"""
import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "training_data" / "elements.json"

METAL_FAMILIES = {
    "alkali metal", "alkaline earth metal", "transition metal",
    "post-transition metal", "lanthanide", "actinide",
}
NONMETAL_FAMILIES = {"nonmetal", "halogen", "noble gas"}


def load_elements():
    els = json.loads(DATA.read_text(encoding="utf-8"))
    if len(els) != 118:
        raise SystemExit(f"expected 118 elements, got {len(els)}")
    nums = [e["number"] for e in els]
    if sorted(nums) != list(range(1, 119)):
        raise SystemExit("elements.json must contain atomic numbers 1-118 exactly once")
    return els


def fam_with_article(fam):
    """a/an before family name."""
    if fam[0] in "aeiou":
        return f"an {fam}"
    return f"a {fam}"


def fam_plural(fam):
    return fam + ("es" if fam.endswith("s") else "s")


def metal_class(e):
    fam = e["family"]
    if fam in METAL_FAMILIES:
        return "metal"
    if fam == "metalloid":
        return "metalloid"
    if fam in NONMETAL_FAMILIES:
        return "nonmetal"
    return fam


def list_answer(prefix, names, max_names=6):
    """Short list for reverse lookups; never dump 30+ names in one answer."""
    if not names:
        return prefix.rstrip() + " none."
    if len(names) == 1:
        return f"{prefix} {names[0]}."
    if len(names) <= max_names:
        return f"{prefix} " + ", ".join(names[:-1]) + ", and " + names[-1] + "."
    ex = names[:max_names]
    return f"{prefix} " + ", ".join(ex[:-1]) + ", and " + ex[-1] + ", among others."


# ── Phrasing templates ───────────────────────────────────────────────────
NUM_Q = ["What is the atomic number of {name}?",
         "What number is {name}?",
         "{name} is which element number?",
         "What's {name}'s atomic number?",
         "What number is {name} on the periodic table?",
         "Where is {name} on the periodic table?"]
SYM_Q = ["What is the symbol for {name}?",
         "What is {name}'s symbol?",
         "What is the chemical symbol of {name}?",
         "What's the symbol for {name}?",
         "How do you write {name}?",
         "What's {name}'s chemical symbol?"]
BYNUM_Q = ["What is element {num}?",
           "Which element is number {num}?",
           "What element has atomic number {num}?",
           "What's element number {num}?",
           "What element is {num}?",
           "Which element is atomic number {num}?"]
BYSYM_Q = ["What element is {sym}?",
           "Which element has the symbol {sym}?",
           "{sym} is which element?",
           "What does {sym} stand for?",
           "Which element is {sym}?"]
FAM_Q = ["What type of element is {name}?",
         "What family is {name} in?",
         "What category is {name}?",
         "Is {name} a metal?",
         "Is {name} a metal or nonmetal?",
         "What kind of element is {name}?"]
ABOUT_Q = ["Tell me about {name}.", "What is {name}?", "Describe {name}.",
           "Tell me about the element {name}.",
           "Give me info on {name}.",
           "What can you tell me about {name}?"]

CAPABILITIES = [
    (["What can you do?", "What can you answer?", "What are your capabilities?",
      "What are you capable of?"],
     "I answer facts about the 118 elements: names, symbols, numbers, families, groups, periods, blocks, and room-temperature state."),
    (["What questions can I ask?", "What can I ask you?", "What should I ask?",
      "Help", "What can you help with?"],
     "Try What is element 26, What is the symbol for Gold, or Which elements are noble gases."),
    (["What do you know about?", "What topics do you cover?", "What is your domain?"],
     "I know the periodic table: element properties, families, groups, periods, blocks, and phase at room temperature."),
    (["Who are you?", "What is this model?", "What are you?"],
     "I am a periodic table guide for the 118 known chemical elements."),
]


def about_answer(e):
    base = f"{e['name']} is element {e['number']}, {fam_with_article(e['family'])}"
    if e["phase"] and e["family"] != "noble gas":
        base += f", and is a {e['phase'].lower()} at room temperature"
    return base + "."


def family_answer(e):
    return f"{e['name']} is {fam_with_article(e['family'])}."


def metal_answer(e):
    cls = metal_class(e)
    if cls == "metal":
        return f"Yes, {e['name']} is a metal."
    if cls == "metalloid":
        return f"{e['name']} is a metalloid, not a metal or a nonmetal."
    return f"No, {e['name']} is a nonmetal."


class Corpus:
    def __init__(self):
        self.blocks = []
        self._seen = set()

    def qa(self, q, a):
        key = (q, a)
        if key in self._seen:
            return
        self._seen.add(key)
        self.blocks.append([f"Q: {q}", f"A: {a}"])

    def qa_variants(self, qs, a):
        for q in qs:
            self.qa(q, a)
            casual = q.lower().rstrip(" ?.")
            if casual and casual != q:
                self.qa(casual, a)

    def prose(self, text):
        self.blocks.append([text])

    def write(self, path, seed=1234):
        rng = random.Random(seed)
        rng.shuffle(self.blocks)
        lines = []
        for b in self.blocks:
            lines.extend(b)
            lines.append("")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return len(self.blocks)


def main():
    ap = argparse.ArgumentParser(description="Generate periodic-table training corpus")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "training_data" / "elements_rich.txt")
    ap.add_argument("--tokens-out", type=Path,
                    default=Path(__file__).parent.parent / "training_data" / "elements_special_tokens.txt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    els = load_elements()
    c = Corpus()

    for e in els:
        name, sym, num = e["name"], e["symbol"], e["number"]
        c.qa_variants([q.format(name=name) for q in NUM_Q],
                      f"{name} has atomic number {num}.")
        c.qa_variants([q.format(name=name) for q in SYM_Q],
                      f"The symbol for {name} is {sym}.")
        c.qa_variants([q.format(num=num) for q in BYNUM_Q],
                      f"Element {num} is {name}, symbol {sym}.")
        c.qa_variants([q.format(sym=sym) for q in BYSYM_Q],
                      f"{sym} is {name}, element number {num}.")
        c.qa_variants([q.format(name=name) for q in FAM_Q if "metal" not in q.lower()],
                      family_answer(e))
        c.qa_variants([f"Is {name} a metal?",
                       f"Is {name} a metal or nonmetal?"],
                      metal_answer(e))
        if e["group"]:
            c.qa_variants([f"What group is {name} in?",
                           f"What period and group is {name} in?"],
                          f"{name} is in group {e['group']}, period {e['period']}.")
        else:
            c.qa_variants([f"What period is {name} in?"],
                          f"{name} is in period {e['period']}.")
        c.qa_variants([f"What block is {name} in?",
                       f"Is {name} in the {e['block']}-block?"],
                      f"{name} is in the {e['block']}-block.")
        if e["phase"]:
            c.qa_variants([f"What state is {name} at room temperature?",
                           f"Is {name} a solid, liquid, or gas?"],
                          f"{name} is a {e['phase'].lower()} at room temperature.")
        c.qa_variants([q.format(name=name) for q in ABOUT_Q], about_answer(e))

    # ── Reverse lookups by family ─────────────────────────────────────────
    by_fam = {}
    for e in els:
        by_fam.setdefault(e["family"], []).append(e["name"])
    for fam, names in by_fam.items():
        c.qa_variants([f"How many {fam_plural(fam)} are there?"],
                      f"There are {len(names)} {fam_plural(fam)} among the 118 elements.")
        ans = list_answer(f"The {fam_plural(fam)} are", names)
        if len(names) > 6:
            ans = list_answer(f"{fam_plural(fam).capitalize()} include", names)
        c.qa_variants([f"Which elements are {fam_plural(fam)}?",
                       f"Name some {fam_plural(fam)}.",
                       f"Give me a {fam}.",
                       f"What are the {fam_plural(fam)}?",
                       f"what are {fam_plural(fam)}?",
                       f"List the {fam_plural(fam)}.",
                       f"Show me the {fam_plural(fam)}."],
                      ans)

    # ── Reverse lookups by period, group, block (from elements.json) ─────
    by_period = {}
    by_group = {}
    by_block = {}
    for e in els:
        by_period.setdefault(e["period"], []).append(e["name"])
        by_group.setdefault(e["group"], []).append(e["name"])
        by_block.setdefault(e["block"], []).append(e["name"])
    for period in sorted(by_period):
        names = by_period[period]
        c.qa_variants([f"Which elements are in period {period}?",
                       f"What elements are in period {period}?",
                       f"Name the elements in period {period}."],
                      list_answer(f"Period {period} includes", names))
    for group in sorted(by_group):
        names = by_group[group]
        c.qa_variants([f"Which elements are in group {group}?",
                       f"What elements are in group {group}?",
                       f"Name the elements in group {group}."],
                      list_answer(f"Group {group} includes", names))
    for block in sorted(by_block):
        names = by_block[block]
        label = f"{block}-block"
        c.qa_variants([f"Which elements are in the {label}?",
                       f"What elements are in the {label}?",
                       f"Name the {label} elements."],
                      list_answer(f"The {label} includes", names))

    # ── Reverse lookups by room-temp phase ────────────────────────────────
    gases = [e["name"] for e in els if e["phase"] == "Gas"]
    c.qa_variants(["Which elements are gases at room temperature?",
                   "Name the gaseous elements."],
                  list_answer("The gaseous elements are", gases, max_names=8))
    c.qa_variants(["Which elements are liquid at room temperature?",
                   "Name the liquid elements.",
                   "What elements are liquids?"],
                  "Only Mercury and Bromine are liquid at room temperature.")

    # ── Group families overview ───────────────────────────────────────────
    GROUPS = [("group 1", "the alkali metals, plus hydrogen"),
              ("group 2", "the alkaline earth metals"),
              ("group 17", "the halogens"),
              ("group 18", "the noble gases")]
    for g, desc in GROUPS:
        c.qa_variants([f"What is {g}?", f"What family is {g}?"],
                      f"{g.capitalize()} is {desc}.")

    # ── Capabilities / help ───────────────────────────────────────────────
    for qs, a in CAPABILITIES:
        c.qa_variants(qs, a)

    # ── General facts ─────────────────────────────────────────────────────
    GENERAL = [
        (["How many elements are there?", "How many chemical elements exist?"],
         "There are 118 known chemical elements."),
        (["What is the lightest element?", "What is the first element?"],
         "Hydrogen is the lightest element, atomic number 1."),
        (["What is the heaviest element?", "What is the last element?"],
         "Oganesson is the heaviest element, atomic number 118."),
        (["What is the most abundant element in the universe?"],
         "Hydrogen is the most abundant element in the universe."),
        (["What is the most abundant element in Earth's crust?"],
         "Oxygen is the most abundant element in Earth's crust."),
        (["What is the periodic table?"],
         "The periodic table arranges the chemical elements by atomic number into periods and groups."),
        (["What is an atomic number?", "What does the atomic number mean?"],
         "The atomic number is the number of protons in an element's nucleus."),
        (["What is a period?", "What is a row on the periodic table?"],
         "A period is a horizontal row of the periodic table. There are seven periods."),
        (["What is a group?", "What is a column on the periodic table?"],
         "A group is a vertical column of the periodic table. There are eighteen groups."),
    ]
    for qs, a in GENERAL:
        c.qa_variants(qs, a)

    # ── Prose ─────────────────────────────────────────────────────────────
    PROSE = [
        "The periodic table organizes all 118 known chemical elements by atomic number. Rows are called periods and columns are called groups. Elements in the same group share similar chemical properties.",
        "Most elements are metals. The metals include the alkali metals, alkaline earth metals, transition metals, post-transition metals, lanthanides, and actinides. Nonmetals, metalloids, halogens, and noble gases make up the rest.",
        "At room temperature most elements are solid. Eleven elements are gases, and only two, mercury and bromine, are liquid. The noble gases in group 18 are colorless and very unreactive.",
    ]
    for p in PROSE:
        c.prose(p)

    n = c.write(args.out, seed=args.seed)
    qa = sum(1 for b in c.blocks if len(b) == 2)
    print(f"Wrote {args.out}")
    print(f"  blocks: {n}  (Q&A: {qa}, prose: {n - qa})  elements: {len(els)}")

    names = [e["name"] for e in els]
    header = ("# Element names only, kept whole by the tokenizer so long names\n"
              "# don't fragment. Symbols are EXCLUDED on purpose: many collide\n"
              "# with English words and would corrupt tokenization. Pass with\n"
              "# --special-tokens. One token per line; blank lines / # ignored.\n\n")
    args.tokens_out.write_text(header + "\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {args.tokens_out}  ({len(names)} name tokens; symbols excluded to avoid word collisions)")


if __name__ == "__main__":
    main()
