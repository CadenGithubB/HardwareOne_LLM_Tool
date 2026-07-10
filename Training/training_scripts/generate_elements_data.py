#!/usr/bin/env python3
"""Generate a periodic-table (118 elements) training corpus.

Q:/A: pairs in the same format as the other datasets, with short answers that
fit the device's ~45-token context. Reads training_data/elements.json (built
from the Bowserinator Periodic-Table-JSON dataset, filtered to elements 1-118,
families normalized, room-temp phase only asserted for naturally-occurring
elements <=94 where it is well established).

Usage:
    python training_scripts/generate_elements_data.py
"""
import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).parent.parent / "training_data" / "elements.json"


def load_elements():
    return json.loads(DATA.read_text(encoding="utf-8"))


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


def fam_plural(fam):
    return fam + ("es" if fam.endswith("s") else "s")


def about_answer(e):
    base = f"{e['name']} is element {e['number']}, a {e['family']}"
    # Skip the phase clause for noble gases (saying "a noble gas, a gas" is redundant).
    if e["phase"] and e["family"] != "noble gas":
        base += f", and is a {e['phase'].lower()} at room temperature"
    return base + "."


class Corpus:
    def __init__(self):
        self.blocks = []
        self._seen = set()  # dedup exact (question, answer) pairs

    def qa(self, q, a):
        key = (q, a)
        if key in self._seen:
            return
        self._seen.add(key)
        self.blocks.append([f"Q: {q}", f"A: {a}"])

    def qa_variants(self, qs, a):
        # Emit each question formally AND a casual lowercase/unpunctuated variant
        # so the model matches casual typing (HardwareOne gold standard style).
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
        c.qa_variants([q.format(name=name) for q in FAM_Q],
                      f"{name} is a {e['family']}.")
        # group / period
        if e["group"]:
            c.qa_variants([f"What group is {name} in?",
                           f"What period and group is {name} in?"],
                          f"{name} is in group {e['group']}, period {e['period']}.")
        else:
            c.qa_variants([f"What period is {name} in?"],
                          f"{name} is in period {e['period']}.")
        # block
        c.qa_variants([f"What block is {name} in?"],
                      f"{name} is in the {e['block']}-block.")
        # phase (only where well established)
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
        if len(names) <= 7:
            listing = ", ".join(names[:-1]) + ", and " + names[-1]
            ans = f"The {fam_plural(fam)} are {listing}."
        else:
            ex = names[:6]
            ans = f"{fam_plural(fam).capitalize()} include " + ", ".join(ex[:-1]) + ", and " + ex[-1] + "."
        c.qa_variants([f"Which elements are {fam_plural(fam)}?",
                       f"Name some {fam_plural(fam)}.",
                       f"Give me a {fam}.",
                       f"What are the {fam_plural(fam)}?",
                       f"what are {fam_plural(fam)}?",
                       f"List the {fam_plural(fam)}.",
                       f"Show me the {fam_plural(fam)}."],
                      ans)

    # ── Reverse lookups by room-temp phase ────────────────────────────────
    gases = [e["name"] for e in els if e["phase"] == "Gas"]
    c.qa_variants(["Which elements are gases at room temperature?",
                   "Name the gaseous elements."],
                  "The gaseous elements are " + ", ".join(gases[:-1]) + ", and " + gases[-1] + ".")
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

    # Special tokens: element names + multi-letter symbols (single-letter
    # symbols are already atomic and risk matching inside words).
    # NAMES ONLY. Symbols are intentionally excluded: 24 of them (At, In, As,
    # No, Be, He, Ne, Na, Si, Co, Ti, Sc, Re, Os, Po, Pa, Pb, Sb, Am, Ar, Cs,
    # Md, Pm, Ds) are also common English words and, as special tokens, would
    # hijack those words throughout the corpus (e.g. "At room temperature" ->
    # the Astatine token). HardwareOne's special tokens are long lowercase
    # commands precisely to avoid this kind of collision.
    names = [e["name"] for e in els]
    header = ("# Element names only, kept whole by the tokenizer so long names\n"
              "# don't fragment. Symbols are EXCLUDED on purpose: many collide\n"
              "# with English words and would corrupt tokenization. Pass with\n"
              "# --special-tokens. One token per line; blank lines / # ignored.\n\n")
    args.tokens_out.write_text(header + "\n".join(names) + "\n", encoding="utf-8")
    print(f"Wrote {args.tokens_out}  ({len(names)} name tokens; symbols excluded to avoid word collisions)")


if __name__ == "__main__":
    main()
