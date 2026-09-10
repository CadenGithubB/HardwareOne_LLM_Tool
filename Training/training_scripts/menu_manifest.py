"""menu_manifest.py — guided-input menu bookkeeping for the LLM generators.

Emits ``menu_manifest.json`` (spec: docs/LLM_GUIDED_MENU_SPEC.md, section 2). A
generator registers a handful of groups; each group has question TEMPLATES (an
archetype phrasing with a single ``{}`` entity slot, or no slot for a canned
question) plus an ENTITY roster. The converter auto-loads this manifest next to
``*domain_vocab.txt`` and bakes it into the ``.bin`` so the firmware can offer a
"pick a question" menu instead of free-text. The composed ``template + entity``
string must be a BYTE-EXACT line the model was trained on — that is the whole
point — so templates come from the FIRST phrasing of each ``*_Q`` list and
entities keep their corpus casing.

This is a small, self-contained helper the standalone generators
(generate_pokemon_data.py / generate_elements_data.py) import directly. The kit
mirrors the same MenuBuilder inside build_your_own_model/corpus_lib.py so a
copied kit stays self-contained; keep the two in sync with the spec's caps.
"""
import json
from pathlib import Path

MENU_VERSION = 1
SLOT = "{}"  # the single entity slot; 0 or 1 per template

# Caps (spec section 2). The converter and firmware re-validate; we fail here
# first with a readable error so bad data never reaches a device.
MAX_GROUPS = 8
MAX_TEMPLATES_PER_GROUP = 64
MAX_ENTITIES_PER_GROUP = 1024
MAX_NAME_BYTES = 32
MAX_Q_BYTES = 120
MAX_ENTITY_BYTES = 48


def _bytelen(s):
    return len(str(s).encode("utf-8"))


class MenuGroup:
    """One menu group: question templates + a shared entity roster."""

    def __init__(self, name):
        self.name = name
        self.templates = []   # list of {"q": str, "label": str(optional)}
        self.entities = []    # list of str

    def template(self, q, label=None):
        """Register one question archetype. ``q`` is a corpus-exact phrasing with
        at most one ``{}`` slot; ``label`` is an optional short (<=20 char)
        display form for small screens (OLED). Returns self for chaining."""
        if q.count(SLOT) > 1:
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
    """Collects groups and serializes menu_manifest.json."""

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
        if len(self.groups) > MAX_GROUPS:
            errs.append(f"{len(self.groups)} groups > cap {MAX_GROUPS}")
        for g in self.groups:
            if _bytelen(g.name) > MAX_NAME_BYTES:
                errs.append(f"group name {g.name!r} > {MAX_NAME_BYTES} bytes")
            if len(g.templates) > MAX_TEMPLATES_PER_GROUP:
                errs.append(f"group {g.name!r}: {len(g.templates)} templates "
                            f"> cap {MAX_TEMPLATES_PER_GROUP}")
            if len(g.entities) > MAX_ENTITIES_PER_GROUP:
                errs.append(f"group {g.name!r}: {len(g.entities)} entities "
                            f"> cap {MAX_ENTITIES_PER_GROUP}")
            for t in g.templates:
                if t["q"].count(SLOT) > 1:
                    errs.append(f"group {g.name!r}: template {t['q']!r} has >1 slot")
                if _bytelen(t["q"]) > MAX_Q_BYTES:
                    errs.append(f"group {g.name!r}: q {t['q']!r} > {MAX_Q_BYTES} bytes")
            for e in g.entities:
                if _bytelen(e) > MAX_ENTITY_BYTES:
                    errs.append(f"group {g.name!r}: entity {e!r} > {MAX_ENTITY_BYTES} bytes")
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
        entities) counts for the caller to print."""
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


# ── Module-level convenience (spec section 11 names menu_group/write_menu) ──
# A generator that prefers free functions over holding a MenuBuilder can use
# these; they operate on one shared default builder.
_default = MenuBuilder()


def menu_group(name):
    return _default.menu_group(name)


def write_menu(out_path):
    return _default.write_menu(out_path)


def reset_menu():
    global _default
    _default = MenuBuilder()
