#!/usr/bin/env python3
"""
scan_contamination.py — catch a model that talks about the wrong domain.

Trained a Pokemon model but it says "HardwareOne"? This finds out WHERE — in the
corpus you trained on, in the weights themselves, or in the .bin that's actually
on the device — instead of guessing. It scans three things for words that should
NOT appear:

  --corpus FILE.txt   the training text you fed the trainer
  --model  DIR        a trained HuggingFace model dir (loads it and generates)
  --bin    FILE.bin   a converted LLM1 model.bin (reads its EMBEDDED vocab)

Tell it what must not appear, via a built-in preset or an explicit list:

  --preset hardwareone            # built-in HardwareOne vocabulary
  --forbidden pikachu,kanto,...   # your own comma-separated words

Or ask which domain an artifact actually is (scores it against every preset):

  --identify

Exit code is non-zero if any forbidden word is found, so it can gate training/CI.

Examples
--------
  # Did my Pokemon corpus pick up any HardwareOne text?
  python scan_contamination.py --corpus training_data/pokemon_kanto.txt --preset hardwareone

  # Does my TRAINED MODEL actually SAY anything HardwareOne?
  python scan_contamination.py --model ./out_kanto_pokemon_master --preset hardwareone

  # Which domain is the model.bin that's really on my device?
  python scan_contamination.py --bin /path/to/model.bin --identify
"""
import argparse
import re
import sys
from pathlib import Path

# ── Domain vocabularies ─────────────────────────────────────────────────────
# Lowercase substrings. Kept specific enough not to false-positive across
# domains (e.g. "sensor"/"thermal" are HardwareOne markers but also appear in
# Pokedex flavor text, so they are weighted by count, not treated as proof).
PRESETS = {
    "hardwareone": [
        "hardwareone", "hardware one", "openwifi", "wifistatus", "wifiadd",
        "espnow", "blesecret", "neopixel", "ledeffect", "gamepad", "esp32",
        "firmware", "i2c", " oled", "help agent", "wifiscan", "wifilist",
        "openhttp", "openble", "llmload", "thermal camera", "mlx90640",
    ],
    "pokemon": [
        "pokemon", "pikachu", "charizard", "bulbasaur", "squirtle", "kanto",
        "pokedex", "evolve", "gym leader", "gym badge", "team rocket",
        "elite four", "mewtwo", "eevee", "type matchup", "trainer",
    ],
    "elements": [
        "periodic table", "atomic number", "atomic mass", "isotope",
        "electron", "proton", "neutron", "noble gas", "valence",
        "hydrogen", "helium", "oxygen", "carbon", "the element",
    ],
}

# Identity/scope answers are where a mis-swapped template shows up first.
DEFAULT_PROBES = [
    "Who are you?",
    "What are you?",
    "What can you do?",
    "What can you answer?",
    "What can you help with?",
    "What do you know about?",
    "What is your domain?",
    "Help",
]


def build_forbidden(args):
    words = []
    for p in args.preset:
        if p not in PRESETS:
            sys.exit(f"unknown preset '{p}' (choices: {', '.join(PRESETS)})")
        words += PRESETS[p]
    if args.forbidden:
        words += [w.strip().lower() for w in args.forbidden.split(",") if w.strip()]
    # de-dup, keep order
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def scan_text(text, forbidden):
    """Return {word: [example snippet, ...]} for every forbidden word found."""
    low = text.lower()
    hits = {}
    for w in forbidden:
        idx = low.find(w)
        if idx >= 0:
            start = max(0, idx - 30)
            end = min(len(text), idx + len(w) + 30)
            snippet = text[start:end].replace("\n", " ").strip()
            hits[w] = snippet
    return hits


def iter_bin_strings(path, min_len=3):
    """Yield runs of printable ASCII from a binary file (like `strings`)."""
    data = Path(path).read_bytes()
    run = bytearray()
    for b in data:
        if 32 <= b < 127:
            run.append(b)
        else:
            if len(run) >= min_len:
                yield run.decode("ascii", "ignore")
            run = bytearray()
    if len(run) >= min_len:
        yield run.decode("ascii", "ignore")


def identify(text):
    """Score text against every preset; return sorted (domain, count)."""
    low = text.lower()
    scores = []
    for name, words in PRESETS.items():
        c = sum(low.count(w) for w in words)
        scores.append((name, c))
    scores.sort(key=lambda x: -x[1])
    return scores


def report(kind, name, hits, identity_scores=None):
    print("=" * 72)
    print(f"SCAN: {kind}  {name}")
    print("=" * 72)
    if identity_scores is not None:
        print("Domain fingerprint (embedded-vocab / text word counts):")
        for dom, c in identity_scores:
            bar = "#" * min(40, c)
            print(f"  {dom:14s} {c:6d}  {bar}")
        top = identity_scores[0]
        if top[1] == 0:
            print("  -> no known-domain vocabulary detected")
        else:
            print(f"  -> looks like: {top[0].upper()}")
        print()
    if not hits:
        print("CLEAN — no forbidden words found.")
        return 0
    print(f"CONTAMINATED — {len(hits)} forbidden word(s) found:")
    for w, snip in hits.items():
        print(f"  [{w}]  …{snip}…")
    return 1


def scan_corpus(path, forbidden, do_identify):
    text = Path(path).read_text(errors="ignore")
    hits = scan_text(text, forbidden)
    ids = identify(text) if do_identify else None
    return report("corpus", path, hits, ids)


def scan_bin(path, forbidden, do_identify):
    text = "\n".join(iter_bin_strings(path))
    hits = scan_text(text, forbidden)
    ids = identify(text) if do_identify else None
    return report("bin (embedded vocab)", path, hits, ids)


def scan_model(model_dir, forbidden, probes, do_identify, max_new_tokens=40):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("--model needs torch + transformers (install requirements.txt)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    torch.manual_seed(0)
    outputs = []
    print(f"(generating {len(probes)} probe answers from {model_dir} …)")
    for q in probes:
        prompt = f"Q: {q}\nA:"                       # firmware framing
        ids = tok(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, repetition_penalty=1.3,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        ans = ans.split("Q:")[0].strip()            # firmware stops at next Q:
        outputs.append((q, ans))
        print(f"  Q: {q}\n     -> {ans!r}")
    joined = "\n".join(a for _, a in outputs)
    hits = scan_text(joined, forbidden)
    ids = identify(joined) if do_identify else None
    return report("model output", model_dir, hits, ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", type=Path, help="training text file to scan")
    src.add_argument("--model", type=Path, help="trained HF model dir (loads + generates)")
    src.add_argument("--bin", dest="binfile", type=Path, help="converted LLM1 model.bin to scan")
    ap.add_argument("--preset", action="append", default=[],
                    help=f"forbidden-word preset ({', '.join(PRESETS)}); repeatable")
    ap.add_argument("--forbidden", default="", help="extra comma-separated forbidden words")
    ap.add_argument("--identify", action="store_true",
                    help="also report which domain the artifact looks like")
    ap.add_argument("--probe", action="append", default=[],
                    help="extra probe question for --model (repeatable)")
    args = ap.parse_args()

    # --identify with no preset/forbidden is still useful (just fingerprint).
    forbidden = build_forbidden(args)
    if not forbidden and not args.identify:
        sys.exit("give --preset/--forbidden, or --identify")

    if args.corpus:
        rc = scan_corpus(args.corpus, forbidden, args.identify)
    elif args.binfile:
        rc = scan_bin(args.binfile, forbidden, args.identify)
    else:
        probes = DEFAULT_PROBES + args.probe
        rc = scan_model(args.model, forbidden, probes, args.identify)
    sys.exit(rc)


if __name__ == "__main__":
    main()
