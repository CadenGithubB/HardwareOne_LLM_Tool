#!/usr/bin/env python3
"""
eval_qa_accuracy.py — how often does a trained model give the EXACT answer,
decoding the way the ESP32 firmware does?

The trainers print a handful of sample answers; this scores many questions
with the device's decoding: greedy, its repetition penalty, stop on the Q:/A:
tokens or EOS, at most 80 new tokens, answer cut after 2 sentences. It reports
exact-match accuracy and word overlap (F1) for
  * trained questions (a sample from --text), and
  * held-out phrasings (--val-text, e.g. the kit's val.txt): how the model
    copes with wordings it never saw,
per question type when a categories.json (written by the build-your-own-model
kit) sits next to --text or is given with --categories.

Simulate the device more closely:
  --zero-linear-biases  for models trained before the trainers froze these
                        biases: the converter drops them, so the device runs
                        without them
  --int8-group 128      quantize the weights exactly like the converter (INT8,
                        one scale per group of values; use the group size you
                        convert with)
Compare decoding settings with several --rep-penalty values, and
--penalty-scope generated to penalize only the answer's own tokens (the
device, like Hugging Face, may also penalize the question's tokens).

Usage:
  python training_scripts/eval_qa_accuracy.py --model ./out_mymodel \\
      --text training_data/corpus.txt --val-text training_data/val.txt \\
      --rep-penalty 1.5 1.2 1.0 --zero-linear-biases --int8-group 128
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_WORD = re.compile(r"[a-z0-9']+")


# ── Pure helpers (no torch) ──────────────────────────────────────────────────
def read_pairs(paths):
    """(question, marker, answer) for every Q:/A: or Q:/Do: block in the files."""
    pairs = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for block in text.split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) == 2 and lines[0].startswith("Q: "):
                marker, _, answer = lines[1].partition(": ")
                if marker in ("A", "Do") and answer.strip():
                    pairs.append((lines[0][3:].strip(), marker + ":", answer.strip()))
    return pairs


def cut_sentences(text, n):
    """Keep at most n sentences, counting sentence ends the way the kit's
    checks do (an abbreviation like "Lt." counts as one)."""
    if n <= 0:
        return text
    for i, m in enumerate(_SENTENCE_END.finditer(text), 1):
        if i == n:
            return text[:m.end()]
    return text


def normalize(text):
    return " ".join(text.split())


def word_f1(pred, ref):
    p, r = _WORD.findall(pred.lower()), _WORD.findall(ref.lower())
    if not p or not r:
        return float(p == r)
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def score(pairs, answer_fn, max_sentences):
    """Run answer_fn(question, marker) on every pair; returns one result dict each."""
    results = []
    for q, marker, ref in pairs:
        got = normalize(cut_sentences(answer_fn(q, marker), max_sentences))
        want = normalize(cut_sentences(ref, max_sentences))
        results.append({"question": q, "expected": want, "got": got,
                        "exact": got == want, "f1": word_f1(got, want)})
    return results


def summarize(results, categories):
    """Overall and per-category exact match / F1."""
    def stats(items):
        n = len(items)
        return {"n": n,
                "exact": sum(r["exact"] for r in items) / n if n else 0.0,
                "f1": sum(r["f1"] for r in items) / n if n else 0.0}
    by_cat = {}
    for r in results:
        by_cat.setdefault(categories.get(r["question"], "other"), []).append(r)
    return {"all": stats(results),
            "by_category": {c: stats(items) for c, items in by_cat.items()}}


def print_table(title, summaries):
    """summaries: {penalty_label: summarize(...)} for one question set."""
    labels = list(summaries)
    first = summaries[labels[0]]
    print(f"\n== {title} ==")
    print("  " + " " * 24 + "".join(f"{lab:>20s}" for lab in labels))
    row = "".join(f"{s['all']['exact']:>12.1%} F1 {s['all']['f1']:.2f}" for s in summaries.values())
    print(f"  {'all (n=' + str(first['all']['n']) + ')':24s}{row}")
    if len(first["by_category"]) > 1:
        for cat in first["by_category"]:
            cells = "".join(f"{s['by_category'][cat]['exact']:>20.1%}" for s in summaries.values())
            print(f"  {cat[:17] + ' (n=' + str(first['by_category'][cat]['n']) + ')':24s}{cells}")


# ── Model side (torch) ───────────────────────────────────────────────────────
def zero_linear_biases(model):
    """Same as the trainers' zero_linear_biases(): the converter exports only
    the weights of c_attn, c_proj, c_fc and mlp.c_proj, so the device runs
    their biases as zero."""
    n = 0
    for block in model.transformer.h:
        for layer in (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
            if getattr(layer, "bias", None) is not None:
                layer.bias.data.zero_()
                n += layer.bias.numel()
    return n


def simulate_int8(model, group):
    """Round-trip the weights through the converter's INT8 scheme (index.html
    quantize worker): symmetric, one scale per `group` consecutive values of
    each matrix in [out, in] layout, scale = max|x| / 127, round half up,
    clamp to [-128, 127]. The embedding (also the output layer when tied) and
    Q/K/V/O/up/down are quantized; LayerNorms and position embeddings stay FP32."""
    import torch

    def fake(w_out_in):
        flat = w_out_in.reshape(-1)
        n = flat.numel()
        x = torch.cat([flat, flat.new_zeros((-n) % group)]).view(-1, group)
        absmax = x.abs().amax(dim=1, keepdim=True)
        scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
        q = torch.clamp(torch.floor(x / scale + 0.5), -128, 127)
        return (q * scale).view(-1)[:n].view_as(w_out_in)

    with torch.no_grad():
        wte = model.transformer.wte.weight                     # [vocab, dim]
        wte.copy_(fake(wte))
        dim = model.config.n_embd
        for block in model.transformer.h:
            w = block.attn.c_attn.weight                       # Conv1D [in=dim, out=3*dim]
            for i in range(3):                                 # Q, K, V, each as [out, in]
                cols = slice(i * dim, (i + 1) * dim)
                w[:, cols] = fake(w[:, cols].t().contiguous()).t()
            for layer in (block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
                layer.weight.copy_(fake(layer.weight.t().contiguous()).t())


def make_answer_fn(model, tokenizer, device, rep_penalty, scope, max_new, context):
    """Greedy decoding like the firmware; returns answer_fn(question, marker)."""
    import torch

    stop_ids = {tokenizer.eos_token_id}
    for marker in ("Q:", "A:"):
        enc = tokenizer.encode(marker, add_special_tokens=False)
        if enc:
            stop_ids.add(enc[0])
    stop_ids.discard(None)

    def answer(question, marker):
        prompt = tokenizer.encode(f"Q: {question}\n{marker}", add_special_tokens=False)
        if len(prompt) >= context:
            return ""
        ids = torch.tensor([prompt], device=device)
        past, out_ids = None, []
        seen = set(prompt) if scope == "all" else set()
        for _ in range(max_new):
            if len(prompt) + len(out_ids) >= context:
                break
            with torch.no_grad():
                out = model(input_ids=ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            if rep_penalty != 1.0 and seen:
                idx = torch.tensor(sorted(seen), device=logits.device)
                vals = logits[idx]
                logits[idx] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
            nxt = int(torch.argmax(logits))
            if nxt in stop_ids:
                break
            out_ids.append(nxt)
            seen.add(nxt)
            ids = torch.tensor([[nxt]], device=device)
        return tokenizer.decode(out_ids, skip_special_tokens=False)

    return answer


def main():
    ap = argparse.ArgumentParser(description="Exact-match accuracy with the device's decoding.")
    ap.add_argument("--model", type=Path, required=True, help="Trained model folder (the trainer's --out).")
    ap.add_argument("--text", type=Path, nargs="+", required=True, help="Training corpus file(s).")
    ap.add_argument("--val-text", type=Path, nargs="*", default=[],
                    help="Held-out Q&A file(s), e.g. the kit's val.txt.")
    ap.add_argument("--categories", type=Path, default=None,
                    help="categories.json (question -> type). Default: next to the first --text file.")
    ap.add_argument("--sample", type=int, default=400,
                    help="How many trained questions to score (random, seeded); 0 = all. Default 400.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rep-penalty", type=float, nargs="+", default=[1.5],
                    help="Repetition penalty value(s) to compare (device default: 1.5).")
    ap.add_argument("--penalty-scope", choices=("all", "generated"), default="all",
                    help="Penalize tokens of the question and answer (all, like Hugging Face) or only "
                         "the answer's own tokens (generated).")
    ap.add_argument("--max-new-tokens", type=int, default=80, help="Device hard cap (default 80).")
    ap.add_argument("--max-sentences", type=int, default=2, help="Device sentence limit (default 2).")
    ap.add_argument("--context", type=int, default=None,
                    help="Context cap in tokens (prompt + answer). Default: the model's n_positions.")
    ap.add_argument("--zero-linear-biases", action="store_true",
                    help="Zero the linear-layer biases the converter drops (for models trained before "
                         "the trainers froze them).")
    ap.add_argument("--int8-group", type=int, default=0, metavar="N",
                    help="Simulate the converter's INT8 weights with this group size (e.g. 128).")
    ap.add_argument("--show-failures", type=int, default=8,
                    help="Wrong answers to print per question set (first penalty only).")
    ap.add_argument("--json", type=Path, default=None, help="Also write the full results here.")
    args = ap.parse_args()
    for path in [args.model, *args.text, *args.val_text]:
        if not path.exists():
            sys.exit(f"Not found: {path}")

    trained = read_pairs(args.text)
    held_out = read_pairs(args.val_text)
    if not trained and not held_out:
        sys.exit("No Q&A pairs found in --text / --val-text.")
    total_trained = len(trained)
    if args.sample and len(trained) > args.sample:
        trained = random.Random(args.seed).sample(trained, args.sample)

    cat_path = args.categories or args.text[0].parent / "categories.json"
    categories = json.loads(cat_path.read_text(encoding="utf-8")) if cat_path.is_file() else {}

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install -r Training/requirements.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(str(args.model))
    model = GPT2LMHeadModel.from_pretrained(str(args.model)).to(device).eval()
    notes = []
    if args.zero_linear_biases:
        notes.append(f"linear biases zeroed ({zero_linear_biases(model):,} values), like the device")
    if args.int8_group:
        simulate_int8(model, args.int8_group)
        notes.append(f"INT8 weights simulated (group {args.int8_group}), like the converter")
    context = min(args.context or model.config.n_positions, model.config.n_positions)

    print(f"Model: {args.model}" + (f"  [{'; '.join(notes)}]" if notes else "  [weights as trained]"))
    print(f"Decoding: greedy; stop on Q:/A:/EOS; <= {args.max_new_tokens} new tokens; "
          f"<= {args.max_sentences} sentences; context {context}; "
          f"repetition penalty on {'question + answer' if args.penalty_scope == 'all' else 'answer'} tokens")
    if categories:
        print(f"Question types from {cat_path}")

    sets = [(f"trained questions ({len(trained)} of {total_trained})", "trained", trained)]
    if held_out:
        sets.append((f"held-out phrasings ({len(held_out)}, never trained)", "held_out", held_out))
    report = {"model": str(args.model), "notes": notes, "results": {}}
    tables = {key: {} for _t, key, _p in sets}
    for penalty in args.rep_penalty:
        answer_fn = make_answer_fn(model, tokenizer, device, penalty, args.penalty_scope,
                                   args.max_new_tokens, context)
        label = f"penalty {penalty:g}"
        report["results"][label] = {}
        for _title, key, pairs in sets:
            results = score(pairs, answer_fn, args.max_sentences)
            summary = summarize(results, categories)
            tables[key][label] = summary
            report["results"][label][key] = {**summary, "items": results}

    for title, key, _pairs in sets:
        print_table(title, tables[key])
    first = f"penalty {args.rep_penalty[0]:g}"
    for title, key, _pairs in sets:
        wrong = [r for r in report["results"][first][key]["items"] if not r["exact"]]
        if wrong and args.show_failures:
            print(f"\nWrong answers — {title}, {first} (first {min(len(wrong), args.show_failures)} "
                  f"of {len(wrong)}):")
            for r in wrong[:args.show_failures]:
                print(f"  Q: {r['question']}\n     expected: {r['expected']}\n     got:      {r['got']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
