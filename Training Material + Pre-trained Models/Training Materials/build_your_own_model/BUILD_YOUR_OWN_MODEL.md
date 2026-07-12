# Build Your Own Tiny-LLM Model — on Any Topic

This folder is a **template for generating training data** for a tiny (~6M
parameter) on-device language model. Pick a topic, hand this folder to an AI,
and it fills in the data. Run one command to produce the corpus, then train.

**Files**
- `corpus_lib.py` — reusable machinery. You never edit this.
- `TEMPLATE.py` — the fill-in-the-blanks generator. Ships with a 4-planet
  example so it runs as-is; you replace the data with your topic.
- `BUILD_YOUR_OWN_MODEL.md` — this file (guide + the AI prompt).

---

## Quickstart

1. **Run the example** to see it work:
   ```
   python TEMPLATE.py --out training_data/corpus.txt --tokens-out training_data/special_tokens.txt
   ```
2. **Give an AI your topic.** Open your AI of choice, attach this folder (or
   paste `TEMPLATE.py` + this file), and use the prompt below with your topic.
3. **Run the AI's filled-in `TEMPLATE.py`** with the same command.
4. **Train** on the resulting `corpus.txt` + `special_tokens.txt` (see the
   bottom of this file).

---

## The AI prompt (copy-paste, insert your topic)

> You are building a training corpus for a **tiny (~6M parameter) on-device
> language model** on the topic: **[YOUR TOPIC HERE]**.
>
> The model is a **memorizer, not a reasoner** — it can only answer a fact it
> was shown, and it cannot derive anything at run time. Your job is to write a
> generator (by editing `TEMPLATE.py`, which imports `corpus_lib.py`) that emits
> a rich, correct, self-consistent set of Q&A about the topic.
>
> Fill in the six "FILL IN" sections of `TEMPLATE.py`:
> 1. **Entities** — the things the model should know, each with a canonical name
>    and a consistent set of short attributes.
> 2. **Attribute questions** — 3–6 distinct phrasings per attribute.
> 3. **Reverse/aggregate lookups** — "which entities have attribute = value?"
>    (write these out; the model can't derive them). Lead answers with a count.
> 4. **Relationships** — entity-to-entity links.
> 5. **Lore** — a handful of short, factually-dense background passages, each
>    bridged with "tell me about X" questions.
> 6. **Precomputed reasoning** — anything needing a rule (comparisons,
>    superlatives, "what beats what"): compute it in Python and bake in the
>    answer. **Double-check every computed fact by hand — a wrong one becomes a
>    confidently-wrong model.**
>
> Follow the **Rules** and cover the **Pattern catalog** below. Aim for a few
> hundred to a few thousand distinct facts, each under many phrasings. When
> done, running `python TEMPLATE.py ...` must print `conflicts dropped: 0` or
> close to it, and the answers must be factually correct.

---

## Pattern catalog (the question types a good corpus covers)

Instantiate as many of these as your topic supports. Each is shown in the
running planet example.

| Pattern | Example question | Notes |
|---|---|---|
| **Identity / describe** | "What is Earth?" / "Tell me about Earth." | One clean description per entity. |
| **Attribute lookup** | "How many moons does Mars have?" | Per entity, per attribute. Many phrasings. |
| **Reverse / aggregate** | "Which planets are rocky?" | Write the LIST out; the model can't derive it. Lead with a count. |
| **Relationship** | "What comes after Earth?" | Entity → entity. Include the reverse ("before"). |
| **Multi-hop / chain** | "What does X's parent lead to?" | Precompute the chain. |
| **Superlative / extreme** | "Which planet has the most moons?" | Compute the max/min/first/last. |
| **Comparison** | "Which is bigger, Earth or Mars?" | Compute it. Pick a FEW meaningful pairs, never all N×N. |
| **Count / statistic** | "How many rocky planets are there?" | Aggregate. |
| **Categorization** | "What are the categories of X?" | If the domain groups things. |
| **Derived reasoning** | "What is X weak to?" | Apply a RULE (a chart/formula) to the data in Python. Verify. |
| **Lore / background** | "Tell me about the Solar System." | Short prose, bridged with "tell me about X". |
| **Procedural / how-to** | "How do you do X?" | If the domain has procedures/steps. |

---

## Rules (the hard-won lessons — follow these)

1. **Many phrasings per fact, not lowercase/punctuation duplicates.** Real
   people ask the same thing many ways ("How many moons does Mars have?" /
   "Does Mars have moons?" / "What's Mars's moon count?"). Give distinct
   wordings. Do NOT add `mars`/`Mars?`/`MARS` variants — casing is handled at
   inference, and lowercased names fragment into tokens the model can't use.
2. **One answer per question.** The machinery enforces this (first-write-wins).
   If two blocks emit the same question with different answers, the later one is
   dropped. Don't rely on it — design questions so each has one true answer.
3. **Short, factually-dense answers.** A tiny model memorizes tight text far
   better than long flowery prose, and short answers drift less.
4. **Precompute all reasoning.** The model cannot compare, aggregate, or apply
   rules at run time. Compute those in Python and store the answers as flat
   facts. This is the single biggest lever for making it look "smart".
5. **Verify computed facts.** A wrong precomputed fact (a bad comparison, a
   wrong rule table) trains the model to be confidently wrong. Hand-check a
   sample of every computed category.
6. **Whole-word entity names.** Return your entity names from `build()`; they
   become special tokens so names stay intact instead of fragmenting.
7. **Lead aggregate answers with a count** ("There are 4 rocky planets: ...") so
   a truncated answer is still useful.
8. **Bridge lore with "tell me about X".** A passage trained only as bare prose
   has no path from a question to it. Give each passage its questions.
9. **Avoid combinatorial explosion.** N entities have N² pairs — don't emit all
   of them. Do superlatives, neighbours, and a few meaningful comparisons, not
   every pair.
10. **Coverage isn't free.** A ~6M model has limited capacity. Core facts under
    many phrasings beat sprawling, rarely-asked coverage. When in doubt, deepen
    (more phrasings of the facts that matter) rather than widen.

---

## STEP 3: train

Once your generator prints a healthy corpus, train the model. This kit no longer
bundles a trainer — use the canonical tiny-LLM trainer in the repo's `Training/`
folder (`Training/train_tiny_model_gpu.py` for GPU, `Training/train_tiny_model.py`
for CPU):

```
python Training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training_data/corpus.txt \
    --special-tokens training_data/special_tokens.txt \
    --epochs 250 --lr 6e-4 --batch-size 64 \
    --val-frac 0.1 --early-stopping-patience 5 \
    --out ./out_mymodel
```

- `--special-tokens` keeps your entity names whole.
- `--val-frac 0.1 --early-stopping-patience 5` stops when it stops improving —
  faster, and avoids overfitting.
- Adjust the preset for your size/hardware (see the trainer's `--help`).
- After saving the model, the trainer also writes a `domain_vocab.txt` into
  `--out` — a word-list extracted from your corpus (pass `--no-domain-vocab` to
  skip it). The browser converter auto-loads that file into its "Domain words"
  field, where it powers an on-device **refusal gate**: any prompt that contains
  none of your domain words is answered with the converter's "Refusal answer"
  instead of a hallucination. Clearing both converter fields disables the gate.
  To build or tune the word-list by hand, use
  `Training/training_scripts/extract_domain_vocab.py`.

The model learns to stop on its own (EOS is trained), so at inference you can
let it terminate naturally rather than hard-capping length.

---

## How much data?

- **Small topic** (tens of entities): a few hundred to ~2,000 facts. Lean on
  phrasings and reasoning to add depth.
- **Larger topic** (~150 entities like the Kanto Pokédex example this template
  is derived from): ~1,500 distinct facts × several phrasings ≈ 10k–13k Q&A.

Start small, train, test the exact questions you care about, then add the
categories that came up short. Iterating on the data beats fiddling with the
model.
