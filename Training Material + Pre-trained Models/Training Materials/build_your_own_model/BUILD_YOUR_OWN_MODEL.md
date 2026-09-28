# Build Your Own Tiny-LLM Model — on Any Topic

This folder is a **template for generating training data** for a tiny (~6M
parameter) on-device language model. Pick a topic, hand this folder to an AI,
and it fills in the facts and how they're asked. Run one command to build and
check the corpus, review the facts it flags, then train.

**Files**
- `facts.json` — the facts the model will learn, each with a `source`. Ships
  with a 4-planet example; you replace it with your topic.
- `TEMPLATE.py` — the fill-in-the-blanks generator: how each fact is phrased as
  questions and answers, plus `verify()` checks on the facts. Runs as-is on the
  example.
- `corpus_lib.py` — reusable machinery and checks. You never edit this.
- `BUILD_YOUR_OWN_MODEL.md` — this file (guide + the AI prompt).

---

## Hard limits

| Limit | Why |
|---|---|
| Answers: at most 2 sentences and ~30 words, on one line | The device stops generating after 2 sentences, and a whole Q&A pair has to fit the model's short context. |
| Prose passages: at most ~55 words, on one line | Prose is trained in 128-token blocks; longer text gets cut. |
| One answer per question | A model trained on two answers to one question blends them. |
| Every fact has a `source`, or is marked `"unverified"` | The model repeats its training data with full confidence, right or wrong. |
| The model can't reason | Comparisons, superlatives, counts and lists must be computed in Python and stored as facts. |
| Guided menu: ≤8 groups; ≤64 templates and ≤1024 entities per group; group name ≤32 B, question ≤120 B, entity ≤48 B; ≤32 KB in total | The converter and firmware reject anything bigger. |

Every run checks these (see [What a run checks and writes](#what-a-run-checks-and-writes)).

---

## Quickstart

Run every command from this folder:

```
cd "Training Material + Pre-trained Models/Training Materials/build_your_own_model"
```

1. **Run the example** to see it work:
   ```
   python TEMPLATE.py --strict
   ```
   It checks the data and writes everything to `training_data/` (see
   [What a run checks and writes](#what-a-run-checks-and-writes)).
2. **Give an AI your topic.** Open your AI of choice, attach this folder (or
   paste `facts.json`, `TEMPLATE.py` and this file), and use the prompt below
   with your topic.
3. **Run the AI's version** with the same command until it passes. Then read
   `training_data/UNVERIFIED.md` and check every fact it lists.
4. **Train** on the result (see [STEP 3](#step-3-train)).

---

## The AI prompt (copy-paste, insert your topic)

> You are building a training corpus for a **tiny (~6M parameter) on-device
> language model** on the topic: **[YOUR TOPIC HERE]**.
>
> The model is a **memorizer, not a reasoner** — it can only answer a fact it
> was shown, and it cannot derive anything at run time. It also repeats
> whatever it was trained on with full confidence, so **a wrong fact in the
> data becomes a confidently wrong model.**
>
> Follow the **Hard limits**, the **Rules**, and the **Pattern catalog** in
> `BUILD_YOUR_OWN_MODEL.md`. Aim for a few hundred to a few thousand distinct
> facts, each under many phrasings. Work in this order:
>
> 1. **Agree the scope with the user.** Which entities does the model cover —
>    and is that list complete? If it isn't, every superlative must say so
>    ("the most moons *of the inner planets*").
> 2. **Put the facts in `facts.json`, each with a `source`.** Write the
>    entities (each with a canonical name, a consistent set of short
>    attributes, and a one-sentence `desc`), any entity-to-entity
>    relationships, and a handful of short lore passages. Take facts from a
>    reliable reference and put its URL or citation in `source`. If a fact
>    comes from your own memory, set `source` to `"unverified"`. **Never
>    invent or guess a source**: "unverified" is an honest answer, and a person
>    checks those facts before training.
> 3. **Fill in the "FILL IN" sections of `TEMPLATE.py`** (it imports
>    `corpus_lib.py`):
>    - **Attribute questions** — 3–6 distinct phrasings per attribute.
>    - **Reverse/aggregate lookups** — "which entities have attribute = value?"
>      (write these out; the model can't derive them). Lead answers with a count.
>    - **Relationship phrasings** — how each kind of link in `facts.json` is
>      asked and answered.
>    - **Precomputed reasoning** — anything needing a rule (comparisons,
>      superlatives, "what beats what"): compute it in Python, bake in the
>      answer, and mark it `computed=True`. **Double-check every computed fact
>      by hand — a wrong one becomes a confidently-wrong model.**
>    - **Help** — what kinds of questions the model can answer.
>    - Optionally, one or two `heldout` phrasings per question list. They are
>      never trained; they test the model on wordings it hasn't seen.
>
>    Make every answer read correctly for every value — "a"/"an", plurals,
>    zero, yes/no. Use `{a_value}` or an answer function (see **Writing
>    answers**).
> 4. **Write `verify()` checks** for whatever a machine can check: required
>    fields, value ranges, duplicates, and agreement with a downloaded
>    reference dataset if you have one.
> 5. **Run `python TEMPLATE.py --strict`** and fix what it reports until it
>    passes.
> 6. **Read what you made:** about 30 random Q&As from
>    `training_data/corpus.txt`, plus every computed answer in
>    `training_data/UNVERIFIED.md`. Fix anything wrong, awkward or
>    ungrammatical, then run again.
> 7. **Report to the user:** the stats the run printed, the facts
>    `UNVERIFIED.md` lists for them to check, and anything left off the menu.
>
> **You're done when:**
> - [ ] `python TEMPLATE.py --strict` finishes without errors or warnings.
> - [ ] Every `facts.json` entry cites a real source, or is marked
>   `"unverified"` and reported to the user.
> - [ ] You've read a sample of `corpus.txt` and every computed answer.
> - [ ] Superlatives claim only what the data covers, and list every tie.
> - [ ] The help answer matches what the model can actually answer.

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
| **Superlative / extreme** | "Which inner planet has the most moons?" | Compute the max/min/first/last. List every tie. Scope it to what the data covers. |
| **Comparison** | "Which is bigger, Earth or Mars?" / "Is Mars bigger than Earth?" | Compute it. Ask both ways round; yes/no questions get "Yes"/"No". Pick a FEW meaningful pairs, never all N×N. |
| **Count / statistic** | "How many rocky planets are there?" | Aggregate. |
| **Categorization** | "What are the categories of X?" | If the domain groups things. |
| **Derived reasoning** | "What is X weak to?" | Apply a RULE (a chart/formula) to the data in Python. Verify. |
| **Lore / background** | "Tell me about the Solar System." | Short prose, bridged with "tell me about X". |
| **Procedural / how-to** | "How do you do X?" | If the domain has procedures/steps. |
| **Help / capabilities** | "What can you do?" | Say what kinds of questions the model can answer. |

---

## Rules (the hard-won lessons — follow these)

1. **Source every fact; never invent a source.** Give each `facts.json` entry
   the URL or citation it came from, or `"unverified"` if it came from memory.
   `UNVERIFIED.md` lists the unverified ones for a person to check. A made-up
   citation is worse than none: it hides the fact from that check.
2. **Many phrasings per fact, not lowercase/punctuation duplicates.** Real
   people ask the same thing many ways ("How many moons does Mars have?" /
   "Does Mars have moons?" / "What's Mars's moon count?"). Give distinct
   wordings. Do NOT add `mars`/`Mars?`/`MARS` variants — casing is handled at
   inference, and lowercased names fragment into tokens the model can't use.
3. **One answer per question.** The machinery enforces this (first-write-wins):
   if two blocks emit the same question with different answers, the later one
   is dropped, and the run lists every such conflict (`--strict` fails on
   them). Don't rely on it — design questions so each has one true answer.
4. **Short, factually-dense answers, each on one line.** At most 2 sentences
   and about 30 words: the device stops generating after 2 sentences, and a
   whole Q&A pair has to fit the model's short context. Standalone prose: at
   most about 55 words. A tiny model memorizes tight text far better than long
   flowery prose, and short answers drift less. The run warns past these
   limits and rejects line breaks.
5. **Precompute all reasoning.** The model cannot compare, aggregate, or apply
   rules at run time. Compute those in Python and store the answers as flat
   facts. This is the single biggest lever for making it look "smart".
6. **Verify computed facts.** A wrong precomputed fact (a bad comparison, a
   wrong rule table) trains the model to be confidently wrong. Mark them
   `computed=True` so `UNVERIFIED.md` lists them, and hand-check a sample of
   every group.
7. **Claim only what the data covers.** A superlative over a partial list is
   wrong in the real world: with 4 of the 8 planets, "Which planet has the most
   moons?" would teach "Mars" (it's Saturn). Scope the question ("Which inner
   planet…") and list every entity tied for first.
8. **Answers must read right for every value.** `"{name} is a {value}."`
   becomes "is a actor", and `"{value} moon(s)"` becomes "0 moon(s)" — and the
   model repeats them word for word. Use `{a_value}`, `count_phrase()` and
   answer functions (see [Writing answers](#writing-answers)), and give a yes/no
   question a "Yes" or "No" that matches the question: "Is Mars bigger than
   Earth?" needs "No, Mars is smaller than Earth."
9. **Whole-word names.** Return your entity names from `build()`, plus
   multi-word category values ("rocky planet"); they become special tokens so
   they stay intact instead of fragmenting. Leave out single common words: a
   special token also matches inside other words ("pop" would cut "popular"
   apart), and the run warns when one does.
10. **Lead aggregate answers with a count** ("There are 4 rocky planets: ...") so
    a truncated answer is still useful. `{list}` names at most 6 and ends
    "and N more", which keeps long lists within the limits.
11. **Bridge lore with "tell me about X".** A passage trained only as bare prose
    has no path from a question to it. Give each passage its questions.
12. **Avoid combinatorial explosion.** N entities have N² pairs — don't emit all
    of them. Do superlatives, neighbours, and a few meaningful comparisons, not
    every pair.
13. **Coverage isn't free.** A ~6M model has limited capacity. Core facts under
    many phrasings beat sprawling, rarely-asked coverage. When in doubt, deepen
    (more phrasings of the facts that matter) rather than widen.

---

## Writing answers

An `"answer"` in `ATTRIBUTE_QUESTIONS`, `REVERSE_LOOKUPS` or
`RELATIONSHIP_QUESTIONS` is either a template or a function:

```python
"answer": "{name} is {a_value}.",                        # "Mercury is a rocky planet."
"answer": lambda name, value: f"{name} has {count_phrase(value, 'moon')}.",
                                                         # "no moons" / "one moon" / "2 moons"
```

Use a function whenever the wording depends on the value. A function receives
only the fields it names as parameters:

| Answer for | Fields |
|---|---|
| An attribute | `name`, `value`, `a_value` (the value with "a"/"an") |
| A reverse lookup | `count`, `value`, `list` (at most 6 names, then "and N more"), `names` (all of them) |
| A relationship | the row's `from`, `rel` and `to` (as a function, take `**r` and read `r["from"]`, since `from` is a Python keyword) |

Helpers in `corpus_lib.py`:
- `with_article("actor")` → "an actor"; also "an R&B singer", "a unicorn".
  It's a heuristic: write the article yourself for odd cases like "a NASA
  mission".
- `count_phrase(n, "moon")` → "no moons" / "one moon" / "2 moons" (pass the
  plural when it isn't just "+s": `count_phrase(n, "person", "people")`).
- `list_some(names)` → "A, B, C, D, E, F, and 5 more"; `list_join(names)` →
  "A, B, and C".

---

## What a run checks and writes

Every run of `python TEMPLATE.py` checks the data **before writing anything**.
It prints corpus stats (Q&A pairs, facts, phrasings per fact, longest answer)
and the result of each check:

| Level | Fails the run? | What it covers |
|---|---|---|
| **Error** | Always | `facts.json` is well-formed; `verify()` finds no problems; no line break inside a question, answer or passage; every guided-menu question is a trained corpus question with exactly one answer; the menu fits the converter's caps. |
| **Warning** | Only with `--strict` | Conflicting answers (each listed); answers over 2 sentences or 30 words; prose over 55 words; held-out phrasings that are also trained; whole-word tokens that would split a longer word. |
| **Note** | Never | `facts.json` entries without a source; questions left off the menu. |

When the run fails, nothing is written. Otherwise it writes to `training_data/`
(next to `--out`):

| File | For |
|---|---|
| `corpus.txt` | The training corpus — the trainer's `--text`. |
| `special_tokens.txt` | Entity names kept whole in the tokenizer — `--special-tokens`. |
| `test_prompts.txt` | Questions the trainer asks the finished model (`--qa-test-prompts`): trained phrasings from every question type, plus held-out ones. |
| `val.txt` | Held-out phrasings with their answers, never trained — `--val-text`. Only written when you add `heldout` phrasings. |
| `menu_manifest.json` | The device's guided "pick a question" menu (see below). |
| `UNVERIFIED.md` | Facts without a source, and every computed answer, for a person to check before training. |

---

## STEP 3: train

Once the generator passes, train the model with the canonical tiny-LLM trainer
in the repo's `Training/` folder (`train_tiny_model_gpu.py` for GPU,
`train_tiny_model.py` for CPU). From this folder:

```
python ../../../Training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training_data/corpus.txt \
    --special-tokens training_data/special_tokens.txt \
    --qa-test-prompts training_data/test_prompts.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 \
    --out ./out_mymodel
```

- These settings match the training commands the catalog gives for every
  shipped model.
- `--special-tokens` keeps your entity names whole.
- `--qa-test-prompts` makes the trainer finish by asking the model the
  questions in `test_prompts.txt` — check that it gets the trained ones right.
- **Optional early stopping:** add `--val-text training_data/val.txt` (needs
  `heldout` phrasings). `--epochs` becomes a ceiling, and training stops once
  the model stops improving on phrasings it has never seen, while all of
  `corpus.txt` is still trained. Don't use `--val-frac` on this kind of corpus:
  it takes a random share of your Q&A pairs out of training, and a memorizer
  never learns a fact it wasn't shown.
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
let it terminate naturally rather than hard-capping length. That relies on each
Q&A pair fitting the trainer's 128-token block, which the answer limits above
keep you well within.

---

## Guided-input menu (optional, automatic)

`TEMPLATE.py` also emits a `menu_manifest.json` next to your corpus, built from
the SAME `facts.json` entities and `ATTRIBUTE_QUESTIONS` / `REVERSE_LOOKUPS` /
lore you already filled in: for each question list, the first phrasing whose
only placeholder is the entity (`{name}`, or `{value}` for reverse lookups)
becomes a menu template with a single `{}` slot, and your entity names become
the roster. On the device this drives a "pick a question" menu (group →
question → entity) so users don't have to type — and because each template is
one of your trained phrasings, every composed question is one the model trained
on verbatim.

- The run checks that promise: it composes every template × entity and fails
  if one isn't a corpus question, or is a question with conflicting answers.
- An attribute that some entities lack is left off the menu (a note says
  which): a group's entities share its questions, so the menu would offer
  questions the model never saw. The same goes for an attribute with no
  phrasing that uses only `{name}`.
- Nothing extra to do: a normal `TEMPLATE.py` run writes `menu_manifest.json`
  beside `corpus.txt`. Pass `--menu-only` to regenerate JUST the menu (the corpus
  and tokens are left untouched), or `--menu-out <path>` to redirect it.
- The trainer copies `menu_manifest.json` into `--out` automatically when it sits
  next to the corpus (or pass `--menu <path>`), so the browser converter
  auto-loads it exactly like `domain_vocab.txt`.
- Caps (the converter and firmware also enforce them; the run checks them
  first): ≤8 groups, ≤64 templates and ≤1024 entities per group, group name
  ≤32 B, question ≤120 B, entity ≤48 B, and ≤32 KB for the whole encoded menu.
  A template has at most one `{}` slot; a slotless template is a canned
  question. No menu is fine — the device just falls back to free-text.

---

## How much data?

- **Small topic** (tens of entities): a few hundred to ~2,000 facts. Lean on
  phrasings and reasoning to add depth.
- **Larger topic** (~150 entities like the Kanto Pokédex example this template
  is derived from): ~1,500 distinct facts × several phrasings ≈ 10k–13k Q&A.

Start small, train, test the exact questions you care about, then add the
categories that came up short. Every run prints how many facts you have and how
many phrasings each gets. Iterating on the data beats fiddling with the model.
