# ESP32 On-Device LLM — Output-Quality Plan

**For:** the agent working on the HardwareOne firmware.
**Repo:** `/Users/morgan/esp/hardwareone-idf/` — engine in `components/hardwareone/`.
**Goal:** get better answers out of the tiny on-device LLM **without changing the
model** (weights/architecture are frozen; reconverting `model.bin` is out of
scope). Every change here is inference-time and runtime — no retraining.

> **Locate code by FILE + SYMBOL, never by line number.** The engine was recently
> split into multiple files and will keep moving. Use symbol search. Locations
> below were verified at authoring time but treat them as starting points.

---

## 1. The thesis (read this first)

A 192-dim / ~3K-vocab model is a **lossy, blurry compressor**. Its dominant
failure is **hallucination — confidently wrong specifics** (wrong numbers, wrong
names, made-up commands), not blandness. So split the work by what each tool is
actually good at:

| Job | Owner | Why |
|---|---|---|
| **Knowledge** (facts, exact values) | a **fact table in flash** | exact bytes can't be muddled; weights can |
| **Language** (phrasing, glue) | the **LLM** | what it's relatively good at |
| **Validity** (commands, structure) | **constrained decoding** | a hard guarantee weights can't give |
| **Repetition / drift** | **samplers** | cheap token-level cleanup |

The plan moves the model **off** "be a database" (which it fails at) and **onto**
"be a language interface" — while hard constraints (a fact table, a command mask)
handle correctness, because **you cannot fix a hard-correctness problem with soft
training pressure on a tiny model.**

The three things that actually move the product: **Phase 0 (casing)** is free and
helps today; **Phase 3 (command mask)** and **Phase 4 (fact table)** are the real
wins. Everything else is cheap warmups.

---

## 2. Device constraints — these are load-bearing, do not design against fantasy numbers

1. **Runtime context window ≈ 41 tokens, NOT `seq_len`=128.** The KV cache is
   PSRAM-capped and auto-fit shrinks it; on this device it lands ~41. Prompts
   already eat ~6–7 tokens. **Anything injected into the prompt competes with the
   answer inside ~41 tokens** and cannot be raised without PSRAM you don't have.
   → This is why injecting facts into context (grounded generation) is *not* the
   default retrieval mode (§5, Phase 4).
2. **PSRAM is full** (~400KB free = the reserve). **Nothing new may be
   PSRAM-resident alongside the model** — no second model, no larger KV cache.
   **Flash/LittleFS is fair game** — the fact table lives in flash, read on demand.
3. **Throughput ≈ 1.7 tok/s.** A ~40-token answer ≈ 24s. **No technique that
   multiplies forward passes per answer** (rules out best-of-N, majority vote).
   Techniques that *reduce* generation (template answers) are a bonus.
4. **The model is confidently wrong.** It emits false answers at `top_prob=1.000`
   (e.g. "200 metres" for a value it was never given). A prior that confident
   **outshouts injected context**, and **confident-wrong is undetectable at decode
   time** — no per-token signal separates it from confident-right.
5. **Command names are single vocab tokens.** Training pre-split command names
   (e.g. `ntpsync` = tok 105), which makes the command mask (Phase 3) a single
   masked step for most commands — no multi-token grammar needed.
6. **Casing matters.** `who` → tok 1558 vs `Who` → tok 2387; lowercase-led
   questions answer measurably worse. The model trained on title-cased `Q:` lines.

Everything new is **flag-gated**: default off must be byte-identical to today, so
each change is A/B-able and instantly revertible.

---

## 3. Engine map (verified locations)

`components/hardwareone/`:

| Area | File | Key symbols |
|---|---|---|
| Sampling | `System_LLM_Sampler.cpp` | `sample`, `sample_argmax`, `sample_topp`, `sample_mirostat2` |
| Generation loop + logit prep + Do: mode | `System_LLM.cpp` | `llmGenerate`, rep-penalty/content-boost/taper block, sampling call site, Do: handling |
| Unified chat funnel (web + BLE + OLED) | `System_LLMChat.cpp` | `chatBeginTurn`, `chatResolveParams` — **prompt building + dispatcher live here** |
| Model load / PSRAM | `System_LLM_Model.cpp` | loader |
| Kernels | `System_LLM_Kernels.cpp` | forward primitives |
| Tokenizer | `System_LLM_Tokenizer.cpp` | encode/decode |
| Public API / config | `System_LLM.h` | `llmGenerate` sig, `LLMGenParams`, `LLMStatus`, `LLM_DEFAULT_*` |
| Device command registry | `System_Command.cpp` / `.h` | **`gCommands` / `gCommandsCount`** (`const CommandEntry**`, backed by `commandRegistry[]`) |

UIs `OLED_Mode_LLM.cpp` and `WebPage_LLM.cpp` funnel through `System_LLMChat.cpp`
— put shared logic (casing, dispatcher) there so all three UIs get it.

**Adding a tunable** requires editing **three** places (mirror how `topp` /
`repPenalty` are done): a `#define LLM_DEFAULT_*` in `System_LLM.h`, the
`llmGenerate(...)` signature + default, and the `LLMGenParams` struct.

---

## 4. The work, phased

Each phase is independent and flag-gated. Build in this order (dependency +
risk), but spend your *attention* on Phase 3 and Phase 4 — that's the product.

---

### Phase 0 — Prompt casing fix · free · do first

- **Goal:** title-case the first word of the user's question before tokenizing.
- **Why:** measured — lowercase question words tokenize differently and answer
  worse; the model trained on title-cased `Q:` lines. This is a format-adherence
  fix, ~3 lines, zero risk, helps immediately.
- **Where:** the prompt builder in `System_LLMChat.cpp` (where the `Q: … A:`
  string is assembled before `llmGenerate`).
- **Approach:** uppercase the first alphabetic character of the question
  (optionally normalize to the training `Q:` casing convention).
- **Done when:** lowercase-led questions answer like their title-cased equivalents
  on the test prompts.

---

### Phase 1 — min-p sampling · cheap · A/B only

- **Goal:** add min-p alongside top-p in `sample()`.
- **Why:** min-p keeps tokens with `prob ≥ min_p × prob_of_top_token` — a bar that
  scales with the model's confidence, truncating the plausible-but-wrong tail
  better than top-p's fixed cumulative quota.
- **Reality check:** on-device logs show `nucleus=1, top_prob=1.000` at most
  tokens on the peaked temp-0.5 factual distributions, where **min-p ≈ top-p and
  changes little.** Its value only shows at higher temperature, which you don't
  want for factual QA. Its only real home is an open/conversational lane run
  warmer. **Bill it "neutral, A/B it," not a win.** Keep it because it's cheap.
- **Where:** `sample()` in `System_LLM_Sampler.cpp`. It already computes
  `max_prob` post-softmax — reuse it.
- **Approach:** after softmax, before the top-p/categorical branch, zero every
  `prob < min_p × max_prob`, then **renormalize**, then let existing code run.
- **Gotchas:** renormalize after zeroing (the categorical fallback walks a CDF
  expecting sum≈1.0; `sample_topp` renormalizes via its own cumsum, the
  categorical branch does not). Applies to the **non-mirostat** path only.
- **Config:** `LLM_DEFAULT_MIN_P 0.0f` (0 = disabled). Thread `minP` through the
  three places.
- **Done when:** `min_p=0` is byte-identical to today; `min_p=0.1` shows no
  regression and any improvement is captured by the A/B on the test prompts.

---

### Phase 2 — Confidence backstop · small · only useful with Phase 4

- **Goal:** from the single pass we already run (no extra passes — §2.3), compute
  a confidence signal and use it for graceful fallback.
- **Hard limit — do not oversell this:** confidence catches **open-domain
  flailing** (model unsure, flat distribution). It does **NOT** catch
  confident-wrong — that scores as high confidence (§2.4) and is undetectable at
  decode time. So confidence is a **backstop for the no-fact lane only**, useful
  *exclusively paired with retrieval*: "low confidence **and** no fact retrieved →
  fall back / say I'm not sure." The residual confident-wrong risk is **managed,
  not detected** — shrink it with fact-table coverage and conservative
  out-of-domain scoping ("I can only help with …").
- **Where:** generation loop in `System_LLM.cpp` (sampling call site) + `LLMStatus`
  in `System_LLM.h`.
- **Approach:** have the sampler report the chosen token's post-softmax prob
  (out-param, or recompute before mutation); accumulate `mean_logprob`; expose on
  `LLMStatus`. `sample_mirostat2` already computes `surprise_bits = -log2(p)` —
  reuse the concept.
- **Done when:** out-of-domain prompts score measurably lower confidence than
  in-domain on the test sets — enough to set a threshold.

---

### Phase 3 — Command mask (constrained decoding) · **PRODUCT WIN**

- **Goal:** in Do: mode, make it **impossible** for the model to emit an invalid
  command by masking illegal tokens' logits before sampling.
- **Why it must be runtime, not trained in:** training only biases the model
  toward valid commands (soft, probabilistic — what you have now, still
  hallucinates). Zeroing illegal logits is a **hard guarantee** — the model
  literally cannot emit a fake command. The guarantee is the whole point, and it
  only exists at decode time. Bonus: reading the live registry means **new
  commands are covered automatically with zero retraining**; baking the list into
  weights would force a retrain on every command change.
- **Where:** the logit-prep block in `System_LLM.cpp` (same region as rep penalty
  / content boost), before the `sample()` call.
- **Approach:**
  - Drive the mask from the **full device registry — `gCommands` /
    `gCommandsCount`** (in `System_Command.cpp`), iterating `gCommands[i]->name`.
    **NOT `llmCommands[]`** — that's only the LLM module's own handful of commands
    (`llmstatus`, `llmgenerate`, …); using it makes Do: mode silently suggest only
    LLM commands.
  - Set illegal tokens to `LOGIT_CLAMP_MIN` / `-INFINITY` so softmax zeroes them.
  - **Most commands are single tokens (§2.5)** → the mask is a single masked
    decode step. Only multi-token command names need a small trie. Far simpler
    than general GBNF.
- **Done when:** 100% of Do:-mode outputs are valid commands from the registry;
  flag off = unchanged. (May also *reduce* latency by cutting wasted tokens.)
- **Note:** this same logit-masking machinery powers Phase 4's constrained-slot.

---

### Phase 4 — Retrieval / fact table · **PRODUCT WIN** (biggest accuracy gain)

- **Goal:** stop asking the model to *recall* facts. Keep facts as **exact bytes
  in flash** and answer factual queries from the table — instantly and exactly —
  using the model only where it adds value and fits the ~41-token budget.
- **This is NOT "match question to a list."** One unified path; the model still
  generates on non-factual traffic. **Retrieval match-strength** picks the route,
  not a hand-maintained list.

**4.0 — Viability gate (≈10-min experiment, BEFORE building grounded-gen).**
Inject a fact whose value deliberately contradicts the model's prior (e.g. put
"ESPNOW range is 50 m" in context, ask the range). If the output uses 50, context
wins; if it stays 200, the prior wins.
- Prior wins / overflow → **do not build grounded-gen.** Ship template-fill +
  constrained-slot only.
- Context reliably wins → grounded-gen is viable as a *secondary* mode for short
  facts that fit ~41 tokens.
Template-fill and constrained-slot are viable **regardless** of this result.

**Routing** (in `chatBeginTurn` / `chatResolveParams`, `System_LLMChat.cpp`, so
web + BLE + OLED all get it):

```
query → retrieve(fact_table) → {match, score}
  ├─ strong match, factual           → TEMPLATE-FILL     (no model; instant, exact)   [PRIMARY]
  ├─ strong match, want model voice   → CONSTRAINED-SLOT  (Phase 3 mask forces value)  [robust]
  ├─ strong match + 4.0 gate passed   → GROUNDED-GEN      (inject short fact, ~41 cap)  [experimental]
  └─ weak / no match                  → FREE-GEN          (model generates, as today)
                                         └─ low confidence + no fact → "I'm not sure" (Phase 2)
```

**Mode viability** (why the order above):

| Mode | Context cost | Can be ignored? | Use for |
|---|---|---|---|
| **Template-fill** (no model) | zero | n/a | **workhorse** — direct factual, all critical exact values |
| **Constrained-slot** (force value via Phase 3 mask) | none injected | no (forced) | model-voice **with** exact facts |
| **Grounded-gen** (inject fact as text) | competes in ~41 | **yes** | experimental only, gated by 4.0 |

Rule: **template the digits, constrain the values, generate only the soft prose.**

- **Where:** new module `LLM_Retrieval.{h,cpp}` (matcher + fact store + template
  formatter); dispatcher in `System_LLMChat.cpp`; data on LittleFS
  (`/system/llm/facts.json` or packed binary).
- **Approach:** flash-resident key→fields store, read on demand (never bulk-loaded
  into PSRAM); cheapest viable matcher first (normalized keyword/alias → score);
  build template-fill, then constrained-slot, then grounded-gen only if 4.0 passes.
- **Honest limit:** multi-fact compositional reasoning ("can ESPNOW and Wi-Fi
  coexist?" → two injected facts) **does not fit ~41 tokens.** Handle via a
  precomputed template answer or accept free-gen.
- **Done when:** factual test prompts return correct specifics with retrieval on
  vs. muddled with it off; bare-definition queries return *faster* (template
  path); non-factual prompts behave as today; PSRAM usage unchanged.

---

### Phase 5 — DRY anti-repetition · as-needed

- **Goal:** kill broken-record loops more surgically than the flat rep penalty, and
  retire the content-exemption workaround.
- **Why:** rep penalty blanket-penalizes tokens (hence the content-exemption hack);
  DRY penalizes *continuing a repeated sequence*, scaled by match length — kills
  loops without suppressing on-topic repetition.
- **Where:** the logit-prep block in `System_LLM.cpp`, alongside/replacing
  rep-penalty logic. Reference the llama.cpp DRY implementation for the curve.
- **Do this only if** repetition is still visible after Phase 1, or fold it in with
  Phase 1 (shared area).

---

## 5. Sequencing & where the value is

0. **Phase 0 (casing)** — free, do immediately.
1. **Phase 1 (min-p)** — cheap; A/B, expect little at temp-0.5.
2. **Phase 2 (confidence)** — small; only meaningful with Phase 4.
3. **Phase 3 (command mask → `gCommands`)** — product win; also the foundation for 4.
4. **Phase 4 (fact table)** — run the 4.0 gate first; template-fill, then
   constrained-slot, then grounded-gen only if the gate passes.
5. **Phase 5 (DRY)** — only if repetition persists.

**The product value is in Phase 3 + Phase 4 template-fill.** Phases 0–2 and 5 are
cheap warmups. And **Phase 4's real cost is fact-table curation (§7), not code.**

Every phase: flag-gated, default off = byte-identical, validated against
`training/training_data/*_test_prompts.txt`
(`{elements,pokemon}_test_prompts.txt` + the HW1 factual set). Grid-search sampler
knobs rather than guessing; score on repetition rate, confidence, fact-accuracy,
and latency.

---

## 6. Out of scope — do not implement (and why)

- **Best-of-N / self-consistency / majority vote** — multiplies forward passes;
  70s+/answer at 1.7 tok/s (§2.3). Its purpose (detect bad answers) is recovered
  for free by Phase 2.
- **Multi-model router (domains resident together)** — cannot fit PSRAM (§2.2).
  The surviving "routing" is Phase 4's method-dispatcher on one loaded model.
- **Grounded-gen as the default retrieval mode** — ruled out by the ~41-token
  ceiling (§2.1) + confidently-wrong prior (§2.4); allowed only as a gated
  experiment after 4.0.
- **Baking the command list / fact constraints into the weights** — gives only a
  soft bias, never a guarantee, and forces a retrain on every change. Correctness
  belongs in runtime constraints (Phases 3–4), not weights.

Raise any of these with the human before reviving — the constraints are firm.

---

## 7. Decisions needed from the human (before Phase 4)

1. **Fact-table contents & source** — which domains/facts? Reuse
   `training/training_data/*.json`?
2. **Data format** — JSON (easy, heavier parse) vs. packed binary (lean)? Flash
   budget?
3. **Matcher fidelity** — keyword/alias enough, or do we need tiny on-device
   embeddings for fuzzy questions?
4. **Latency policy** — is an instant template answer (no model voice) acceptable
   for direct factual queries?
5. **Confidence-fallback UX (Phase 2)** — canned "I'm not sure" vs. visible
   retrieval retry?
6. **4.0 gate result** — does context beat the prior on the contradiction test?
   This single experiment decides whether grounded-gen exists at all.

---

## 8. Honest end-state — what "fully built" buys

- **Facts → excellent**, via the flash template table — a curated FAQ that mostly
  bypasses the model. **The win is curating the data, not writing inference code.**
  The hard, human work is §7.
- **Do: commands → excellent** — constrained decoding makes them unhallucinatable
  (Phase 3, pointed at `gCommands`).
- **Open / conversational chat → still mediocre.** This is the residual lane after
  facts and commands are routed away, and it's where the model wanders. Samplers
  nudge it; nothing here fixes it — **and that's fine, because facts + commands are
  the actual product.**

The LLM's job legitimately **shrinks to glue**: phrasing the narrow stuff that
fits ~41 tokens and handling open chat as best it can. That's a good product
shape — go in knowing the engineering wins are Phase 3 + Phase 4, and the data
curation is the real cost.

> Keep the ground-truth loop tight: every empirical finding so far (the 41-token
> ceiling, confident-wrong, casing, single-token commands) overrode an assumption.
> When in doubt, run the 10-minute experiment before building.
