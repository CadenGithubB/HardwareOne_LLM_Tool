# Training Materials Catalog

A catalog of **training packages** for the on-device LLM models built with this
tool. Each package is a zip of just the `training_data/` needed to reproduce a model —
the corpus, special-tokens, test-prompts, and any topic data. The trainers,
validation scripts, and the single guide live once in `Training/` and are shared
across every package.

> **The main repo `training/` folder does not ship example data.** Either
> write your own dataset and train with the repo trainers (see
> [`training/INSTRUCTIONS.txt`](../../training/INSTRUCTIONS.txt)), grab a
> pre-trained `.bin` from [`../Trained + Ready Models/`](../Trained%20+%20Ready%20Models/),
> unzip one of the data packages below and train it with a repo trainer, or
> use the [`build_your_own_model/`](build_your_own_model/) kit to have an AI
> generate a package for any topic (see next section).

> Trained, ready-to-deploy `model.bin` files live in [`../Trained + Ready Models/`](../Trained%20+%20Ready%20Models/).
> This folder holds the **source materials** that produce them.

## Build your own — any topic (AI-assisted)

Don't see your topic in the catalog below? The
[`build_your_own_model/`](build_your_own_model/) kit is a fill-in-the-blanks
generator you hand to an AI (Claude, ChatGPT, …) with any topic. Following its
`BUILD_YOUR_OWN_MODEL.md`, the AI writes a complete data generator — facts under
many phrasings, reverse/aggregate lookups, precomputed reasoning, and lore. You
generate the corpus in the kit, then train it with the canonical `Training/`
trainer (`Training/train_tiny_model_gpu.py`). It's the fastest way to a model on a
topic no package covers. Every package below is built from these same patterns.

## How to use a package

1. Unzip it anywhere — you get a `training_data/` folder (corpus, special-tokens,
   test-prompts, and any topic data):
   `unzip <package>.zip -d my_model`
2. `pip install -r Training/requirements.txt` (add a CUDA build of torch for GPU training)
3. Train with the canonical `Training/` trainer, using the command in `Training/INSTRUCTIONS.txt` or the notes below.
   After it saves the model, the trainer also extracts a domain word-list from the
   corpus and writes it as `domain_vocab.txt` in the output folder (pass
   `--no-domain-vocab` to skip; build or tune one by hand with
   `Training/training_scripts/extract_domain_vocab.py`)
4. Convert the output folder to `model.bin` via the browser tool (`index.html` at
   the repo root, INT8, group size 128). The converter auto-loads any
   `domain_vocab.txt` from the selected folder into its **Domain words** field, and
   also offers a manual **Domain words** file input and a **Refusal answer** field.
   On device, when the model carries a domain word-list and the gate is enabled, a
   prompt that contains none of those words is refused with the Refusal answer
   instead of being answered; clearing both fields disables the gate
5. Copy `model.bin` to the device, or compare against a pre-trained `.bin` in
   `../Trained + Ready Models/` if one exists for that model

---

## Catalog

| Model | Domain | Preset | Training data | Trained `.bin` | Package |
|---|---|---|---|---|---|
| **HardwareOne Help Agent** | HardwareOne ESP32-S3 firmware help (Q&A + `Do:` CLI command suggestions) | HW1HelpAgent192_deep (~7.5 MB INT8) | ~890 Q&A + 1,123 `Do:` pairs, 27 topics | [`HardwareOneHelpAgent.bin`](../Trained%20+%20Ready%20Models/HardwareOneHelpAgent.bin) (6.5 MB) | [`hardwareone-help-agent/`](hardwareone-help-agent/hardwareone_training_package.zip) |
| **Kanto Pokemon Master** | Generation-1 / Kanto knowledge: the original 151 Pokémon + the Kanto region (pure knowledge Q&A) | HW1HelpAgent192_deep (~7.5 MB INT8) | 5,825 Q&A + 9 prose | _not yet trained_ | [`kanto-pokemon-master/`](kanto-pokemon-master/kanto_pokemon_master_training_package.zip) |
| **Periodic Table Guide** | The 118 chemical elements: number, symbol, family, group/period, room-temp state (pure knowledge Q&A) | HW1HelpAgent192_deep (~7.5 MB INT8) | 10,000 Q&A + 3 prose | _not yet trained_ | [`periodic-table-guide/`](periodic-table-guide/periodic_table_guide_training_package.zip) |
| **World Pop Culture** | Worldwide music, film, sports, business, and media figures, plus companies, platforms, leagues/awards, and world cities and countries (pure knowledge Q&A) | HW1HelpAgent192_deep (~7.5 MB INT8) | 2,174 Q&A + 4 prose | _not yet trained_ | [`pop-culture-guide/`](pop-culture-guide/pop_culture_training_package.zip) |

---

## Model notes

### HardwareOne Help Agent
The original shipped model. Answers questions about the HardwareOne firmware and
suggests CLI commands via `Do:` pairs. Data is hand-curated in
`training_data/hardwareone_rich.txt`; the package includes the full suite of data
quality validators.

Train:
```bash
python train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training_data/hardwareone_rich.txt \
    --special-tokens training_data/hardwareone_special_tokens.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 \
    --out ./out_HW1HelpAgent192_deep
```

### Kanto Pokemon Master
A knowledge agent for the original 151 Pokémon and the Kanto region. The data is
**generated** from a single fact table (`Training/training_scripts/generate_pokemon_data.py`)
so facts never contradict across phrasings. Covers, per Pokémon: type, Pokédex number, category, **Pokedex entry text**,
evolution (method + level/stone/trade), **where to catch in Kanto**, reverse
dex lookup ("what Pokemon is #25?"), and "about". Plus capability/help Q&A
("What can you do?", "Help"). Plus the Kanto world: gyms, Elite Four, type chart (super-effective,
weaknesses, immunities), reverse type lookups, locations, characters, items, HMs,
and mechanics.

All Gen-1 authentic (no abilities/held items; Clefairy/Jigglypuff/Mr. Mime are
Normal/Psychic, Magnemite is Electric). Types and evolutions were **verified
against PokéAPI for all 151** via `Training/training_scripts/verify_pokemon_data.py`.
Tokenizer reaches ~2,315 of the 3,072 BPE budget (151 names kept whole + subwords).

Train:
```bash
python Training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text my_model/training_data/pokemon_kanto.txt \
    --special-tokens my_model/training_data/pokemon_special_tokens.txt \
    --qa-test-prompts my_model/training_data/pokemon_test_prompts.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 \
    --out ./out_kanto_pokemon_master
```

The 151 Pokémon names are kept whole in the tokenizer (`pokemon_special_tokens.txt`)
so names tokenize atomically instead of as partial fragments.

When converted, name the deployable model `KantoPokemonMaster.bin` in
`../Trained + Ready Models/`.

### Periodic Table Guide
A knowledge agent for the 118 chemical elements. Generated from a single fact
table (`training_data/elements.json`, derived from the Bowserinator
Periodic-Table-JSON dataset) by `Training/training_scripts/generate_elements_data.py`.
Covers, per element: atomic number, symbol (both directions), family, group and
period, block, metal/nonmetal classification, and room-temp state — plus reverse
lookups by family, period, group, and block; gases/liquids at room temperature;
capability/help Q&A; and general facts. All per-element answers are derived from
`elements.json` (no hand-written uses or discovery trivia).

Data is cleaned for the device's short context: families normalized, group-17
reclassified as halogens, and room-temp phase asserted only for the
naturally-occurring elements (≤94) where it is well established (so "liquid at
room temperature" correctly answers Mercury and Bromine, not the data's
predicted Copernicium). Tokenizer reaches ~1,178 of the 3,072 budget; the 118
element names are kept whole (symbols excluded to avoid English word collisions).

Train:
```bash
python Training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text my_model/training_data/elements_rich.txt \
    --special-tokens my_model/training_data/elements_special_tokens.txt \
    --qa-test-prompts my_model/training_data/elements_test_prompts.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 --compile \
    --out ./out_periodic_table
```

When converted, name the deployable model `PeriodicTableGuide.bin` in
`../Trained + Ready Models/`.

### World Pop Culture
A knowledge agent for world pop culture. **Generated** from a single fact table
(`Training/training_scripts/generate_pop_culture_data.py`) so facts never contradict
across phrasings. Covers ~82 famous people (musicians, actors and directors,
athletes across 10 sports, business and media figures) with profession, field,
primary genre/sport, and home country; plus 13 companies, 10 platforms, 9
leagues/awards, music genres and sports, and 26 world cities and countries
(continent, and each country's capital). Includes reverse lookups by field, role,
genre, sport, and country/continent. All 185 proper nouns are kept whole in the
tokenizer so names, places, and companies never fragment.

**STARTER data:** pop culture is open and fast-changing, so this uses only
rock-stable facts (profession, genre/sport, nationality, capitals) - verify
against a current source before training.

Train:
```bash
python Training/train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text my_model/training_data/pop_culture_rich.txt \
    --special-tokens my_model/training_data/pop_culture_special_tokens.txt \
    --qa-test-prompts my_model/training_data/pop_culture_test_prompts.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 --vocab-size 3072 \
    --out ./out_pop_culture
```

When converted, name the deployable model `WorldPopCulture.bin` in
`../Trained + Ready Models/`.

---

## Adding a new model to the catalog

1. Build the training-data package zip — just `training_data/` (corpus, special-tokens, test-prompts, and any topic data). The trainers, scripts, `requirements.txt`, and `INSTRUCTIONS.txt` stay in `Training/`.
2. Create a `kebab-case/` subfolder here and drop the zip inside.
3. Add a row to the **Catalog** table and a short **Model notes** section above.
4. When the `.bin` is trained, place it in `../Trained + Ready Models/` and link it.
