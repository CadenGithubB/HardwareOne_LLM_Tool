# Training Materials Catalog

A catalog of **training packages** for the on-device LLM models built with this
tool. Each package is a self-contained zip with the training data, trainers, validation
scripts, and instructions needed to reproduce a model from scratch.

> **The main repo `training/` folder does not ship example data.** Either
> write your own dataset and train with the repo trainers (see
> [`training/INSTRUCTIONS.txt`](../../training/INSTRUCTIONS.txt)), grab a
> pre-trained `.bin` from [`../Trained + Ready Models/`](../Trained%20+%20Ready%20Models/),
> or unzip one of the packages below and train from inside the unzip directory.

> Trained, ready-to-deploy `model.bin` files live in [`../Trained + Ready Models/`](../Trained%20+%20Ready%20Models/).
> This folder holds the **source materials** that produce them.

## How to use a package

1. Unzip it anywhere — the zip is fully self-contained (trainers + `training_data/` + scripts):
   `unzip <package>.zip -d my_model && cd my_model`
2. `pip install -r requirements.txt` (add a CUDA build of torch for GPU training)
3. Train with the command in that package's `INSTRUCTIONS.txt` / the notes below
4. Convert the output folder to `model.bin` via the browser tool (`index.html` at
   the repo root, INT8, group size 128)
5. Copy `model.bin` to the device, or compare against a pre-trained `.bin` in
   `../Trained + Ready Models/` if one exists for that model

---

## Catalog

| Model | Domain | Preset | Training data | Trained `.bin` | Package |
|---|---|---|---|---|---|
| **HardwareOne Help Agent** | HardwareOne ESP32-S3 firmware help (Q&A + `Do:` CLI command suggestions) | HW1HelpAgent192_deep (~7.5 MB INT8) | ~890 Q&A + 1,123 `Do:` pairs, 27 topics | [`HardwareOneHelpAgent.bin`](../Trained%20+%20Ready%20Models/HardwareOneHelpAgent.bin) (6.5 MB) | [`hardwareone-help-agent/`](hardwareone-help-agent/hardwareone_training_package.zip) |
| **Kanto Pokemon Master** | Generation-1 / Kanto knowledge: the original 151 Pokémon + the Kanto region (pure knowledge Q&A) | HW1HelpAgent192_deep (~7.5 MB INT8) | 3,998 Q&A + 9 prose | _not yet trained_ | [`kanto-pokemon-master/`](kanto-pokemon-master/kanto_pokemon_master_training_package.zip) |
| **Periodic Table Guide** | The 118 chemical elements: number, symbol, family, group/period, room-temp state (pure knowledge Q&A) | HW1HelpAgent192_deep (~7.5 MB INT8) | 2,734 Q&A + 3 prose | _not yet trained_ | [`periodic-table-guide/`](periodic-table-guide/periodic_table_guide_training_package.zip) |

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
**generated** from a single fact table (`training_scripts/generate_pokemon_data.py`)
so facts never contradict across phrasings. Covers, per Pokémon: type, Pokédex
number, category, evolution (method + level/stone/trade + item interactions), and
"about". Plus the Kanto world: gyms, Elite Four, type chart (super-effective,
weaknesses, immunities), reverse type lookups, locations, characters, items, HMs,
and mechanics.

All Gen-1 authentic (no abilities/held items; Clefairy/Jigglypuff/Mr. Mime are
Normal/Psychic, Magnemite is Electric). Types and evolutions were **verified
against PokéAPI for all 151** via `training_scripts/verify_pokemon_data.py`.
Tokenizer reaches ~2,315 of the 3,072 BPE budget (151 names kept whole + subwords).

Train:
```bash
python train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training_data/pokemon_kanto.txt \
    --special-tokens training_data/pokemon_special_tokens.txt \
    --qa-test-prompts training_data/pokemon_test_prompts.txt \
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
Periodic-Table-JSON dataset) by `training_scripts/generate_elements_data.py`.
Covers, per element: atomic number, symbol (both directions), family, group and
period, block, and room-temp state — plus reverse lookups (which elements are
noble gases, halogens, gases/liquids…) and general facts.

Data is cleaned for the device's short context: families normalized, group-17
reclassified as halogens, and room-temp phase asserted only for the
naturally-occurring elements (≤94) where it is well established (so "liquid at
room temperature" correctly answers Mercury and Bromine, not the data's
predicted Copernicium). Tokenizer reaches ~1,178 of the 3,072 budget; the 118
names and multi-letter symbols are kept whole.

Train:
```bash
python train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text training_data/elements_rich.txt \
    --special-tokens training_data/elements_special_tokens.txt \
    --qa-test-prompts training_data/elements_test_prompts.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 --compile \
    --out ./out_periodic_table
```

When converted, name the deployable model `PeriodicTableGuide.bin` in
`../Trained + Ready Models/`.

---

## Adding a new model to the catalog

1. Build the training package zip (data + trainers + scripts + INSTRUCTIONS).
2. Create a `kebab-case/` subfolder here and drop the zip inside.
3. Add a row to the **Catalog** table and a short **Model notes** section above.
4. When the `.bin` is trained, place it in `../Trained + Ready Models/` and link it.
