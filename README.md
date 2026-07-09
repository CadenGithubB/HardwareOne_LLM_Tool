# HardwareOne LLM Tool

**Train tiny language models on your PC and deploy them to ESP32-S3**

Part of the [Hardware One](https://github.com/CadenGithubB/HardwareOne) ecosystem — a self-contained IoT platform with WiFi, sensors, ESP-NOW mesh networking, MQTT, and local AI inference.

---

## What This Tool Does

HardwareOne LLM Tool trains ultra-compact GPT-2 style language models on a PC and converts them to run entirely on the ESP32-S3 microcontroller using only 8MB of PSRAM. No cloud, no internet required at runtime — the model runs locally on the device.

**The trainer is domain-agnostic — you bring your own training data.** The same toolchain can produce a firmware help agent, a Pokédex, or any other tiny knowledge model. HardwareOne is the reference example; see [`Training Material + Pre-trained Models/`](Training%20Material%20%2B%20Pre-trained%20Models/) for ready-to-use training packages and deployable models.

**Training happens on your PC. The model runs on the ESP32. Nothing is trained on the device.**

### Key Features

- **Tiny Models**: 4K vocab, 16 layers, ~7.3 MB quantized
- **PC-Based Training**: Train on any machine with Python and PyTorch (GPU strongly recommended)
- **INT8 Quantization**: Browser-based converter produces a single `model.bin` for the device
- **Q&A Optimized**: Boundary-aware packing prevents answer bleed across training blocks
- **Do: Command Suggestions**: Model can suggest CLI commands the user can edit and execute
- **Hardware One Integration**: Drop the converted `model.bin` on the SD card and it runs

---

## Quick Start

You have three paths — pick one:

| Path | When to use |
|------|-------------|
| **A. Pre-trained model** | Fastest — grab a ready `model.bin` and deploy |
| **B. Your own dataset** | Custom domain — write your data, train with `training/` |
| **C. Example package** | Reproduce or tweak a shipped model — unzip a training package and train |

See `training/INSTRUCTIONS.txt` for the full walkthrough. Summary below.

### Path A — Deploy a pre-trained model

1. Download a `.bin` from [`Trained + Ready Models/`](Training%20Material%20%2B%20Pre-trained%20Models/Trained%20%2B%20Ready%20Models/) (e.g. `HardwareOneHelpAgent.bin`)
2. Copy it to `/sd/llm/` on the SD card (or upload via the web Files page)
3. Load from the LLM tab or CLI: `llm load /sd/llm/HardwareOneHelpAgent.bin`

No training or conversion needed.

### Path B — Train on your own dataset

#### 1. Install dependencies

```bash
cd training
pip install -r requirements.txt
```

For GPU training (recommended):
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

#### 2. Create your data

Write a plain `.txt` file of blocks separated by blank lines:

- `Q:` / `A:` pairs (short answers, ~30 words or less)
- optional `Q:` / `Do:` command pairs (if your agent suggests CLI commands)
- prose paragraphs (background context)

Reinforce each fact with several question phrasings. If your domain has multi-word
terms that must tokenize as one unit, list them one-per-line in a separate file and
pass `--special-tokens that_file.txt`.

Each [training package](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/) zip includes a `training_data/template.txt` you can copy as a starting point.

#### 3. Train

```bash
cd training

python train_tiny_model_gpu.py \
    --preset HW1HelpAgent192_deep \
    --text /path/to/YOUR_DATA.txt \
    --qa-test-prompts /path/to/YOUR_TEST_PROMPTS.txt \
    --epochs 250 --lr 3e-4 --batch-size 16 \
    --out ./out_mymodel
    # add: --special-tokens /path/to/YOUR_TOKENS.txt   (optional)
```

CPU training (slower, use if no GPU):

```bash
python train_tiny_model.py \
    --preset HW1HelpAgent192_deep \
    --text /path/to/YOUR_DATA.txt \
    --epochs 400 --batch-size 8 --lr 3e-4 \
    --out ./out_mymodel
```

Training takes ~30–60 minutes on a modern GPU. CPU training works but is much slower (many hours).

#### 4. Convert to ESP32 format

1. Open `index.html` in Chrome/Edge/Firefox
2. Drag the output folder (`./out_mymodel`) onto the page
3. Select **INT8 quantization**, group size **128**
4. Click **Convert**, then **Download** — saves `model.bin`

#### 5. Deploy

Copy `model.bin` to `/sd/llm/` on the SD card or upload via the web Files page. Load it from the LLM tab or CLI.

### Path C — Train from an example package

1. Pick a package from [`Training Materials/`](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/) (HardwareOne Help Agent, Kanto Pokemon Master, Periodic Table Guide)
2. Unzip it anywhere, e.g. `unzip hardwareone_training_package.zip -d my_model && cd my_model`
3. The zip is self-contained — it includes trainers, `training_data/`, validation scripts, and `INSTRUCTIONS.txt`
4. `pip install -r requirements.txt` (add a CUDA build of torch for GPU training)
5. Train using the command in that package's `INSTRUCTIONS.txt` or the [catalog README](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/README.md)
6. Convert with `index.html` (repo root) and deploy as in Path B

---

## What's Included

### Training (`training/`)
- `train_tiny_model_gpu.py` — GPU training script (recommended)
- `train_tiny_model.py` — CPU training script
- `INSTRUCTIONS.txt` — Detailed training guide and preset reference
- `requirements.txt` — Python dependencies
- `training_scripts/` — Data validation and analysis tools (run against your own data)

This folder ships **trainers and tools only** — no example data. Bring your own
`.txt` dataset (Path B) or unzip a [training package](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/) (Path C).

### Example Models (`Training Material + Pre-trained Models/`)
- [`Training Materials/`](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/) — Self-contained training packages (data + scripts + instructions) to reproduce each model
- [`Trained + Ready Models/`](Training%20Material%20%2B%20Pre-trained%20Models/Trained%20%2B%20Ready%20Models/) — Pre-trained, ready-to-deploy `model.bin` files

See the [catalog README](Training%20Material%20%2B%20Pre-trained%20Models/Training%20Materials/README.md) for per-model train commands and package downloads.

### Training Scripts (`training/training_scripts/`)
- `run_all_checks.py` — Run all data quality checks at once
- `deep_error_analysis.py` — 12-check structural and content analysis
- `validate_training_data.py` — Format and command validation
- `shuffle_training_data.py` — Randomize training block order
- `check_hallucinated_sensors.py` — Flag invalid chip names
- `check_answer_consistency.py` — Verify facts match across answers
- `answer_frequency_balance.py` — Report answer repetition balance
- `find_near_duplicate_answers.py` — Find 75%+ word-overlap answer pairs
- `topic_coverage_report.py` — Count Q&A pairs per topic
- `prose_analysis.py` — Check prose lengths and topic coverage

### Converter (root)
- `index.html` — Browser-based INT8 quantization converter
- `tokenizer.js` — Tokenizer used by the converter

### Technical Docs (`training/technical_docs/`)
- [`architecture_comparison.html`](https://cadengithubb.github.io/HardwareOne_LLM_Tool/training/technical_docs/architecture_comparison.html) — Visual comparison of model architectures
- [`transformer_deep_dive.html`](https://cadengithubb.github.io/HardwareOne_LLM_Tool/training/technical_docs/transformer_deep_dive.html) — Detailed transformer internals reference

---

## Model Presets

| Preset | Vocab | Layers | Dim | FFN | PSRAM (INT8) | Notes |
|--------|-------|--------|-----|-----|--------------|-------|
| **HW1HelpAgent192_deep** | 4K | 16 | 192 | 512 | ~7.7 MB | **Recommended** — best quality/size tradeoff |
| HW1HelpAgent | 4K | 22 | 128 | 768 | ~7.5 MB | Proven fallback, wide FFN |
| HW1HelpAgent192 | 4K | 12 | 192 | 768 | ~7.5 MB | Wider per-layer but shallower |

All presets target 8MB PSRAM on ESP32-S3. See `training/INSTRUCTIONS.txt` for the full preset list.

---

## Training Philosophy

### Boundary-Aware Q&A Packing
Q&A pairs are packed into fixed 128-token training blocks without splitting any pair across a boundary. Before this fix, ~39% of pairs were corrupted by being split mid-answer. The model now learns clean, complete Q&A associations.

### Three Training Pair Types
- **Q:/A:** — Standard question-answer pairs for knowledge retrieval
- **Q:/Do:** — Question paired with a short CLI command suggestion (e.g. `Do: opentof`)
- **Prose passages** — Topic descriptions that provide background context

### Instruction Verb Diversity
Answers use varied instruction verbs ("Type X", "Run X", "Use X", "The command is X") to prevent the model from over-predicting any single high-frequency token pattern.

### Topic Vocabulary Isolation
Question phrasings are varied for reinforcement (5 copies per answer) while keeping vocabulary strictly within each topic — WiFi questions don't use MQTT words, sensor questions don't use networking terms.

---

## Performance

**Typical results on ESP32-S3 @ 240MHz with HW1HelpAgent192_deep:**
- **Inference speed**: 2-4 tokens/second (INT8)
- **Model size**: ~7.3 MB (fits in 8 MB PSRAM with ~733 KB headroom)
- **Context window**: 128 tokens
- **Use case**: Single-turn domain Q&A — "What is ESP-NOW?", "How do I set the MQTT broker?"

---

## Integration with Hardware One Firmware

The trained model integrates with [Hardware One firmware](https://github.com/CadenGithubB/HardwareOne):

```bash
# CLI usage
llm load              # Load model.bin from SD card
llm generate What is ESP-NOW?
llm models            # List available models
llm status            # Check model state
```

**Web UI**: Navigate to the LLM tab for a chat interface with token-per-second stats.

---

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (CPU or CUDA)
- **8GB+ RAM** (16GB recommended for GPU training)
- **Modern browser** (Chrome/Edge/Firefox for the converter)
- **ESP32-S3** with 8MB PSRAM (for deployment)

---

## License

MIT License — See LICENSE file for details.

---

## Related

- **[Hardware One Firmware](https://github.com/CadenGithubB/HardwareOne)** — ESP32-S3 IoT platform with LLM inference support

---

## Links

| | |
|---|---|
| 📐 **[Architecture Comparison](https://cadengithubb.github.io/HardwareOne_LLM_Tool/training/technical_docs/architecture_comparison.html)** | Visual breakdown of model size and shape tradeoffs |
| 🧠 **[Transformer Deep Dive](https://cadengithubb.github.io/HardwareOne_LLM_Tool/training/technical_docs/transformer_deep_dive.html)** | How prompts move through the model — heads, attention, FFN, KV cache |
| 💾 **[Hardware One Firmware](https://github.com/CadenGithubB/HardwareOne)** | The ESP32-S3 platform this model runs on |

---

**Built for Hardware One — Local AI, No Cloud Required**
