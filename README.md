# HardwareOne LLM Tool

**Train and deploy tiny language models for ESP32-S3 on-device inference**

Part of the [Hardware One](https://github.com/yourusername/HardwareOne) ecosystem — a self-contained IoT platform with WiFi, sensors, ESP-NOW mesh networking, MQTT, and local AI inference.

---

## What This Tool Does

HardwareOne LLM Tool trains ultra-compact GPT-2 style language models that run entirely on the ESP32-S3 microcontroller using only 8MB of PSRAM. No cloud, no internet required — the model runs locally on the device.

### Key Features

- **Tiny Models**: 4K-8K vocab, 12-20 layers, ~6-7 MB quantized
- **On-Device Training**: Train custom models for your specific domain
- **INT8 Quantization**: Browser-based converter for efficient inference
- **Q&A Optimized**: Boundary-aware training prevents answer bleed
- **Two-Phase Training**: Positive examples + negative corrections
- **Hardware One Integration**: Drop-in model.bin files for the firmware

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU training (recommended):
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Train Your Model

```bash
python train_tiny_model.py \
    --preset narrow3 \
    --text training_data/hardwareone_qa.txt \
           training_data/hardwareone_qa_comprehensive.txt \
           training_data/hardwareone_qa_expanded.txt \
           training_data/hardwareone_qa_v2.txt \
           training_data/hardwareone_qa_paraphrases.txt \
           training_data/hardwareone_qa_natural.txt \
           training_data/hardwareone_rich.txt \
           training_data/hardwareone_qa_troubleshooting.txt \
    --negatives training_data/hardwareone_qa_negatives.txt \
    --epochs 300 --batch-size 8 --lr 3e-4 \
    --neg-epochs 30 \
    --out ./out_model
```

Training takes ~30 minutes on RTX A4000 GPU, 8-12 hours on CPU.

### 3. Convert to ESP32 Format

1. Open `index.html` in Chrome/Edge/Firefox
2. Drag the `./out_model` folder onto the page
3. Select **INT8 quantization**, group size **128**
4. Click **Convert**, then **Download** → saves `model.bin`

### 4. Deploy to Hardware One

Copy `model.bin` to the SD card root or `/system/llm/` on internal flash. The firmware auto-loads it on boot.

---

## What's Included

### Training Data (Hardware One Domain)
- `hardwareone_qa.txt` — Core Q&A pairs with command examples
- `hardwareone_qa_comprehensive.txt` — Architecture and technical details
- `hardwareone_qa_expanded.txt` — Deep dives into LLM, sensors, GPS, OTA
- `hardwareone_qa_v2.txt` — Alternative phrasings for WiFi, ESP-NOW, MQTT
- `hardwareone_qa_paraphrases.txt` — Casual/conversational variations
- `hardwareone_qa_natural.txt` — Natural language with varied answers
- `hardwareone_rich.txt` — Comprehensive command reference with corrections
- `hardwareone_qa_troubleshooting.txt` — Problem-solution pairs
- `hardwareone_qa_negatives.txt` — Corrects common misconceptions

### Tools
- `train_tiny_model.py` — Training script with two-phase learning
- `index.html` — Browser-based INT8 quantization converter
- `INSTRUCTIONS.txt` — Detailed training guide and preset reference

---

## Model Presets

| Preset   | Vocab | Layers | Dim | FFN | PSRAM (INT8) | Notes |
|----------|-------|--------|-----|-----|--------------|-------|
| **narrow3** | 4K | 18 | 128 | 768 | ~6.9 MB | **Recommended** — wider FFN for better fact storage |
| narrow2  | 4K | 20 | 128 | 640 | ~6.9 MB | More layers, narrower FFN |
| stretch  | 8K | 18 | 128 | 512 | ~7.4 MB | Larger vocab for diverse domains |
| leaner   | 8K | 15 | 128 | 512 | ~6.0 MB | Smaller model, larger vocab |

See `INSTRUCTIONS.txt` for full preset details.

---

## Training Philosophy

### Boundary-Aware Q&A Training
Each Q&A pair is treated as an independent training block. The model learns to:
- Stop generation when it emits a second `Q:`
- Prevent answers from bleeding into unrelated questions
- Maintain context within a single Q&A exchange

### Two-Phase Learning
1. **Phase 1 (300 epochs)**: Learn positive Q&A associations
2. **Phase 2 (30 epochs)**: Apply negative corrections to distinguish similar concepts

Example negative correction:
```
Q: Is ESP-NOW the same as WiFi?
A: ESP-NOW is a direct device-to-device radio link. It connects ESP32 units 
   to each other without any router or access point.
```

This prevents the model from conflating ESP-NOW, WiFi, BLE, and MQTT.

---

## Performance

**Typical Results on ESP32-S3 @ 240MHz:**
- **Inference speed**: 2-8 tokens/second (INT8)
- **Model size**: 6.9 MB (fits in 8 MB PSRAM)
- **Context window**: 128-256 tokens (maintains conversation history across multiple Q&A exchanges)
- **Accuracy**: Domain-specific Q&A with 85-95% relevance

---

## Integration with Hardware One Firmware

The trained model integrates seamlessly with [Hardware One firmware](https://github.com/yourusername/HardwareOne):

```bash
# CLI usage
llm load              # Load model.bin from SD card or internal flash
llm generate What is ESP-NOW?
llm models            # List available models
llm status            # Check model state
```

**Web UI**: Navigate to the LLM tab for a chat interface with token-per-second stats.

---

## Requirements

- **Python 3.8+**
- **PyTorch 2.0+** (CPU or CUDA)
- **8GB+ RAM** (16GB recommended for training)
- **Modern browser** (Chrome/Edge/Firefox for converter)

---

## License

MIT License — See LICENSE file for details.

---

## Related Projects

- **[Hardware One Firmware](https://github.com/yourusername/HardwareOne)** — ESP32-S3 IoT platform
- **[Hardware One Web UI](https://github.com/yourusername/HardwareOne-WebUI)** — Dashboard and controls
- **[Hardware One Docs](https://github.com/yourusername/HardwareOne-Docs)** — Complete documentation

---

## Contributing

Contributions welcome! Please open an issue or PR for:
- New training data for different domains
- Model architecture improvements
- Quantization optimizations
- Bug fixes and documentation

---

**Built for Hardware One — Local AI, No Cloud Required**
