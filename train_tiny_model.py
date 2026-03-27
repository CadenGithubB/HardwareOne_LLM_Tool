#!/usr/bin/env python3
"""
Train a tiny GPT-2–compatible LM for the esp32-llm-converter (browser INT8 → model.bin).

Install (once):
  pip install torch transformers datasets tokenizers accelerate

Example:
  # 1) Put some plain text in data.txt (more text = better tokenizer + model)
  python train_tiny_model.py --text data.txt --out ./my_tiny_gpt2

  # 2) Or stream a slice of TinyStories (downloads ~data~ on first run)
  python train_tiny_model.py --dataset tiny_stories --max_samples 50000 --out ./my_tiny_gpt2

  # 3) Open index.html → drop the output folder or zip contents → Convert

Defaults target a few million parameters and a small vocab so model.bin can fit ESP32 flash.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train tiny GPT-2 for esp32-llm-converter")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--text",
        type=Path,
        help="Plain-text file to train tokenizer + model on (UTF-8)",
    )
    src.add_argument(
        "--dataset",
        choices=("tiny_stories",),
        help="Built-in dataset name (downloads via Hugging Face datasets)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for checkpoint")
    p.add_argument("--vocab-size", type=int, default=2048, help="BPE vocab size (smaller → smaller model.bin)")
    p.add_argument("--n-embd", type=int, default=64, help="Hidden size (dim)")
    p.add_argument("--n-layer", type=int, default=4, help="Transformer blocks")
    p.add_argument("--n-head", type=int, default=8, help="Attention heads (must divide n-embd)")
    p.add_argument("--n-inner", type=int, default=None, help="FFN hidden (default: 4 * n-embd)")
    p.add_argument("--seq-len", type=int, default=128, help="Context length (blocks for LM)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-samples", type=int, default=None, help="Cap training rows (TinyStories)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_bpe_tokenizer(text_paths: list[Path], vocab_size: int, out_dir: Path) -> "GPT2TokenizerFast":
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2TokenizerFast

    tokenizer_core = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer_core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>", "<unk>"],
    )
    tokenizer_core.train([str(p) for p in text_paths], trainer)
    tok_path = out_dir / "tokenizer.json"
    tokenizer_core.save(str(tok_path))

    hf_tok = GPT2TokenizerFast(tokenizer_file=str(tok_path))
    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_tok.save_pretrained(out_dir)
    return hf_tok


def load_text_dataset(args: argparse.Namespace) -> tuple[list[Path], "Dataset"]:
    """Returns (temp_files_to_cleanup_or_empty, hf_dataset)."""
    from datasets import Dataset, load_dataset

    if args.text:
        if not args.text.is_file():
            sys.exit(f"Not a file: {args.text}")
        return [], Dataset.from_dict({"text": [args.text.read_text(encoding="utf-8", errors="replace")]})

    assert args.dataset == "tiny_stories"
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_samples:
        n = min(len(ds), args.max_samples)
        ds = ds.select(range(n))
    return [], ds


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.n_embd % args.n_head != 0:
        sys.exit(f"n-embd ({args.n_embd}) must be divisible by n-head ({args.n_head})")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import (
            DataCollatorForLanguageModeling,
            GPT2Config,
            GPT2LMHeadModel,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nInstall: pip install torch transformers datasets tokenizers accelerate")

    temp_files: list[Path] = []
    text_paths: list[Path] = []

    if args.text:
        text_paths = [args.text.resolve()]
        ds_raw, _ = load_text_dataset(args)
    else:
        _, hf_train = load_text_dataset(args)
        chunk_path = out_dir / "_train_corpus.txt"
        with chunk_path.open("w", encoding="utf-8") as f:
            for row in hf_train:
                f.write(row["text"].strip() + "\n")
        temp_files.append(chunk_path)
        text_paths = [chunk_path]
        ds_raw = hf_train

    print("Training BPE tokenizer…")
    tokenizer = train_bpe_tokenizer(text_paths, args.vocab_size, out_dir)

    n_inner = args.n_inner if args.n_inner is not None else 4 * args.n_embd
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=n_inner,
        bos_token_id=eos_id,
        eos_token_id=eos_id,
    )
    model = GPT2LMHeadModel(config)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )

    if args.text:
        tok_ds = ds_raw.map(tokenize, batched=False, remove_columns=["text"])
    else:
        nproc = max(1, min(8, (args.max_samples or 50000) // 1000 + 1))
        tok_ds = ds_raw.map(
            tokenize,
            batched=True,
            remove_columns=ds_raw.column_names,
            num_proc=nproc,
        )

    block_size = args.seq_len

    def group_texts(examples):
        # Concatenate only token ids (ignore attention_mask boundaries for simplicity)
        all_ids: list[int] = []
        for ids in examples["input_ids"]:
            all_ids.extend(ids)
        total_length = len(all_ids)
        if total_length < block_size:
            return {"input_ids": [], "labels": []}
        total_length = (total_length // block_size) * block_size
        chunks = [all_ids[i : i + block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    rm_cols = [c for c in tok_ds.column_names if c != "input_ids"]
    lm_ds = tok_ds.map(
        group_texts,
        batched=True,
        batch_size=10_000,
        remove_columns=rm_cols,
    )
    lm_ds = lm_ds.filter(lambda ex: len(ex["input_ids"]) > 0)
    if len(lm_ds) == 0:
        sys.exit(
            "No training blocks after grouping — use longer text, more TinyStories samples, or lower --seq-len."
        )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer_ckpt"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=10_000,
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=use_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds,
        data_collator=collator,
    )

    print("Training…")
    trainer.train()

    print(f"Saving to {out_dir} …")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for p in temp_files:
        try:
            p.unlink()
        except OSError:
            pass

    print("Done. Contents:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
