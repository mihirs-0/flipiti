#!/usr/bin/env python3
"""
trial_train_gpt2.py – Karpathy‑style sanity run
================================================
* Converts a small .txt corpus → uint16 .bin (one‑time)
* Memory‑maps the .bin with packed, fixed‑length blocks
* Trains a *tiny* GPT‑2 from scratch for a handful of steps
* Prints a sample completion at the end

Assumptions
-----------
- Tokenizer lives at ``tokenizers/gpt2-reverse-hf``
- Raw text is ``data/train_reversed.txt``
- Special tokens: <PAD>=0, <UNK>=1, <EOS>=2, <BOS>=3

Run:
    python trial_train_gpt2.py \
        --txt data/train_reversed.txt \
        --tokenizer tokenizers/gpt2-reverse-hf \
        --bin data/train_reversed.bin
"""
import argparse, os, pathlib, struct, sys
from pathlib import Path

import numpy as np
import torch
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ---------- Helpers ---------------------------------------------------------

def txt_to_bin(txt_path: Path, bin_path: Path, tokenizer: PreTrainedTokenizerFast):
    """Tokenize `txt_path` and write uint16 IDs to `bin_path` (Karpathy style)."""
    ids = []
    with txt_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            ids.extend(tokenizer.encode(line))
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"[info] wrote {len(arr):,} tokens → {bin_path}")


class PackedDataset(torch.utils.data.Dataset):
    """Memory‑mapped packed dataset: each item is a block of `block_size` tokens."""

    def __init__(self, bin_path: Path, block_size: int = 128):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block = block_size

    def __len__(self):
        # one training example per position where we have block_size+1 tokens
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        chunk = torch.from_numpy(self.data[idx : idx + self.block + 1].astype(np.int64))
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


# ---------- Main ------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="Karpathy‑style tiny GPT‑2 trial run")
    p.add_argument("--txt", type=Path, required=True, help="Input .txt file (small)")
    p.add_argument("--tokenizer", type=Path, required=True, help="HF tokenizer dir")
    p.add_argument("--bin", type=Path, required=True, help="Output .bin path")
    p.add_argument("--steps", type=int, default=10, help="Trainer max steps")
    p.add_argument("--block_size", type=int, default=128)
    args = p.parse_args(argv)

    # 1) Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    assert tokenizer.pad_token_id == 0  # sanity

    # 2) Build .bin if needed
    if not args.bin.exists():
        args.bin.parent.mkdir(parents=True, exist_ok=True)
        txt_to_bin(args.txt, args.bin, tokenizer)
    else:
        print(f"[info] using existing {args.bin}")

    # 3) Dataset
    ds = PackedDataset(args.bin, block_size=args.block_size)

    # 4) Tiny GPT‑2 config
    cfg = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=3,
        eos_token_id=2,
        pad_token_id=0,
    )

    # 5) Model (try FlashAttention‑2 if supported)
    try:
        model = GPT2LMHeadModel(cfg, attn_implementation="flash_attention_2")
        print("[info] FlashAttention‑2 enabled (GPU w/ Hopper/Ampere)")
    except TypeError:
        model = GPT2LMHeadModel(cfg)
        print("[info] FlashAttention‑2 unavailable; using standard attention")

    # 6) TrainingArguments – run just a few steps
    targs = TrainingArguments(
        output_dir="tmp_trial_run",
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        max_steps=args.steps,
        logging_steps=1,
        report_to="none",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    # 7) Quick sample
    prompt = "ehT"  # first word reversed: "The"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Move model and inputs to the same device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    out = model.generate(input_ids, max_new_tokens=40, do_sample=True)
    print("\n=== SAMPLE ===")
    print(tokenizer.decode(out[0]))


if __name__ == "__main__":
    main() 