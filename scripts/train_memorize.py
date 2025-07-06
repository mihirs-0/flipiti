#!/usr/bin/env python3
"""
train_memorize.py – Train a tiny GPT-2 to overfit a single sentence
===================================================================
Pipeline sanity check: train a minimal GPT-2 model to memorize exactly one sentence.
The goal is for the model to perfectly reproduce the input text.

Usage:
    python scripts/train_memorize.py

This script will:
1. Load the sentence from data/memorize.txt
2. Tokenize it with a custom tokenizer  
3. Train a tiny GPT-2 model to overfit
4. Generate samples to verify memorization
"""
import argparse
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# ────────────────────────────────────────────────────────────────────────────────
# Data utilities
# ────────────────────────────────────────────────────────────────────────────────

def txt_to_bin(txt_path: Path, bin_path: Path, tokenizer: PreTrainedTokenizerFast):
    """Tokenise a .txt file and dump raw uint16 IDs."""
    ids: List[int] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                ids.extend(tokenizer.encode(line))
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    print(f"[✓] wrote {len(ids):,} tokens → {bin_path}")


class PackedDataset(torch.utils.data.Dataset):
    """Flat‑packed dataset à la Karpathy: rolling windows from mem‑mapped array."""

    def __init__(self, bin_path: Path, block_size: int = 32):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block - 1)

    def __getitem__(self, idx):
        # For memorization, we want to repeat the same sequence
        # Use modulo to cycle through the available data
        start_idx = idx % max(1, len(self.data) - self.block)
        chunk = torch.from_numpy(
            self.data[start_idx : start_idx + self.block + 1].astype(np.int64)
        )
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


# ────────────────────────────────────────────────────────────────────────────────
# Callback for logging samples and metrics
# ────────────────────────────────────────────────────────────────────────────────

class MemorizationCallback(TrainerCallback):
    """Log samples and metrics every N steps to track memorization progress."""

    def __init__(self, tokenizer, target_text: str, log_every: int = 100):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.log_every = log_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.log_every != 0:
            return
        if model is None:
            return
        
        model.eval()
        with torch.no_grad():
            # Generate from the beginning of the target text
            first_word = self.target_text.split()[0]
            prompt_ids = self.tokenizer(first_word, return_tensors="pt").input_ids.to(model.device)
            
            # Generate the full sequence
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=20,
                do_sample=False,  # Use greedy decoding for deterministic output
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
            
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Log to wandb
            wandb.log({
                "step": step,
                "generated_text": generated_text,
                "target_text": self.target_text,
                "matches_target": generated_text.strip() == self.target_text.strip(),
            }, step=step)
            
            print(f"\n=== Step {step} ===")
            print(f"Target:    {self.target_text}")
            print(f"Generated: {generated_text}")
            print(f"Match: {generated_text.strip() == self.target_text.strip()}")
        
        model.train()


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Fixed paths and parameters
    txt_path = Path("data/memorize.txt")
    bin_path = Path("data/memorize.bin")
    tokenizer_path = Path("tokenizers/gpt2-forward-hf")
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Load target text
    target_text = txt_path.read_text().strip()
    print(f"Target text: {target_text}")
    
    # Convert to binary format
    if not bin_path.exists():
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        txt_to_bin(txt_path, bin_path, tokenizer)
    
    # Create dataset
    train_ds = PackedDataset(bin_path, block_size=32)
    print(f"Dataset size: {len(train_ds)} samples")
    
    # Minimal GPT-2 config as specified
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=32,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=3,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Initialize wandb
    wandb.init(
        project="memorization-test",
        name="memorize-single-sentence",
        config={
            "target_text": target_text,
            "vocab_size": tokenizer.vocab_size,
            "model_params": model.num_parameters(),
            "n_positions": 32,
            "n_embd": 64,
            "n_layer": 2,
            "n_head": 2,
        }
    )
    
    # Training arguments as specified
    training_args = TrainingArguments(
        output_dir="tmp_memorize_run",
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        max_steps=1000,
        logging_steps=100,
        report_to="wandb",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
        save_steps=500,
        overwrite_output_dir=True,
    )
    
    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Callback for logging samples
    callback = MemorizationCallback(tokenizer, target_text, log_every=100)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[callback],
    )
    
    print("Starting training...")
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        # Test generation starting from first word
        first_word = target_text.split()[0]
        prompt_ids = tokenizer(first_word, return_tensors="pt").input_ids.to(model.device)
        
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        print(f"Target text:    {target_text}")
        print(f"Generated text: {generated_text}")
        print(f"Perfect match:  {generated_text.strip() == target_text.strip()}")
        
        # Log final results
        wandb.log({
            "final_generated_text": generated_text,
            "final_target_text": target_text,
            "final_perfect_match": generated_text.strip() == target_text.strip(),
        })
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main() 