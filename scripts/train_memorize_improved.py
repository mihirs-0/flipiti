#!/usr/bin/env python3
"""
train_memorize_improved.py – Improved version with better tokenization handling
==============================================================================
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
    EarlyStoppingCallback,
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
                # Add BOS token at the beginning
                encoded = tokenizer.encode(line)
                ids.extend(encoded)
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    print(f"[✓] wrote {len(ids):,} tokens → {bin_path}")


class MemorizationDataset(torch.utils.data.Dataset):
    """Dataset that cycles through the same sequence multiple times for memorization."""

    def __init__(self, bin_path: Path, block_size: int = 32):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.sequence_length = len(self.data)
        
        # For memorization, we want many repetitions of the same sequence
        self.num_repetitions = 100

    def __len__(self):
        return self.num_repetitions

    def __getitem__(self, idx):
        # Always return the same sequence for memorization
        if self.sequence_length <= self.block_size:
            # If sequence is short, pad it
            sequence = np.pad(self.data, (0, self.block_size + 1 - self.sequence_length), 
                            mode='constant', constant_values=0)
        else:
            # If sequence is long, truncate it
            sequence = self.data[:self.block_size + 1]
        
        sequence = torch.from_numpy(sequence.astype(np.int64))
        return {"input_ids": sequence[:-1], "labels": sequence[1:]}


# ────────────────────────────────────────────────────────────────────────────────
# Callback for logging samples and metrics
# ────────────────────────────────────────────────────────────────────────────────

class MemorizationCallback(TrainerCallback):
    """Log samples and metrics every N steps to track memorization progress."""

    def __init__(self, tokenizer, target_text: str, target_tokens: List[int], log_every: int = 100):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.target_tokens = target_tokens
        self.log_every = log_every
        self.best_match_step = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.log_every != 0:
            return
        if model is None:
            return
        
        model.eval()
        with torch.no_grad():
            # Generate from the beginning token
            first_token = torch.tensor([[self.target_tokens[0]]], device=model.device)
            
            # Generate the exact number of tokens we need
            output_ids = model.generate(
                first_token,
                max_new_tokens=len(self.target_tokens) - 1,
                do_sample=False,  # Use greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                top_p=1.0,
            )[0]
            
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # Check if the generated tokens match exactly
            generated_tokens = output_ids.tolist()
            exact_match = generated_tokens == self.target_tokens
            
            # Log to wandb
            wandb.log({
                "step": step,
                "generated_text": generated_text,
                "target_text": self.target_text,
                "exact_token_match": exact_match,
                "text_matches_target": generated_text.strip() == self.target_text.strip(),
                "generated_tokens": generated_tokens,
                "target_tokens": self.target_tokens,
            }, step=step)
            
            print(f"\n=== Step {step} ===")
            print(f"Target:    {self.target_text}")
            print(f"Generated: {generated_text}")
            print(f"Target tokens:    {self.target_tokens}")
            print(f"Generated tokens: {generated_tokens}")
            print(f"Exact match: {exact_match}")
            print(f"Text match: {generated_text.strip() == self.target_text.strip()}")
            
            # Early stopping if perfect match
            if exact_match and self.best_match_step is None:
                self.best_match_step = step
                print(f"🎉 Perfect memorization achieved at step {step}!")
                control.should_training_stop = True
        
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
    
    # Load target text and analyze tokenization
    target_text = txt_path.read_text().strip()
    target_tokens = tokenizer.encode(target_text)
    
    print(f"Target text: {target_text}")
    print(f"Target tokens: {target_tokens}")
    print(f"Tokenized: {tokenizer.tokenize(target_text)}")
    print(f"Decoded: {tokenizer.decode(target_tokens)}")
    
    # Convert to binary format
    if not bin_path.exists():
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        txt_to_bin(txt_path, bin_path, tokenizer)
    
    # Create dataset
    train_ds = MemorizationDataset(bin_path, block_size=32)
    print(f"Dataset size: {len(train_ds)} samples")
    
    # Even smaller GPT-2 config for faster memorization
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=32,
        n_embd=32,  # Smaller embedding
        n_layer=2,
        n_head=2,
        bos_token_id=1,  # Correct BOS token
        eos_token_id=2,  # Correct EOS token
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Initialize wandb
    wandb.init(
        project="memorization-test",
        name="memorize-improved",
        config={
            "target_text": target_text,
            "target_tokens": target_tokens,
            "vocab_size": tokenizer.vocab_size,
            "model_params": model.num_parameters(),
            "n_positions": 32,
            "n_embd": 32,
            "n_layer": 2,
            "n_head": 2,
        }
    )
    
    # Training arguments with higher learning rate for faster memorization
    training_args = TrainingArguments(
        output_dir="tmp_memorize_improved",
        per_device_train_batch_size=4,  # Larger batch size
        learning_rate=5e-3,  # Higher learning rate
        max_steps=2000,
        logging_steps=50,
        report_to="wandb",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
        save_steps=1000,
        overwrite_output_dir=True,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
    )
    
    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Callback for logging samples
    callback = MemorizationCallback(tokenizer, target_text, target_tokens, log_every=50)
    
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
        # Test generation starting from first token
        first_token = torch.tensor([[target_tokens[0]]], device=model.device)
        
        output_ids = model.generate(
            first_token,
            max_new_tokens=len(target_tokens) - 1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.1,
        )[0]
        
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_tokens = output_ids.tolist()
        
        print(f"Target text:      {target_text}")
        print(f"Generated text:   {generated_text}")
        print(f"Target tokens:    {target_tokens}")
        print(f"Generated tokens: {generated_tokens}")
        print(f"Perfect match:    {generated_tokens == target_tokens}")
        print(f"Text match:       {generated_text.strip() == target_text.strip()}")
        
        # Log final results
        wandb.log({
            "final_generated_text": generated_text,
            "final_target_text": target_text,
            "final_perfect_match": generated_tokens == target_tokens,
            "final_text_match": generated_text.strip() == target_text.strip(),
            "final_generated_tokens": generated_tokens,
            "final_target_tokens": target_tokens,
        })
    
    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main() 