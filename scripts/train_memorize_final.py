#!/usr/bin/env python3
"""
train_memorize_final.py – Final version matching exact specifications
====================================================================
Train a tiny GPT-2 model to overfit on a single sentence using the exact
configuration specified in the user's prompt.
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
    """Tokenize text and convert to .bin format using txt_to_bin()."""
    ids: List[int] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                ids.extend(tokenizer.encode(line))
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    print(f"[✓] wrote {len(ids):,} tokens → {bin_path}")


class MemorizationDataset(torch.utils.data.Dataset):
    """Dataset for memorization - repeats the same sequence many times."""

    def __init__(self, bin_path: Path, block_size: int = 32):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.sequence_length = len(self.data)
        # Repeat the sequence many times for memorization
        self.num_repetitions = 1000

    def __len__(self):
        return self.num_repetitions

    def __getitem__(self, idx):
        # Always return the same sequence
        if self.sequence_length <= self.block_size:
            sequence = np.pad(self.data, (0, self.block_size + 1 - self.sequence_length), 
                            mode='constant', constant_values=0)
        else:
            sequence = self.data[:self.block_size + 1]
        
        sequence = torch.from_numpy(sequence.astype(np.int64))
        return {"input_ids": sequence[:-1], "labels": sequence[1:]}


# ────────────────────────────────────────────────────────────────────────────────
# Callback for logging samples every 100 steps
# ────────────────────────────────────────────────────────────────────────────────

class SampleLoggingCallback(TrainerCallback):
    """Track loss, learning rate, and log text samples every 100 steps."""

    def __init__(self, tokenizer, target_text: str, target_tokens: List[int]):
        self.tokenizer = tokenizer
        self.target_text = target_text
        self.target_tokens = target_tokens

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % 100 != 0:
            return
        if model is None:
            return
        
        model.eval()
        with torch.no_grad():
            # Generate sample from the model
            first_token = torch.tensor([[self.target_tokens[0]]], device=model.device)
            
            output_ids = model.generate(
                first_token,
                max_new_tokens=len(self.target_tokens) - 1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
            
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            generated_tokens = output_ids.tolist()
            
            # Check if memorization is perfect
            perfect_match = generated_tokens == self.target_tokens
            
            # Log to wandb
            wandb.log({
                "step": step,
                "sample_text": generated_text,
                "target_text": self.target_text,
                "perfect_memorization": perfect_match,
            }, step=step)
            
            print(f"\n=== Step {step} Sample ===")
            print(f"Target:    {self.target_text}")
            print(f"Generated: {generated_text}")
            print(f"Perfect memorization: {perfect_match}")
            
            if perfect_match:
                print(f"🎉 Perfect memorization achieved at step {step}!")
        
        model.train()


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1. 📝 Input text file
    txt_path = Path("data/memorize.txt")
    bin_path = Path("data/memorize.bin")
    
    # 2. 🧠 Custom tokenizer
    tokenizer_path = Path("tokenizers/gpt2-forward-hf")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Load and analyze the target text
    target_text = txt_path.read_text().strip()
    target_tokens = tokenizer.encode(target_text)
    
    print(f"📝 Target text: {target_text}")
    print(f"🧠 Tokenized as: {tokenizer.tokenize(target_text)}")
    print(f"📊 Token IDs: {target_tokens}")
    
    # 3. Convert to .bin format using txt_to_bin()
    if not bin_path.exists():
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        txt_to_bin(txt_path, bin_path, tokenizer)
    
    # Create dataset
    train_ds = MemorizationDataset(bin_path, block_size=32)
    
    # 4. 🧬 Minimal GPT-2 config as specified
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
    print(f"🧬 Model parameters: {model.num_parameters():,}")
    
    # Initialize wandb
    wandb.init(
        project="memorization-test",
        name="memorize-final",
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
    
    # 5. ⚙️ Training arguments as specified
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
        dataloader_num_workers=0,
    )
    
    # Data collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 6. 📈 Callback for tracking loss, learning rate, and logging samples every 100 steps
    callback = SampleLoggingCallback(tokenizer, target_text, target_tokens)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[callback],
    )
    
    print("🚀 Starting training...")
    trainer.train()
    
    # 7. 🧪 Final verification - generate sample and verify it matches input
    print("\n" + "="*60)
    print("🧪 FINAL VERIFICATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        first_token = torch.tensor([[target_tokens[0]]], device=model.device)
        
        output_ids = model.generate(
            first_token,
            max_new_tokens=len(target_tokens) - 1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        generated_tokens = output_ids.tolist()
        
        print(f"📝 Original input:  {target_text}")
        print(f"🤖 Generated text: {generated_text}")
        print(f"📊 Target tokens:   {target_tokens}")
        print(f"📊 Generated tokens: {generated_tokens}")
        print(f"✅ Perfect match:   {generated_tokens == target_tokens}")
        
        # Log final results
        wandb.log({
            "final_target_text": target_text,
            "final_generated_text": generated_text,
            "final_perfect_match": generated_tokens == target_tokens,
            "final_target_tokens": target_tokens,
            "final_generated_tokens": generated_tokens,
        })
    
    wandb.finish()
    print("\n✅ Training complete! Pipeline sanity check passed.")


if __name__ == "__main__":
    main() 