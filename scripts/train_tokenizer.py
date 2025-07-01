#!/usr/bin/env python3
"""
Tokenizer training script for forward and reverse text.

Trains BPE tokenizers on both forward and reverse text to enable
fair comparison between language modeling directions.
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2TokenizerFast


def train_tokenizer(text_files: list, direction: str, vocab_size: int = 50257) -> Tokenizer:
    """Train a BPE tokenizer on text files."""
    print(f"Training {direction} tokenizer with vocab size {vocab_size}...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|unk|>", "<|pad|>", "<|eos|>", "<|bos|>"]
    )
    
    # Train tokenizer
    tokenizer.train(text_files, trainer)
    
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_path: Path, direction: str):
    """Save tokenizer in HuggingFace format."""
    print(f"Saving {direction} tokenizer to {output_path}")
    
    # Convert to HuggingFace format
    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.pad_token = "<|pad|>"
    wrapped_tokenizer.eos_token = "<|eos|>"
    wrapped_tokenizer.bos_token = "<|bos|>"
    
    # Save
    wrapped_tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer for language modeling")
    parser.add_argument("--direction", choices=["forward", "reverse"], required=True,
                       help="Direction of text for tokenizer training")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--output-dir", default="tokenizers/", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Vocabulary size")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find training files
    text_files = list(data_dir.glob(f"*{args.direction}*.txt"))
    if not text_files:
        print(f"No {args.direction} text files found in {data_dir}")
        return
    
    print(f"Found {len(text_files)} files for {args.direction} tokenizer training")
    
    # Train tokenizer
    tokenizer = train_tokenizer([str(f) for f in text_files], args.direction, args.vocab_size)
    
    # Save tokenizer
    save_path = output_dir / f"gpt2-{args.direction}"
    save_tokenizer(tokenizer, save_path, args.direction)
    
    print(f"Tokenizer training complete! Saved to {save_path}")


if __name__ == "__main__":
    main() 