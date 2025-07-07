#!/usr/bin/env python3
"""
GPT-2 training script for forward and reverse language modeling.

Trains GPT-2 small models from scratch on forward and reverse text
using identical hyperparameters for fair comparison.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb


class LanguageModelingDataset:
    """Dataset for language modeling with custom tokenizer."""
    
    def __init__(self, text_files: list, tokenizer, block_size: int = 512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # TODO: Implement dataset loading and tokenization
        pass


def create_model_config(vocab_size: int) -> GPT2Config:
    """Create GPT-2 small configuration."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=3
    )


def setup_training_args(output_dir: Path, direction: str) -> TrainingArguments:
    """Setup training arguments."""
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        logging_steps=100,
        save_steps=5000,
        eval_steps=5000,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=f"gpt2-{direction}",
        dataloader_num_workers=4,
        fp16=torch.cuda.is_available(),
    )


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 language model")
    parser.add_argument("--config", choices=["forward", "reverse"], required=True,
                       help="Model configuration (forward or reverse)")
    parser.add_argument("--data-dir", default="data/", help="Data directory")
    parser.add_argument("--tokenizer-dir", default="tokenizers/", help="Tokenizer directory")
    parser.add_argument("--output-dir", default="models/", help="Output directory")
    parser.add_argument("--wandb-project", default="reverse-language-modeling", 
                       help="Weights & Biases project name")
    
    args = parser.parse_args()
    
    # Setup directories
    data_dir = Path(args.data_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    output_dir = Path(args.output_dir) / f"gpt2-{args.config}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"gpt2-{args.config}",
        config={
            "direction": args.config,
            "model": "gpt2-small",
            "training": "from_scratch"
        }
    )
    
    # Load tokenizer
    tokenizer_path = tokenizer_dir / f"gpt2-{args.config}"
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    print(f"Loaded tokenizer from {tokenizer_path}")
    
    # Create model
    config = create_model_config(len(tokenizer))
    model = GPT2LMHeadModel(config)
    print(f"Created model with {model.num_parameters()} parameters")
    
    # Load datasets
    train_files = list(data_dir.glob(f"train_{args.config}*.txt"))
    eval_files = list(data_dir.glob(f"eval_{args.config}*.txt"))
    
    # TODO: Create datasets
    # train_dataset = LanguageModelingDataset(train_files, tokenizer)
    # eval_dataset = LanguageModelingDataset(eval_files, tokenizer)
    
    # Setup training
    training_args = setup_training_args(output_dir, args.config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # TODO: Initialize trainer and start training
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    # )
    
    # trainer.train()
    # trainer.save_model()
    
    print(f"Training setup complete! Model will be saved to {output_dir}")


if __name__ == "__main__":
    main() 