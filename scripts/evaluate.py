#!/usr/bin/env python3
"""
Evaluation script for forward and reverse language models.

Computes perplexity, generates text samples, and compares
model performance across different metrics.
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


def calculate_perplexity(model, tokenizer, text_data: list, device: str = "cuda"):
    """Calculate perplexity on text data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(text_data, desc="Calculating perplexity"):
            # Tokenize text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Calculate loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Accumulate
            seq_len = inputs["input_ids"].shape[1]
            total_loss += loss.item() * seq_len
            total_tokens += seq_len
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item()


def generate_samples(model, tokenizer, prompts: list, max_length: int = 100, num_samples: int = 5):
    """Generate text samples from model."""
    model.eval()
    samples = []
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        for i in range(num_samples):
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Sample {i+1}: {generated_text}")
            samples.append({
                "prompt": prompt,
                "generated": generated_text,
                "sample_id": i
            })
    
    return samples


def evaluate_model(model_path: Path, tokenizer_path: Path, test_data: list, direction: str):
    """Evaluate a single model."""
    print(f"Evaluating {direction} model...")
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokenizer, test_data, device)
    print(f"{direction} model perplexity: {perplexity:.2f}")
    
    # Generate samples
    test_prompts = [
        "The quick brown fox",
        "In a galaxy far far away",
        "The meaning of life is",
        "Once upon a time",
        "The future of artificial intelligence"
    ]
    
    samples = generate_samples(model, tokenizer, test_prompts)
    
    return {
        "direction": direction,
        "perplexity": perplexity,
        "samples": samples
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models")
    parser.add_argument("--models-dir", default="models/", help="Models directory")
    parser.add_argument("--tokenizers-dir", default="tokenizers/", help="Tokenizers directory")
    parser.add_argument("--test-data", default="data/test.txt", help="Test data file")
    parser.add_argument("--output-file", default="evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    tokenizers_dir = Path(args.tokenizers_dir)
    
    # Load test data
    # TODO: Implement test data loading
    test_data = ["This is a sample test sentence."]  # Placeholder
    
    results = {}
    
    # Evaluate forward model
    forward_model_path = models_dir / "gpt2-forward"
    forward_tokenizer_path = tokenizers_dir / "gpt2-forward"
    
    if forward_model_path.exists() and forward_tokenizer_path.exists():
        results["forward"] = evaluate_model(
            forward_model_path, forward_tokenizer_path, test_data, "forward"
        )
    
    # Evaluate reverse model
    reverse_model_path = models_dir / "gpt2-reverse"
    reverse_tokenizer_path = tokenizers_dir / "gpt2-reverse"
    
    if reverse_model_path.exists() and reverse_tokenizer_path.exists():
        results["reverse"] = evaluate_model(
            reverse_model_path, reverse_tokenizer_path, test_data, "reverse"
        )
    
    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete! Results saved to {args.output_file}")
    
    # Print summary
    if "forward" in results and "reverse" in results:
        print("\n=== COMPARISON SUMMARY ===")
        print(f"Forward perplexity:  {results['forward']['perplexity']:.2f}")
        print(f"Reverse perplexity:  {results['reverse']['perplexity']:.2f}")
        
        perplexity_diff = results['reverse']['perplexity'] - results['forward']['perplexity']
        print(f"Difference:          {perplexity_diff:+.2f}")


if __name__ == "__main__":
    main() 