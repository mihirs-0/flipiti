#!/usr/bin/env python3
"""
Text generation script for forward and reverse language models.

Generates text samples from trained models for qualitative analysis
and comparison of generation patterns.
"""

import argparse
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model_and_tokenizer(model_path: Path, tokenizer_path: Path, device: str):
    """Load model and tokenizer."""
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    
    tokenizer = GPT2TokenizerFast.from_pretrained(str(tokenizer_path))
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 150, 
                 temperature: float = 0.8, top_p: float = 0.9, num_samples: int = 3):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    generated_texts = []
    
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    return generated_texts


def interactive_generation(model, tokenizer, direction: str):
    """Interactive text generation session."""
    print(f"\n=== Interactive Generation ({direction} model) ===")
    print("Enter prompts to generate text. Type 'quit' to exit.\n")
    
    while True:
        prompt = input(f"{direction.capitalize()} prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            continue
        
        print(f"\nGenerating from '{prompt}'...\n")
        
        try:
            generated_texts = generate_text(model, tokenizer, prompt)
            
            for i, text in enumerate(generated_texts, 1):
                print(f"Sample {i}:")
                print(f"{text}\n")
                print("-" * 80)
        
        except Exception as e:
            print(f"Error generating text: {e}")


def batch_generation(model, tokenizer, prompts: list, direction: str, output_file: Path):
    """Generate text for a batch of prompts."""
    print(f"Generating text for {len(prompts)} prompts using {direction} model...")
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Processing prompt {i}/{len(prompts)}: {prompt[:50]}...")
        
        try:
            generated_texts = generate_text(model, tokenizer, prompt)
            
            results.append({
                "prompt": prompt,
                "direction": direction,
                "generated_samples": generated_texts
            })
        
        except Exception as e:
            print(f"Error with prompt '{prompt}': {e}")
            results.append({
                "prompt": prompt,
                "direction": direction,
                "error": str(e)
            })
    
    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text from language models")
    parser.add_argument("--direction", choices=["forward", "reverse", "both"], 
                       default="both", help="Which model(s) to use")
    parser.add_argument("--models-dir", default="models/", help="Models directory")
    parser.add_argument("--tokenizers-dir", default="tokenizers/", help="Tokenizers directory")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--prompts-file", help="File with prompts for batch generation")
    parser.add_argument("--output-file", default="generated_samples.json", 
                       help="Output file for batch generation")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    tokenizers_dir = Path(args.tokenizers_dir)
    
    # Default prompts if no file provided
    default_prompts = [
        "The quick brown fox",
        "In a world where artificial intelligence",
        "The secret to happiness is",
        "Once upon a time in a distant galaxy",
        "The future of technology will",
        "Climate change is",
        "The most important lesson I learned",
        "In the year 2050, humans will"
    ]
    
    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = default_prompts
    
    # Process each direction
    directions = ["forward", "reverse"] if args.direction == "both" else [args.direction]
    
    for direction in directions:
        model_path = models_dir / f"gpt2-{direction}"
        tokenizer_path = tokenizers_dir / f"gpt2-{direction}"
        
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        if not tokenizer_path.exists():
            print(f"Tokenizer not found: {tokenizer_path}")
            continue
        
        print(f"\nLoading {direction} model...")
        model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device)
        
        if args.interactive:
            interactive_generation(model, tokenizer, direction)
        else:
            output_file = Path(f"{direction}_{args.output_file}")
            batch_generation(model, tokenizer, prompts, direction, output_file)


if __name__ == "__main__":
    main() 