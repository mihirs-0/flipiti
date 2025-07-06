#!/usr/bin/env python3
"""
Convert raw tokenizer files to HuggingFace format
"""

import os
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

def convert_tokenizer_to_hf(tokenizer_path, output_dir):
    """
    Convert a raw tokenizer file to HuggingFace format
    
    Args:
        tokenizer_path: Path to the raw tokenizer.json file
        output_dir: Directory to save the HuggingFace format tokenizer
    """
    print(f"Converting {tokenizer_path} to HuggingFace format...")
    
    # Load the raw tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Wrap into HuggingFace interface
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        pad_token="<PAD>",
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in HF format
    hf_tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Tokenizer saved to {output_dir}")
    print(f"   - tokenizer.json")
    print(f"   - special_tokens_map.json")
    print(f"   - tokenizer_config.json")

def main():
    # Convert both tokenizers
    tokenizers_to_convert = [
        ("tokenizers/tokenizer_forward_10gb.json", "tokenizers/gpt2-forward-hf"),
        ("tokenizers/tokenizer_reverse_10gb.json", "tokenizers/gpt2-reverse-hf"),
    ]
    
    for tokenizer_path, output_dir in tokenizers_to_convert:
        if os.path.exists(tokenizer_path):
            convert_tokenizer_to_hf(tokenizer_path, output_dir)
        else:
            print(f"⚠️  Warning: {tokenizer_path} not found, skipping...")
    
    print("\n✅ All tokenizers converted successfully!")

if __name__ == "__main__":
    main() 