#!/usr/bin/env python3
"""
Dataset preparation script for forward vs. reverse language modeling.

This script downloads and preprocesses text data for training both 
forward and reverse language models.
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_dataset(dataset_name: str, output_dir: Path):
    """Download and save raw dataset."""
    print(f"Downloading {dataset_name}...")
    # TODO: Implement dataset download
    pass


def create_reverse_text(text: str) -> str:
    """Reverse the order of characters/tokens in text."""
    # TODO: Implement text reversal logic
    return text[::-1]  # Simple character reversal for now


def preprocess_and_split(input_file: Path, output_dir: Path):
    """Process raw text and create forward/reverse splits."""
    print(f"Processing {input_file}...")
    
    # TODO: Implement preprocessing:
    # 1. Clean and normalize text
    # 2. Split into train/validation sets
    # 3. Create reverse versions
    # 4. Save processed files
    
    pass


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for language modeling")
    parser.add_argument("--dataset", default="openwebtext", help="Dataset to use")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    parser.add_argument("--size", default="10M", help="Dataset size (e.g., 10M, 100M)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Preparing {args.dataset} dataset...")
    
    # TODO: Implement full pipeline
    download_dataset(args.dataset, output_dir)
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main() 