#!/usr/bin/env python3
"""
prepare_dataset.py – Stream and save OpenWebText forward and reversed datasets.
"""

import argparse
import unicodedata
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def stream_openwebtext(sample_gb: float, reverse: bool = False):
    ds = load_dataset("openwebtext", split="train", streaming=True)
    bytes_seen = 0
    gb_limit = sample_gb * (1024 ** 3)

    for ex in ds:
        text = ex["text"]
        text = unicodedata.normalize("NFKC", text)
        if reverse:
            text = text[::-1]
        bytes_seen += len(text.encode("utf-8"))
        yield text.strip().replace("\n", " ")
        if bytes_seen >= gb_limit:
            break

def save_text(filepath: Path, stream):
    with filepath.open("w", encoding="utf-8") as f:
        for line in tqdm(stream, desc=f"[writing {filepath.name}]"):
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data", help="Output folder for .txt files")
    parser.add_argument("--sample-gb", type=float, default=0.01, help="Total size to stream (GB)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_text(out_dir / "train_forward.txt", stream_openwebtext(args.sample_gb, reverse=False))
    save_text(out_dir / "train_reversed.txt", stream_openwebtext(args.sample_gb, reverse=True))

    print("[done] Saved forward and reversed corpora.")

if __name__ == "__main__":
    main() 