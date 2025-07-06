#!/usr/bin/env python
"""
Train a Byte-Pair-Encoding (BPE) tokenizer on (a slice of) OpenWebText.
Usage examples
--------------
# Forward tokenizer (no reversal)
python scripts/train_tokenizer.py \
    --out tokenizers/tokenizer_forward.json \
    --vocab_size 32000 \
    --sample_gb 10

# Reversed tokenizer (character-level flip)
python scripts/train_tokenizer.py \
    --out tokenizers/tokenizer_reverse.json \
    --vocab_size 32000 \
    --sample_gb 10 \
    --reverse
"""
import argparse, itertools, unicodedata, random, os, json, math, tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers

def iter_owt(sample_gb: int, reverse: bool):
    """
    Stream roughly `sample_gb` of OpenWebText.
    Uses HF streaming to avoid downloading the full 40 GB.
    """
    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
    bytes_seen = 0
    gb_limit = sample_gb * (1024**3)
    rng = random.Random(42)

    for ex in ds:
        text = ex["text"]
        if reverse:
            text = text[::-1]          # simple character flip
        # Normalize to NFKC for consistency
        text = unicodedata.normalize("NFKC", text)
        bytes_seen += len(text.encode("utf-8"))
        yield text
        if bytes_seen >= gb_limit:
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to save tokenizer JSON")
    ap.add_argument("--vocab_size", type=int, default=32_000)
    ap.add_argument("--sample_gb", type=int, default=10,
                    help="How many gigabytes of OWT to stream")
    ap.add_argument("--reverse", action="store_true",
                    help="Flip each document's characters")
    args = ap.parse_args()

    # --- Build tokenizer skeleton ---
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    )

    print(f"[INFO] Streaming ≈{args.sample_gb} GB of OpenWebText "
          f"({'reversed' if args.reverse else 'forward'})…")
    tokenizer.train_from_iterator(
        iter_owt(args.sample_gb, args.reverse),
        trainer=trainer,
        length=args.vocab_size * 200  # rough iterator length hint
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tokenizer.save(args.out)
    print(f"[DONE] Saved tokenizer to {args.out}")

if __name__ == "__main__":
    main() 