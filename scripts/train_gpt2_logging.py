#!/usr/bin/env python3
"""
train_gpt2_logging.py
=====================
A drop‑in replacement for *trial_train_gpt2.py* that **streams rich metrics to Weights &
Biases (wandb)**:

*   Training loss & perplexity
*   Gradient norm (from Trainer logs)
*   Token‑level entropy (confidence)
*   Attention entropy (how concentrated heads are)
*   Periodic generated samples

Run ✨
-----
>>> python train_gpt2_logging.py \
        --txt data/train_reversed.txt \
        --tokenizer tokenizers/gpt2-reverse-hf \
        --bin data/train_reversed.bin \
        --eval_txt data/eval_reversed.txt \
        --steps 5000 \
        --block_size 128 \
        --run_name char_flip_trial

The script will push everything to **wandb** under project *reverse-gpt2*.
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
    """Tokenise a .txt file and dump raw uint16 IDs."""
    ids: List[int] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                ids.extend(tokenizer.encode(line))
    np.array(ids, dtype=np.uint16).tofile(bin_path)
    print(f"[✓] wrote {len(ids):,} tokens → {bin_path}")


class PackedDataset(torch.utils.data.Dataset):
    """Flat‑packed dataset à la Karpathy: rolling windows from mem‑mapped array."""

    def __init__(self, bin_path: Path, block_size: int = 128):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block = block_size

    def __len__(self):
        return len(self.data) - self.block - 1

    def __getitem__(self, idx):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.block + 1].astype(np.int64)
        )
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


# ────────────────────────────────────────────────────────────────────────────────
# Callback for extra metrics
# ────────────────────────────────────────────────────────────────────────────────

class ExtraMetricsCallback(TrainerCallback):
    """Compute token‑entropy, attention‑entropy & sample generation every eval step."""

    def __init__(self, eval_texts: List[str], tokenizer, sample_prompt: str, block_size: int, log_every: int = 500):
        self.eval_texts = eval_texts
        self.tokenizer = tokenizer
        self.sample_prompt = sample_prompt
        self.block_size = block_size
        self.log_every = log_every

    # utility ------------------------------------------------------------------
    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        ent = -(probs * logp).sum(dim=-1)  # (batch, seq)
        return ent.mean().item()

    @staticmethod
    def _attention_entropy(attn: torch.Tensor):
        # attn: (batch, heads, seq, seq)
        probs = attn.mean(dim=1)  # average over heads -> (batch, seq, seq)
        logp = torch.log(probs + 1e-12)
        ent = -(probs * logp).sum(dim=-1)  # (batch, seq)
        return ent.mean().item()

    # hook ---------------------------------------------------------------------
    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step == 0 or step % self.log_every != 0:
            return
        if model is None:
            return
        model.eval()
        with torch.no_grad():
            # Token‑entropy & attention‑entropy over a few eval texts
            token_ent, attn_ent = [], []
            for txt in self.eval_texts:
                inp = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=self.block_size).to(model.device)
                out = model(**inp, output_attentions=True)
                token_ent.append(self._entropy_from_logits(out.logits))
                attn_ent.append(self._attention_entropy(out.attentions[-1]))  # last layer
            wandb.log({
                "token_entropy": float(np.mean(token_ent)),
                "attention_entropy": float(np.mean(attn_ent)),
            }, step=step)

            # quick sample generation -----------------------------------------
            sample_ids = model.generate(
                self.tokenizer(self.sample_prompt, return_tensors="pt").input_ids.to(model.device),
                max_new_tokens=40,
                do_sample=True,
            )[0]
            sample_text = self.tokenizer.decode(sample_ids)
            wandb.log({"sample": wandb.Html(f"<pre>{sample_text}</pre>")}, step=step)
        model.train()


# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--eval_txt", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--run_name", default="char_flip_run")
    ap.add_argument("--project", default="reverse-gpt2")
    args = ap.parse_args(argv)

    # tokenizer ----------------------------------------------------------------
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)

    # data prep ----------------------------------------------------------------
    if not args.bin.exists():
        args.bin.parent.mkdir(parents=True, exist_ok=True)
        txt_to_bin(args.txt, args.bin, tok)

    train_ds = PackedDataset(args.bin, args.block_size)

    # tiny GPT‑2 ---------------------------------------------------------------
    cfg = GPT2Config(
        vocab_size=len(tok),
        n_positions=args.block_size,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=3,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)

    # wandb --------------------------------------------------------------------
    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    # training args ------------------------------------------------------------
    tr_args = TrainingArguments(
        output_dir="ckpts/" + args.run_name,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        max_steps=args.steps,
        logging_steps=20,
        report_to=["wandb"],
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # eval texts ---------------------------------------------------------------
    eval_texts = [l.strip() for l in Path(args.eval_txt).read_text().splitlines() if l.strip()][:8]
    cb = ExtraMetricsCallback(eval_texts, tok, sample_prompt="ehT", block_size=args.block_size, log_every=100)

    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[cb],
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main() 