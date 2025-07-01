# Forward vs. Reverse Language Modeling with GPT-2

A research project comparing the performance and characteristics of forward (left-to-right) versus reverse (right-to-left) language modeling using GPT-2-small architecture.

## 🎯 Objective

This project investigates the differences in:
- **Learning dynamics** between forward and reverse text generation
- **Perplexity and loss curves** during training
- **Generated text quality** and characteristics  
- **Tokenization impact** on bidirectional language understanding

## 📁 Project Structure

```
├── data/           # Training datasets and preprocessed text
├── tokenizers/     # Custom tokenizers for forward/reverse text
├── models/         # Trained model checkpoints and configurations
├── scripts/        # Training and evaluation scripts
├── analysis/       # Jupyter notebooks for comparison analysis
├── logs/           # Training logs and experiment tracking
└── requirements.txt # Project dependencies
```

## 🛠 Setup

1. **Clone and setup environment:**
```bash
git clone <repo-url>
cd reverse-language-modeling
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Prepare dataset:**
```bash
python scripts/prepare_dataset.py
```

3. **Train tokenizers:**
```bash
python scripts/train_tokenizer.py --direction forward
python scripts/train_tokenizer.py --direction reverse
```

4. **Train models:**
```bash
python scripts/train_gpt2.py --config forward
python scripts/train_gpt2.py --config reverse
```

## 📊 Analysis

Comparative analysis notebooks are available in `analysis/`:
- `compare.ipynb` - Side-by-side model comparison
- Loss curves, perplexity analysis, and generation samples

## 🧪 Experiments

- **Dataset**: Common Crawl subset / OpenWebText
- **Architecture**: GPT-2 small (124M parameters)
- **Training**: From scratch with identical hyperparameters
- **Evaluation**: Perplexity, BLEU scores, human evaluation

## 📈 Tracking

Experiments are tracked using:
- **Weights & Biases** for metrics and visualizations
- **TensorBoard** for loss curves and training dynamics
- **Local logs** in `logs/` directory

## 🔍 Key Research Questions

1. Does reverse language modeling learn different linguistic patterns?
2. How do training dynamics differ between directions?
3. What impact does tokenization direction have on model performance?
4. Can reverse models capture different aspects of language structure?

## 📝 Development Log

See `logs/dev-log.md` for detailed progress and findings.

## 🤝 Contributing

This is a research project. Feel free to open issues for discussion or suggestions.

## 📄 License

MIT License - see LICENSE file for details.
