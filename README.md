# Reverse-Token Autoregressive Modeling

Can flipping the direction of text sequences help models learn *effect-to-cause* reasoning?  
This project investigates whether reversing tokens leads to different convergence behavior and model representations in GPT-style autoregressive training.

Inspired by:
- [Arrow of Time in LMs](https://arxiv.org/abs/2304.00643)
- [Reverse Modeling](https://arxiv.org/abs/2309.10664)

## Goals 
- Build character-reversed dataset
- Train fresh tokenizer
- Train GPT2-small on forward + reversed
- Compare perplexity and convergence behavior
- Writeup results + share on GitHub
