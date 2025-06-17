# French-english-NMT

A PyTorch-based implementation of a Transformer encoder-decoder model for translating French to English, built from scratch.

## What I Built

- A custom Transformer model (encoder + decoder) using Multi-Head Attention, LayerNorm, and Feed-Forward layers.
- Implemented greedy and beam search decoding from scratch.
- Used the Hansards French-English corpus for training.
- Evaluated translation quality using BLEU scores.

## Training

Trained using teacher forcing and cross-entropy loss. Beam search decoding improved final BLEU scores over greedy decoding.

## Dataset

Canadian Hansards parallel corpus of French-English sentence pairs.

