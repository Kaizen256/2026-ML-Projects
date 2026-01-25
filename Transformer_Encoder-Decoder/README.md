# Transformer Encoder-Decoder for English-to-Japanese Translation

This project implements a Transformer based sequence-to-sequence model from scratch to translate English sentences into Japanese. The entire architecture, including multi-head attention, positional encoding, masking, and training with a learning rate scheduler is manually implemented using PyTorch. This project is meant to give me a deeper understanding of Transformers so I can tackle Swin Transformer.


## Architecture

- Follows the original Transformer encoder–decoder design.
- English input is word-level, Japanese output is character-level.
- Uses sinusoidal positional encoding.
- Includes custom masking, embedding, and projection layers.
- Output is generated with beam search.

## Preprocessing

- English cleaned to lowercase alphanumerics.
- Japanese cleaned to keep Hiragana, Katakana, Kanji, and punctuation.
- Words with freq > 2 are kept.
- Vocab built with special tokens.

## Model

- 5 encoder and 5 decoder layers.
- 180-dim embeddings.
- 6 heads.
- Feedforward size 720.

## Training

- Optimizer: Adam(lr=1e-4, betas=(0.9, 0.98), eps=1e-9).
- Learning rate scheduler with warmup steps.
- Cross entropy loss ignoring padding.
- Early stopping after loss degrades.

## Translation

- Beam search used instead of greedy decoding.
- Outputs character-level Japanese.
- Produces good translations after enough epochs.

## Example

Input: "This worked pretty well"
Output: "これ、かなりうまく動くよ。"
Translated: "This works pretty well."