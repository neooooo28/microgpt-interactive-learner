# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

A from-scratch implementation of GPT in a single Python file (`microgpt.py`). No external ML dependencies — everything (autograd, attention, training, inference) is built from scalar operations. Educational purpose: teach transformer architecture at the most fundamental level.

## Architecture

`microgpt.py` follows a progressive disclosure structure:

1. **Value** — Autograd engine (scalar-level backpropagation)
2. **Tokenizer** — Character-level tokenization
3. **Model Components** — Embedding, multi-head attention, MLP, layer norm
4. **GPT Model** — Full transformer decoder
5. **Training Loop** — Forward pass, loss computation, backprop, parameter updates
6. **Inference** — Text generation with temperature sampling

## Commands

```bash
python microgpt.py      # Train on embedded text + generate output
python3 microgpt.py     # Same (use whichever Python is available)
```

## Companion Files

- `educational_guide.html` — Interactive single-page visualization (~70KB)
- `plan_educational_guide.md` — Planning document for the guide
- `README.md` — Project documentation

## Key Constraints

- **Single file**: Keep `microgpt.py` self-contained — no imports beyond standard library
- **No ML dependencies**: No numpy, torch, tensorflow, etc. — everything from scratch
- **Educational priority**: Readability and clarity always win over performance
- **Variable naming**: Match mathematical notation where possible (Q, K, V, not query_matrix)
