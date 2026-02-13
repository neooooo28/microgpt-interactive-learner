# microgpt.py — A Complete GPT in Pure Python

A single-file, dependency-free implementation of training and inference for a GPT language model. Written by [@karpathy](https://github.com/karpathy), this [script](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) is the entire algorithm — everything else (CUDA, PyTorch, distributed training) is just efficiency.

The model trains on a dataset of names and learns to generate new, plausible-sounding names from scratch.

### :bulb: [**Interactive Visual Guide**](https://htmlpreview.github.io/?https://github.com/neooooo28/microgpt-interactive-learner/blob/main/educational_guide.html)

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset and Tokenizer](#dataset-and-tokenizer)
3. [Autograd Engine (`Value`)](#autograd-engine-value)
4. [Model Parameters](#model-parameters)
5. [Model Architecture (`gpt()`)](#model-architecture-gpt)
6. [Training Loop](#training-loop)
7. [Inference](#inference)
8. [Hyperparameters Summary](#hyperparameters-summary)

---

## Overview

The script proceeds in a straight line through these stages:

1. **Load data** — download a names dataset and shuffle it.
2. **Build a tokenizer** — map characters to integer token IDs.
3. **Define autograd** — a minimal `Value` class that tracks computation graphs and backpropagates gradients.
4. **Initialize parameters** — create all weight matrices for a small GPT.
5. **Define the model** — a function `gpt()` that maps a token + position to logits over what comes next.
6. **Train** — for 1000 steps, pick a document, forward it through the model, compute loss, backpropagate, and update weights with Adam.
7. **Generate** — sample 20 new names from the trained model.

---

## Dataset and Tokenizer

```
Lines 14–27
```

**Dataset loading (lines 14–21):**
- If `input.txt` doesn't exist locally, it is downloaded from the [makemore](https://github.com/karpathy/makemore) repository. This file contains ~32,000 human names, one per line.
- Each non-empty line becomes a document (a name). The list is shuffled with a fixed seed (`random.seed(42)`) for reproducibility.

**Tokenizer (lines 23–27):**
- All unique characters across every name are collected and sorted. Each character becomes a token with an integer ID (its index in the sorted list). For example, `'a'` might be token 0, `'b'` token 1, etc.
- A special **BOS (Beginning of Sequence)** token is added with ID equal to the number of unique characters. BOS serves as both the start-of-sequence and end-of-sequence marker.
- `vocab_size` = number of unique characters + 1 (for BOS). For the names dataset this is 27 (26 lowercase letters + BOS).

---

## Autograd Engine (`Value`)

```
Lines 29–72
```

The `Value` class implements **scalar-level automatic differentiation** (autograd). Every scalar in the computation is wrapped in a `Value` node that remembers:

| Field | Purpose |
|-------|---------|
| `data` | The scalar float value (forward pass result) |
| `grad` | The accumulated gradient of the final loss w.r.t. this node (filled during backward pass) |
| `_children` | Tuple of `Value` nodes that were inputs to the operation that created this node |
| `_local_grads` | Tuple of partial derivatives of this node's output w.r.t. each child |

### Supported operations

| Operation | Forward | Local gradient (d_out/d_child) |
|-----------|---------|------|
| `a + b` | `a.data + b.data` | `(1, 1)` |
| `a * b` | `a.data * b.data` | `(b.data, a.data)` — product rule |
| `a ** n` | `a.data ** n` | `n * a.data^(n-1)` — power rule |
| `a.log()` | `log(a.data)` | `1 / a.data` |
| `a.exp()` | `exp(a.data)` | `exp(a.data)` — exp is its own derivative |
| `a.relu()` | `max(0, a.data)` | `1 if a.data > 0, else 0` |

Subtraction, division, and negation are all derived from addition, multiplication, and power.

### Backward pass (lines 59–72)

`loss.backward()` performs **reverse-mode automatic differentiation**:

1. **Topological sort** — a depth-first traversal builds an ordering of all nodes such that every node appears after its children.
2. **Seed** — the loss node's gradient is set to 1 (d_loss/d_loss = 1).
3. **Propagate** — iterate through nodes in reverse topological order. For each node, multiply its gradient by each local gradient and accumulate into the corresponding child's `.grad`. This is the **chain rule** applied recursively.

---

## Model Parameters

```
Lines 74–90
```

### Hyperparameters

| Name | Value | Meaning |
|------|-------|---------|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 1 | Number of transformer blocks |
| `block_size` | 16 | Maximum sequence length |
| `head_dim` | 4 | Dimension per head (`n_embd // n_head`) |

### Weight matrices

All parameters are initialized as 2D lists of `Value` objects, drawn from a Gaussian distribution with mean 0 and standard deviation 0.08.

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| `wte` | (vocab_size, n_embd) | Token embedding table |
| `wpe` | (block_size, n_embd) | Positional embedding table |
| `lm_head` | (vocab_size, n_embd) | Final projection from hidden state to logits |
| `layer{i}.attn_wq` | (n_embd, n_embd) | Query projection for attention |
| `layer{i}.attn_wk` | (n_embd, n_embd) | Key projection for attention |
| `layer{i}.attn_wv` | (n_embd, n_embd) | Value projection for attention |
| `layer{i}.attn_wo` | (n_embd, n_embd) | Output projection for attention |
| `layer{i}.mlp_fc1` | (4*n_embd, n_embd) | MLP first layer (expand 1x to 4x) |
| `layer{i}.mlp_fc2` | (n_embd, 4*n_embd) | MLP second layer (contract 4x to 1x) |

All parameters are flattened into a single list `params` for the optimizer.

---

## Model Architecture (`gpt()`)

```
Lines 92–144
```

The model follows the **GPT-2** architecture with three simplifications:
- **RMSNorm** instead of LayerNorm (no learned scale/bias, no mean subtraction)
- **No biases** anywhere
- **ReLU** instead of GeLU activation

### Helper functions

- **`linear(x, w)`** — matrix-vector multiply: for each row in `w`, compute the dot product with `x`. Returns a list of `Value`.
- **`softmax(logits)`** — numerically stable softmax: subtract max, exponentiate, normalize. Returns a probability distribution.
- **`rmsnorm(x)`** — Root Mean Square normalization: divide each element by the RMS of the vector (plus epsilon for stability).

### Forward pass (`gpt(token_id, pos_id, keys, values)`)

The function processes **one token at a time** (not a full sequence at once). It uses a KV-cache (`keys` and `values` lists) to store past key/value vectors for attention.

**Step-by-step:**

1. **Embedding (lines 109–112):**
   - Look up the token embedding from `wte[token_id]` (a vector of size `n_embd`).
   - Look up the position embedding from `wpe[pos_id]`.
   - Add them element-wise to get the combined embedding.
   - Apply RMSNorm.

2. **For each transformer layer (lines 114–141):**

   **a) Multi-Head Self-Attention (lines 115–134):**
   - Save a residual copy of `x`.
   - Apply RMSNorm to `x`.
   - Project `x` into query (`q`), key (`k`), and value (`v`) vectors using linear layers.
   - Append `k` and `v` to the KV-cache for this layer (this is how the model "remembers" previous tokens).
   - For each of the 4 attention heads:
     - Slice the head's portion of `q`, all cached `k`s, and all cached `v`s.
     - Compute attention scores: `dot(q_h, k_h[t]) / sqrt(head_dim)` for each cached position `t`.
     - Apply softmax to get attention weights.
     - Compute weighted sum of value vectors.
   - Concatenate all head outputs.
   - Project through the output matrix `attn_wo`.
   - Add the residual connection.

   **b) MLP / Feed-Forward Network (lines 135–141):**
   - Save a residual copy of `x`.
   - Apply RMSNorm.
   - Project through `mlp_fc1` (expand from 16 to 64 dimensions).
   - Apply ReLU activation.
   - Project through `mlp_fc2` (contract from 64 back to 16 dimensions).
   - Add the residual connection.

3. **Output (line 143):**
   - Project the final hidden state through `lm_head` to produce logits over the full vocabulary.
   - Return the logits (unnormalized log-probabilities for each possible next token).

### Why causal masking isn't needed

Traditional GPT implementations use a causal mask to prevent tokens from attending to future positions. Here, because the model processes one token at a time and only stores past keys/values in the KV-cache, future tokens are simply not present — causal masking is implicit.

---

## Training Loop

```
Lines 146–184
```

### Optimizer setup (lines 147–149)

The **Adam** optimizer is used with:
- Learning rate: 0.01 (with linear decay to 0 over training)
- beta1 = 0.85 (momentum/first moment)
- beta2 = 0.99 (second moment / adaptive learning rate)
- epsilon = 1e-8 (numerical stability)
- Two buffers `m` and `v` (first and second moment estimates), one entry per parameter.

### Training step (lines 153–184)

For each of 1000 steps:

1. **Select a document (lines 156–158):**
   - Pick the next document in round-robin order (`docs[step % len(docs)]`).
   - Tokenize it: convert each character to its token ID.
   - Surround with BOS tokens: `[BOS, char1, char2, ..., charN, BOS]`. The trailing BOS acts as the end-of-sequence target.
   - Clamp sequence length to `block_size`.

2. **Forward pass (lines 160–168):**
   - Initialize fresh KV-caches.
   - For each position in the sequence:
     - Feed the current token and position into `gpt()` to get logits.
     - Apply softmax to get probabilities.
     - Compute the **cross-entropy loss** for this position: `-log(prob[target_token])`. This measures how surprised the model is by the correct next token.
   - Average the per-position losses to get the final loss for this document.

3. **Backward pass (line 172):**
   - Call `loss.backward()`, which propagates gradients through the entire computation graph back to every parameter.

4. **Parameter update (lines 174–182):**
   - Compute the decayed learning rate: `lr * (1 - step/num_steps)`, linearly decaying from 0.01 to 0.
   - For each parameter, apply the Adam update:
     - Update first moment: `m = beta1 * m + (1 - beta1) * grad`
     - Update second moment: `v = beta2 * v + (1 - beta2) * grad^2`
     - Bias-correct both moments (dividing by `1 - beta^t`).
     - Update parameter: `p -= lr * m_hat / (sqrt(v_hat) + epsilon)`
   - Zero out the gradient for the next step.

5. **Log** the step number and loss.

---

## Inference

```
Lines 186–200
```

After training, the model generates 20 new names by **autoregressive sampling**:

1. Start with the BOS token and empty KV-caches.
2. At each position (up to `block_size`):
   - Run `gpt()` to get logits for the next token.
   - Divide logits by `temperature` (0.5) — lower temperature makes the distribution sharper/more confident, higher temperature makes it more random.
   - Apply softmax to get a probability distribution.
   - **Sample** a token from this distribution using `random.choices` (weighted random selection).
   - If the sampled token is BOS, the name is complete — stop.
   - Otherwise, append the corresponding character to the output.
3. Print the generated name.

The temperature of 0.5 means the model will produce relatively conservative, high-confidence samples — names that closely resemble the training data rather than wild hallucinations.

---

## Hyperparameters Summary

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 1 | Transformer layers |
| `block_size` | 16 | Max sequence length |
| `learning_rate` | 0.01 | Initial learning rate |
| `beta1` | 0.85 | Adam first moment decay |
| `beta2` | 0.99 | Adam second moment decay |
| `num_steps` | 1000 | Training iterations |
| `temperature` | 0.5 | Sampling temperature |
| Init std | 0.08 | Weight initialization std dev |

---

*This is the complete algorithm. Everything else is just efficiency.*
