# Plan: Interactive Educational Guide for microgpt.py

An interactive, single-page HTML visualization that lets users explore every component of microgpt.py — from individual `Value` nodes to full training steps — through hands-on interaction.

---

## 1. Goals

- **Teach by doing** — every concept should be explorable, not just readable.
- **Single HTML file** — no build step, no dependencies beyond a browser. All CSS and JS inline.
- **Faithful to the code** — the visualizations should map directly to microgpt.py, not to a generic "transformer explainer."
- **Progressive disclosure** — start simple (tokenizer), build to complex (full training loop). Users can jump to any section.

---

## 2. Target Audience

Beginners to intermediate learners who can read Python but want to build intuition for what a GPT is actually doing at the scalar level.

---

## 3. Page Structure

The page is organized as a vertical scrolling guide with **7 interactive sections**, each corresponding to a stage of the script. A sticky sidebar nav lets users jump between sections.

### Section 1: Dataset & Tokenizer
- **Display:** Show a sample of 10 names from the dataset.
- **Interactive element:** A text input where the user types a name. As they type, a live panel shows the character-to-token-ID mapping and the full token sequence `[BOS, ...ids, BOS]`.
- **Highlight:** The BOS token is visually distinct (colored border) to emphasize its dual start/end role.

### Section 2: The Value Node & Autograd
- **Display:** A small computation graph (e.g., `a * b + c`) rendered as an SVG directed acyclic graph (DAG).
- **Interactive element:**
  - Sliders for `a`, `b`, `c` values. As the user adjusts them, the forward values update in real-time on each node.
  - A "Backward" button that animates gradient flow: highlights propagate from the output node back through the graph, filling in `.grad` values on each node one at a time.
- **Preset examples:** Buttons to load specific sub-graphs: "Addition", "Multiplication", "Log + Exp (softmax fragment)", "ReLU".

### Section 3: Model Parameters
- **Display:** A schematic diagram of all weight matrices as colored rectangles, sized proportionally (e.g., `wte` is 27x16, `mlp_fc1` is 64x16).
- **Interactive element:** Hover over any matrix to see its name, shape, purpose, and a heatmap of its current values (initialized randomly on page load). Click to zoom into the matrix.
- **Counter:** A running total of parameters at the bottom, broken down by component.

### Section 4: The GPT Architecture
- **Display:** A vertical block diagram of the `gpt()` function:
  - Token Embed + Position Embed -> Add -> RMSNorm
  - Transformer Block: Attention (with Q/K/V projections, multi-head split, softmax, weighted sum) + MLP (fc1 -> ReLU -> fc2)
  - Residual connections shown as bypass arrows
  - Final `lm_head` projection -> logits
- **Interactive element:**
  - A "Step Through" mode: the user clicks "Next" to advance through the forward pass one operation at a time. At each step, the active operation highlights in the diagram and a side panel shows the actual vector values (small enough to display at n_embd=16).
  - Dropdowns to select which token/position is being processed.

### Section 5: Attention Deep Dive
- **Display:** A dedicated visualization of multi-head attention for a short sequence (e.g., "cat").
  - An attention weight heatmap (queries on one axis, keys on the other) for each of the 4 heads, displayed as small matrices.
  - Arrows from each query to keys, with thickness proportional to attention weight.
- **Interactive element:**
  - Click on any cell in the attention matrix to see the dot-product computation broken down.
  - Toggle between heads to compare what different heads "look at."
  - A slider to artificially set attention to uniform vs. peaked, showing its effect on the output.

### Section 6: Training Loop
- **Display:** A dashboard with:
  - The current training document (highlighted characters showing the input/target pairs).
  - A loss curve chart that updates in real-time as training steps are simulated.
  - The Adam optimizer state: a mini bar chart showing `m` and `v` for a few selected parameters.
- **Interactive element:**
  - A "Train 1 Step" button that runs a single forward + backward + update step (using a JS re-implementation of the core loop, operating on small toy data for speed).
  - A "Train 10 Steps" / "Train 100 Steps" button for faster progression.
  - A speed slider controlling animation pace.
  - Pause at any step to inspect: hover over any parameter to see its value, gradient, and moment estimates.

### Section 7: Inference / Generation
- **Display:** A live generation panel.
  - Shows the BOS token, then each sampled token appearing one at a time with a typing animation.
  - Below each token, a mini bar chart shows the probability distribution the model used to sample it.
- **Interactive element:**
  - A temperature slider (0.1 to 2.0) — regenerate with different temperatures to see the effect.
  - A "Sample Again" button to generate a new name.
  - Click on any token in the generated sequence to see the full probability distribution and which token was sampled.

---

## 4. Technical Implementation Plan

### 4.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Layout/styling | Vanilla CSS + CSS Grid | No framework needed; keep the file self-contained |
| Interactivity | Vanilla JavaScript (ES6+) | No React/Vue overhead; direct DOM manipulation |
| Graphs/diagrams | SVG (hand-built + dynamic) | Lightweight, scalable, animatable |
| Charts (loss curve, bar charts) | Canvas 2D API | Fast rendering for real-time updates |
| Math rendering | Inline HTML/CSS (no MathJax) | Keep it dependency-free; only a few formulas needed |

### 4.2 File Structure

Everything lives in a single file: `educational_guide.html`

```
educational_guide.html
├── <style> ... </style>           (~300 lines of CSS)
├── <body>
│   ├── Sidebar navigation
│   ├── Section 1: Tokenizer
│   ├── Section 2: Autograd
│   ├── Section 3: Parameters
│   ├── Section 4: Architecture
│   ├── Section 5: Attention
│   ├── Section 6: Training
│   └── Section 7: Inference
└── <script> ... </script>         (~800-1200 lines of JS)
    ├── Value class (port from Python)
    ├── Model definition (port from Python)
    ├── Tokenizer logic
    ├── Visualization renderers
    ├── Interaction handlers
    └── Animation engine
```

### 4.3 Implementation Phases

**Phase 1 — Scaffolding & Static Layout**
- Set up the HTML structure with all 7 sections.
- Implement sidebar navigation with scroll-spy (highlight the active section).
- Style the page: dark theme (to match code editor feel), monospace fonts for code, clean typography for prose.
- Add static explanatory text and placeholder diagrams for each section.

**Phase 2 — Core Engine (JS Port)**
- Port the `Value` class to JavaScript, including the `backward()` method.
- Port the `linear`, `softmax`, `rmsnorm`, and `gpt` functions.
- Port the tokenizer (hardcode the 26-letter vocabulary or load it from a small inline constant).
- Port the Adam optimizer update step.
- Initialize a small model with random weights (same hyperparameters: n_embd=16, n_head=4, etc.).
- Verify correctness by running a few forward passes and checking outputs are reasonable.

**Phase 3 — Section Interactivity (one section at a time)**

| Section | Key interactive work |
|---------|---------------------|
| 1. Tokenizer | Text input with live token display; character-to-ID mapping table |
| 2. Autograd | SVG graph renderer; slider inputs; backward animation with timed highlights |
| 3. Parameters | Matrix heatmap renderer (Canvas); hover tooltips; zoom modal |
| 4. Architecture | Block diagram SVG; step-through state machine; value inspector panel |
| 5. Attention | Attention heatmap renderer; head selector tabs; cell click detail view |
| 6. Training | Loss chart (Canvas); step executor; parameter inspector |
| 7. Inference | Token-by-token generation with animation; temperature slider; probability bar charts |

**Phase 4 — Polish & Accessibility**
- Add keyboard navigation (arrow keys to step through architecture, Enter to train).
- Responsive layout for tablet/desktop (mobile is secondary).
- Add a brief intro/hero section at the top explaining what the page is.
- Add tooltips and "?" icons for jargon (e.g., "RMSNorm", "residual connection").
- Performance optimization: debounce slider inputs, use requestAnimationFrame for animations.
- Test across Chrome, Firefox, Safari.

---

## 5. Design Principles

- **Color-coded data flow:** Use consistent colors throughout — e.g., blue for embeddings, green for attention, orange for MLP, red for loss/gradients.
- **Numbers are visible:** Since n_embd=16 is small enough, always show actual numeric values rather than abstract shapes.
- **Code alongside visuals:** Each section includes a collapsible "Show Python Code" panel that highlights the exact lines from microgpt.py being visualized.
- **No jargon without explanation:** Every technical term is either defined inline or has a hover tooltip.
