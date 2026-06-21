# Explain a Concept

Explain a concept from the codebase at multiple levels of abstraction.

`$ARGUMENTS` should be the concept (e.g., "attention", "backprop", "softmax", "autograd", "layer norm", "embedding").

## Steps

1. Find the relevant code in `microgpt.py`
2. Explain at three levels:

### Level 1: Intuitive
- What does this do in plain English?
- What real-world analogy helps explain it?

### Level 2: Mathematical
- What is the mathematical formula?
- What are the inputs and outputs?
- Why does this formula work?

### Level 3: Code Mapping
- Which lines in `microgpt.py` implement this?
- How does the code map to the math?
- What design decisions were made and why?

If the concept spans multiple parts of the file, trace the full path.
