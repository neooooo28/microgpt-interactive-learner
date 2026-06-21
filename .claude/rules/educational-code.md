# Educational Code Rules

## Readability First

- Code must be readable above all else — this is a teaching tool
- Comments explain the "why" and the math, not just the "what"
- Use whitespace and section headers to create visual structure

## Naming

- Variable names should match mathematical notation: `Q`, `K`, `V` for attention, `W` for weights
- Use descriptive names for non-mathematical concepts: `token_embedding`, `position_embedding`
- Class names describe the concept: `MultiHeadAttention`, `LayerNorm`

## Single-File Constraint

- Keep `microgpt.py` as a single self-contained file
- No external dependencies beyond Python standard library
- No numpy, no torch, no ML frameworks — everything from scratch
- The file tells a story from simple (autograd) to complex (full GPT)

## Progressive Disclosure

- Build from simple to complex — each section builds on the previous
- A reader should be able to understand the code top-to-bottom
- When modifying, preserve this progressive structure
- Don't introduce forward references (using something before it's defined)

## Performance

- This code is intentionally slow — it operates at the scalar level for clarity
- Never optimize at the cost of readability
- If a change makes the code faster but harder to understand, don't make it
