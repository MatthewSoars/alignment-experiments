# Experiment 01 — Decoder-only LM from Scratch

**Goal:**  
Fill in later

**Scope:**  
- Also fill in later

## Implementation Summary
- Fill in later


## Development Notes

### Day 1 — Embeddings and Representations
Key learnings:
- Embedding layers are simple lookup tables indexed by IDs.
- Torch initializes embedding tables automatically with random values.
- Lookup is read-only; learning occurs later during training.
- `d_model` controls the capacity of each token representation (amount of columns of embedding).
- `max_seq_len` controls the amount of tokens the model can look at once, currently visualizing as a chunk which only knows about the singular one - used in the positional embedding
- `vocab_size` controls the amount of distinct tokens id's the model knows about - used in the token embedding

Questions:
- How attention / self attention mixes these representations across tokens.