# Four Chunk Option B

Diversity-first 4-window summary:
- start with `1` seed anchor and `1` recent anchor
- fill the remaining slots using a diversity-first objective over pooled window descriptors
- candidate quality still contributes as a smaller bonus

This variant also uses a copied `variant_dit_model.py` that compresses `4 x 64 -> 64 -> 16`, then appends the last frame.
