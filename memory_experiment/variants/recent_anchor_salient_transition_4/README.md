# Four Chunk Option A

Rule-based 4-window summary:
- `1` seed anchor from the original conditioning video
- `1` recent anchor from the tail of history
- `1` highest-saliency window
- `1` novelty window chosen relative to the first three

This variant also uses a copied `variant_dit_model.py` that compresses `4 x 64 -> 64 -> 16`, then appends the last frame.
