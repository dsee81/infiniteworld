# Five Chunk Option B

Diversity-first 5-window summary:
- start with `1` seed anchor and `1` recent anchor
- fill the remaining three slots using a diversity-first objective over pooled window descriptors
- candidate quality still contributes as a smaller bonus

This variant uses the 5-window compressor path: `5 x 64 -> 80 -> 20`, then appends the last frame.
