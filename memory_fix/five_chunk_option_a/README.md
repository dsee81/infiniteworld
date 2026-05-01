# Five Chunk Option A

Rule-based 5-window summary:
- `1` seed anchor from the original conditioning video
- `1` recent anchor from the tail of history
- `2` high-saliency windows
- `1` novelty window chosen relative to the first four

This variant uses the 5-window compressor path: `5 x 64 -> 80 -> 20`, then appends the last frame.
