# Four Chunk Action + Visual Hybrid

This is the 4-window counterpart to `action_visual_hybrid`.

Policy:
- reserve `1` slot for seed-video memory, scored visually
- reserve `1` coverage slot near the temporal midpoint
- select `1` high-saliency slot using visual motion plus action-based signals when available
- select `1` novelty slot to diversify the final chosen windows

It uses the 4-window compressor path: `4 x 64 -> 64 -> 16`, then appends the last frame.
