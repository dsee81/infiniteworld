# Action + Visual Hybrid Variant

This experiment keeps the main Infinite-World repo untouched and changes only the copied inference entrypoint in this folder.

Policy:
- reserve `1` slot for seed-video memory, scored visually
- reserve `1` coverage slot near the temporal midpoint
- select `2` high-saliency slots using visual motion plus action-based signals when available
- select `1` novelty slot to diversify the final 5 chosen windows

Important detail:
- windows from the original conditioning video do not have historical action labels
- they are scored with visual cues only and are not penalized for missing actions

The selected five 64-frame windows are concatenated into a 320-frame history clip before the normal model compression path runs.

Run from this folder, for example:

```bash
cd /root/dataDisk/skebin_temp_storage/infiniteworld/memory_fix/action_visual_hybrid
python3 infworld_inference.py
```
