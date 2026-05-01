# Last 5 Chunks Variant

This experiment keeps the main Infinite-World repo untouched.

What changes:
- uses a copied `infworld_inference.py`
- uses a copied `infworld_config.yaml`
- swaps in `variant_dit_model.py` as the model target
- replaces the long-history chunk selector with a simple policy that always takes the most recent five 64-frame history windows

Behavior:
- if history is longer than 320 latent frames, it uses the final 320 frames
- if history is shorter than 320 latent frames, it left-pads to 320 and still splits into five 64-frame windows
- the rest of the compression path stays the same: `5 x 64 -> 5 x 16 -> 80 -> 20`, then append the last frame

Run from this folder, for example:

```bash
cd /root/dataDisk/skebin_temp_storage/infiniteworld/memory_fix/last5_chunks
python infworld_inference.py
```
