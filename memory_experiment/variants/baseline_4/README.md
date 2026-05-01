# Uniform Spacing 4

This variant keeps the baseline-style uniform temporal spacing policy, but compresses from 4 history windows instead of 5.

History path:
- no inference-time preselection
- model-side 4-window evenly spaced sliding coverage
- `4 x 64 -> 64 -> 16`, then append the last frame
