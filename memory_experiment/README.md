# Memory Experiment Suite

This suite compares how Infinite-World behaves under different history-memory constructions while keeping the rest of inference fixed.

Experiment families:
- `baseline_4`: uniform-spacing baseline adapted to a 4-chunk compressor
- `baseline_5`: default Infinite-World memory path
- `recent_tail_4`: last four contiguous 64-frame windows
- `recent_tail_5`: last five contiguous 64-frame windows
- `recent_anchor_salient_transition_4`: two most recent anchors + one salient + one transition slot
- `recent_anchor_salient_transition_5`: two most recent anchors + two salient + one transition slot

Batch outputs are written to:
- `/mnt/shared_storage/dsee/memory_experiment/videos`
- `/mnt/shared_storage/dsee/memory_experiment/logs`
- `/mnt/shared_storage/dsee/memory_experiment/results`

Main entrypoints:
- [run_memory_experiment_batch.py](/root/dataDisk/skebin_temp_storage/infiniteworld/memory_experiment/run_memory_experiment_batch.py)
- [summarize_memory_experiment.py](/root/dataDisk/skebin_temp_storage/infiniteworld/memory_experiment/summarize_memory_experiment.py)
