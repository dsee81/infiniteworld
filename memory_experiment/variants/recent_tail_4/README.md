# Four Chunk Contiguous Tail

This diagnostic variant uses the simplest 4-chunk memory summary:

- always take the last four non-overlapping 64-frame windows
- treat them as one contiguous 256-frame tail
- front-pad only when the history is shorter than 256 frames

Why this variant exists:
- it tests whether the rewind artifacts come from stitching disjoint or overlapping windows into a fake timeline
- if this variant reduces rewind blips, the main problem is likely temporal inconsistency in the learned memory summary rather than the 4-chunk budget itself
