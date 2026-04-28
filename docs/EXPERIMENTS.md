# Running Your Own Infinite-World Experiments (Self-Serve)

This repo’s main inference entrypoint is:

- `scripts/infworld_inference.py`

It reads a prompt list YAML (`INFWORLD_PROMPTS_YAML`) where each task is:

```
[prompt, cond_path, action_json_path]               # optional output_name omitted
[prompt, cond_path, action_json_path, output_name]  # recommended
```

`cond_path` can be an image (`.png/.jpg/...`) or a video (`.mp4/...`).  
The `action_json_path` must be a JSON list of per-frame actions:

```json
[
  {"move": "go forward", "view": "turn left"},
  {"move": "go forward", "view": "turn left"}
]
```

## 1) The Controls You Can Vary (No Code Changes)

All of these are environment variables consumed by `scripts/infworld_inference.py`:

- `INFWORLD_PROMPTS_YAML`: YAML file containing `prompts: ...` list
- `INFWORLD_OUTPUT_BASE`: output folder (default `./outputs`)
- `INFWORLD_SEED`: base seed (default `42`)
- `INFWORLD_SEED_PER_TASK`: `1` to reseed per task based on `task_idx` (default `0`)
- `INFWORLD_SEED_TASK_STRIDE`: multiplier used when reseeding per task (default `1000`)
- `INFWORLD_NUM_CHUNKS`: number of generation chunks (default `13`)
- `INFWORLD_NUM_SAMPLING_STEPS`: sampling steps (default `30`)
- `INFWORLD_TEXT_CFG_SCALE`: CFG guidance (default `5.0`)
- `INFWORLD_SHIFT`: scheduler shift (default `7`)
- `INFWORLD_MAX_TASKS`: cap how many prompt entries to run (`0` means all)
- `INFWORLD_HIGH_QUALITY_SAVE`: `1` (default) or `0`
- `INFWORLD_CONTEXT_PARALLEL_SIZE`: for multi-GPU context parallel (default `1`)

Condition (input image/video context):

- `INFWORLD_COND_CLIP_LEN`:
  - `1` (default): condition on one frame
  - `16` / `32` / `81` / ...: condition on that many frames from the video
  - `all`: condition on the entire video
- `INFWORLD_COND_CLIP_MODE`: `first` (default) or `last` (only meaningful when `COND_CLIP_LEN` is an int)
- `INFWORLD_COND_WINDOW_FRAMES`: if > 0, only the last N frames of history are fed as conditioning each chunk (helps avoid OOM)

Note: if `cond_path` is an image, inference always uses 1 condition frame regardless of `INFWORLD_COND_CLIP_LEN`.

Per-task overrides: the generated `prompts.yaml` may optionally specify `cond_clip_len` and `cond_clip_mode` per task
(supported by `scripts/infworld_inference.py`). If present, they override the env vars for that task.

## 2) Quick Start (One-Off Run)

1. Create a prompt YAML, for example `prompts/my_run.yaml`:

```yaml
prompts:
  - - "A realistic outdoor scene continuing from the provided context."
    - "./assets/example_case/0001.jpg"
    - "./assets/example_case/0001.json"
    - "my_output_name"
```

2. Run inference:

```bash
export INFWORLD_PROMPTS_YAML=prompts/my_run.yaml
export INFWORLD_NUM_CHUNKS=13
export INFWORLD_NUM_SAMPLING_STEPS=30
export INFWORLD_TEXT_CFG_SCALE=5.0
export INFWORLD_SEED=42
bash infer_local.sh 1
```

## 3) Action JSON: What The Model Expects

Valid `move` tokens:

- `no-op`
- `go forward`
- `go back`
- `go left`
- `go right`
- `go forward and go left`
- `go forward and go right`
- `go back and go left`
- `go back and go right`
- `uncertain`

Valid `view` tokens:

- `no-op`
- `turn up`
- `turn down`
- `turn left`
- `turn right`
- `turn up and turn left`
- `turn up and turn right`
- `turn down and turn left`
- `turn down and turn right`
- `uncertain`

### Chunk Overlap (Important)

The inference loop consumes actions in windows of `chunk_frames=81` with a **1-frame overlap** between chunks.

If you generate `N` chunks and condition on `C` frames, the action length that will be used is:

```
total_frames = C + 80 * N
```

Default `C=1`, `N=13` => `1041` action frames.

If your action JSON is shorter than needed, inference pads missing actions as `no-op`.  
If it is longer, the extra tail is ignored.

## 4) Generate Actions By Chunk (Recommended)

Use:

- `scripts/infworld_actiongen.py`

Example spec:

- `experiments/templates/action_spec_example.yaml`

Generate the JSON:

```bash
python scripts/infworld_actiongen.py \
  --spec experiments/templates/action_spec_example.yaml \
  --out outputs/actions/my_actions.json
```

Then reference `outputs/actions/my_actions.json` in your prompt YAML.

## 5) Run Many Experiments From One Suite YAML (Recommended)

Use:

- `scripts/infworld_suite.py`

Edit:

- `experiments/templates/suite_example.yaml`

Create a run directory:

```bash
python scripts/infworld_suite.py --suite experiments/templates/suite_example.yaml
```

This creates:

- `experiments/runs/<run_id>/prompts.yaml`
- `experiments/runs/<run_id>/actions/*.json`
- `experiments/runs/<run_id>/run.sh`

Run it:

```bash
bash experiments/runs/<run_id>/run.sh 1
```

## 6) PBS Job Submission (Recommended)

Templates:

- `experiments/templates/pbs_suite_1gpu.pbs`
- `experiments/templates/pbs_suite_4gpu.pbs`

Typical flow:

1. Edit your suite YAML (for example, start from `experiments/templates/suite_example.yaml`).
2. In the PBS file, set `SUITE_YAML=...` to point at your suite.
3. Submit:

```bash
qsub experiments/templates/pbs_suite_1gpu.pbs
```

What the PBS script does:

- runs `python scripts/infworld_suite.py ...` to generate a per-job run directory
- runs `bash <run_dir>/run.sh <num_gpus>` to launch inference

## 6) What To Vary In Your Sweeps

Common, high-impact sweeps:

- Text prompt phrasing (camera motion cues, environment continuity cues)
- Action schedule shapes (constant, phase changes, loops)
- `INFWORLD_NUM_CHUNKS` (video length)
- `INFWORLD_NUM_SAMPLING_STEPS` (quality vs speed)
- `INFWORLD_TEXT_CFG_SCALE` (prompt adherence vs artifacts)
- `INFWORLD_COND_CLIP_LEN` (how much input-video context to provide)
- `INFWORLD_SEED` (stochasticity / reproducibility)
