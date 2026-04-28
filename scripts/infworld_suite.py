#!/usr/bin/env python3
"""
Infinite World - Experiment Suite Builder
=========================================

Goal: run your own experiments without editing code.

This script reads a single suite YAML, materializes a run directory:
  - prompts.yaml (format expected by scripts/infworld_inference.py)
  - actions/*.json (generated per task)
  - run.sh (exports env vars + launches inference)

Typical usage:
  python scripts/infworld_suite.py --suite experiments/templates/suite_example.yaml
  bash experiments/runs/<run_id>/run.sh 1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

# When executed as `python scripts/infworld_suite.py`, Python puts the `scripts/`
# directory on sys.path, so importing the sibling module works without packaging.
from infworld_actiongen import generate_actions_from_spec

_VID_EXTS = {".mp4", ".avi", ".webm", ".mov", ".mkv", ".m4v"}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top-level: {path}")
    return data


def _sanitize_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._-")
    return s or "run"


def _write_yaml(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, width=120)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _relpath_from_repo_root(repo_root: str, path: str) -> str:
    # Inference is typically launched from repo root, so keep prompt paths relative.
    return os.path.relpath(os.path.abspath(path), start=os.path.abspath(repo_root))


def _make_run_id(suite_name: str) -> str:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_name(suite_name)}_{ts}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained Infinite-World experiment run directory.")
    ap.add_argument("--suite", required=True, help="Suite YAML file.")
    ap.add_argument(
        "--runs-dir",
        default="experiments/runs",
        help="Base directory for created runs (relative to repo root).",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run id (folder name). If omitted uses <suite_name>_<timestamp>.",
    )
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    suite_path = os.path.abspath(args.suite)
    suite = _read_yaml(suite_path)

    suite_name = str(suite.get("suite_name", os.path.splitext(os.path.basename(suite_path))[0]))
    run_id = args.run_id or _make_run_id(suite_name)

    runs_dir = args.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(repo_root, runs_dir)
    run_dir = os.path.join(runs_dir, run_id)
    actions_dir = os.path.join(run_dir, "actions")
    os.makedirs(actions_dir, exist_ok=True)

    defaults = suite.get("defaults", {}) or {}
    env_defaults = defaults.get("env", {}) or {}
    if not isinstance(env_defaults, dict):
        raise ValueError("defaults.env must be a mapping")

    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("suite.tasks must be a non-empty list")

    prompts: List[List[str]] = []
    prompts_mixed: List[Any] = []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            raise ValueError(f"tasks[{i}] must be a mapping")

        prompt = t.get("prompt")
        cond_path = t.get("cond_path")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"tasks[{i}].prompt must be a non-empty string")
        if not isinstance(cond_path, str) or not cond_path.strip():
            raise ValueError(f"tasks[{i}].cond_path must be a non-empty string")

        output_name = t.get("output_name")
        if output_name is None:
            output_name = f"{i:04d}_{_sanitize_name(prompt[:40])}"
        output_name = _sanitize_name(str(output_name))

        task_cond_clip_len = t.get("cond_clip_len")
        task_cond_clip_mode = t.get("cond_clip_mode")

        action_spec = t.get("actions")
        if not isinstance(action_spec, dict):
            raise ValueError(
                f"tasks[{i}].actions must be a mapping (actiongen spec). "
                "See experiments/templates/suite_example.yaml."
            )

        # Allow per-task override of num_chunks based on suite env, but keep spec explicit if provided.
        # If the user relies on INFWORLD_NUM_CHUNKS, they can set it in defaults.env and omit it here.
        # We still require actiongen to know num_chunks, so fill from env if missing.
        action_spec = json.loads(json.dumps(action_spec))  # deep copy (JSON-safe)
        action_infer = action_spec.get("inference", {}) or {}
        if "num_chunks" not in action_infer:
            if "INFWORLD_NUM_CHUNKS" in env_defaults:
                action_infer["num_chunks"] = int(env_defaults["INFWORLD_NUM_CHUNKS"])
            else:
                action_infer["num_chunks"] = 13
        action_spec["inference"] = action_infer

        # Wire cond path/clip_len into actiongen if user didn't specify.
        cond_block = action_spec.get("cond", {}) or {}
        cond_block.setdefault("path", cond_path)
        ext = os.path.splitext(str(cond_path))[1].lower()
        if ext in _IMG_EXTS:
            # Images are always single-frame conditions in inference.
            cond_block.setdefault("clip_len", 1)
        elif ext in _VID_EXTS:
            if task_cond_clip_len is not None:
                cond_block.setdefault("clip_len", task_cond_clip_len)
            else:
                cond_block.setdefault("clip_len", env_defaults.get("INFWORLD_COND_CLIP_LEN", 1))
        else:
            # Unknown extension: default to 1 to avoid action shifting.
            cond_block.setdefault("clip_len", 1)
        action_spec["cond"] = cond_block

        actions, meta = generate_actions_from_spec(action_spec)
        action_json_path = os.path.join(actions_dir, f"{i:04d}_{output_name}.json")
        with open(action_json_path, "w") as f:
            json.dump(actions, f)

        cond_path_rel = _relpath_from_repo_root(repo_root, os.path.join(repo_root, cond_path))
        action_path_rel = _relpath_from_repo_root(repo_root, action_json_path)

        # If task-level condition overrides exist, write dict format (inference supports both).
        if task_cond_clip_len is not None or task_cond_clip_mode is not None:
            entry: Dict[str, Any] = {
                "prompt": prompt,
                "cond_path": cond_path_rel,
                "action_path": action_path_rel,
                "output_name": output_name,
            }
            if task_cond_clip_len is not None:
                entry["cond_clip_len"] = task_cond_clip_len
            if task_cond_clip_mode is not None:
                entry["cond_clip_mode"] = task_cond_clip_mode
            prompts_mixed.append(entry)
        else:
            prompts_mixed.append([prompt, cond_path_rel, action_path_rel, output_name])

    prompts_yaml_obj = {"prompts": prompts_mixed}
    prompts_yaml_path = os.path.join(run_dir, "prompts.yaml")
    _write_yaml(prompts_yaml_path, prompts_yaml_obj)

    # Generate run.sh
    output_base = env_defaults.get("INFWORLD_OUTPUT_BASE")
    if not output_base:
        output_base = _relpath_from_repo_root(repo_root, os.path.join(run_dir, "outputs"))

    env_lines = []
    env_lines.append("#!/usr/bin/env bash")
    env_lines.append("set -euo pipefail")
    env_lines.append("")
    env_lines.append("# Auto-generated by scripts/infworld_suite.py")
    env_lines.append(f'REPO_ROOT="{repo_root}"')
    env_lines.append('cd "$REPO_ROOT"')
    env_lines.append("")
    env_lines.append("# ---- Inference knobs (edit freely) ----")
    env_lines.append(f'export INFWORLD_PROMPTS_YAML="{_relpath_from_repo_root(repo_root, prompts_yaml_path)}"')
    env_lines.append(f'export INFWORLD_OUTPUT_BASE="{output_base}"')
    for k, v in env_defaults.items():
        if k in {"INFWORLD_PROMPTS_YAML", "INFWORLD_OUTPUT_BASE"}:
            continue
        # Let PBS or the caller override any knob by exporting it before calling run.sh.
        # We only provide defaults here.
        env_lines.append(f'export {k}="${{{k}:-{v}}}"')
    env_lines.append("")
    env_lines.append('NUM_GPUS="${1:-1}"')
    env_lines.append('bash infer_local.sh "$NUM_GPUS"')
    env_lines.append("")

    run_sh_path = os.path.join(run_dir, "run.sh")
    _write_text(run_sh_path, "\n".join(env_lines))
    os.chmod(run_sh_path, 0o755)

    # Human-friendly summary
    print("[InfWorld][suite] Created:", run_dir)
    print("[InfWorld][suite] Prompts YAML:", os.path.relpath(prompts_yaml_path, start=repo_root))
    print("[InfWorld][suite] Run script:", os.path.relpath(run_sh_path, start=repo_root))
    print("[InfWorld][suite] Example:")
    print("  bash", os.path.relpath(run_sh_path, start=repo_root), "1")


if __name__ == "__main__":
    main()
