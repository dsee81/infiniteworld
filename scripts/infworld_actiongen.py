#!/usr/bin/env python3
"""
Infinite World - Action / Camera Trajectory Generator
=====================================================

This script generates the JSON action format expected by `scripts/infworld_inference.py`:

    [
      {"move": "go forward", "view": "turn left"},
      {"move": "go forward", "view": "turn left"},
      ...
    ]

Key detail (matches inference):
  - Inference slices actions in windows of `chunk_frames` (default 81) with an
    overlap of 1 frame between consecutive chunks. That means a run with
    `num_chunks=N` and `cond_frames=C` (number of condition frames) will
    *consume* actions up to index:

      total_frames = C + (chunk_frames - overlap) * N

    With defaults C=1, chunk_frames=81, overlap=1, total_frames = 1 + 80*N.

This generator produces exactly that length by:
  - Prefixing (C-1) "no-op" actions (for multi-frame video conditioning),
  - Creating N chunks of length chunk_frames,
  - Stitching them with the requested overlap (dropping the first `overlap`
    frames of every chunk after the first).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml


MOVE_TOKENS = [
    "no-op",
    "go forward",
    "go back",
    "go left",
    "go right",
    "go forward and go left",
    "go forward and go right",
    "go back and go left",
    "go back and go right",
    "uncertain",
]

VIEW_TOKENS = [
    "no-op",
    "turn up",
    "turn down",
    "turn left",
    "turn right",
    "turn up and turn left",
    "turn up and turn right",
    "turn down and turn left",
    "turn down and turn right",
    "uncertain",
]

MOVE_SET = set(MOVE_TOKENS)
VIEW_SET = set(VIEW_TOKENS)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top-level: {path}")
    return data


def _video_frame_count(path: str) -> int:
    try:
        import cv2  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required only for cond.clip_len='all' (to count video frames). "
            "Install opencv-python or set cond.clip_len to an integer."
        ) from e

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RuntimeError(f"Could not determine frame count for: {path}")
    return n


def _validate_token(kind: str, token: str, allowed: set) -> None:
    if token not in allowed:
        raise ValueError(
            f"Invalid {kind} token: {token!r}. Allowed: {sorted(allowed)}"
        )


def _constant_frames(move: str, view: str, n: int) -> List[Dict[str, str]]:
    return [{"move": move, "view": view} for _ in range(n)]


def _stitch_chunks(chunks: List[List[Dict[str, str]]], overlap: int) -> List[Dict[str, str]]:
    if not chunks:
        return []
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    out: List[Dict[str, str]] = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.extend(c)
        else:
            out.extend(c[overlap:])
    return out


def compute_required_total_frames(cond_frames: int, num_chunks: int, chunk_frames: int, overlap: int) -> int:
    if cond_frames < 1:
        raise ValueError("cond_frames must be >= 1")
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be >= 1")
    if overlap < 0 or overlap >= chunk_frames:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_frames")
    # (cond_frames-1) prefix + stitched chunks: chunk_frames + (N-1)*(chunk_frames-overlap)
    return (cond_frames - 1) + chunk_frames + (num_chunks - 1) * (chunk_frames - overlap)


@dataclass(frozen=True)
class ChunkSpec:
    move: str
    view: str
    repeat: int = 1


def _expand_chunk_schedule(chunks: List[Dict[str, Any]], num_chunks: int) -> List[ChunkSpec]:
    expanded: List[ChunkSpec] = []
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"chunks[{i}] must be a mapping")
        move = str(c.get("move", "no-op"))
        view = str(c.get("view", "no-op"))
        repeat = int(c.get("repeat", 1))
        if repeat < 1:
            raise ValueError(f"chunks[{i}].repeat must be >= 1")
        _validate_token("move", move, MOVE_SET)
        _validate_token("view", view, VIEW_SET)
        expanded.extend([ChunkSpec(move=move, view=view, repeat=1) for _ in range(repeat)])

    if len(expanded) != num_chunks:
        raise ValueError(
            f"Chunk schedule expands to {len(expanded)} chunks, but num_chunks={num_chunks}. "
            "Fix repeats or num_chunks."
        )
    return expanded


def generate_actions_from_spec(spec: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Spec schema (YAML):

      cond:
        path: ./path/to/cond.mp4  # optional (used only to infer frames when clip_len='all')
        clip_len: 1              # int >= 1 or 'all'
      inference:
        num_chunks: 13
        chunk_frames: 81
        overlap: 1
      prefix:
        move: no-op              # optional, default no-op
        view: no-op              # optional, default no-op
      chunks:
        - move: go forward
          view: turn left
          repeat: 2
        - move: go back
          view: turn right
          repeat: 11

    Returns: (actions, meta)
    """
    cond = spec.get("cond", {}) or {}
    inference = spec.get("inference", {}) or {}
    prefix = spec.get("prefix", {}) or {}

    num_chunks = int(inference.get("num_chunks", 13))
    chunk_frames = int(inference.get("chunk_frames", 81))
    overlap = int(inference.get("overlap", 1))

    cond_path = cond.get("path")
    clip_len = cond.get("clip_len", 1)
    if isinstance(clip_len, str) and clip_len.strip().lower() == "all":
        if not cond_path:
            raise ValueError("cond.clip_len='all' requires cond.path to infer video length")
        cond_frames = _video_frame_count(str(cond_path))
    else:
        cond_frames = int(clip_len)
        if cond_frames < 1:
            raise ValueError("cond.clip_len must be >= 1 or 'all'")

    prefix_move = str(prefix.get("move", "no-op"))
    prefix_view = str(prefix.get("view", "no-op"))
    _validate_token("move", prefix_move, MOVE_SET)
    _validate_token("view", prefix_view, VIEW_SET)

    chunks = spec.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("spec.chunks must be a non-empty list")

    schedule = _expand_chunk_schedule(chunks, num_chunks=num_chunks)

    # Prefix (C-1) actions for multi-frame cond video context.
    actions: List[Dict[str, str]] = []
    if cond_frames > 1:
        actions.extend(_constant_frames(prefix_move, prefix_view, cond_frames - 1))

    chunk_actions = [_constant_frames(s.move, s.view, chunk_frames) for s in schedule]
    actions.extend(_stitch_chunks(chunk_actions, overlap=overlap))

    meta = {
        "cond_frames": cond_frames,
        "num_chunks": num_chunks,
        "chunk_frames": chunk_frames,
        "overlap": overlap,
        "total_frames": len(actions),
        "required_total_frames_formula": compute_required_total_frames(
            cond_frames=cond_frames, num_chunks=num_chunks, chunk_frames=chunk_frames, overlap=overlap
        ),
    }
    return actions, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Infinite-World action JSON from a chunk schedule YAML.")
    ap.add_argument("--spec", required=True, help="Path to action spec YAML.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = ap.parse_args()

    spec = _read_yaml(args.spec)
    actions, meta = generate_actions_from_spec(spec)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(actions, f, indent=2 if args.pretty else None)

    print(
        "[InfWorld][actiongen] Wrote",
        args.out,
        "| total_frames=",
        meta["total_frames"],
        "| cond_frames=",
        meta["cond_frames"],
        "| num_chunks=",
        meta["num_chunks"],
        "| chunk_frames=",
        meta["chunk_frames"],
        "| overlap=",
        meta["overlap"],
    )


if __name__ == "__main__":
    main()
