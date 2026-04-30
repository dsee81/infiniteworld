from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class ActionStats:
    num_frames: int
    move_counts: dict[str, int]
    view_counts: dict[str, int]
    non_noop_ratio: float
    summary: str


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMG_EXTS


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VID_EXTS


def load_action_stats(action_json_path: str | Path) -> ActionStats:
    with open(action_json_path, "r", encoding="utf-8") as f:
        actions = json.load(f)

    move_counts: dict[str, int] = {}
    view_counts: dict[str, int] = {}
    non_noop = 0
    runs = []
    prev = None
    start = 0

    for i, action in enumerate(actions):
        move = action["move"]
        view = action["view"]
        move_counts[move] = move_counts.get(move, 0) + 1
        view_counts[view] = view_counts.get(view, 0) + 1
        if move != "no-op" or view != "no-op":
            non_noop += 1
        cur = (move, view)
        if prev is None:
            prev = cur
            start = i
        elif cur != prev:
            runs.append((start, i - 1, prev))
            start = i
            prev = cur
    if actions:
        runs.append((start, len(actions) - 1, prev))

    run_summary = "; ".join(
        f"frames {s}-{e}: move={move}, view={view}" for s, e, (move, view) in runs[:12]
    )
    summary = (
        f"{len(actions)} action frames. "
        f"Move counts: {move_counts}. "
        f"View counts: {view_counts}. "
        f"Runs: {run_summary}"
    )
    return ActionStats(
        num_frames=len(actions),
        move_counts=move_counts,
        view_counts=view_counts,
        non_noop_ratio=(non_noop / len(actions)) if actions else 0.0,
        summary=summary,
    )


def sample_video_frames(video_path: str | Path, max_frames: int = 8) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = max_frames
    indices = np.linspace(0, max(total - 1, 0), num=min(max_frames, max(total, 1)), dtype=int)
    wanted = set(int(i) for i in indices)
    idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def sample_condition_frames(path: str | Path, max_frames: int = 4) -> list[np.ndarray]:
    if is_image_path(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return []
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
    if is_video_path(path):
        return sample_video_frames(path, max_frames=max_frames)
    return []


def frame_to_data_url(
    frame: np.ndarray,
    *,
    max_side: int | None = None,
    jpeg_quality: int = 85,
) -> str:
    if max_side is not None and max_side > 0:
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame for VLM request.")
    data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"
