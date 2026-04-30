from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


MOVE_MAP = {
    "noop": "no-op",
    "F": "go forward",
    "B": "go back",
    "L": "go left",
    "R": "go right",
    "FL": "go forward and go left",
    "FR": "go forward and go right",
    "BL": "go back and go left",
    "BR": "go back and go right",
    "UNC": "uncertain",
}

VIEW_MAP = {
    "noop": "no-op",
    "U": "turn up",
    "D": "turn down",
    "L": "turn left",
    "R": "turn right",
    "UL": "turn up and turn left",
    "UR": "turn up and turn right",
    "DL": "turn down and turn left",
    "DR": "turn down and turn right",
    "UNC": "uncertain",
}

VIDEO_SUFFIX = "_ori"
PROMPT_SUFFIX = "_prompt.txt"
TRAJ_SUFFIX = "_traj.txt"
CHUNK_FRAMES = 81


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _normalize_token(token: str) -> str:
    token = token.strip()
    if token.lower() == "noop":
        return "noop"
    return token.upper()


def _expand_actions(traj_path: Path) -> tuple[str, list[dict[str, str]], int, str]:
    lines = [line.strip() for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Trajectory file needs a natural-language line plus at least one chunk line: {traj_path}")

    trajectory_nl = lines[0]
    actions: list[dict[str, str]] = []
    chunk_specs: list[str] = []
    total_chunks = 0

    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(
                f"Invalid trajectory line {idx} in {traj_path}. Expected: '<num_chunks> <move> <view>'"
            )
        num_chunks = int(parts[0])
        move_token = _normalize_token(parts[1])
        view_token = _normalize_token(parts[2])
        if move_token not in MOVE_MAP:
            raise ValueError(f"Unknown move token {move_token!r} in {traj_path}")
        if view_token not in VIEW_MAP:
            raise ValueError(f"Unknown view token {view_token!r} in {traj_path}")

        total_chunks += num_chunks
        chunk_specs.append(
            f"{num_chunks} chunk(s): move={MOVE_MAP[move_token]}, view={VIEW_MAP[view_token]}"
        )
        for _ in range(num_chunks * CHUNK_FRAMES):
            actions.append({"move": MOVE_MAP[move_token], "view": VIEW_MAP[view_token]})

    enhanced_summary = (
        f"{trajectory_nl} "
        f"Structured chunk plan: {'; '.join(chunk_specs)}. "
        f"Total continuation chunks: {total_chunks} ({total_chunks * CHUNK_FRAMES} action frames)."
    )
    return trajectory_nl, actions, total_chunks, enhanced_summary


def build_dataset(input_dir: Path, dataset_path: Path, actions_dir: Path, cond_clip_len: int) -> None:
    dataset_rows: list[dict[str, object]] = []
    actions_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.stem.endswith(VIDEO_SUFFIX)]
    )
    if not video_files:
        raise ValueError(f"No '*{VIDEO_SUFFIX}.mp4' style video files found in {input_dir}")

    for video_path in video_files:
        stem_base = video_path.stem[: -len(VIDEO_SUFFIX)]
        prompt_path = input_dir / f"{stem_base}{PROMPT_SUFFIX}"
        traj_path = input_dir / f"{stem_base}{TRAJ_SUFFIX}"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Missing prompt file for {video_path.name}: {prompt_path.name}")
        if not traj_path.exists():
            raise FileNotFoundError(f"Missing trajectory file for {video_path.name}: {traj_path.name}")

        base_prompt = _read_text(prompt_path)
        trajectory_nl, actions, total_chunks, enhanced_summary = _expand_actions(traj_path)
        action_json_path = actions_dir / f"{stem_base}.json"
        action_json_path.write_text(json.dumps(actions, indent=2), encoding="utf-8")

        dataset_rows.append(
            {
                "base_prompt": base_prompt,
                "condition_path": os.path.relpath(str(video_path), str(PROJECT_ROOT)),
                "action_path": os.path.relpath(str(action_json_path), str(PROJECT_ROOT)),
                "condition_context": base_prompt,
                "action_trajectory": enhanced_summary,
                "target_traits": (
                    "Preserve the source scene identity, respect the described camera and action trajectory, "
                    "and keep motion coherent across the generated continuation."
                ),
                "output_name": f"{stem_base}_gepa",
                "cond_clip_len": cond_clip_len,
                "cond_clip_mode": "uniform",
                "action_start_mode": "chunk_aligned",
                "num_chunks": total_chunks,
                "trajectory_description": trajectory_nl,
            }
        )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(dataset_rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a GEPA dataset manifest and action JSONs from an input folder."
    )
    parser.add_argument(
        "--input-dir",
        default="/root/dataDisk/skebin_temp_storage/input_data",
        help="Folder containing *_ori video, *_prompt.txt, and *_traj.txt files.",
    )
    parser.add_argument(
        "--dataset-out",
        default="infworld_gepa/input_data_dataset.json",
        help="Output dataset JSON path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--actions-dir",
        default="infworld_gepa/generated_actions",
        help="Where to write generated action JSON files, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--cond-clip-len",
        type=int,
        default=150,
        help="How many conditioning video frames to sample uniformly from each input video.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    dataset_out = Path(args.dataset_out)
    if not dataset_out.is_absolute():
        dataset_out = PROJECT_ROOT / dataset_out
    actions_dir = Path(args.actions_dir)
    if not actions_dir.is_absolute():
        actions_dir = PROJECT_ROOT / actions_dir

    build_dataset(input_dir, dataset_out, actions_dir, cond_clip_len=args.cond_clip_len)
    print("dataset_written:", str(dataset_out))
    print("actions_dir:", str(actions_dir))


if __name__ == "__main__":
    main()
