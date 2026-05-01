from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT.parent / "input_data"
SHARED_ROOT = Path("/mnt/shared_storage/dsee/memory_fix_batch")
PROMPTS_ROOT = SHARED_ROOT / "prompts"
VIDEOS_ROOT = SHARED_ROOT / "videos"
LOGS_ROOT = SHARED_ROOT / "logs"
RESULTS_ROOT = SHARED_ROOT / "results"
DEFAULT_ACTION_ROOT = REPO_ROOT / "infworld_gepa" / "generated_actions"

sys.path.insert(0, str(REPO_ROOT))

from infworld_gepa.config import load_settings
from infworld_gepa.metric_suite import (
    LocalVLMJudge,
    MetricInput,
    MockVLMJudge,
    OpticalFlowMetric,
    SemanticConsistencyMetric,
    TrajectoryAdherenceMetric,
)


@dataclass(frozen=True)
class Experiment:
    name: str
    folder: str
    family: str
    memory_chunks: int


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("recent_tail_4", "four_chunk_contiguous_tail", "recent_tail", 4),
    Experiment("recent_tail_5", "last5_chunks", "recent_tail", 5),
    Experiment("uniform_spacing_4", "uniform_spacing_4", "uniform_spacing", 4),
    Experiment("uniform_spacing_5", "uniform_spacing_5", "uniform_spacing", 5),
    Experiment("option_a_4", "four_chunk_option_a", "option_a", 4),
    Experiment("option_a_5", "five_chunk_option_a", "option_a", 5),
    Experiment("option_b_4", "four_chunk_option_b", "option_b", 4),
    Experiment("option_b_5", "five_chunk_option_b", "option_b", 5),
    Experiment("action_visual_4", "four_chunk_action_visual_hybrid", "action_visual", 4),
    Experiment("action_visual_5", "action_visual_hybrid", "action_visual", 5),
)


def _discover_video_stems() -> list[str]:
    return sorted(
        p.stem[:-4]
        for p in INPUT_ROOT.glob("*_ori.mp4")
        if p.stem.endswith("_ori")
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_prompt_yaml(stem: str, prompt: str, cond_path: Path, action_path: Path, output_name: str) -> Path:
    PROMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    prompt_path = PROMPTS_ROOT / f"{output_name}.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "prompts:",
                "  - prompt: |",
                *[f"      {line}" if line else "" for line in prompt.splitlines()],
                f"    cond_path: {cond_path}",
                f"    action_path: {action_path}",
                f"    output_name: {output_name}",
                "    cond_clip_len: 150",
                "    cond_clip_mode: uniform",
                "    action_start_mode: chunk_aligned",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return prompt_path


def _result_csv_path() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT / "batch_results.csv"


def _append_csv_row(row: dict) -> None:
    csv_path = _result_csv_path()
    fieldnames = [
        "timestamp_utc",
        "experiment",
        "family",
        "memory_chunks",
        "video_stem",
        "num_generation_chunks",
        "runtime_seconds",
        "runtime_minutes",
        "status",
        "video_path",
        "log_path",
        "metrics_mode",
        "optical_flow_score",
        "traj_score",
        "semantic_score",
        "composite_score",
        "traj_feedback",
        "semantic_feedback",
        "optical_feedback",
        "error",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_jsonl_row(row: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_ROOT / "batch_results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _maybe_existing_row(video_path: Path) -> bool:
    return video_path.exists()


def _evaluate_video(
    prompt: str,
    cond_path: Path,
    action_path: Path,
    video_path: Path,
    use_live_vlm: bool,
) -> tuple[dict, str]:
    settings = load_settings()
    metric_input = MetricInput(
        prompt=prompt,
        condition_path=str(cond_path),
        action_path=str(action_path),
    )

    metrics_mode = "mock"
    judge = MockVLMJudge()
    if use_live_vlm and settings.has_local_vlm:
        try:
            judge = LocalVLMJudge(settings)
            metrics_mode = "live_vlm"
        except Exception:
            judge = MockVLMJudge()
            metrics_mode = "mock_fallback"

    optical = OpticalFlowMetric().evaluate(metric_input, str(video_path))
    traj = TrajectoryAdherenceMetric(judge).evaluate(metric_input, str(video_path))
    semantic = SemanticConsistencyMetric(judge).evaluate(metric_input, str(video_path))
    metrics = {
        "optical_flow": {
            "score": optical.score,
            "feedback": optical.feedback,
            "details": optical.details,
        },
        "traj_adherence": {
            "score": traj.score,
            "feedback": traj.feedback,
            "details": traj.details,
        },
        "semantic_consistency": {
            "score": semantic.score,
            "feedback": semantic.feedback,
            "details": semantic.details,
        },
        "composite_score": (0.2 * optical.score) + (0.4 * traj.score) + (0.4 * semantic.score),
    }
    return metrics, metrics_mode


def _run_inference(
    exp: Experiment,
    prompt_yaml: Path,
    output_base: Path,
    log_path: Path,
    gpu: str,
    num_generation_chunks: int,
) -> subprocess.CompletedProcess:
    variant_dir = REPO_ROOT / "memory_fix" / exp.folder
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "INFWORLD_MAX_TASKS": "1",
            "INFWORLD_NUM_CHUNKS": str(num_generation_chunks),
            "INFWORLD_PROMPTS_YAML": str(prompt_yaml),
            "INFWORLD_OUTPUT_BASE": str(output_base),
        }
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(
            [sys.executable, "-u", "infworld_inference.py"],
            cwd=variant_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )


def _iter_experiments(selected: set[str] | None) -> Iterable[Experiment]:
    for exp in EXPERIMENTS:
        if selected and exp.name not in selected:
            continue
        yield exp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run memory-fix experiment batches and score outputs.")
    parser.add_argument("--gpu", default="6", help="Single visible GPU id to use for inference.")
    parser.add_argument("--num-generation-chunks", type=int, default=7, help="How many continuation chunks to generate.")
    parser.add_argument("--videos", nargs="*", help="Optional list of input video stems, e.g. 39 44.")
    parser.add_argument("--experiments", nargs="*", help="Optional list of experiment names to run.")
    parser.add_argument("--use-live-vlm", action="store_true", help="Use the configured local VLM for trajectory/semantic scoring.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many runs; 0 means no limit.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose target mp4 already exists.")
    args = parser.parse_args()

    selected_videos = set(args.videos or _discover_video_stems())
    selected_experiments = set(args.experiments or [])

    SHARED_ROOT.mkdir(parents=True, exist_ok=True)
    PROMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    run_counter = 0
    for exp in _iter_experiments(selected_experiments):
        for stem in sorted(selected_videos):
            cond_path = INPUT_ROOT / f"{stem}_ori.mp4"
            prompt_path = INPUT_ROOT / f"{stem}_prompt.txt"
            action_path = DEFAULT_ACTION_ROOT / f"{stem}.json"
            if not cond_path.exists() or not prompt_path.exists() or not action_path.exists():
                print(f"[Batch] Skipping {stem}/{exp.name}: missing input file(s).", flush=True)
                continue

            output_name = f"{stem}_{exp.name}_h{args.num_generation_chunks}"
            prompt_yaml = _write_prompt_yaml(
                stem=stem,
                prompt=_read_text(prompt_path),
                cond_path=cond_path,
                action_path=action_path,
                output_name=output_name,
            )

            output_base = VIDEOS_ROOT / exp.name
            video_path = output_base / "infworld-ckpt0-step30-cfg5.0" / f"{output_name}.mp4"
            log_path = LOGS_ROOT / exp.name / f"{output_name}.log"

            if args.skip_existing and _maybe_existing_row(video_path):
                print(f"[Batch] Skipping existing output: {video_path}", flush=True)
                continue

            run_counter += 1
            print(f"[Batch] ({run_counter}) Running {exp.name} on {stem} -> {video_path}", flush=True)
            started = time.time()
            proc = _run_inference(
                exp=exp,
                prompt_yaml=prompt_yaml,
                output_base=output_base,
                log_path=log_path,
                gpu=args.gpu,
                num_generation_chunks=args.num_generation_chunks,
            )
            finished = time.time()
            runtime_seconds = finished - started

            row = {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
                "experiment": exp.name,
                "family": exp.family,
                "memory_chunks": exp.memory_chunks,
                "video_stem": stem,
                "num_generation_chunks": args.num_generation_chunks,
                "runtime_seconds": round(runtime_seconds, 3),
                "runtime_minutes": round(runtime_seconds / 60.0, 3),
                "status": "ok" if proc.returncode == 0 and video_path.exists() else "failed",
                "video_path": str(video_path),
                "log_path": str(log_path),
                "metrics_mode": "",
                "optical_flow_score": "",
                "traj_score": "",
                "semantic_score": "",
                "composite_score": "",
                "traj_feedback": "",
                "semantic_feedback": "",
                "optical_feedback": "",
                "error": "" if proc.returncode == 0 else f"inference_returncode={proc.returncode}",
            }

            per_run_metrics_path = RESULTS_ROOT / f"{output_name}.json"

            if row["status"] == "ok":
                try:
                    prompt = _read_text(prompt_path)
                    metrics, metrics_mode = _evaluate_video(
                        prompt=prompt,
                        cond_path=cond_path,
                        action_path=action_path,
                        video_path=video_path,
                        use_live_vlm=args.use_live_vlm,
                    )
                    row.update(
                        {
                            "metrics_mode": metrics_mode,
                            "optical_flow_score": round(metrics["optical_flow"]["score"], 4),
                            "traj_score": round(metrics["traj_adherence"]["score"], 4),
                            "semantic_score": round(metrics["semantic_consistency"]["score"], 4),
                            "composite_score": round(metrics["composite_score"], 4),
                            "traj_feedback": metrics["traj_adherence"]["feedback"],
                            "semantic_feedback": metrics["semantic_consistency"]["feedback"],
                            "optical_feedback": metrics["optical_flow"]["feedback"],
                        }
                    )
                    per_run_metrics_path.write_text(
                        json.dumps(
                            {
                                "row": row,
                                "metrics": metrics,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    row["metrics_mode"] = "evaluation_failed"
                    row["error"] = (row["error"] + "; " if row["error"] else "") + f"evaluation={type(exc).__name__}: {exc}"
            else:
                per_run_metrics_path.write_text(
                    json.dumps({"row": row}, indent=2),
                    encoding="utf-8",
                )

            _append_csv_row(row)
            _append_jsonl_row(row)
            print(
                f"[Batch] Completed {exp.name} on {stem}: status={row['status']} "
                f"runtime={row['runtime_minutes']} min",
                flush=True,
            )

            if args.limit and run_counter >= args.limit:
                print(f"[Batch] Reached limit={args.limit}; stopping.", flush=True)
                return


if __name__ == "__main__":
    main()
