from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT.parent / "input_data"
OUTPUT_ROOT = Path("/mnt/shared_storage/dsee/memory_experiment")
PROMPTS_ROOT = OUTPUT_ROOT / "prompts"
VIDEOS_ROOT = OUTPUT_ROOT / "videos"
LOGS_ROOT = OUTPUT_ROOT / "logs"
RESULTS_ROOT = OUTPUT_ROOT / "results"
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
    Experiment("baseline_4", "baseline_4", "baseline", 4),
    Experiment("baseline_5", "baseline_5", "baseline", 5),
    Experiment("recent_tail_4", "recent_tail_4", "recent_tail", 4),
    Experiment("recent_tail_5", "recent_tail_5", "recent_tail", 5),
    Experiment("recent_anchor_salient_transition_4", "recent_anchor_salient_transition_4", "anchor_salient_transition", 4),
    Experiment("recent_anchor_salient_transition_5", "recent_anchor_salient_transition_5", "anchor_salient_transition", 5),
)


CSV_LOCK = threading.Lock()


def _discover_video_stems() -> list[str]:
    return sorted(
        p.stem[:-4]
        for p in INPUT_ROOT.glob("*_ori.mp4")
        if p.stem.endswith("_ori")
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _write_prompt_yaml(prompt: str, cond_path: Path, action_path: Path, output_name: str) -> Path:
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
        "gpu",
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
    with CSV_LOCK:
        exists = csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _append_jsonl_row(row: dict) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    jsonl_path = RESULTS_ROOT / "batch_results.jsonl"
    with CSV_LOCK:
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _evaluate_video(prompt: str, cond_path: Path, action_path: Path, video_path: Path, use_live_vlm: bool) -> tuple[dict, str]:
    settings = load_settings()
    metric_input = MetricInput(prompt=prompt, condition_path=str(cond_path), action_path=str(action_path))

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
        "optical_flow": {"score": optical.score, "feedback": optical.feedback, "details": optical.details},
        "traj_adherence": {"score": traj.score, "feedback": traj.feedback, "details": traj.details},
        "semantic_consistency": {"score": semantic.score, "feedback": semantic.feedback, "details": semantic.details},
        "composite_score": (0.2 * optical.score) + (0.4 * traj.score) + (0.4 * semantic.score),
    }
    return metrics, metrics_mode


def _run_inference(exp: Experiment, prompt_yaml: Path, output_base: Path, log_path: Path, gpu: str, num_generation_chunks: int) -> subprocess.CompletedProcess:
    variant_dir = REPO_ROOT / "memory_experiment" / "variants" / exp.folder
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


def _build_jobs(selected_videos: list[str], selected_experiments: set[str] | None) -> list[tuple[Experiment, str]]:
    jobs: list[tuple[Experiment, str]] = []
    for exp in EXPERIMENTS:
        if selected_experiments and exp.name not in selected_experiments:
            continue
        for stem in selected_videos:
            jobs.append((exp, stem))
    return jobs


def _process_job(exp: Experiment, stem: str, gpu: str, num_generation_chunks: int, use_live_vlm: bool, skip_existing: bool) -> dict:
    cond_path = INPUT_ROOT / f"{stem}_ori.mp4"
    prompt_path = INPUT_ROOT / f"{stem}_prompt.txt"
    action_path = DEFAULT_ACTION_ROOT / f"{stem}.json"
    output_name = f"{stem}_{exp.name}_h{num_generation_chunks}"
    output_base = VIDEOS_ROOT / exp.name
    video_path = output_base / "infworld-ckpt0-step30-cfg5.0" / f"{output_name}.mp4"
    log_path = LOGS_ROOT / exp.name / f"{output_name}.log"

    row = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": exp.name,
        "family": exp.family,
        "memory_chunks": exp.memory_chunks,
        "video_stem": stem,
        "num_generation_chunks": num_generation_chunks,
        "gpu": gpu,
        "runtime_seconds": "",
        "runtime_minutes": "",
        "status": "skipped",
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
        "error": "",
    }

    if not cond_path.exists() or not prompt_path.exists() or not action_path.exists():
        row["status"] = "failed"
        row["error"] = "missing_input_files"
        return row

    if skip_existing and video_path.exists():
        row["status"] = "skipped_existing"
        return row

    prompt = _read_text(prompt_path)
    prompt_yaml = _write_prompt_yaml(prompt=prompt, cond_path=cond_path, action_path=action_path, output_name=output_name)

    started = time.time()
    proc = _run_inference(exp, prompt_yaml, output_base, log_path, gpu, num_generation_chunks)
    finished = time.time()
    runtime_seconds = finished - started
    row["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished))
    row["runtime_seconds"] = round(runtime_seconds, 3)
    row["runtime_minutes"] = round(runtime_seconds / 60.0, 3)
    row["status"] = "ok" if proc.returncode == 0 and video_path.exists() else "failed"
    if proc.returncode != 0:
        row["error"] = f"inference_returncode={proc.returncode}"

    per_run_metrics_path = RESULTS_ROOT / f"{output_name}.json"
    if row["status"] == "ok":
        try:
            metrics, metrics_mode = _evaluate_video(prompt, cond_path, action_path, video_path, use_live_vlm)
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
            per_run_metrics_path.write_text(json.dumps({"row": row, "metrics": metrics}, indent=2), encoding="utf-8")
        except Exception as exc:
            row["metrics_mode"] = "evaluation_failed"
            row["error"] = (row["error"] + "; " if row["error"] else "") + f"evaluation={type(exc).__name__}: {exc}"
            per_run_metrics_path.write_text(json.dumps({"row": row}, indent=2), encoding="utf-8")
    else:
        per_run_metrics_path.write_text(json.dumps({"row": row}, indent=2), encoding="utf-8")

    return row


def _run_gpu_queue(gpu: str, jobs: list[tuple[Experiment, str]], num_generation_chunks: int, use_live_vlm: bool, skip_existing: bool) -> list[dict]:
    rows = []
    for exp, stem in jobs:
        row = _process_job(
            exp=exp,
            stem=stem,
            gpu=gpu,
            num_generation_chunks=num_generation_chunks,
            use_live_vlm=use_live_vlm,
            skip_existing=skip_existing,
        )
        _append_csv_row(row)
        _append_jsonl_row(row)
        print(
            f"[MemoryExperiment] {exp.name} on {stem} gpu={gpu} "
            f"status={row['status']} runtime_min={row['runtime_minutes']}",
            flush=True,
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the memory experiment suite across all input videos.")
    parser.add_argument("--gpus", nargs="+", default=["4", "7"], help="GPU ids to use concurrently.")
    parser.add_argument("--num-generation-chunks", type=int, default=7, help="Continuation horizon in chunks.")
    parser.add_argument("--videos", nargs="*", help="Optional list of input video stems, e.g. 39 44.")
    parser.add_argument("--experiments", nargs="*", help="Optional list of experiment names to run.")
    parser.add_argument("--use-live-vlm", action="store_true", help="Use the configured local VLM for trajectory/semantic scoring.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs whose target mp4 already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of jobs.")
    args = parser.parse_args()

    selected_videos = sorted(set(args.videos or _discover_video_stems()))
    selected_experiments = set(args.experiments or [])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROMPTS_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(selected_videos, selected_experiments)
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"[MemoryExperiment] jobs={len(jobs)} videos={selected_videos} gpus={args.gpus}", flush=True)

    gpu_queues: dict[str, list[tuple[Experiment, str]]] = {gpu: [] for gpu in args.gpus}
    for idx, job in enumerate(jobs):
        gpu = args.gpus[idx % len(args.gpus)]
        gpu_queues[gpu].append(job)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = []
        for gpu, gpu_jobs in gpu_queues.items():
            if not gpu_jobs:
                continue
            future = executor.submit(
                _run_gpu_queue,
                gpu,
                gpu_jobs,
                num_generation_chunks=args.num_generation_chunks,
                use_live_vlm=args.use_live_vlm,
                skip_existing=args.skip_existing,
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
