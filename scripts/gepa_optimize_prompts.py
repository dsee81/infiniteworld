from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import dspy
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld_gepa.config import build_deepseek_lm, load_settings
from infworld_gepa.hard_prompt import InfWorldHardPromptProgram, summarize_action_runs
from infworld_gepa.metric_suite import FiveMetricSuite, MetricInput
from infworld_gepa.pipeline_components import PromptEditor, SceneSummarizer


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, width=120)


def _sha_text(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _coerce_relpath(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return os.path.relpath(str(path), PROJECT_ROOT)
    return value


def _load_dataset_rows(dataset_path: Path) -> list[dict[str, Any]]:
    raw = _read_json(dataset_path)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset JSON must be a non-empty list of task objects.")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {idx} must be an object.")
        required = ["base_prompt", "condition_path", "action_path"]
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"Dataset row {idx} missing required fields: {', '.join(missing)}")

        action_path = Path(PROJECT_ROOT, _coerce_relpath(str(row["action_path"]))).resolve()
        action_summary = row.get("action_trajectory") or summarize_action_runs(action_path)
        condition_context = row.get("condition_context") or ""
        target_traits = row.get("target_traits") or (
            "Preserve scene identity, follow the requested trajectory, and stay visually coherent."
        )

        normalized = dict(row)
        normalized["condition_path"] = _coerce_relpath(str(row["condition_path"]))
        normalized["action_path"] = _coerce_relpath(str(row["action_path"]))
        normalized["condition_context"] = condition_context
        normalized["action_trajectory"] = action_summary
        normalized["target_traits"] = target_traits
        normalized["num_chunks"] = row.get("num_chunks")
        normalized["action_start_mode"] = row.get("action_start_mode", "buffer")
        rows.append(normalized)
    return rows


def load_examples_from_rows(rows: list[dict[str, Any]]) -> list[dspy.Example]:
    examples: list[dspy.Example] = []
    for row in rows:
        examples.append(
            dspy.Example(
                base_prompt=row["base_prompt"],
                condition_context=row["condition_context"],
                scene_summary=row.get("scene_summary", ""),
                action_trajectory=row["action_trajectory"],
                target_traits=row["target_traits"],
                condition_path=row["condition_path"],
                action_path=row["action_path"],
                cond_clip_len=row.get("cond_clip_len"),
                cond_clip_mode=row.get("cond_clip_mode"),
                num_chunks=row.get("num_chunks"),
                action_start_mode=row.get("action_start_mode", "buffer"),
                output_name=row.get("output_name"),
            ).with_inputs(
                "base_prompt",
                "condition_context",
                "scene_summary",
                "action_trajectory",
                "target_traits",
            )
        )
    return examples


def select_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_index: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = rows
    if dataset_index is not None:
        selected = [selected[dataset_index]]
    if limit is not None:
        selected = selected[:limit]
    return selected


def enrich_rows_with_scene_summaries(
    rows: list[dict[str, Any]],
    *,
    settings,
    cache_dir: Path,
    sample_frames: int = 30,
) -> list[dict[str, Any]]:
    summarizer = SceneSummarizer(settings, cache_dir=cache_dir, sample_frames=sample_frames)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        summary = summarizer.summarize(
            str(Path(PROJECT_ROOT, row["condition_path"]).resolve()),
            base_prompt=row["base_prompt"],
            action_trajectory=row["action_trajectory"],
        )
        normalized = dict(row)
        normalized["scene_summary"] = summary.get("summary", "")
        normalized["scene_summary_details"] = summary
        enriched.append(normalized)
    return enriched


class RealInfWorldMetricRunner:
    def __init__(
        self,
        *,
        settings,
        output_root: Path,
        num_sampling_steps: int,
        num_chunks: int,
        text_cfg_scale: float,
        seed: int,
        max_tasks: int,
        cond_window_frames: int,
        high_quality_save: bool,
        inference_cuda_visible_devices: str | None = None,
        prompt_editor: PromptEditor | None = None,
    ) -> None:
        self.settings = settings
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.eval_root = self.output_root / "eval_runs"
        self.eval_root.mkdir(parents=True, exist_ok=True)
        self.metric_cache_path = self.output_root / "metric_cache.json"
        self.metric_cache = self._load_cache()
        self.metric_suite = FiveMetricSuite(settings)
        self.num_sampling_steps = num_sampling_steps
        self.num_chunks = num_chunks
        self.text_cfg_scale = text_cfg_scale
        self.seed = seed
        self.max_tasks = max_tasks
        self.cond_window_frames = cond_window_frames
        self.high_quality_save = high_quality_save
        self.inference_cuda_visible_devices = inference_cuda_visible_devices
        self.prompt_editor = prompt_editor
        self.prompt_history: dict[str, list[dict[str, Any]]] = {}

    def _load_cache(self) -> dict[str, Any]:
        if self.metric_cache_path.exists():
            return _read_json(self.metric_cache_path)
        return {}

    def _save_cache(self) -> None:
        _write_json(self.metric_cache_path, self.metric_cache)

    def _build_task_payload(self, example, prompt_text: str, cache_key: str) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "prompt": prompt_text,
            "cond_path": getattr(example, "condition_path"),
            "action_path": getattr(example, "action_path"),
            "output_name": getattr(example, "output_name", None) or f"gepa_{cache_key}",
        }
        cond_clip_len = getattr(example, "cond_clip_len", None)
        cond_clip_mode = getattr(example, "cond_clip_mode", None)
        if cond_clip_len is not None:
            entry["cond_clip_len"] = cond_clip_len
        if cond_clip_mode is not None:
            entry["cond_clip_mode"] = cond_clip_mode
        action_start_mode = getattr(example, "action_start_mode", None)
        if action_start_mode:
            entry["action_start_mode"] = action_start_mode
        return entry

    def _find_video_path(self, run_output_dir: Path, output_name: str) -> Path:
        matches = sorted(run_output_dir.rglob(f"{output_name}.mp4"))
        if not matches:
            raise FileNotFoundError(
                f"No output video named {output_name}.mp4 found under {run_output_dir}"
            )
        return matches[0]

    def _run_inference(self, task_yaml_path: Path, run_output_dir: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "INFWORLD_PROMPTS_YAML": str(task_yaml_path),
                "INFWORLD_OUTPUT_BASE": str(run_output_dir),
                "INFWORLD_NUM_SAMPLING_STEPS": str(self.num_sampling_steps),
                "INFWORLD_NUM_CHUNKS": str(self.num_chunks),
                "INFWORLD_TEXT_CFG_SCALE": str(self.text_cfg_scale),
                "INFWORLD_SEED": str(self.seed),
                "INFWORLD_MAX_TASKS": str(self.max_tasks),
                "INFWORLD_COND_WINDOW_FRAMES": str(self.cond_window_frames),
                "INFWORLD_HIGH_QUALITY_SAVE": "1" if self.high_quality_save else "0",
            }
        )
        if self.inference_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.inference_cuda_visible_devices
            env["INFWORLD_CONTEXT_PARALLEL_SIZE"] = "1"
        return subprocess.run(
            [sys.executable, "scripts/infworld_inference.py"],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def evaluate_prompt(self, example, prompt_text: str) -> dspy.Prediction:
        task_num_chunks = getattr(example, "num_chunks", None) or self.num_chunks
        history_key = f"{getattr(example, 'condition_path')}::{getattr(example, 'action_path')}"
        original_prompt_text = prompt_text
        editor_details = None
        if self.prompt_editor is not None:
            editor_details = self.prompt_editor.revise(
                base_prompt=getattr(example, "base_prompt"),
                generated_prompt=prompt_text,
                scene_summary=getattr(example, "scene_summary", ""),
                action_trajectory=getattr(example, "action_trajectory"),
                target_traits=getattr(example, "target_traits"),
                previous_attempts=self.prompt_history.get(history_key, []),
            )
            prompt_text = editor_details.get("revised_prompt", prompt_text).strip() or prompt_text

        cache_key = _sha_text(
            getattr(example, "condition_path"),
            getattr(example, "action_path"),
            prompt_text,
            str(self.num_sampling_steps),
            str(task_num_chunks),
            str(self.text_cfg_scale),
            str(self.seed),
        )
        if cache_key in self.metric_cache:
            cached = self.metric_cache[cache_key]
            return dspy.Prediction(score=cached["score"], feedback=cached["feedback"])

        run_dir = self.eval_root / cache_key
        prompts_yaml_path = run_dir / "prompts.yaml"
        logs_path = run_dir / "inference.log"
        run_output_dir = run_dir / "outputs"
        task_entry = self._build_task_payload(example, prompt_text, cache_key)
        _write_yaml(prompts_yaml_path, {"prompts": [task_entry]})

        original_num_chunks = self.num_chunks
        self.num_chunks = int(task_num_chunks)
        try:
            proc = self._run_inference(prompts_yaml_path, run_output_dir)
        finally:
            self.num_chunks = original_num_chunks
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        logs_path.write_text(
            proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            feedback = (
                "InfiniteWorld inference failed. "
                f"Return code={proc.returncode}. See {logs_path}."
            )
            self.metric_cache[cache_key] = {
                "score": 0.0,
                "feedback": feedback,
                "video_path": None,
                "log_path": str(logs_path),
                "raw_prompt": original_prompt_text,
                "edited_prompt": prompt_text,
                "editor": editor_details,
            }
            self._save_cache()
            return dspy.Prediction(score=0.0, feedback=feedback)

        output_name = task_entry["output_name"]
        video_path = self._find_video_path(run_output_dir, output_name)
        metric_input = MetricInput(
            prompt=prompt_text,
            condition_path=task_entry["cond_path"],
            action_path=task_entry["action_path"],
        )
        suite_result = self.metric_suite.evaluate(metric_input, str(video_path))
        feedback_parts = []
        if editor_details and editor_details.get("critique"):
            feedback_parts.append(f"editor={editor_details['critique']}")
        feedback_parts.extend(
            [f"{metric.name}={metric.score:.3f}: {metric.feedback}" for metric in suite_result.metrics]
        )
        feedback = " ".join(feedback_parts)
        self.metric_cache[cache_key] = {
            "score": suite_result.total_score,
            "feedback": feedback,
            "video_path": str(video_path),
            "log_path": str(logs_path),
            "metrics": suite_result.as_dict(),
            "raw_prompt": original_prompt_text,
            "edited_prompt": prompt_text,
            "editor": editor_details,
        }
        self.prompt_history.setdefault(history_key, []).append(
            {
                "raw_prompt": original_prompt_text,
                "edited_prompt": prompt_text,
                "score": suite_result.total_score,
                "feedback": feedback,
            }
        )
        self._save_cache()
        return dspy.Prediction(score=suite_result.total_score, feedback=feedback)


def build_real_metric(runner: RealInfWorldMetricRunner):
    def _metric(example, pred, trace=None, pred_name=None, pred_trace=None):
        prompt_text = (getattr(pred, "optimized_prompt", "") or "").strip()
        if not prompt_text:
            return dspy.Prediction(score=0.0, feedback="The optimized prompt was empty.")
        return runner.evaluate_prompt(example, prompt_text)

    return _metric


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEPA hard-prompt optimization on real InfiniteWorld outputs.")
    parser.add_argument("--dataset", required=True, help="Path to a JSON dataset of prompt examples.")
    parser.add_argument("--auto", default=None, choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int, default=12)
    parser.add_argument("--val-dataset", help="Optional held-out validation dataset JSON.")
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=None,
        help="Run only a single dataset row by 0-based index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many dataset rows to keep after any index selection.",
    )
    parser.add_argument("--output-root", default="outputs/gepa_optimize", help="Where to store optimization artifacts.")
    parser.add_argument("--num-sampling-steps", type=int, default=2, help="Short inference steps for inner-loop GEPA.")
    parser.add_argument("--num-chunks", type=int, default=1, help="Short inference chunks for inner-loop GEPA.")
    parser.add_argument("--text-cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--cond-window-frames", type=int, default=0)
    parser.add_argument("--high-quality-save", action="store_true")
    parser.add_argument("--task-max-tokens", type=int, default=384)
    parser.add_argument("--reflection-max-tokens", type=int, default=2048)
    parser.add_argument("--candidate-budget", type=int, default=None, help="Approximate number of proposal candidates per iteration.")
    parser.add_argument("--iteration-budget", type=int, default=None, help="Approximate number of outer GEPA iterations.")
    parser.add_argument("--num-threads", type=int, default=1, help="DSPy evaluation threads. Keep at 1 for GPU stability unless intentionally parallelizing.")
    parser.add_argument("--inference-cuda-visible-devices", default=None, help="CUDA_VISIBLE_DEVICES value for InfiniteWorld inference subprocesses, e.g. '5'.")
    parser.add_argument("--vbench-device", default=None, help="Device for VBench metrics, e.g. 'cuda:4'.")
    parser.add_argument("--scene-summary-frames", type=int, default=30, help="How many uniformly sampled condition frames to send to the local VLM scene summarizer.")
    args = parser.parse_args()

    settings = load_settings()
    if args.vbench_device:
        settings.vbench_device = args.vbench_device
    task_lm = build_deepseek_lm(
        settings.deepseek_model,
        settings,
        temperature=0.7,
        max_tokens=args.task_max_tokens,
    )
    reflection_lm = build_deepseek_lm(
        settings.reflection_model,
        settings,
        temperature=0.7,
        max_tokens=args.reflection_max_tokens,
    )

    dspy.configure(lm=task_lm)
    output_root = Path(args.output_root)
    scene_summary_cache_dir = output_root / "scene_summaries"
    raw_train_rows = select_rows(
        _load_dataset_rows(Path(args.dataset)),
        dataset_index=args.dataset_index,
        limit=args.limit,
    )
    train_rows = enrich_rows_with_scene_summaries(
        raw_train_rows,
        settings=settings,
        cache_dir=scene_summary_cache_dir,
        sample_frames=args.scene_summary_frames,
    )
    trainset = load_examples_from_rows(train_rows)
    if args.val_dataset:
        raw_val_rows = select_rows(
            _load_dataset_rows(Path(args.val_dataset)),
            dataset_index=args.dataset_index,
            limit=args.limit,
        )
        val_rows = enrich_rows_with_scene_summaries(
            raw_val_rows,
            settings=settings,
            cache_dir=scene_summary_cache_dir,
            sample_frames=args.scene_summary_frames,
        )
        valset = load_examples_from_rows(val_rows)
    else:
        valset = trainset
    program = InfWorldHardPromptProgram()

    effective_metric_calls = args.max_metric_calls
    if args.candidate_budget is not None or args.iteration_budget is not None:
        if args.candidate_budget is None or args.iteration_budget is None:
            raise ValueError("Use --candidate-budget and --iteration-budget together.")
        valset_size = len(valset)
        trainset_size = len(trainset)
        eval_set_size = trainset_size + valset_size
        # User-facing approximation: baseline full pass + candidate passes across requested iterations.
        effective_metric_calls = eval_set_size * (1 + args.candidate_budget * args.iteration_budget)

    runner = RealInfWorldMetricRunner(
        settings=settings,
        output_root=output_root,
        num_sampling_steps=args.num_sampling_steps,
        num_chunks=args.num_chunks,
        text_cfg_scale=args.text_cfg_scale,
        seed=args.seed,
        max_tasks=args.max_tasks,
        cond_window_frames=args.cond_window_frames,
        high_quality_save=args.high_quality_save,
        inference_cuda_visible_devices=args.inference_cuda_visible_devices,
        prompt_editor=PromptEditor(settings),
    )
    optimizer_kwargs = {
        "metric": build_real_metric(runner),
        "reflection_lm": reflection_lm,
        "track_stats": True,
        "num_threads": args.num_threads,
    }
    if args.auto is not None:
        optimizer_kwargs["auto"] = args.auto
    else:
        optimizer_kwargs["max_metric_calls"] = effective_metric_calls

    optimizer = dspy.GEPA(**optimizer_kwargs)
    optimized = optimizer.compile(program, trainset=trainset, valset=valset)

    print("optimization_complete")
    if args.candidate_budget is not None and args.iteration_budget is not None:
        print("requested_candidate_budget:", args.candidate_budget)
        print("requested_iteration_budget:", args.iteration_budget)
        print("effective_max_metric_calls:", effective_metric_calls)
    print("artifact_root:", str(Path(args.output_root).resolve()))
    print("metric_cache:", str((Path(args.output_root) / "metric_cache.json").resolve()))
    if trainset:
        example = trainset[0]
        pred = optimized(
            base_prompt=example.base_prompt,
            condition_context=example.condition_context,
            scene_summary=example.scene_summary,
            action_trajectory=example.action_trajectory,
            target_traits=example.target_traits,
        )
        print("optimized_prompt:", getattr(pred, "optimized_prompt", ""))
    if hasattr(optimized, "detailed_results"):
        print("detailed_results_available: True")


if __name__ == "__main__":
    main()
