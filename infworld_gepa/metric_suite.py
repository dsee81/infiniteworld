from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from openai import OpenAI

from .config import GepaSettings, build_local_vlm_client
from .video_utils import (
    ActionStats,
    frame_to_data_url,
    load_action_stats,
    sample_condition_frames,
    sample_video_frames,
)


@dataclass
class MetricInput:
    prompt: str
    condition_path: str
    action_path: str


@dataclass
class MetricResult:
    name: str
    score: float
    feedback: str
    details: dict


@dataclass
class MetricSuiteResult:
    total_score: float
    metrics: list[MetricResult]

    def as_dict(self) -> dict:
        return {
            "total_score": self.total_score,
            "metrics": [
                {
                    "name": m.name,
                    "score": m.score,
                    "feedback": m.feedback,
                    "details": m.details,
                }
                for m in self.metrics
            ],
        }


class VideoMetric(Protocol):
    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        ...


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_json_loads(text: str) -> dict:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


class LocalVLMJudge:
    def __init__(
        self,
        settings: GepaSettings,
        *,
        client: OpenAI | None = None,
        model: str | None = None,
        max_generated_frames: int = 8,
        max_condition_frames: int = 3,
    ) -> None:
        self.settings = settings
        self.client = client or build_local_vlm_client(settings)
        self.model = model or settings.local_vlm_model
        self.max_generated_frames = max_generated_frames
        self.max_condition_frames = max_condition_frames

    def _build_multimodal_message(
        self,
        intro_text: str,
        condition_frames: list[np.ndarray],
        generated_frames: list[np.ndarray],
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": intro_text}]
        for idx, frame in enumerate(condition_frames):
            content.append({"type": "text", "text": f"Condition frame {idx + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame)}})
        for idx, frame in enumerate(generated_frames):
            content.append({"type": "text", "text": f"Generated frame {idx + 1}:"})
            content.append({"type": "image_url", "image_url": {"url": frame_to_data_url(frame)}})
        return content

    def _chat_json(self, system_prompt: str, content: list[dict]) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return _safe_json_loads(response.choices[0].message.content)

    def judge_trajectory(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        action_stats = load_action_stats(metric_input.action_path)
        condition_frames = sample_condition_frames(
            metric_input.condition_path, max_frames=self.max_condition_frames
        )
        generated_frames = sample_video_frames(video_path, max_frames=self.max_generated_frames)
        content = self._build_multimodal_message(
            (
                f"Prompt: {metric_input.prompt}\n"
                f"Trajectory summary: {action_stats.summary}\n"
                "Score how well the generated video follows the intended motion path and camera path. "
                "Return JSON with keys overall_score, motion_path_score, camera_path_score, confidence, evidence, issues."
            ),
            condition_frames,
            generated_frames,
        )
        system_prompt = (
            "You are a strict video-generation evaluator. Use only visible evidence from the provided "
            "frames plus the explicit trajectory specification. Scores must be between 0 and 1. "
            "Return JSON only."
        )
        data = self._chat_json(system_prompt, content)
        score = _clamp01(data.get("overall_score", 0.0))
        feedback = "; ".join(data.get("issues", [])) or "Trajectory looks acceptable."
        return MetricResult(
            name="traj_adherence",
            score=score,
            feedback=feedback,
            details=data,
        )

    def judge_semantics(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        condition_frames = sample_condition_frames(
            metric_input.condition_path, max_frames=self.max_condition_frames
        )
        generated_frames = sample_video_frames(video_path, max_frames=self.max_generated_frames)
        content = self._build_multimodal_message(
            (
                f"Prompt: {metric_input.prompt}\n"
                "Score semantic consistency between the condition input and the generated video. "
                "Consider scene identity, objects, layout, and whether the generated clip still matches the prompt. "
                "Return JSON with keys overall_score, scene_match_score, prompt_match_score, confidence, evidence, issues."
            ),
            condition_frames,
            generated_frames,
        )
        system_prompt = (
            "You are a strict semantic-faithfulness evaluator for image-to-video or video-to-video generation. "
            "Use only visible evidence from the provided frames. Scores must be between 0 and 1. "
            "Return JSON only."
        )
        data = self._chat_json(system_prompt, content)
        score = _clamp01(data.get("overall_score", 0.0))
        feedback = "; ".join(data.get("issues", [])) or "Semantic consistency looks acceptable."
        return MetricResult(
            name="semantic_consistency",
            score=score,
            feedback=feedback,
            details=data,
        )


class MockVLMJudge(LocalVLMJudge):
    def __init__(self) -> None:
        self.settings = None
        self.client = None
        self.model = "mock-vlm"
        self.max_generated_frames = 0
        self.max_condition_frames = 0

    def judge_trajectory(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        return MetricResult(
            name="traj_adherence",
            score=0.78,
            feedback="Mock judge: trajectory mostly matches with minor camera drift.",
            details={
                "overall_score": 0.78,
                "motion_path_score": 0.8,
                "camera_path_score": 0.76,
                "confidence": 0.6,
                "evidence": ["Forward motion is visible.", "Camera turn is somewhat weaker later."],
                "issues": ["Late camera turn is weaker than requested."],
            },
        )

    def judge_semantics(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        return MetricResult(
            name="semantic_consistency",
            score=0.81,
            feedback="Mock judge: scene identity is mostly preserved.",
            details={
                "overall_score": 0.81,
                "scene_match_score": 0.82,
                "prompt_match_score": 0.8,
                "confidence": 0.62,
                "evidence": ["Campus walkway look is preserved.", "Lighting remains plausible."],
                "issues": [],
            },
        )


class VBenchDimensionMetric:
    def __init__(
        self,
        settings: GepaSettings,
        dimension: str,
        *,
        device: str = "cuda",
        read_frame: bool = False,
        local: bool = False,
    ) -> None:
        self.settings = settings
        self.dimension = dimension
        self.device = device
        self.read_frame = read_frame
        self.local = local

    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        import torch
        from vbench import VBench

        full_info = Path(".venv/lib/python3.10/site-packages/vbench/VBench_full_info.json").resolve()
        output_root = Path(self.settings.metric_cache_dir) / "vbench" / self.dimension
        output_root.mkdir(parents=True, exist_ok=True)
        name = f"{Path(video_path).stem}_{self.dimension}"
        evaluator = VBench(self.device, str(full_info), str(output_root))
        original_torch_load = torch.load

        def _torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_compat
        try:
            results = evaluator.evaluate(
                videos_path=video_path,
                name=name,
                prompt_list=[metric_input.prompt],
                dimension_list=[self.dimension],
                local=self.local,
                read_frame=self.read_frame,
                mode="custom_input",
                imaging_quality_preprocessing_mode="shorter_centercrop",
            )
        finally:
            torch.load = original_torch_load

        if results is None:
            results_path = output_root / f"{name}_eval_results.json"
            if not results_path.exists():
                raise RuntimeError(
                    f"VBench did not return results in-memory and no results file was found at {results_path}."
                )
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)

        raw = results[self.dimension]
        if isinstance(raw, tuple) and len(raw) >= 1:
            score = float(raw[0])
            details = {"raw": raw[1] if len(raw) > 1 else None}
        elif isinstance(raw, list) and len(raw) >= 1:
            score = float(raw[0])
            details = {"raw": raw[1] if len(raw) > 1 else None}
        else:
            score = float(raw)
            details = {"raw": raw}
        return MetricResult(
            name=self.dimension,
            score=_clamp01(score),
            feedback=f"VBench {self.dimension} score={score:.4f}",
            details=details,
        )


class OpticalFlowMetric:
    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        action_stats = load_action_stats(metric_input.action_path)
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        cap.release()

        if len(frames) < 2:
            return MetricResult(
                name="optical_flow",
                score=0.0,
                feedback="Not enough frames to compute optical flow.",
                details={},
            )

        magnitudes = []
        h, w = frames[0].shape[:2]
        diag = math.sqrt(h * h + w * w)
        for prev, curr in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(mag) / max(diag, 1.0)))

        actual = float(np.percentile(magnitudes, 75))
        expected = 0.002 + 0.02 * action_stats.non_noop_ratio
        score = 1.0 - min(1.0, abs(actual - expected) / max(expected, 1e-6))
        return MetricResult(
            name="optical_flow",
            score=_clamp01(score),
            feedback=(
                f"Observed normalized flow={actual:.6f}, expected around {expected:.6f} "
                f"from action activity ratio {action_stats.non_noop_ratio:.3f}."
            ),
            details={
                "actual_flow": actual,
                "expected_flow": expected,
                "per_frame_flow_mean": magnitudes[:32],
            },
        )


class TrajectoryAdherenceMetric:
    def __init__(self, judge: LocalVLMJudge) -> None:
        self.judge = judge

    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        return self.judge.judge_trajectory(metric_input, video_path)


class SemanticConsistencyMetric:
    def __init__(self, judge: LocalVLMJudge) -> None:
        self.judge = judge

    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricResult:
        return self.judge.judge_semantics(metric_input, video_path)


class FiveMetricSuite:
    def __init__(
        self,
        settings: GepaSettings,
        *,
        judge: LocalVLMJudge | None = None,
        vbench_device: str | None = None,
    ) -> None:
        resolved_vbench_device = vbench_device or settings.vbench_device
        self.judge = judge or LocalVLMJudge(settings)
        self.metrics: list[tuple[VideoMetric, float]] = [
            (VBenchDimensionMetric(settings, "aesthetic_quality", device=resolved_vbench_device), 0.10),
            (VBenchDimensionMetric(settings, "motion_smoothness", device=resolved_vbench_device), 0.15),
            (OpticalFlowMetric(), 0.15),
            (TrajectoryAdherenceMetric(self.judge), 0.35),
            (SemanticConsistencyMetric(self.judge), 0.25),
        ]

    def evaluate(self, metric_input: MetricInput, video_path: str) -> MetricSuiteResult:
        results: list[MetricResult] = []
        total = 0.0
        total_weight = 0.0
        for metric, weight in self.metrics:
            try:
                result = metric.evaluate(metric_input, video_path)
            except Exception as exc:
                metric_name = getattr(metric, "dimension", None) or getattr(metric, "__class__", type(metric)).__name__
                result = MetricResult(
                    name=str(metric_name),
                    score=0.0,
                    feedback=f"Metric failed: {exc}",
                    details={"error": type(exc).__name__, "message": str(exc)},
                )
            results.append(result)
            total += result.score * weight
            total_weight += weight
        return MetricSuiteResult(
            total_score=(total / total_weight) if total_weight else 0.0,
            metrics=results,
        )
