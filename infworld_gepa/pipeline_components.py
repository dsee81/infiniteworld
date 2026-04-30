from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import GepaSettings, build_deepseek_client, build_local_vlm_client
from .video_utils import frame_to_data_url, sample_condition_frames


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]
    return json.loads(text)


def _path_signature(path: str | Path) -> str:
    p = Path(path)
    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(p.resolve()).encode("utf-8"))
    h.update(str(stat.st_mtime_ns).encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    return h.hexdigest()[:24]


class SceneSummarizer:
    def __init__(
        self,
        settings: GepaSettings,
        *,
        cache_dir: str | Path,
        model: str | None = None,
        sample_frames: int = 30,
    ) -> None:
        self.settings = settings
        self.client = build_local_vlm_client(settings)
        self.model = model or settings.local_vlm_model
        self.sample_frames = sample_frames
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def summarize(
        self,
        condition_path: str,
        *,
        base_prompt: str = "",
        action_trajectory: str = "",
    ) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{_path_signature(condition_path)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        frames = sample_condition_frames(condition_path, max_frames=self.sample_frames)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Summarize this conditioning scene for prompt optimization. "
                    "Return JSON with keys: summary, location_type, key_landmarks, spatial_layout, "
                    "lighting, moving_entities, continuity_anchors, camera_start_pose. "
                    "Keep summary concise but concrete."
                    f"\nBase prompt reference: {base_prompt}"
                    f"\nTrajectory reference: {action_trajectory}"
                ),
            }
        ]
        for idx, frame in enumerate(frames):
            content.append({"type": "text", "text": f"Condition frame {idx + 1}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": frame_to_data_url(frame, max_side=384, jpeg_quality=60)
                    },
                }
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict scene summarizer for video-generation conditioning inputs. "
                        "Use only visible evidence from the frames and return JSON only."
                    ),
                },
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        data = _safe_json_loads(response.choices[0].message.content)
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data


class PromptEditor:
    def __init__(
        self,
        settings: GepaSettings,
        *,
        model: str | None = None,
        history_limit: int = 5,
        max_tokens: int = 900,
    ) -> None:
        self.settings = settings
        self.client = build_deepseek_client(settings)
        self.model = (model or settings.reflection_model).replace("openai/", "")
        self.history_limit = history_limit
        self.max_tokens = max_tokens

    def revise(
        self,
        *,
        base_prompt: str,
        generated_prompt: str,
        scene_summary: str,
        action_trajectory: str,
        target_traits: str,
        previous_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        trimmed_history = previous_attempts[-self.history_limit :]
        history_text = json.dumps(trimmed_history, indent=2) if trimmed_history else "[]"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a prompt editor for InfiniteWorld. Critique the current candidate prompt using "
                        "the scene summary, trajectory, and prior attempted prompts with their results. "
                        "Return JSON only with keys critique, keep, revised_prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Base prompt:\n{base_prompt}\n\n"
                        f"Scene summary:\n{scene_summary}\n\n"
                        f"Trajectory:\n{action_trajectory}\n\n"
                        f"Target traits:\n{target_traits}\n\n"
                        f"Current generated prompt:\n{generated_prompt}\n\n"
                        f"Previous attempts and results:\n{history_text}\n\n"
                        "If the current prompt is already strong, set keep=true and return the original prompt as revised_prompt. "
                        "Otherwise, make the revised_prompt more explicit about camera motion, scene continuity, and key landmarks."
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=self.max_tokens,
        )
        data = _safe_json_loads(response.choices[0].message.content)
        revised = str(data.get("revised_prompt", generated_prompt)).strip() or generated_prompt
        data["revised_prompt"] = revised
        data["keep"] = bool(data.get("keep", False))
        data["critique"] = str(data.get("critique", "")).strip()
        return data
