from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import dspy


class RewritePrompt(dspy.Signature):
    """Rewrite an InfiniteWorld prompt so it is concrete, cinematic, and faithful to the scene and trajectory."""

    base_prompt = dspy.InputField(desc="The current or baseline InfiniteWorld text prompt.")
    condition_context = dspy.InputField(desc="Short description of the starting image or clip context.")
    scene_summary = dspy.InputField(desc="Structured VLM summary of the conditioning video or image.")
    action_trajectory = dspy.InputField(desc="Natural-language summary of movement and camera trajectory.")
    target_traits = dspy.InputField(desc="What the optimized prompt should emphasize.")
    optimized_prompt = dspy.OutputField(desc="A stronger hard prompt for InfiniteWorld inference.")


class InfWorldHardPromptProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rewrite = dspy.ChainOfThought(RewritePrompt)

    def forward(
        self,
        *,
        base_prompt: str,
        condition_context: str,
        scene_summary: str,
        action_trajectory: str,
        target_traits: str,
    ):
        return self.rewrite(
            base_prompt=base_prompt,
            condition_context=condition_context,
            scene_summary=scene_summary,
            action_trajectory=action_trajectory,
            target_traits=target_traits,
        )


def keyword_feedback_metric(example, pred):
    text = getattr(pred, "optimized_prompt", "") or ""
    text_l = text.lower()
    required = list(getattr(example, "required_keywords", []))
    banned = list(getattr(example, "banned_keywords", []))

    hits = [kw for kw in required if kw.lower() in text_l]
    misses = [kw for kw in required if kw.lower() not in text_l]
    violations = [kw for kw in banned if kw.lower() in text_l]

    score = 0.0
    if required:
        score += len(hits) / len(required)
    else:
        score += 1.0
    if violations:
        score -= min(1.0, len(violations) * 0.25)
    score = max(0.0, min(1.0, score))

    feedback_parts = []
    if hits:
        feedback_parts.append(f"Kept desired traits: {', '.join(hits)}.")
    if misses:
        feedback_parts.append(f"Missing desired traits: {', '.join(misses)}.")
    if violations:
        feedback_parts.append(f"Avoid banned terms: {', '.join(violations)}.")
    if not feedback_parts:
        feedback_parts.append("Prompt looks aligned with the requested traits.")

    return dspy.Prediction(score=score, feedback=" ".join(feedback_parts))


def summarize_action_runs(action_json_path: str | Path) -> str:
    with open(action_json_path, "r", encoding="utf-8") as f:
        actions = json.load(f)

    if not actions:
        return "No actions."

    move_counts = Counter(a["move"] for a in actions)
    view_counts = Counter(a["view"] for a in actions)

    runs = []
    prev = None
    start = 0
    for i, action in enumerate(actions):
        cur = (action["move"], action["view"])
        if prev is None:
            prev = cur
            start = i
        elif cur != prev:
            runs.append((start, i - 1, prev))
            start = i
            prev = cur
    runs.append((start, len(actions) - 1, prev))

    run_lines = [
        f"frames {s}-{e}: move={move}, view={view}" for s, e, (move, view) in runs[:10]
    ]
    return (
        f"{len(actions)} action frames. "
        f"Move counts: {dict(move_counts)}. "
        f"View counts: {dict(view_counts)}. "
        f"Runs: {'; '.join(run_lines)}"
    )
