from __future__ import annotations

import argparse
import os
import sys

import dspy
from dspy.utils.dummies import DummyLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld_gepa.config import build_deepseek_lm, load_settings
from infworld_gepa.hard_prompt import InfWorldHardPromptProgram, keyword_feedback_metric


def run_offline_smoke() -> None:
    offline_lm = DummyLM(
        [
            {
                "reasoning": "Keep the prompt grounded in the scene and make the motion explicit.",
                "optimized_prompt": (
                    "A serene campus walkway lined with modern glass buildings, "
                    "green ivy, soft dappled sunlight, and a steady forward dolly "
                    "through the path with stable composition and realistic detail."
                ),
            }
        ]
    )
    dspy.configure(lm=offline_lm)
    program = InfWorldHardPromptProgram()
    pred = program(
        base_prompt="A serene campus walkway.",
        condition_context="Glass buildings, ivy, empty benches, maple trees.",
        scene_summary="A campus walkway with ivy-covered glass buildings, benches, and soft daylight.",
        action_trajectory="The camera moves forward with no view rotation.",
        target_traits="Preserve realism, continuity, and explicit forward motion.",
    )
    example = dspy.Example(
        base_prompt="A serene campus walkway.",
        condition_context="Glass buildings, ivy, empty benches, maple trees.",
        scene_summary="A campus walkway with ivy-covered glass buildings, benches, and soft daylight.",
        action_trajectory="The camera moves forward with no view rotation.",
        target_traits="Preserve realism, continuity, and explicit forward motion.",
        required_keywords=["forward", "realistic", "ivy"],
        banned_keywords=["spaceship"],
    ).with_inputs("base_prompt", "condition_context", "scene_summary", "action_trajectory", "target_traits")
    metric = keyword_feedback_metric(example, pred)
    print("offline_prediction:", pred.optimized_prompt)
    print("offline_score:", metric.score)
    print("offline_feedback:", metric.feedback)


def run_live_smoke() -> None:
    settings = load_settings()
    live_lm = build_deepseek_lm(
        settings.deepseek_model,
        settings,
        temperature=0.0,
        max_tokens=64,
        cache=False,
    )
    prompt = "Reply with exactly the single word OK."
    messages = [{"role": "user", "content": prompt}]
    response = live_lm(messages=messages)
    text = response[0] if isinstance(response, list) else str(response)
    print("live_response:", text.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for DSPy + GEPA scaffold.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Also perform a tiny live DeepSeek API call using .env credentials.",
    )
    args = parser.parse_args()

    settings = load_settings()
    print("dspy_version:", dspy.__version__)
    print("has_gepa:", hasattr(dspy, "GEPA"))
    print("env_file: .env at repo root")
    print("api_key_present:", settings.has_api_key)
    print("deepseek_model:", settings.deepseek_model)
    print("reflection_model:", settings.reflection_model)

    run_offline_smoke()

    if args.live:
        run_live_smoke()
    elif settings.has_api_key:
        print("live_smoke_skipped: pass --live to test the DeepSeek API key.")
    else:
        print("live_smoke_skipped: DEEPSEEK_API_KEY is not set yet.")


if __name__ == "__main__":
    main()
