from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infworld_gepa.config import load_settings
from infworld_gepa.metric_suite import FiveMetricSuite, MetricInput, MockVLMJudge, OpticalFlowMetric


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for InfiniteWorld GEPA metric suite.")
    parser.add_argument(
        "--video",
        default="outputs/quick_sample_flash/infworld-ckpt0-step2-cfg5.0/0000_A_serene_campus_walkway_lined_.mp4",
        help="Video path to score.",
    )
    parser.add_argument(
        "--use-live-vlm",
        action="store_true",
        help="Use the configured local VLM instead of the mock judge.",
    )
    parser.add_argument(
        "--with-vbench",
        action="store_true",
        help="Run the full five-metric suite, including VBench metrics.",
    )
    args = parser.parse_args()

    settings = load_settings()
    metric_input = MetricInput(
        prompt="A serene campus walkway lined with modern glass buildings, green ivy, and soft dappled sunlight.",
        condition_path="assets/example_case/0001.jpg",
        action_path="assets/example_case/0001.json",
    )
    video_path = args.video

    print("video_exists:", Path(video_path).exists())
    print("local_vlm_model:", settings.local_vlm_model)
    print("local_vlm_api_base:", settings.local_vlm_api_base)

    optical = OpticalFlowMetric().evaluate(metric_input, video_path)
    print("optical_flow_score:", optical.score)
    print("optical_flow_feedback:", optical.feedback)

    judge = None if args.use_live_vlm else MockVLMJudge()

    if args.with_vbench:
        suite = FiveMetricSuite(settings, judge=judge)
        result = suite.evaluate(metric_input, video_path)
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print("vbench_skipped: pass --with-vbench to run aesthetic and motion_smoothness.")
        if args.use_live_vlm:
            from infworld_gepa.metric_suite import LocalVLMJudge, SemanticConsistencyMetric, TrajectoryAdherenceMetric

            live_judge = LocalVLMJudge(settings)
            traj = TrajectoryAdherenceMetric(live_judge).evaluate(metric_input, video_path)
            semantic = SemanticConsistencyMetric(live_judge).evaluate(metric_input, video_path)
        else:
            from infworld_gepa.metric_suite import SemanticConsistencyMetric, TrajectoryAdherenceMetric

            traj = TrajectoryAdherenceMetric(MockVLMJudge()).evaluate(metric_input, video_path)
            semantic = SemanticConsistencyMetric(MockVLMJudge()).evaluate(metric_input, video_path)
        print(
            json.dumps(
                {
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
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
