from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dspy
from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GepaSettings:
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "openai/deepseek-chat"
    reflection_model: str = "openai/deepseek-chat"
    deepseek_timeout_s: float = 120.0
    local_vlm_api_base: str = "http://127.0.0.1:8001/v1"
    local_vlm_api_key: str = "EMPTY"
    local_vlm_model: str = "Qwen/Qwen2.5-VL-32B-Instruct"
    local_vlm_timeout_s: float = 120.0
    metric_cache_dir: str = str(REPO_ROOT / "outputs" / "gepa_metrics")
    vbench_device: str = "cuda:4"

    @property
    def has_api_key(self) -> bool:
        return bool(self.deepseek_api_key.strip())

    @property
    def has_local_vlm(self) -> bool:
        return bool(self.local_vlm_api_base.strip()) and bool(self.local_vlm_model.strip())


def load_settings() -> GepaSettings:
    load_dotenv(REPO_ROOT / ".env")
    return GepaSettings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "openai/deepseek-chat"),
        reflection_model=os.getenv("GEPA_REFLECTION_MODEL", "openai/deepseek-chat"),
        deepseek_timeout_s=float(os.getenv("DEEPSEEK_TIMEOUT_S", "120")),
        local_vlm_api_base=os.getenv("LOCAL_VLM_API_BASE", "http://127.0.0.1:8001/v1"),
        local_vlm_api_key=os.getenv("LOCAL_VLM_API_KEY", "EMPTY"),
        local_vlm_model=os.getenv("LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"),
        local_vlm_timeout_s=float(os.getenv("LOCAL_VLM_TIMEOUT_S", "120")),
        metric_cache_dir=os.getenv(
            "GEPA_METRIC_CACHE_DIR", str(REPO_ROOT / "outputs" / "gepa_metrics")
        ),
        vbench_device=os.getenv("GEPA_VBENCH_DEVICE", "cuda:4"),
    )


def build_deepseek_lm(
    model: str,
    settings: GepaSettings,
    *,
    temperature: float = 0.7,
    max_tokens: int = 512,
    cache: bool = False,
) -> dspy.LM:
    if not settings.has_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Put it in the repo-root .env file first."
        )

    return dspy.LM(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
        api_key=settings.deepseek_api_key,
        api_base=settings.deepseek_base_url,
    )


def build_local_vlm_client(settings: GepaSettings) -> OpenAI:
    if not settings.has_local_vlm:
        raise RuntimeError(
            "LOCAL_VLM_API_BASE and LOCAL_VLM_MODEL must be set to use the local VLM judge."
        )
    return OpenAI(
        api_key=settings.local_vlm_api_key,
        base_url=settings.local_vlm_api_base,
        timeout=settings.local_vlm_timeout_s,
    )


def build_deepseek_client(settings: GepaSettings) -> OpenAI:
    if not settings.has_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Put it in the repo-root .env file first."
        )
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=settings.deepseek_timeout_s,
    )
