# InfiniteWorld GEPA Prompt Optimization Scaffold

This repo includes a small DSPy/GEPA scaffold for hard-prompt optimization.

## DeepSeek key location

Create a repo-root `.env` file next to `README.md`:

```bash
cp .env.example .env
```

Then fill in:

```env
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=openai/deepseek-chat
GEPA_REFLECTION_MODEL=openai/deepseek-chat
```

## Files

- `infworld_gepa/config.py`: `.env` loading and DeepSeek-backed `dspy.LM` construction.
- `infworld_gepa/hard_prompt.py`: DSPy signature/module for rewriting InfiniteWorld prompts.
- `scripts/gepa_smoke_test.py`: offline and optional live connectivity smoke test.
- `scripts/gepa_optimize_prompts.py`: starter CLI for GEPA hard-prompt optimization.
