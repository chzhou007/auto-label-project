# AutoLabel Generation Module

This folder contains the industrial anomaly image-generation and auto-label pipeline for `autolabel`.
Run it from the repository root with the package entrypoint; the old `vlm_wan_autolabel/src/main.py` entry remains as a compatibility wrapper.

## Install

```bash
python -m pip install -r autolabel/modules/generation/requirements.txt
```

## Environment

Do not hardcode API keys. The CLI automatically loads `.env` from the repository root and `autolabel/modules/generation/.env` before reading environment variables. Current shell exports still take precedence over `.env` values.

Create a local `.env` file, which is ignored by git:

```bash
DASHSCOPE_API_KEY=your_api_key_here
QWEN_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_WAN_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

If live Wan editing is unavailable, use `--dry-run` to validate grid generation, dummy Qwen selection, local synthetic edit, diff localization, mask/crop writing, metadata construction, and RequiredFieldsV1 validation.

## API Notes

The implementation follows the current Alibaba Cloud Model Studio documentation:

- Qwen API reference: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference/
- Wan2.7 image generation/editing API reference: https://www.alibabacloud.com/help/en/model-studio/wan-image-generation-and-editing-api-reference
- Model Studio rate limits: https://www.alibabacloud.com/help/en/model-studio/rate-limit

Key maintenance notes:

- Qwen supports OpenAI-compatible vision calls and structured output with `response_format={"type":"json_object"}` when the prompt explicitly asks for JSON. Image token cost is tied to input pixels; the pipeline now sends smaller VLM previews by default and also passes `max_pixels`.
- Model Studio rate limits are account/model scoped. 429 can be caused by RPM/TPM or short request bursts, so the clients retry 429/5xx/timeout with exponential backoff and expose separate VLM/Wan concurrency caps.
- Wan image APIs support both synchronous multimodal generation and asynchronous image-generation tasks. Async tasks return `task_id`, are polled through `/api/v1/tasks/{task_id}`, and returned image URLs expire, so generated files are downloaded immediately.

## Supported Anomaly Types

```text
diesel_leak
oil_leak
coolant_leak
water_leak
```

`water_leak` is clear-water leakage: pure white / transparent whitish / bright reflective clean water. It must not look like yellow diesel, silver-green coolant, black-brown oil, fluorescent liquid, muddy water, foam, high-pressure spray, or flooding.
It is now prompted as an extremely small, early, controlled local transparent wet trace or thin water film. A visible wall/pipe leak point is not required; a small bare wet mark on floor or near an equipment base is allowed.

## Flow

```text
clean industrial image
  -> 4x4 grid preview for Qwen only
  -> Qwen selects top3 coarse grid candidates outside timestamp/site-label overlay regions
  -> no fine 3x3 grid by default
  -> top3 coarse candidates are tried sequentially or speculatively
  -> Wan edits the clean original image with bbox_list
  -> diff localization creates mask and final_bbox
  -> crop, metadata, RequiredFieldsV1 validation, optional Label Studio export
```

Wan never receives the grid preview. Final `objects[].box` is always the diff localizer `final_bbox`, not a grid bbox.

## Run

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --vlm-model qwen3.6-plus \
  --image-model wan2.7-image-pro \
  --mode balanced \
  --grid-layout 4x4 \
  --edit-bbox-expand-ratio 0.20 \
  --crop-expand-ratio 0.10 \
  --num-generations-per-candidate 1 \
  --candidate-grid-count 3 \
  --max-candidate-grids 3 \
  --sensitive-top-left-ratio 0.30,0.12 \
  --sensitive-bottom-right-ratio 0.42,0.12 \
  --workers 4 \
  --vlm-concurrency 2 \
  --wan-submit-concurrency 4 \
  --wan-poll-concurrency 8 \
  --download-concurrency 8 \
  --skip-existing \
  --benchmark
```

Speed mode:

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --mode speed \
  --workers 6 \
  --vlm-concurrency 2 \
  --wan-submit-concurrency 6 \
  --candidate-strategy sequential \
  --skip-existing \
  --benchmark
```

Quality mode:

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --mode quality \
  --candidate-strategy speculative \
  --speculative-top-k 2 \
  --skip-existing \
  --benchmark
```

Dry-run:

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks_water_leak_test.csv \
  --image-root data/raw/images \
  --output-root data/processed \
  --vlm-model qwen3.6-plus \
  --image-model wan2.7-image-pro \
  --grid-layout 4x4 \
  --edit-bbox-expand-ratio 0.20 \
  --crop-expand-ratio 0.10 \
  --num-generations-per-candidate 1 \
  --dry-run \
  --benchmark \
  --export-labelstudio
```

Benchmark summaries are written to:

```text
data/processed/metadata/logs/run_summary_*.json
```
