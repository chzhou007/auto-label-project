# VLM + Wan Industrial Anomaly AutoLabel Pipeline

The implementation now lives in `autolabel/modules/generation/`. This folder keeps compatibility wrappers for the previous `python src/main.py` entry.
It should still be run and changed as an isolated generation module; avoid modifying direct-labeling services, shared APIs, frontend/backend code, or global project structure unless integration requires it.

## Flow

```text
normal industrial image
  -> 4x4 grid preview for Qwen VLM only
  -> Qwen3.6-Plus selects top3 coarse candidate grid cells outside timestamp/site-label overlay regions
  -> no local 3x3 refinement by default
  -> top3 coarse grids become expanded_edit_bbox candidates
  -> Wan2.7-Image-Pro edits the clean original image with bbox_list
  -> original/generated ROI difference creates mask and final_bbox
  -> anomaly crop is saved
  -> optional Qwen VLM review
  -> AutoLabelSample metadata is built and strictly validated
  -> optional Label Studio export
```

Wan input is always the clean original image, never the grid preview. Final `objects[].box` comes from diff localization, not from the grid bbox or expanded edit bbox.
Fine grid remains available only as an explicit compatibility option; the production default is top3 coarse-grid generation.

## Install

From this module folder:

```bash
python -m pip install -r requirements.txt
```

Do not use plain `pip install` in this project environment.

## Environment

Copy `.env.example` to your local environment manager and set:

```bash
DASHSCOPE_API_KEY=your_api_key_here
QWEN_VLM_MODEL=qwen3.6-plus
WAN_IMAGE_MODEL=wan2.7-image-pro
```

Live Wan calls require `WAN_IMAGE_EDIT_ENDPOINT` or `DASHSCOPE_WAN_ENDPOINT` in this local wrapper. Without a configured endpoint, use `--dry-run` to validate the complete local data path.

## tasks.csv

Minimum columns:

```csv
task_id,image_path,anomaly_type,source_type
```

Supported `anomaly_type` values:

```text
diesel_leak
oil_leak
coolant_leak
water_leak
```

`water_leakage` is accepted as an input alias and normalized to `water_leak`.

Optional columns include `collection_batch,site,building,floor,room_name,room_type,severity_level`.
Defaults: `source_type=generated`, `severity_level=early`, `room_type=generator_room`.

## Run

Preferred package entry from the repository root:

```bash
python -m autolabel.modules.generation.main \
  --tasks data/tasks.csv \
  --image-root data/images \
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
  --benchmark
```

From this module folder:

```bash
python src/main.py \
  --tasks data/tasks.csv \
  --image-root data/images \
  --output-root data/processed \
  --vlm-model qwen3.6-plus \
  --image-model wan2.7-image-pro \
  --mode balanced \
  --grid-layout 4x4 \
  --edit-bbox-expand-ratio 0.20 \
  --crop-expand-ratio 0.10 \
  --num-generations-per-candidate 1 \
  --candidate-grid-count 3 \
  --max-candidate-grids 3
```

Offline validation:

```bash
python src/main.py \
  --tasks data/tasks.csv \
  --image-root data/images \
  --output-root data/processed \
  --dry-run \
  --benchmark \
  --export-labelstudio
```

Optional flags:

```text
--enable-fine-grid
--enable-vlm-review
--export-labelstudio
--dry-run
--mode speed|balanced|quality
--candidate-grid-count 3
--max-candidate-grids 3
--sensitive-top-left-ratio 0.30,0.12
--sensitive-bottom-right-ratio 0.42,0.12
--candidate-strategy sequential|speculative
--vlm-concurrency 2
--wan-submit-concurrency 4
--wan-poll-concurrency 8
--download-concurrency 8
--benchmark
--severity-level early
```

## Outputs

```text
data/processed/crops/      final bbox crop images
data/processed/masks/      binary PNG masks from diff localization
data/processed/metadata/   AutoLabelSample JSON files, retained generated images, logs, and optional Label Studio import
```

## Metadata Contract

Each successful sample includes RequiredFieldsV1 fields:

- `sample_id`
- `image_asset.image_id`, `image_uri`, `width`, `height`, `source_type`
- non-empty `objects[]`
- `objects[].box` in `xyxy`, from diff localization
- `objects[].geometry_detail.mask_uri` and `mask_format=png`
- `objects[].crop.crop_id` and `crop_uri`
- `objects[].classification.multi_labels[]` with `anomaly_type`
- `workflow.workflow_status=classified`
- `export.export_format=labelstudio`, `export.export_status=not_exported`

Quality scores are stored in `objects[].geometry_detail.generation_params`:

- `background_preservation_score`
- `anomaly_visibility_score`

## Troubleshooting

- If Qwen JSON parsing fails, the client retries once with a stricter JSON-only prompt.
- If Wan changes too much background, the candidate is rejected and the next grid is tried.
- If diff localization finds no valid connected component, no success metadata is written.
- If a crop, mask, generated image, bbox, or label field is missing, RequiredFieldsV1 validation fails and the sample is not exported.
- Use `--dry-run` when API keys or Wan endpoints are unavailable.
