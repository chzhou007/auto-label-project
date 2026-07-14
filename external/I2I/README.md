# VLM Grid + Image Edit 工业异常图生图自动标注流水线

本项目是 AutoLabel 生成分支的外部 I2I 后端。当前默认链路为：

```text
clean original image
-> 生成 4x4 grid preview
-> Qwen3.6-27b 选择候选 grid
-> 将 grid 转为 expanded_edit_bbox
-> Seedream5.0 在 clean original image 上生成局部漏液异常
-> 原图/生成图差分得到初始 bbox/mask/crop/metadata
```

推荐从主仓 `auto_label_project` 启动批量任务，因为主仓会在 ingest 阶段再执行 PGCD/localizer quality gate，并且只导出通过质量门禁的生成样本。本项目直接运行主要用于调试 I2I 生成链路。

## 安装

```powershell
python -m pip install -r requirements.txt
```

## 环境变量

选区模型默认走 OpenAI-compatible `/v1` 接口：

```powershell
$env:QWEN397B_API_KEY="your-qwen-key"
$env:QWEN397B_API_URL="https://deepseek.gds-services.com/v1"
```

生成模型默认走 Ark Seedream：

```powershell
$env:ARK_API_KEY="your-ark-key"
$env:SEEDREAM_BASE_URL="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
```

兼容变量：

```powershell
$env:SEEDREAM_API_KEY="your-ark-key"
$env:SEEDREAM_ENDPOINT="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
$env:ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/plan/v3/images/generations"
```

Seedream 输入图字段默认使用 `image_urls`。如果服务端要求单图字段，可覆盖：

```powershell
$env:SEEDREAM_IMAGE_FIELD="image"
```

旧 DashScope/Wan 仍可通过 `DASHSCOPE_API_KEY`、`DASHSCOPE_VLM_ENDPOINT`、`DASHSCOPE_WAN_ENDPOINT` 兼容运行。

## tasks.csv 格式

必填字段：

```csv
sample_id,image_id,image_uri,anomaly_type,source_type,site,building,floor,room_name,room_type
sample_000001,img_000001,data/images/img_000001.jpg,water_leak,generated,,,,,
```

当前支持：

```text
water_leak
diesel_leak
oil_leak
coolant_leak
```

## 直接运行 I2I

50 张校准：

```powershell
python src/main.py `
  --tasks data/tasks_water_leak.csv `
  --image-root data/images `
  --output-root outputs/water_leak_seedream5_calibration `
  --vlm-model qwen3.6-27b `
  --image-model doubao-seedream-5.0-lite `
  --grid-layout 4x4 `
  --edit-bbox-expand-ratio 0.20 `
  --crop-expand-ratio 0.10 `
  --limit 50 `
  --workers 1 `
  --skip-existing
```

1000 张生产：

```powershell
python src/main.py `
  --tasks data/tasks_water_leak_1000.csv `
  --image-root data/images `
  --output-root outputs/water_leak_seedream5_1000 `
  --vlm-model qwen3.6-27b `
  --image-model doubao-seedream-5.0-lite `
  --grid-layout 4x4 `
  --edit-bbox-expand-ratio 0.20 `
  --crop-expand-ratio 0.10 `
  --workers 1 `
  --skip-existing
```

本地无 key 验证流程：

```powershell
python src/main.py --tasks data/tasks_water_leak.csv --image-root data/images --output-root outputs/dry_run --dry-run
```

## 输出目录

```text
outputs/
  generated_images/
  grid_previews/
  crops/
  masks/
  metadata/
  logs/
```

## 设计约束

- `grid_previews` 只传给 VLM 选区模型，不传给图像生成模型。
- 图像生成模型始终接收 clean original image。
- 对 Wan/DashScope，bbox 通过 `parameters.bbox_list` 传入。
- 对 Seedream/Ark，`/images/generations` 仅视为 reference-generation 接口，生产默认拒绝用于局部编辑，避免整场景小窗贴回原图。只有设置 `SEEDREAM_ALLOW_REFERENCE_GENERATION_DEBUG=1` 时才允许 debug 实验；生产必须配置真正支持局部编辑、mask 或 inpaint 语义的 Seedream endpoint/参数。
- 调试产物写入 `debug/grid_previews`、`debug/crops`、`debug/masks`，不计入最终生成图；`generated_images` 才是每个样本唯一的最终生成图目录。
- `objects[].box` 必须来自原图/生成图差分或主仓 localizer 后处理结果，不能直接使用 grid 框或 expanded_edit_bbox。
- 失败样本只写 failure log，不应混入最终 Label Studio 交付结果。
