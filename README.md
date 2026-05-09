# 自动化标注项目使用说明

本项目用于维护一条自动化标注 DAG，将未标注图片、视频、图像生成结果、检测/分割模型结果、大模型分类结果统一整理为 `AutoLabelSample` 数据结构，并支持导出到 Label Studio。

## 1. DAG 流程

```text
原始素材
  ├─ 未标注图片
  └─ 未标注视频
        ↓
预处理
  └─ 生成图片待处理序列 manifest
        ↓
任务选择
  ├─ 直接标注任务
  │    ├─ 调用检测/分割模型服务
  │    ├─ 得到目标框或分割转框
  │    ├─ 裁剪目标小图 crop
  │    ├─ 可选：VLM 复核 crop 是否包含完整可见人体
  │    └─ 调用 classification.py 做多标签分类
  │
  └─ 图像生成任务
       ├─ 调用 C:\Users\chang\Documents\数据标注\I2I
       ├─ 接收生成图自带框坐标
       └─ 使用生成任务自带分类信息，跳过 classification.py
        ↓
统一 metadata
  └─ AutoLabelSample JSON
        ↓
抽检 / 质检
        ↓
导出
  └─ Label Studio / COCO / YOLO / custom
```

## 2. 项目目录

```text
auto_label_project/
  autolabel/                    # 核心 Python 包
    adapters/                   # 外部系统适配器
      classification_script.py  # classification.py 适配
      crop_reviewer.py          # crop 完整人体复核
      detector_service.py       # 检测/分割服务适配
      i2i_generator.py          # I2I 图像生成适配
    exporters/
      labelstudio.py            # Label Studio 导出
    config_loader.py            # YAML/JSON 配置读取
    cropper.py                  # 目标裁剪
    pipeline.py                 # 直接标注与生成 metadata 接入
    preprocess.py               # 图片/视频预处理
    validators.py               # AutoLabelSample 轻量校验

  configs/
    autolabel.yaml              # 主配置文件
    task_manifest.example.csv   # manifest 示例
    detector_services.example.json
    pipeline.example.json
    taxonomy.example.json

  data/
    raw/images/                 # 原始图片
    raw/videos/                 # 原始视频
    staging/image_sequence/     # 预处理后的图片序列
    processed/crops/            # 裁剪小图
    processed/masks/            # mask
    processed/metadata/         # AutoLabelSample JSON
    processed/i2i_outputs/      # I2I 输出
    exports/labelstudio/        # Label Studio 导出
    qc/                         # 质检记录

  schemas/
    autolabel_sample.schema.json
    autolabel_sample.example.json

  scripts/
    classification.py            # 内置 AutoLabelSample 分类脚本
    run_preprocess.py
    run_i2i_generation.py
    run_direct_pipeline.py
    validate_sample.py
    export_labelstudio.py

  tests/
```

## 3. 环境准备

进入项目目录：

```powershell
cd D:\codex\auto_label_project
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

设置大模型密钥。建议使用环境变量，不要把真实 key 写进仓库；你的服务地址需要填到 `/v1`，不要填到 `/v1/chat/completions`：

```powershell
$env:QWEN397B_API_KEY="你的大模型 Key"
$env:QWEN397B_API_URL="https://deepseek.gds-services.com/vllm-qwen35b/v1"
$env:QWEN_GEOMETRY_API_URL="https://deepseek.gds-services.com/vllm-qwen35b/v1"
$env:QWEN397B_MODEL="aios-smart-eye-vlm"
$env:QWEN_GEOMETRY_MODEL="aios-smart-eye-vlm"

# 只有运行 I2I 生成分支时才需要配置生成模型 key：
$env:DASHSCOPE_API_KEY="你的 DashScope Key"
```

如需本地私有配置，可以复制：

```powershell
Copy-Item configs/autolabel.yaml configs/autolabel.local.yaml
```

`configs/autolabel.local.yaml` 已加入 `.gitignore`，适合填写本机路径或真实 key。

## 4. 主配置文件

主配置文件为：

```text
configs/autolabel.yaml
```

常用配置块如下：

```yaml
credentials:
  dashscope_generation:
    api_key_env: DASHSCOPE_API_KEY
    api_key: ${DASHSCOPE_API_KEY}
  qwen_classifier:
    api_key_env: QWEN397B_API_KEY
    api_key: ${QWEN397B_API_KEY}
    base_url: ${QWEN397B_API_URL:-https://deepseek.gds-services.com/vllm-qwen35b/v1}
  qwen_geometry_vlm:
    api_key_env: QWEN397B_API_KEY
    api_key: ${QWEN397B_API_KEY}
    base_url: ${QWEN_GEOMETRY_API_URL:-https://deepseek.gds-services.com/vllm-qwen35b/v1}

paths:
  i2i_project: C:\Users\chang\Documents\数据标注\I2I
  classification_script: scripts/classification.py
  raw_images_dir: data/raw/images
  raw_videos_dir: data/raw/videos
  image_sequence_manifest: data/staging/image_sequence/manifest.csv
  metadata_dir: data/processed/metadata
  crop_dir: data/processed/crops
  labelstudio_export: data/exports/labelstudio/import.json

modules:
  generation:
    backend: i2i_external
    backends:
      i2i_external:
        project_dir: C:\Users\chang\Documents\数据标注\I2I
  classification:
    backend: external_script
    backends:
      external_script:
        script_path: scripts/classification.py

models:
  generation:
    active_vlm: qwen_grid_selector
    active_image_generator: wan_image_editor
  classification:
    active_model: qwen397b_vlm_classifier
  geometry:
    candidates:
      ppe_person_detector_http:
        model_name: replace_with_detector_name
        endpoint: http://127.0.0.1:8001/detect

detector_services:
  default_task_key: ppe_person
  services:
    ppe_person:
      model_ref: ppe_person_detector_http
```

环境变量支持 `${ENV_NAME}` 和 `${ENV_NAME:-默认值}` 两种写法。

模型选择规则：

- DAG 代码只依赖 `modules`，不直接依赖 `I2I` 或 `classification.py`。
- 当前 `i2i_external`、`external_script` 只是兼容已有代码的 backend；后续可以把实现迁入 `autolabel/modules/` 后替换 backend。
- 生成分支：`generation.vlm_model_key` 和 `generation.image_model_key` 可以覆盖 `models.generation.active_vlm`、`models.generation.active_image_generator`。
- 分类分支：`classification.model_key` 可以覆盖 `models.classification.active_model`。
- 检测/分割分支：`detector_services.services.<task_key>.model_ref` 指向 `models.geometry.candidates` 里的任意模型。
- key、base_url、endpoint 不写死在代码里，统一从 `credentials` 和 `models` 解析。

## 5. 模块化运行入口

推荐使用统一项目入口：

```powershell
python scripts/run_pipeline.py --config configs/autolabel.yaml --branches generation,direct
```

只跑直接标注：

```powershell
python scripts/run_pipeline.py --config configs/autolabel.yaml --branches direct
```

只跑生成分支并使用 dry run：

```powershell
python scripts/run_pipeline.py `
  --config configs/autolabel.yaml `
  --branches generation `
  --dry-run-generation
```

运行导出：

```powershell
python scripts/run_pipeline.py --config configs/autolabel.yaml --branches export
```

旧的 `run_i2i_generation.py`、`run_direct_pipeline.py` 仍保留，主要用于单步调试。

## 6. 一键批处理脚本

如果要对 `data/raw/images/` 和 `data/raw/videos/` 中的一批数据直接跑人体框和分类，可以使用 PowerShell 一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 -DryRunModels
```

`-DryRunModels` 用于无 key 验证数据流，会生成模拟人体框和模拟分类结果。配置真实大模型 key 后，去掉 `-DryRunModels` 即可调用真实 VLM 检测与分类：

```powershell
$env:QWEN397B_API_KEY="你的 key"
$env:QWEN397B_API_URL="https://deepseek.gds-services.com/vllm-qwen35b/v1"
$env:QWEN_GEOMETRY_API_URL="https://deepseek.gds-services.com/vllm-qwen35b/v1"
$env:QWEN_GEOMETRY_MODEL="aios-smart-eye-vlm"
$env:QWEN397B_MODEL="aios-smart-eye-vlm"

powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 -BatchName batch_001
```

默认输出到：

```text
data/runs/<batch_name>/
  manifest.csv
  image_sequence/
  processed/
    metadata/
    crops/
  exports/
    labelstudio_import.json
    labelstudio_config.xml
```

常用参数：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 `
  -BatchName batch_001 `
  -RawImages data/raw/images `
  -RawVideos data/raw/videos `
  -VideoFrameStride 30 `
  -VideoDecodeMode cpu `
  -DryRunModels
```

### Direct pipeline batch / workers

`run_pipeline.py` 的 direct 分支支持按 manifest 行分批提交，并在每个 batch 内并发处理样本：

```powershell
python scripts/run_pipeline.py `
  --config configs/autolabel.yaml `
  --branches direct `
  --batch-size 512 `
  --workers 64
```

一键脚本同样支持：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 `
  -BatchName batch_001 `
  -BatchSize 512 `
  -Workers 64
```

如果要用 `batch_size=32` 起步：

```powershell
python scripts/run_pipeline.py `
  --config configs/autolabel.yaml `
  --branches direct `
  --batch-size 32 `
  --workers 32
```

或使用一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 `
  -BatchName batch_001 `
  -BatchSize 32 `
  -Workers 32
```

`batch_size` 表示每一轮从 manifest 取多少张图进入处理队列；`workers` 表示客户端同时发起多少个样本级任务。对于 OpenAI-compatible / vLLM 服务，通常是多并发请求触发服务端动态 batching，而不是一次 HTTP 请求里塞 512 张图。H20 场景可以先用 `--batch-size 32 --workers 8/16/32` 验证稳定性，再逐步提高到 `--batch-size 512 --workers 32/64`。如果服务端出现 429、连接重置或超时，就先降低 `workers`。分类限速可通过 `classification.delay_seconds: 0` 关闭。

## 7. 准备输入数据

把原始图片放到：

```text
data/raw/images/
```

把原始视频放到：

```text
data/raw/videos/
```

然后运行预处理：

```powershell
python scripts/run_preprocess.py --config configs/autolabel.yaml
```

当前视频抽帧逻辑是按帧序号抽样，不是按秒抽样：`video_frame_stride=30` 表示保存第 `0, 30, 60, ...` 帧。理论抽帧数为：

```text
floor((视频总帧数 - 1) / video_frame_stride) + 1
```

例如 100 秒视频：

```text
30 fps: 100 * 30 = 3000 帧，stride=30 -> 100 张
25 fps: 100 * 25 = 2500 帧，stride=30 -> 84 张
60 fps: 100 * 60 = 6000 帧，stride=30 -> 200 张
```

偏差常见原因包括：视频真实 fps 不是标称 fps、可变帧率 VFR、首尾不足整秒、解码器丢帧或坏帧、`video_max_frames` 截断、以及 GPU/FFmpeg 后端与 OpenCV 后端对损坏帧/时间戳处理不同。

预处理支持 CPU 和 GPU 两种模式，默认保留 CPU：

```yaml
preprocess:
  video_decode_mode: cpu   # cpu / gpu / auto
  ffmpeg_path: ffmpeg
  video_gpu_hwaccel: cuda
  video_gpu_fallback_to_cpu: true
  video_error_policy: skip  # skip / fail
```

命令行开启 GPU：

```powershell
python scripts/run_preprocess.py `
  --config configs/autolabel.yaml `
  --video-decode-mode gpu
```

一键脚本开启 GPU：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_batch.ps1 `
  -VideoDecodeMode gpu `
  -VideoFrameStride 30
```

GPU 模式使用 FFmpeg `-hwaccel cuda` 解码；如果本机 FFmpeg 或显卡驱动不支持 CUDA，默认会回退 CPU。若希望 GPU 失败时直接报错，使用 `--no-video-gpu-fallback` 或 `-NoVideoGpuFallback`。

若某个视频本身损坏、为空文件，或 FFmpeg/OpenCV 都无法打开，`video_error_policy: skip` 会跳过该视频并继续处理其他文件；如果希望发现坏视频就中断批处理，改为 `fail` 或命令行传：

```powershell
python scripts/run_preprocess.py `
  --config configs/autolabel.yaml `
  --video-error-policy fail
```

`moov atom not found` 通常表示 MP4 文件损坏、未完整拷贝，或文件大小为 0。

预处理会生成图片待处理序列和 manifest：

```text
data/staging/image_sequence/manifest.csv
```

manifest 关键字段：

```csv
sample_id,image_id,image_uri,source_type,task_mode,task_key,object_type,anomaly_type
sample_001,img_001,data/staging/image_sequence/img_001.jpg,manual_upload,direct,ppe_person,person,
sample_002,img_002,data/staging/image_sequence/img_002.jpg,generated,generation,,leakage_area,oil_leak
```

其中：

- `task_mode=direct`：走检测/分割模型服务，再走分类。
- `task_mode=generation`：走 I2I 图像生成分支，跳过分类脚本。
- `task_key`：直接标注任务使用，对应 `detector_services.services` 下的 key。
- `anomaly_type`：生成任务使用，例如 `diesel_leak`、`oil_leak`、`coolant_leak`。

## 8. 图像生成分支

图像生成分支使用外部代码：

```text
C:\Users\chang\Documents\数据标注\I2I
```

运行命令：

```powershell
python scripts/run_i2i_generation.py --pipeline-config configs/autolabel.yaml
```

调试时可以使用 dry run：

```powershell
python scripts/run_i2i_generation.py --pipeline-config configs/autolabel.yaml --dry-run
```

脚本会自动从 manifest 中筛选：

```text
task_mode=generation
```

然后传给 I2I 项目。I2I 输出的 metadata 会保留生成图自带框坐标和生成标签，进入本项目后不再调用 `classification.py`。

生成模型选择在 YAML 中维护：

```yaml
models:
  generation:
    active_vlm: qwen_grid_selector
    active_image_generator: wan_image_editor

generation:
  vlm_model_key: qwen_grid_selector
  image_model_key: wan_image_editor
```

要换模型时，新增一个 `models.generation.vlm` 或 `models.generation.image_generators` 条目，再把 active/key 改过去即可。

如需把 I2I 输出 metadata 汇总进本项目 metadata 目录：

```powershell
python scripts/run_i2i_generation.py `
  --pipeline-config configs/autolabel.yaml `
  --ingest-metadata-dir data/processed/metadata
```

## 9. 直接标注分支

直接标注分支会执行：

1. 读取 manifest 中 `task_mode=direct` 的样本。
2. 根据 `task_key` 找到检测/分割模型配置。
3. 调用模型服务得到框。
4. 基于框裁剪 crop。
5. 如果开启 `direct_annotation.crop_review.enabled`，调用 VLM 复核 crop 是否包含完整可见人体。
6. 调用 `classification.py` 对 crop 做多标签分类。
7. 写出 `AutoLabelSample` metadata。

运行命令：

```powershell
python scripts/run_direct_pipeline.py --pipeline-config configs/autolabel.yaml
```

如果只想测试检测/裁剪，不调用分类：

```powershell
python scripts/run_direct_pipeline.py --pipeline-config configs/autolabel.yaml --no-classify
```

输出位置：

```text
data/processed/metadata/
data/processed/crops/
```

## 10. 检测/分割模型配置

检测/分割模型分两层配置：`models.geometry.candidates` 维护可选模型，`detector_services.services` 维护任务路由。

```yaml
models:
  geometry:
    candidates:
      ppe_person_detector_http:
        service_type: detector
        geometry_source: detector
        model_name: replace_with_detector_name
        model_version: replace_with_detector_version
        endpoint: http://127.0.0.1:8001/detect
        method: POST
        timeout_seconds: 60

detector_services:
  default_task_key: ppe_person
  services:
    ppe_person:
      model_ref: ppe_person_detector_http
      request:
        image_field: image_uri
        extra_payload:
          task: ppe_person
      response:
        format: boxes
        boxes_field: boxes
        label_field: label
        bbox_field: bbox
        confidence_field: score
      object_type_map:
        person: person
```

要切换检测/分割模型，只需要把某个 `task_key` 的 `model_ref` 改成另一个 `models.geometry.candidates` key。

### 人体框 VLM 检测

`ppe_person` 默认已经配置为大模型人体框检测：

```yaml
models:
  geometry:
    candidates:
      ppe_person_vlm_labelstudio_detector:
        provider: openai_compatible
        service_type: vlm_detector
        backend: vlm_labelstudio_detector
        geometry_source: detector
        model_name: ${QWEN_GEOMETRY_MODEL:-aios-smart-eye-vlm}
        credential_ref: qwen_geometry_vlm
        base_url: ${QWEN_GEOMETRY_API_URL:-https://deepseek.gds-services.com/vllm-qwen35b/v1}
        use_response_format: true
        response_format_type: json_object
        prompt_version: person_labelstudio_full_body_bbox_v2
        parse_retry_count: 0
        fail_on_parse_error: true

detector_services:
  default_task_key: ppe_person
  services:
    ppe_person:
      model_ref: ppe_person_vlm_labelstudio_detector
      target_label: Person
      default_object_type: person
      prompt: |
        # Role
        你是一个专业的计算机视觉数据标注专家和自动化脚本引擎。
        ...
```

这个子模块要求大模型按 Label Studio 标准输出百分比框：

```json
{
  "value": {
    "x": 10.0,
    "y": 20.0,
    "width": 30.0,
    "height": 40.0,
    "rectanglelabels": ["Person"]
  }
}
```

DAG 内部会自动转换为 `AutoLabelSample.objects[].box` 需要的像素坐标：

```json
{
  "format": "xyxy",
  "x1": 100,
  "y1": 100,
  "x2": 400,
  "y2": 300
}
```

人体框 prompt 采用完整可见人体优先规则：只要头部可见，框顶必须覆盖到头顶/头盔；只要脚或鞋可见，框底必须覆盖到脚底/鞋底/脚尖。只有脚部确实被画面边缘裁掉、被遮挡或不可见时，才允许输出半身/局部人体框。这个约束用于避免后续 crop 截断脚部，影响 `safety_shoes` 等分类标签。

检测阶段默认启用 `response_format: json_object`，并要求模型输出顶层 JSON object，减少模型输出 Thinking Process/自然语言分析的概率。若 VLM 仍没有按 Label Studio JSON 输出，会抛出可识别的 JSON 解析错误，进入 direct pipeline 的重试队列。重试只是运行时机制：如果后续某次重试成功，最终 metadata 只保留成功那一轮的唯一结果；如果连续 3 次仍不合法，则不写入该样本的最终 metadata，避免把“模型输出格式错误”误当成“图片里没有人”，从而生成无框场景图。

转换后的对象会继续进入 crop 和分类阶段：

```text
VLM 百分比框 -> xyxy 像素框 -> crop -> 可选 crop 复核 -> classification -> AutoLabelSample metadata
```

### 可选 crop 完整人体复核

如果发现人体框在场景图里位置大致正确，但裁剪图经常漏掉头部、脚部或只截到半身，可以打开 crop 复核开关。该功能会在 `crop` 之后、`classification` 之前，再把每个 `person` crop 交给 VLM 判断是否包含完整可见人体。

默认关闭：

```yaml
direct_annotation:
  crop_review:
    enabled: false
```

开启方式：

```yaml
direct_annotation:
  crop_review:
    enabled: true
    model_ref: ppe_person_vlm_labelstudio_detector
    target_object_type: person
    prompt_version: crop_full_person_review_v1
    failed_issue_flag: incomplete_person_crop
    record_passed: false
```

复核不通过时，pipeline 不会直接丢弃样本，而是写回对象级质检字段，便于后续筛选和回看：

```json
{
  "quality_check": {
    "qc_sampled": true,
    "qc_status": "failed",
    "issue_flags": ["incomplete_person_crop", "missing_feet"],
    "reviewer": "vlm_crop_reviewer",
    "comment": "crop_full_person_review_v1: 脚部被 crop 截断"
  }
}
```

如果复核 VLM 本身连续输出非法 JSON，该样本会沿用 direct pipeline 的 `json_retry_attempts` 重试队列；重试成功后最终 metadata 只保留成功结果，连续失败则不写入该样本 metadata。如果只是不完整人体判断失败，则只记录 `quality_check`，后续分类仍会继续执行。

使用 `-DryRunModels` 或 `--dry-run-models` 时，如果 crop 复核开关被打开，复核模块会自动返回通过结果，不会真实调用 VLM。

HTTP 服务推荐返回格式：

```json
{
  "boxes": [
    {
      "label": "person",
      "bbox": [320, 180, 640, 920],
      "score": 0.92
    }
  ]
}
```

分割模型也可以返回对象格式：

```json
{
  "objects": [
    {
      "object_type": "ground_area",
      "box": {
        "format": "xyxy",
        "x1": 10,
        "y1": 20,
        "x2": 300,
        "y2": 400
      },
      "polygon": [[10, 20], [300, 20], [300, 400]],
      "mask_uri": "data/processed/masks/sample_001_mask.png",
      "mask_format": "png",
      "confidence": 0.88
    }
  ]
}
```

也支持本地命令式模型：

```yaml
models:
  geometry:
    candidates:
      local_detector_command:
        service_type: detector
        geometry_source: detector
        model_name: replace_with_local_model_name
        command:
          - python
          - C:\path\to\detector_service_cli.py
          - --image
          - "{image_uri}"
          - --output-json
          - "{output_json}"

detector_services:
  services:
    ppe_person:
      model_ref: local_detector_command
```

本地命令需要写出 `{output_json}`，或直接在 stdout 输出 JSON。

## 11. 分类配置

分类脚本路径在。默认使用仓库内置脚本，它的输入/输出都是 `AutoLabelSample`，会把结果写回 `objects[].classification`：

```yaml
paths:
  classification_script: scripts/classification.py
```

分类配置在：

```yaml
models:
  classification:
    active_model: qwen397b_vlm_classifier
    candidates:
      qwen397b_vlm_classifier:
        provider: openai_compatible
        model_name: ${QWEN397B_MODEL:-aios-smart-eye-vlm}
        classifier_type: vlm
        classifier_name: qwen397b_vlm_classifier
        credential_ref: qwen_classifier
        base_url: ${QWEN397B_API_URL:-https://deepseek.gds-services.com/vllm-qwen35b/v1}

classification:
  enabled: true
  model_key: qwen397b_vlm_classifier
  skip_source_types:
    - generated
  delay_seconds: 0.5
  max_tokens: 2000
  parse_retry_count: 0
  use_response_format: true
  text_fallback_enabled: false
  log_parse_fallback: false
```

注意：

- `generated` 来源会跳过分类。
- 非生成图会把完整 `AutoLabelSample` 传给 `classification.py`，脚本内部逐个读取 `objects[].crop.crop_uri` 分类。
- 分类结果会写入 `objects[].classification.multi_labels`。
- 要换分类大模型，新增 `models.classification.candidates` 条目并修改 `classification.model_key`。
- `max_tokens` 用于避免模型输出 `notes` 时被截断导致 `Unterminated string`。
- `use_response_format: true` 会优先请求 JSON object 输出；如果后端不支持，代码会自动回退普通调用。
- `parse_retry_count` 表示分类模块内部 JSON 重试次数；默认交给 direct pipeline 的统一重试队列处理，所以设为 `0`。
- `text_fallback_enabled: false` 表示分类 JSON 不合法时不再使用文本兜底结果，而是抛出 JSON 解析错误进入重试队列。
- `log_parse_fallback: false` 会隐藏“已使用文本兜底解析”的非致命日志；仅当显式开启 `text_fallback_enabled` 时才会用到。

direct pipeline 对检测、可选 crop 复核和分类共用一套 JSON 重试策略：

```yaml
direct_annotation:
  json_retry_attempts: 3
```

同一张样本前几次出现模型 JSON 不合法、后续重试成功时，只写出最终成功结果：

```text
data/processed/metadata/<sample_id>.json
```

同一张样本连续 3 次仍然 JSON 不合法时，不写入 `metadata/<sample_id>.json`，也不额外写入重试状态文件；如果输出目录里存在同名旧结果，pipeline 会移除该样本的旧 metadata，避免旧结果混入本轮最终输出。最终导出只会读取 `metadata/` 中真实成功落盘的样本。

## 12. 主输出结构

主输出不是 crop，也不是 Label Studio 结果，而是完整的 `AutoLabelSample`：

```text
data/processed/metadata/<sample_id>.json
```

crop 图只是 `objects[].crop.crop_uri` 指向的派生产物：

```text
data/processed/crops/<sample_id>_<object_id>.jpg
```

每个 metadata 文件都包含：

```text
sample_id
image_asset
objects[]
  object_id
  object_type
  box
  geometry_source
  geometry_model
  geometry_detail
  crop
  classification
  quality_check                  # 可由 crop_review 或人工抽检写入
qc_policy
workflow
export
```

写出 metadata 前，代码会调用 `normalize_autolabel_sample()` 补齐默认字段，再调用 `validate_sample_contract()` 校验结构。

## 13. 校验 metadata

校验单个文件：

```powershell
python scripts/validate_sample.py --sample schemas/autolabel_sample.example.json
```

校验目录：

```powershell
python scripts/validate_sample.py --metadata-dir data/processed/metadata
```

数据契约文件：

```text
schemas/autolabel_sample.schema.json
```

示例：

```text
schemas/autolabel_sample.example.json
```

## 14. 导出 Label Studio

导出命令：

```powershell
python scripts/export_labelstudio.py --config configs/autolabel.yaml
```

默认输出：

```text
data/exports/labelstudio/import.json
```

如果希望导出后同步更新 metadata 中的导出状态：

```powershell
python scripts/export_labelstudio.py --config configs/autolabel.yaml --update-samples
```

## 15. 测试

运行基础测试：

```powershell
python -m unittest discover -s tests
```

运行 Python 编译检查：

```powershell
python -m compileall autolabel scripts tests
```

## 16. 常见问题

### 1. YAML 无法读取

安装依赖：

```powershell
python -m pip install PyYAML
```

### 2. classification.py 导入失败

确认：

- `paths.classification_script` 路径存在。
- 已安装 `openai`。
- `classification.model_key` 指向的模型配置存在。
- 模型的 `credential_ref` 指向的 key 已设置。

### 3. 检测服务调用失败

确认：

- `detector_services.services.<task_key>.model_ref` 指向的模型配置存在。
- 该模型的 `endpoint` 可访问，或 `command` 可执行。
- manifest 中的 `task_key` 与 YAML 中的服务 key 一致。
- 服务返回字段与 `response` 映射一致。

### 4. 生成任务没有被 I2I 执行

确认 manifest 中存在：

```text
task_mode=generation
```

并且生成任务填写了：

```text
anomaly_type
```

### 5. 生成图为什么没有再分类

这是当前设计：生成图自带框坐标和原始分类信息，因此跳过 `classification.py`，避免重复调用大模型。
