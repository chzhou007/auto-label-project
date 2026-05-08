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
  -DryRunModels
```

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
5. 调用 `classification.py` 对 crop 做多标签分类。
6. 写出 `AutoLabelSample` metadata。

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
        prompt_version: person_labelstudio_bbox_v1

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

转换后的对象会继续进入 crop 和分类阶段：

```text
VLM 百分比框 -> xyxy 像素框 -> crop -> classification -> AutoLabelSample metadata
```

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
```

注意：

- `generated` 来源会跳过分类。
- 非生成图会把完整 `AutoLabelSample` 传给 `classification.py`，脚本内部逐个读取 `objects[].crop.crop_uri` 分类。
- 分类结果会写入 `objects[].classification.multi_labels`。
- 要换分类大模型，新增 `models.classification.candidates` 条目并修改 `classification.model_key`。

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
  quality_check
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
