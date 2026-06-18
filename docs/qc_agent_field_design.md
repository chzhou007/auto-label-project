# 质检 Agent 字段穿透设计

本文先定义质检问题和数据字段，再说明 agent 如何基于字段做判断。核心要求是每个结论都能追溯到原始 metadata 字段、文件路径、标注对象和上游来源，避免只给一条孤立的看图打分。

## 1. 问题定义

质检 agent 的输入是数据生成或自动化标注后的 `AutoLabelSample` metadata。它要回答三个问题：

1. 这条样本能不能被下游使用：结构、图片、crop、mask、框坐标、分类字段是否完整且一致。
2. 这条标注是否疑似错误：框是否越界、过小、重复、crop 是否没覆盖框、已有质检是否失败。
3. 哪些样本需要人工二次质检：规则失败、VLM 语义复核不确定、或通过样本抽样。

质检结果必须保留数据穿透链路：

```text
manifest row / generated output
  -> AutoLabelSample metadata
  -> image / mask / crop 文件
  -> object box / classification / quality_check
  -> QC issue.field_path
  -> report.json + manual_review_queue.csv
```

## 2. Manifest 输入字段

`configs/task_manifest.example.csv` 和实际预处理 manifest 会进入 `make_sample()`，形成 `image_asset.source_context`、`image_asset.scene_context` 和后续对象上下文。

| 字段 | 说明 | QC 用途 |
| --- | --- | --- |
| `sample_id` | 样本唯一 ID | 报告主键、抽样稳定 hash |
| `image_id` | 原图唯一 ID | 人工队列追图 |
| `image_uri` | 原图路径 | 文件存在性、尺寸读取、VLM 复核 |
| `source_type` | `generated` / `cctv` / `manual_upload` 等 | 区分生成样本和直接标注样本 |
| `task_mode` | `direct` / `generation` | 判断来自直接标注还是生成分支 |
| `task_key` | 检测任务 key，如 `ppe_person` | 追溯检测服务 |
| `object_type` | 目标类型，如 `person` / `leakage_area` | VLM 语义复核提示词 |
| `anomaly_type` | 异常类型，如 `diesel_leak` | 生成异常语义复核 |
| `site` | 园区 | 人工队列筛选 |
| `building` | 楼栋 | 人工队列筛选 |
| `floor` | 楼层 | 人工队列筛选 |
| `room_name` | 房间名 | 人工队列筛选 |
| `room_type` | 房间类型 | 语义上下文 |
| `task_group` | 任务组 | 语义上下文 |
| `inspection_content` | 巡检内容 | 生成异常/标注语义核对 |
| `collection_batch` | 采集批次 | 批次返工 |
| `camera_id` | 摄像头 ID | 摄像头维度问题聚合 |
| `capture_time` | 采集时间 | 时间维度问题聚合 |
| `width` | 原图宽 | 与实际图片尺寸核对 |
| `height` | 原图高 | 与实际图片尺寸核对 |

## 3. AutoLabelSample 字段全集

### 3.1 样本级字段

| 字段路径 | 说明 | QC 用途 |
| --- | --- | --- |
| `sample_id` | 样本唯一 ID | 报告主键 |
| `image_asset.image_id` | 原图 ID | 追图、人工队列 |
| `image_asset.image_uri` | 原图路径 | 文件存在性、VLM 复核 |
| `image_asset.width` | metadata 记录图宽 | 与真实图片尺寸核对 |
| `image_asset.height` | metadata 记录图高 | 与真实图片尺寸核对 |
| `image_asset.source_type` | 数据来源类型 | 选择规则和人工分层 |
| `image_asset.source_context.generation_prompt` | 生成提示词 | 生成样本语义追溯，报告只存 preview |
| `image_asset.source_context.generation_model` | 生成模型 | 模型维度问题聚合 |
| `image_asset.source_context.collection_batch` | 批次 | 返工/抽样分层 |
| `image_asset.source_context.camera_id` | 摄像头 | 摄像头维度问题聚合 |
| `image_asset.source_context.capture_time` | 采集时间 | 时间维度问题聚合 |
| `image_asset.scene_context.site` | 园区 | 人工队列筛选 |
| `image_asset.scene_context.building` | 楼栋 | 人工队列筛选 |
| `image_asset.scene_context.floor` | 楼层 | 人工队列筛选 |
| `image_asset.scene_context.room_name` | 房间名 | 人工队列筛选 |
| `image_asset.scene_context.room_type` | 房间类型 | 语义上下文 |
| `image_asset.scene_context.task_group` | 任务组 | 语义上下文 |
| `image_asset.scene_context.inspection_content` | 巡检内容 | 语义上下文 |

### 3.2 对象级字段

| 字段路径 | 说明 | QC 用途 |
| --- | --- | --- |
| `objects[].object_id` | 对象唯一 ID | 对象级追踪、重复 ID 检查 |
| `objects[].object_type` | 对象类型 | 语义复核、分类规则 |
| `objects[].box.format` | 框格式，固定 `xyxy` | 契约校验 |
| `objects[].box.x1` | 左上角 x | 几何合法性 |
| `objects[].box.y1` | 左上角 y | 几何合法性 |
| `objects[].box.x2` | 右下角 x | 几何合法性 |
| `objects[].box.y2` | 右下角 y | 几何合法性 |
| `objects[].geometry_source` | 框来源 | 区分检测、分割、生成、人标 |
| `objects[].geometry_model.model_name` | 出框模型名 | 模型维度问题聚合 |
| `objects[].geometry_model.model_version` | 出框模型版本 | 版本维度问题聚合 |
| `objects[].geometry_model.confidence` | 出框置信度 | 低置信风险排序 |
| `objects[].geometry_detail.polygon` | 多边形 | 分割类质检 |
| `objects[].geometry_detail.mask_uri` | mask 路径 | mask 文件存在性、尺寸核对 |
| `objects[].geometry_detail.mask_format` | mask 格式 | 契约校验 |
| `objects[].geometry_detail.generation_params` | 生成/定位过程参数 | 生成样本追溯；报告只存摘要键 |

### 3.3 Crop 字段

| 字段路径 | 说明 | QC 用途 |
| --- | --- | --- |
| `objects[].crop.crop_id` | crop ID | crop 追踪 |
| `objects[].crop.crop_uri` | crop 文件路径 | 文件存在性、尺寸读取 |
| `objects[].crop.crop_box.format` | crop 框格式 | 契约校验 |
| `objects[].crop.crop_box.x1` | crop 左上角 x | 是否覆盖 object box |
| `objects[].crop.crop_box.y1` | crop 左上角 y | 是否覆盖 object box |
| `objects[].crop.crop_box.x2` | crop 右下角 x | 是否覆盖 object box |
| `objects[].crop.crop_box.y2` | crop 右下角 y | 是否覆盖 object box |
| `objects[].crop.crop_expand_ratio` | crop 扩边比例 | crop 尺寸解释 |
| `objects[].crop.is_valid_crop` | crop 有效标记 | 失败路径追踪 |

### 3.4 分类字段

| 字段路径 | 说明 | QC 用途 |
| --- | --- | --- |
| `objects[].classification.multi_labels[].label_key` | 标签键 | 人工队列和语义复核 |
| `objects[].classification.multi_labels[].label_value` | 标签值 | 人工队列和语义复核 |
| `objects[].classification.multi_labels[].confidence` | 分类置信度 | 风险排序 |
| `objects[].classification.multi_labels[].evidence` | 分类证据 | 人工复核解释 |
| `objects[].classification.classifier_type` | 分类器类型 | 模型/规则/人工来源 |
| `objects[].classification.classifier_name` | 分类器名称 | 模型维度问题聚合 |
| `objects[].classification.classifier_version` | 分类器版本 | 版本维度问题聚合 |
| `objects[].classification.prompt_version` | 分类 prompt 版本 | prompt 维度问题聚合 |
| `objects[].classification.raw_response` | 原始分类返回 | 追溯用；QC 报告只记录是否存在 |

### 3.5 质检、工作流、导出字段

| 字段路径 | 说明 | QC 用途 |
| --- | --- | --- |
| `objects[].quality_check.qc_sampled` | 是否被质检抽中 | 复核状态追踪 |
| `objects[].quality_check.qc_status` | `pending` / `passed` / `failed` 等 | 已有失败/待复核识别 |
| `objects[].quality_check.reviewed_labels[].label_key` | 人工复核标签键 | 纠错记录 |
| `objects[].quality_check.reviewed_labels[].label_value` | 人工复核标签值 | 纠错记录 |
| `objects[].quality_check.issue_flags` | 问题标签 | 问题聚合 |
| `objects[].quality_check.reviewer` | 复核人/agent | 责任链路 |
| `objects[].quality_check.review_time` | 复核时间 | 时序追踪 |
| `objects[].quality_check.comment` | 复核说明 | 人工判断解释 |
| `qc_policy.qc_mode` | 质检模式 | 抽检/全检策略 |
| `qc_policy.sampling_ratio` | 抽样比例 | 人工队列规模 |
| `qc_policy.sampling_method` | 抽样方法 | 抽样可解释性 |
| `qc_policy.fail_policy` | 失败处理方式 | 批次返工策略 |
| `qc_policy.qc_batch_id` | 质检批次 | 批次追踪 |
| `workflow.workflow_status` | DAG 状态 | 是否已到可质检阶段 |
| `workflow.pipeline_id` | 流水线 ID | 版本追踪 |
| `workflow.pipeline_version` | 流水线版本 | 版本追踪 |
| `workflow.created_time` | 创建时间 | 时序追踪 |
| `workflow.updated_time` | 更新时间 | 时序追踪 |
| `export.export_format` | 导出格式 | 下游兼容 |
| `export.export_status` | 导出状态 | 避免重复导出坏数据 |
| `export.export_uri` | 导出路径 | 追踪下游文件 |
| `export.labelstudio_mapping` | Label Studio 映射 | 平台字段追踪 |

## 4. QC 输出字段

如果只有图片和 crop，先运行 `scripts/run_asset_qc.py`。它可以同时读取 manifest 和 `AutoLabelSample` metadata，把图片资产反查到样本与对象。输出是资产级字段：

| 字段 | 说明 |
| --- | --- |
| `asset_kind` | `image` 或 `crop` |
| `asset_uri` | 输入图片路径 |
| `resolved_asset_uri` | 解析后的绝对路径 |
| `file_name` | 文件名 |
| `file_size_bytes` | 文件大小 |
| `width` / `height` | 实际图片尺寸 |
| `aspect_ratio` | 宽高比 |
| `sample_id` | manifest 或 crop 文件名反推的样本 ID |
| `image_id` | manifest 或 crop 文件名反推的图像 ID |
| `source_type` | manifest 中的数据来源 |
| `source_image_uri` | manifest 中的原图路径 |
| `object_type` | crop 文件名或 manifest 中的对象类型 |
| `object_index` | crop 文件名中的对象序号 |
| `issues[].code` | 资产级问题码 |

### 4.1 `report.json`

| 字段 | 说明 |
| --- | --- |
| `qc_run_id` | 本次质检运行 ID |
| `created_time` | 运行时间 |
| `metadata_dir` | 输入 metadata 目录 |
| `sample_paths` | 输入单文件列表 |
| `qc_config.manual_sampling_ratio` | 人工抽样比例 |
| `qc_config.random_seed` | 稳定抽样 seed |
| `qc_config.vlm_review_enabled` | 是否启用 VLM 复核 |
| `summary.total_samples` | 样本数 |
| `summary.total_objects` | 对象数 |
| `summary.passed_samples` | 规则通过样本数 |
| `summary.failed_samples` | 失败样本数 |
| `summary.needs_human_review_samples` | 疑似问题样本数 |
| `summary.manual_review_queue_size` | 人工队列样本数 |
| `summary.issue_counts` | 问题类型计数 |
| `samples[].field_snapshot` | 样本级字段快照 |
| `samples[].objects[].field_snapshot` | 对象级字段快照 |
| `samples[].issues[].field_path` | 问题指向的字段路径 |
| `samples[].objects[].issues[].field_path` | 对象问题指向的字段路径 |

### 4.2 `manual_review_queue.csv`

人工队列 CSV 会展开关键字段，便于人直接筛选：

```text
sample_id,image_id,source_type,collection_batch,camera_id,capture_time,
site,building,floor,room_name,room_type,task_group,inspection_content,
object_id,object_type,geometry_source,geometry_model_name,geometry_model_version,
geometry_confidence,classifier_type,classifier_name,classifier_version,label_summary,
workflow_status,pipeline_id,pipeline_version,export_format,export_status,qc_batch_id,
status,risk_score,issue_codes,issue_field_paths,issue_messages,
metadata_uri,image_uri,resolved_image_uri,crop_uri,resolved_crop_uri,manual_review_reason
```

## 5. Agent 设计

质检 agent 拆成六个稳定模块：

| 模块 | 输入字段 | 输出 |
| --- | --- | --- |
| `DataIntake` | `metadata_dir` / `sample_paths` | metadata 路径列表 |
| `ContractValidator` | `AutoLabelSample` 全字段 | 契约错误，`field_path=$` |
| `AssetResolver` | `image_uri` / `crop_uri` / `mask_uri` | 真实文件路径、缺失问题 |
| `GeometryValidator` | `image width/height`、`objects[].box`、`crop_box` | 越界、过小、极端比例、重复框 |
| `SemanticReviewer` | 原图红框、`object_type`、`multi_labels`、`inspection_content` | 可选 VLM 语义问题 |
| `RiskRouter` | issue severity、抽样策略、`qc_policy` | pass / failed / needs_human_review |
| `ReportWriter` | 字段快照、issue、路径 | JSON 报告和人工队列 |

## 6. 当前默认规则

| 规则 | 依赖字段 | 失败类型 |
| --- | --- | --- |
| metadata 契约校验 | 全字段 | `contract_validation_failed` |
| 原图存在 | `image_asset.image_uri` | `image_file_missing` |
| 原图尺寸一致 | `image_asset.width/height` | `image_dimension_mismatch` |
| object ID 唯一 | `objects[].object_id` | `duplicate_object_id` |
| box 合法 | `objects[].box` | `invalid_box_geometry` / `box_outside_image` |
| box 尺寸阈值 | `objects[].box` | `tiny_box` / `extreme_box_aspect_ratio` |
| 重复框 | `objects[].box` | `duplicate_or_overlapping_box` |
| crop 存在 | `objects[].crop.crop_uri` | `crop_file_missing` |
| crop 尺寸阈值 | `objects[].crop.crop_uri` | `tiny_crop` / `extreme_crop_aspect_ratio` |
| crop 覆盖原框 | `objects[].crop.crop_box` + `objects[].box` | `crop_does_not_cover_box` |
| mask 存在和尺寸 | `objects[].geometry_detail.mask_uri` | `mask_file_missing` / `mask_dimension_mismatch` |
| 分类标签存在 | `objects[].classification.multi_labels` | `classification_labels_missing` |
| 已有质检失败 | `objects[].quality_check.qc_status` | `existing_object_qc_failed` |
| VLM 红框语义复核 | `object_type` + `multi_labels` + `inspection_content` + 原图 | `vlm_qc_target_mismatch` / `vlm_qc_needs_human_review` |

## 7. 设计边界

- 规则质检默认不调用大模型，适合作为每批标注后的第一层闸门。
- VLM 只做疑难语义复核或全量语义复核开关，不替代字段规则。
- 大字段不在报告里重复存全量内容：`generation_prompt` 存 preview，`raw_response` 存是否存在，完整内容通过 `metadata_uri` 回查。
- 所有 issue 必须带 `field_path`，人工复核可以直接知道问题来自哪个字段。
