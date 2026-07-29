# 自动标注质检 Agent 汇报材料

面向带教汇报。核心目标是讲清楚数据字段、质检 agent 做法、全量测试进度、自动化边界、人工复核边界和模型边界。

## 1. 一句话结论

这个仓库本质上是一条工业视觉数据的自动化标注流水线：

```text
原始图片/视频/生成图
  -> 预处理/生成/检测/裁剪/分类
  -> 统一 AutoLabelSample metadata
  -> 质检 agent
  -> Label Studio / 下游训练数据
```

我负责的质检部分按三层处理：

```text
规则能判定的自动判定
规则不能判定的再交给 VLM
VLM 仍不确定的才进入少量人工复核
```

目前已经完成：

- 图片资产级 QC：全量 3780 个图片资产通过。
- metadata 规则 QC：全量 209 个 `AutoLabelSample` 通过。
- 字段穿透设计：每个问题都能追到 `sample_id`、原图、object、box、crop、mask、classification、workflow/export 和具体 `field_path`。

尚未完成：

- H20/VLM 语义质检全量调用测试。代码路径已有，未实际全量连模型跑。

## 2. 项目解决的问题

工业视觉数据标注里有三类成本：

1. 原始图片/视频整理成本。
2. 人工画框和打标签成本。
3. 人工二次质检成本。

本项目要解决的是：

```text
把自动生成、自动检测、自动分类结果统一成一个可追溯的数据结构，
再用规则 + VLM + 少量人工复核把坏数据拦下来。
```

最终让散乱图片和 crop 进入可验证、可导出、可复查的标注样本链路。

## 3. 字段体系

完整字段清单见：

```text
docs/qc_agent_field_design.md
```

汇报时可以按五组讲。

### 3.1 Manifest 输入字段

这些字段来自预处理 manifest，用于说明原图从哪里来、属于什么任务。

```text
sample_id
image_id
image_uri
source_type
task_mode
task_key
object_type
anomaly_type
site
building
floor
room_name
room_type
task_group
inspection_content
collection_batch
camera_id
capture_time
width
height
```

作用：

- `sample_id` / `image_id`：追踪样本。
- `image_uri`：找到原图。
- `source_type`：区分 generated、cctv、manual_upload。
- `task_mode`：区分 direct 和 generation。
- `task_key` / `object_type` / `anomaly_type`：决定后续标注和质检策略。
- `room_type` / `task_group` / `inspection_content`：给 VLM 和人工复核提供语义上下文。
- `width` / `height`：核对真实图片尺寸。

### 3.2 样本级字段

`AutoLabelSample` 的样本级字段：

```text
sample_id
image_asset.image_id
image_asset.image_uri
image_asset.width
image_asset.height
image_asset.source_type
image_asset.source_context.generation_prompt
image_asset.source_context.generation_model
image_asset.source_context.collection_batch
image_asset.source_context.camera_id
image_asset.source_context.capture_time
image_asset.scene_context.site
image_asset.scene_context.building
image_asset.scene_context.floor
image_asset.scene_context.room_name
image_asset.scene_context.room_type
image_asset.scene_context.task_group
image_asset.scene_context.inspection_content
```

质检用途：

- 原图路径、尺寸、来源可验证。
- 生成样本能追到 prompt 和 generation model。
- 工业场景能追到机房、任务组、巡检内容。

### 3.3 对象级字段

每个 `objects[]` 表示一个框、一个 crop、一个分类结果。

```text
objects[].object_id
objects[].object_type
objects[].box.format
objects[].box.x1
objects[].box.y1
objects[].box.x2
objects[].box.y2
objects[].geometry_source
objects[].geometry_model.model_name
objects[].geometry_model.model_version
objects[].geometry_model.confidence
objects[].geometry_detail.polygon
objects[].geometry_detail.mask_uri
objects[].geometry_detail.mask_format
objects[].geometry_detail.generation_params
```

质检用途：

- `box` 做越界、尺寸、比例、重复框检查。
- `geometry_source` 区分检测模型、生成器、人标。
- `geometry_model` 用来按模型聚合问题。
- `mask_uri` 用来检查 mask 文件和尺寸。
- `generation_params` 用来追溯生成图的定位过程。

### 3.4 Crop 字段

```text
objects[].crop.crop_id
objects[].crop.crop_uri
objects[].crop.crop_box.format
objects[].crop.crop_box.x1
objects[].crop.crop_box.y1
objects[].crop.crop_box.x2
objects[].crop.crop_box.y2
objects[].crop.crop_expand_ratio
objects[].crop.is_valid_crop
```

质检用途：

- crop 文件是否存在。
- crop 是否可读。
- crop 尺寸是否异常。
- `crop_box` 是否完整覆盖 `object.box`。

### 3.5 分类与质检字段

```text
objects[].classification.multi_labels[].label_key
objects[].classification.multi_labels[].label_value
objects[].classification.multi_labels[].confidence
objects[].classification.multi_labels[].evidence
objects[].classification.classifier_type
objects[].classification.classifier_name
objects[].classification.classifier_version
objects[].classification.prompt_version
objects[].classification.raw_response

objects[].quality_check.qc_sampled
objects[].quality_check.qc_status
objects[].quality_check.reviewed_labels[].label_key
objects[].quality_check.reviewed_labels[].label_value
objects[].quality_check.issue_flags
objects[].quality_check.reviewer
objects[].quality_check.review_time
objects[].quality_check.comment
```

质检用途：

- 判断分类标签是否缺失。
- 判断标签和任务语义是否一致。
- 读取已有质量状态，避免把失败样本继续导出。
- 把 VLM 或人工复核结果写回对象级 `quality_check`。

### 3.6 工作流和导出字段

```text
qc_policy.qc_mode
qc_policy.sampling_ratio
qc_policy.sampling_method
qc_policy.fail_policy
qc_policy.qc_batch_id

workflow.workflow_status
workflow.pipeline_id
workflow.pipeline_version
workflow.created_time
workflow.updated_time

export.export_format
export.export_status
export.export_uri
export.labelstudio_mapping
```

质检用途：

- 判断样本是否已经到可质检阶段。
- 记录 pipeline 版本。
- 决定抽样比例。
- 避免坏数据导出到 Label Studio。

## 4. 质检 Agent 设计

整体分成六层。

### 4.1 Asset QC：图片资产级检查

入口：

```text
scripts/run_asset_qc.py
```

输入：

- 原始图片/视频帧目录。
- crop 目录。
- generated image 目录。
- mask 目录。
- grid preview 目录。
- manifest。
- metadata。

检查：

```text
图片是否可读
图片尺寸是否可读
crop 是否过小
crop 宽高比是否极端
crop 文件名能否反推原始帧
crop / image / mask 能否回连 manifest 或 metadata
```

它解决的是图片数据本身是否完整、能不能被下游处理。

### 4.2 Contract QC：metadata 契约检查

入口：

```text
scripts/run_qc_agent.py
```

检查：

```text
是否是合法 JSON
是否满足 AutoLabelSample schema
必填字段是否存在
枚举字段是否合法
box / crop / classification 结构是否完整
```

### 4.3 Geometry QC：框和 crop 几何检查

检查：

```text
box 是否越界
box 是否 x2 <= x1 或 y2 <= y1
box 是否太小
box 是否长宽比异常
同图是否有重复框
crop 文件是否存在
crop 尺寸是否异常
crop_box 是否覆盖 object.box
mask 尺寸是否和原图一致
```

这一层最稳定，完全不需要大模型。

### 4.4 Label QC：分类和业务字段检查

检查：

```text
classification.multi_labels 是否存在
label_key / label_value 是否完整
生成样本的 anomaly_type 是否能追溯
inspection_content 是否能作为语义判断依据
已有 quality_check 是否 failed / pending / discarded
```

### 4.5 Semantic QC：VLM 语义复核

这层才需要 H20/VLM。

做法：

1. 在原图上叠加红框。
2. 把红框图、`object_type`、`label_summary`、`inspection_content` 发给 VLM。
3. 要求 VLM 返回结构化 JSON：

```json
{
  "visible_target": true,
  "box_quality": "good",
  "label_match": true,
  "needs_human_review": false,
  "issue_flags": [],
  "reason": ""
}
```

VLM 只负责规则无法判断的视觉语义问题：

```text
红框是否真的框到目标
是否框错对象
是否漏掉头/脚/主要区域
异常区域是否真的像对应异常
标签和图像是否一致
```

Pilot 入口：

```text
scripts/run_vlm_qc_pilot.ps1
docs/vlm_semantic_qc_pilot.md
```

### 4.6 Risk Router：风险分流

最终输出三类：

```text
passed
  自动通过

failed
  明确失败，直接拦截或返工

needs_human_review
  规则不确定、VLM 不确定、或抽样样本
```

对应输出：

```text
report.json
manual_review_queue.csv
```

`manual_review_queue.csv` 只放人工需要看的样本，避免人工二次质检全量铺开。

## 5. 哪些 agent 可以自动做

可以自动完成：

```text
图片可读性检查
图片尺寸检查
crop 可读性检查
crop 尺寸/比例检查
文件路径回连检查
metadata 契约检查
box 越界检查
box 太小/极端比例检查
重复框检查
crop_box 覆盖 box 检查
mask 文件存在性检查
mask 尺寸检查
classification 字段完整性检查
已有 quality_check 状态读取
人工复核队列生成
质检报告生成
```

这些都不需要人工，也不需要大模型。

## 6. 哪些需要 VLM 或人工

需要 VLM 判断：

```text
框里是否真的是目标
人体是否完整可见
框是否漏头/漏脚
泄漏区域是否像真实泄漏
生成异常是否自然
标签是否和图像语义一致
```

必须人工或建议人工：

```text
VLM 输出 uncertain 的样本
VLM 和规则结论冲突的样本
业务定义模糊的异常类型
极少量通过样本抽样
生成图真实性和工业合理性最终抽查
```

人工不应该看全部，只看：

```text
规则失败样本
VLM 不确定样本
抽样样本
```

## 7. 模型能力边界

H20/VLM 能做：

```text
看图判断框内是否有目标
判断人是否被截断
判断图像异常是否大致符合标签
给出自然语言 reason
输出结构化 JSON
```

H20/VLM 不适合单独承担：

```text
文件是否存在
路径是否可追溯
字段是否缺失
box 是否越界
crop_box 是否覆盖 box
mask 尺寸是否一致
批量统计和汇总
稳定执行可审计规则
```

质检不能只靠问大模型。当前采用的做法是：

```text
规则先行，VLM 兜底，人工收尾。
```

## 8. 全量测试结果

测试日期：2026-06-07。

### 8.1 单元测试

命令：

```powershell
python -m unittest discover -s tests
```

结果：

```text
Ran 38 tests
OK
```

编译检查：

```powershell
python -m compileall autolabel scripts tests
```

结果：通过。

### 8.2 Metadata 全量质检

命令：

```powershell
python scripts/run_qc_agent.py `
  --config configs/autolabel.yaml `
  --metadata-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata `
  --asset-base-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs `
  --output-dir data/qc `
  --sampling-ratio 0.05 `
  --disable-vlm
```

结果：

```text
samples=209
objects=209
passed=209
failed=0
needs_human_review=0
manual_queue=15
```

报告：

```text
data/qc/qc_20260607T161507_0800_report.json
data/qc/qc_20260607T161507_0800_manual_review_queue.csv
```

解释：

- 209 个 metadata 规则全部通过。
- 15 个进入人工队列来自 5% 抽样，不能按失败样本理解。
- 当前没有发现字段断链、缺文件、框越界、crop 异常或 mask 异常。

### 8.3 图片资产全量质检

命令：

```powershell
python scripts/run_asset_qc.py `
  --image-dir C:\Users\19310\Desktop\万国数据实习\异常检测\image_sequence\image_sequence `
  --image-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\generated_images `
  --image-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\masks `
  --image-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\grid_previews `
  --crop-dir C:\Users\19310\Desktop\万国数据实习\异常检测\crops\crops `
  --crop-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\crops `
  --manifest C:\Users\19310\Desktop\万国数据实习\异常检测\image_sequence\image_sequence\manifest.csv `
  --metadata-dir C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata `
  --output-dir data/qc
```

结果：

```text
assets=3780
image=2340
crop=1440
passed=3780
failed=0
needs_human_review=0
issues={}
```

报告：

```text
data/qc/asset_qc_20260607T161551_0800_report.json
data/qc/asset_qc_20260607T161551_0800_assets.csv
```

解释：

- 原始帧、生成图、mask、grid preview、直接 crop、生成 crop 全部可读。
- crop 命名均能回连 manifest 或 metadata。
- 没有坏图、极端小图、极端比例 crop。

## 9. 发现的有意思问题

### 9.1 只有 metadata 质检不够

最开始只跑 metadata 会漏掉一批只有图片/crop 的数据。

因此补了 `asset_qc.py`，先把图片资产也纳入：

```text
image_sequence
crops
generated_images
masks
grid_previews
```

这能覆盖上游还没转成 `AutoLabelSample` 的场景。

### 9.2 crop 文件命名有历史变体

实际 crop 文件名里出现过：

```text
_person_1
_bbox_1
_1
_item_1
_unique_id_1
_auto_001
_bbox_person_1
```

这些如果不兼容，就会误判成无法回连原图。

因此 `asset_qc.py` 增加了多种命名解析规则，让 crop 能反推 `sample_id` 和原始帧。

### 9.3 质检报告不能复制所有大字段

`generation_prompt`、`raw_response` 可能很长。如果在报告里全量复制，会导致报告膨胀。

处理方式：

```text
报告里存 preview / present flag / metadata_uri
完整内容通过 metadata_uri 回查
```

这样既有数据穿透，又不会让报告过大。

### 9.4 目前规则质检全过，但不代表语义全对

规则质检能证明：

```text
文件完整
字段完整
路径可追
框和 crop 几何合法
mask 尺寸正常
```

但不能证明：

```text
框一定框到了正确目标
生成异常一定真实
标签一定语义正确
```

这些需要 VLM 或人工抽检。

## 10. 明天汇报建议

建议按这个顺序讲：

1. 项目目标：自动标注流水线，减少人工标注和质检成本。
2. 数据核心：统一成 `AutoLabelSample`，所有字段可追溯。
3. 质检策略：规则先行，VLM 兜底，人工收尾。
4. 当前实现：`asset_qc.py` + `qc_agent.py`。
5. 全量结果：3780 图片资产全过，209 metadata 全过。
6. 边界：尚未做 H20/VLM 全量语义复核。
7. 下一步：用 `scripts/run_vlm_qc_pilot.ps1` 接 H20 做红框语义复核 pilot，再决定是否全量启用。

可以直接说：

> 我现在做的是分层质检体系。图片资产、metadata、框、crop、mask、classification、workflow/export 都会被自动检查。规则无法判断的视觉语义问题，再交给 H20/VLM 做红框复核，最后只把高风险和抽样样本交给人工。

## 11. 可能被问到的问题

### Q1：大模型在质检里占什么位置？

大模型只是最后一层。

```text
文件、字段、路径、box、crop、mask 用规则判断。
视觉语义才问 VLM。
```

### Q2：为什么需要字段穿透？

因为质检失败时必须知道：

```text
哪张图
哪个框
哪个 crop
哪个字段
哪个模型
哪个导出状态
```

否则无法返工，也无法定位上游问题。

### Q3：目前是否全量测试？

规则层已经全量测试：

```text
3780 个图片资产全过
209 个 metadata 全过
38 个单元测试通过
```

语义层尚未全量测试：

```text
H20/VLM 红框判断还没全量跑
```

### Q4：哪些要人工？

人工只看：

```text
规则失败
VLM 不确定
VLM 和规则冲突
通过样本抽样
```

### Q5：下一步怎么做？

1. 配置 H20 key 和 VPN 环境。
2. 对一小批样本跑 VLM 红框复核。
3. 比较 VLM 结论和人工结论。
4. 决定是否全量启用 VLM 语义质检。
