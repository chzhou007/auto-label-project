# 质检 Agent 汇报一页纸

## 1. 我做的是什么

我做的是自动标注结果的分层质检 agent。

核心思路：

```text
规则先行 -> VLM 兜底 -> 人工收尾
```

目标是把数据生成、自动检测、自动分类后的结果，变成可自动检查、可追溯、可导出、可少量人工复核的数据闭环。

## 2. 数据字段怎么穿透

质检结论必须能追到：

```text
sample_id
image_asset: image_id / image_uri / width / height / source_type
source_context: generation_prompt / generation_model / collection_batch / camera_id / capture_time
scene_context: site / building / floor / room_name / room_type / task_group / inspection_content
objects[]: object_id / object_type / box / geometry_source / geometry_model / geometry_detail
crop: crop_id / crop_uri / crop_box / crop_expand_ratio / is_valid_crop
classification: multi_labels / classifier_type / classifier_name / prompt_version / raw_response
quality_check: qc_status / issue_flags / reviewer / comment
workflow: workflow_status / pipeline_id / pipeline_version
export: export_format / export_status / export_uri
```

每个问题都带 `field_path`，例如：

```text
image_asset.image_uri
objects[].box
objects[].crop.crop_uri
objects[].classification.multi_labels
objects[].geometry_detail.mask_uri
```

## 3. 质检 agent 怎么做

### Asset QC

用于检查只有图片/crop 的数据。

能自动做：

```text
图片是否可读
尺寸是否正常
crop 是否过小
crop 宽高比是否异常
crop 是否能通过文件名、manifest、metadata 回连样本
```

入口：

```text
scripts/run_asset_qc.py
```

### Metadata QC

用于检查 `AutoLabelSample` 标注结果。

能自动做：

```text
JSON/schema 契约检查
原图/crop/mask 文件存在性检查
图片尺寸一致性检查
box 越界/过小/极端比例检查
重复框检查
crop_box 是否覆盖 object.box
classification 字段完整性检查
已有 quality_check 状态检查
manual_review_queue.csv 生成
```

入口：

```text
scripts/run_qc_agent.py
```

### VLM Semantic QC

用于规则无法判断的视觉语义问题。

VLM 判断：

```text
红框是否真的框到目标
框是否漏头/漏脚/漏主体
是否框错对象
标签是否和图像一致
生成异常是否像真实工业异常
```

要求 VLM 返回结构化 JSON：

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

Pilot 入口：

```text
scripts/run_vlm_qc_pilot.ps1
docs/vlm_semantic_qc_pilot.md
```

## 4. 哪些自动做，哪些人工做

自动做：

```text
文件完整性
字段完整性
路径回连
box/crop/mask 几何一致性
分类字段完整性
风险汇总
人工队列生成
```

VLM 做：

```text
视觉语义判断
红框质量判断
标签和图像是否一致
```

人工做：

```text
VLM 不确定
规则和 VLM 冲突
业务语义模糊
通过样本抽样
最终工业合理性抽查
```

## 5. 全量测试结果

测试日期：2026-06-07。

单元测试：

```text
python -m unittest discover -s tests
Ran 38 tests
OK
```

metadata 全量规则质检：

```text
samples=209
objects=209
passed=209
failed=0
needs_human_review=0
manual_queue=15
```

图片资产全量质检：

```text
assets=3780
image=2340
crop=1440
passed=3780
failed=0
needs_human_review=0
issues={}
```

最新报告：

```text
data/qc/qc_20260607T161507_0800_report.json
data/qc/qc_20260607T161507_0800_manual_review_queue.csv
data/qc/asset_qc_20260607T161551_0800_report.json
data/qc/asset_qc_20260607T161551_0800_assets.csv
```

## 6. 模型能力边界

VLM 能做：

```text
看图判断目标是否存在
判断红框是否框错/漏框
判断标签语义是否大致匹配
输出 reason 和结构化 JSON
```

VLM 不适合做：

```text
文件是否存在
字段是否缺失
路径是否可追溯
box 是否越界
crop_box 是否覆盖 box
mask 尺寸是否一致
批量统计和可审计规则
```

所以质检 agent 的定位是规则先筛、VLM 处理语义问题、人工做少量兜底。

## 7. 有意思的发现

1. 只做 metadata QC 会漏掉只有图片/crop、还没转成 metadata 的数据，所以必须有 `asset_qc.py`。
2. crop 命名有很多历史变体，例如 `_person_1`、`_bbox_1`、`_item_1`、`_unique_id_1`、`_bbox_person_1`，需要兼容解析，否则会误判断链。
3. 规则 QC 全过只能说明文件、字段、几何链路没问题，不代表语义一定标对，所以后续必须接 VLM 或人工抽样。
4. 报告不能复制所有大字段，`generation_prompt` 和 `raw_response` 应存 preview 或 present flag，完整内容通过 `metadata_uri` 回查。

## 8. 汇报时可以直接说

我现在做的是一个分层质检 agent。第一层检查图片资产是否完整可读，第二层检查 `AutoLabelSample` 字段、box、crop、mask、classification 是否完整且一致，第三层用 VLM 做红框和标签的语义复核，最后只把失败、不确定和抽样样本交给人工。当前规则层已经完成全量测试：3780 个图片资产、209 个 metadata 样本全部通过；VLM 语义层代码路径和 pilot 脚本已有，但还没有全量实际调用 H20 跑完，这是下一步验证重点。
