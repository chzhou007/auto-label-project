# 框标注质检方案

## 已看到的 metadata 字段

本地生成数据的框信息在 `outputs/outputs/metadata/*.json`。每张图目前基本是一张图一个目标，关键字段如下：

- `sample_id`：样本编号。
- `image_asset.image_uri`：生成图路径。
- `image_asset.width`、`image_asset.height`：图像尺寸。
- `image_asset.source_type`：是否为生成图。
- `image_asset.scene_context.inspection_content`：业务异常类型，例如 `diesel_leak`、`oil_leak`、`coolant_leak`。
- `objects[].object_id`：目标编号。
- `objects[].object_type`：目标类别，当前多为 `leakage_area`。
- `objects[].box`：最终标注框，格式是 `xyxy`。
- `objects[].geometry_source`：框来源，当前是 `synthetic_generator`。
- `objects[].geometry_model.model_name`：生成框链路说明，例如 `qwen3.6-plus_grid_selection + wan2.7-image-pro + image_diff_connected_components`。
- `objects[].geometry_detail.mask_uri`：mask 路径。
- `objects[].geometry_detail.generation_params.selected_grid`：VLM 选择的网格。
- `objects[].geometry_detail.generation_params.grid_bbox`：被选网格的坐标。
- `objects[].geometry_detail.generation_params.expanded_edit_bbox`：实际编辑区域。
- `objects[].geometry_detail.generation_params.final_bbox_source`：最终框来源。
- `objects[].geometry_detail.generation_params.vlm_selection.reason`：当时选择网格的理由。
- `objects[].crop.crop_box`、`objects[].crop.crop_uri`：裁剪框和裁剪图。
- `objects[].classification.multi_labels`：异常类型、液体类型、严重程度等标签。

## 质检判断逻辑

我把框质检拆成三层。

第一层是 metadata 规则。它不看语义，只检查坐标链路是否自洽：

- `object.box` 是否合法，是否越界，是否过小，长宽比是否异常。
- `crop_box` 是否包含 `object.box`。
- `object.box` 是否落在 `expanded_edit_bbox` 内。
- `object.box` 是否和 `selected_grid` 的 `grid_bbox` 有基本重叠。
- 如果 mask 是独立证据，再比较 mask 的内容 bbox 和 `object.box` 是否对齐。

第二层是二次定位。用本地 LocateAnything 在 ROI 里重新找一次异常区域，然后和 metadata 的框计算 IoU、中心偏移、面积比。它能给出一个候选替代框，适合发现框过窄、偏移、标到背景等问题；它对提示词和 ROI 很敏感，整图裸跑容易给出很大的框，所以更适合作为召回工具。

第三层是 VLM 红框复核。把原图叠上红色 `object.box`，让 Qwen 判断红框是否框住了目标异常，输出 `good`、`over_inclusive`、`under_inclusive`、`wrong_target`、`uncertain`。它更接近人工质检，适合做最终判定，但有接口延迟和 token 成本。

## 当前测试结果

规则层全量跑了 `outputs/outputs/metadata`：

- 总样本：209
- 规则通过：208
- 规则进入复核：1
- 抽样复核队列：16
- 命中的真实问题码：`box_disconnected_from_selected_grid`

命中的样本是 `sample_000127`：

- `selected_grid = B2`
- `grid_bbox = [480,270,960,540]`
- `object.box = [1007,348,1022,384]`
- 两者 IoU = 0
- `object.box` 仍在 `expanded_edit_bbox = [384,216,1056,594]` 内；问题更准确地说是最终差分框偏到了选中网格外侧。

随后用 SJTU 的 `qwen` 路由做红框复核，模型判断：

- `box_quality = wrong_target`
- `issue_flags = ["wrong_target", "no_visible_anomaly"]`
- 结论：红框框住的是阀体上的螺栓或机械部件，框内没有可见柴油泄漏、液体、污渍或泄漏痕迹。

LocateAnything 也在这个样本的 expanded ROI 上跑通了。它给了一个更大的右上区域，能提示 metadata 框可能过窄或不稳定，但结果偏召回，不适合单独作为失败结论。

## 目前推荐落地策略

主流程建议用 `规则层召回 + Qwen 复核判定`：

1. 先跑规则层，把越界、极小框、crop 不覆盖、框和 selected grid 脱节的样本筛出来。
2. 对规则层可疑样本调用 SJTU Qwen 红框复核。
3. Qwen 判 `wrong_target`、`no_visible_anomaly`、`under_inclusive` 时进入返工清单。
4. Qwen 判 `uncertain` 时进入少量人工复核。
5. LocateAnything 作为辅助：只在 Qwen 不确定或者需要给生成方一个建议替代框时调用。

这样做的原因很简单：规则层便宜、稳定，适合大批量扫；Qwen 对红框语义判断更直接，适合确认框是否标错目标；LocateAnything 能给替代框，但单独使用时容易框大，适合做证据补充。

## 还需要向生成侧确认的信息

本地 metadata 已经够跑第一版质检，但如果要把框问题定位到生成链路，还需要生成侧补充这些字段或口径：

- `mask_uri` 现在到底是真实差分/分割 mask，还是由最终框直接画出来的矩形 mask。
- `final_bbox_source=image_difference_within_selected_grid` 时，为什么最终框允许跑到 `grid_bbox` 外；这是设计允许，还是后处理膨胀带来的副作用。
- 是否能保留原始 diff 连通域列表，避免只保留最终一个框。
- 是否能保留生成前原图、生成后图、差分热力图、最终 mask 四件套。
- 是否能保留每个候选框的面积、置信度、连通域编号和筛选原因。

## 可复现命令

规则层全量：

```powershell
python scripts\run_qc_agent.py --config configs\autolabel.yaml --metadata-dir "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata" --asset-base-dir "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs" --output-dir "C:\Users\19310\Desktop\万国数据实习\异常检测\auto-label-project\auto-label-project\data\qc\bbox_qc_full_rule_after" --disable-vlm
```

SJTU Qwen 红框复核单样本：

```powershell
$env:QWEN397B_API_KEY=$env:SJTU_API_KEY
$env:QWEN_GEOMETRY_API_URL=$env:SJTU_API_BASE_URL
$env:QWEN_GEOMETRY_MODEL="qwen"
python scripts\run_qc_agent.py --config configs\autolabel.yaml --sample "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata\sample_000127.json" --asset-base-dir "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs" --output-dir "C:\Users\19310\Desktop\万国数据实习\异常检测\auto-label-project\auto-label-project\data\qc\bbox_qc_sample127_sjtu_qwen" --enable-vlm
```

LocateAnything ROI 测试：

```powershell
cd D:\LocateAnything-local
$env:HF_HOME="D:\hf_cache"
$env:HUGGINGFACE_HUB_CACHE="D:\hf_cache\hub"
$env:TORCH_HOME="D:\torch_cache"
.\.venv\Scripts\python.exe .\cli.py --image "D:\LocateAnything-local\outputs\bbox_qc_probe\sample_000127_expanded_roi.jpg" --task custom --question "Locate the actual pale yellow diesel leak or wet oil stain near the pipe joint in this industrial CCTV image crop. Return exactly one tight bounding box around the visible leakage only. If no visible leakage exists, return None." --device cuda --dtype fp16 --generation-mode fast --max-new-tokens 512 --temperature 0.0 --json-out "D:\LocateAnything-local\outputs\bbox_qc_probe\sample_000127_locateanything_roi.json" --annotate-out "D:\LocateAnything-local\outputs\bbox_qc_probe\sample_000127_locateanything_roi_overlay.jpg"
```
