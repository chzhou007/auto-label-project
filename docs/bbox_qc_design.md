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

第二层是 bbox crop 证据。针对带教说的“先切图再判断 bbox 是否兜住水渍”，脚本现在支持生成一张三联证据图：

- A：原始整图 + 红色 bbox，用来看 bbox 在全局位置是否合理。
- B：bbox 内部原始像素 crop，只回答“框内有没有目标水渍/污渍/液膜”。
- C：bbox 周边 expanded context crop，红框内是 bbox，红框外是邻近区域，用来回答“框外是否还有同一片水渍延伸”。

这层的核心字段是：

```text
inside_has_target_anomaly
outside_has_same_anomaly
outside_extension
containment_quality
box_quality
needs_human_review
```

推荐口径：

- 框内没有目标异常：`wrong_target`。
- 框内有目标异常、框外同一片异常明显延伸：`under_inclusive`。
- 框内有目标异常，但背景/设备/正常区域占比明显过大：`over_inclusive`。
- 框内有目标异常，框外没有连续延伸，留白可接受：`good`。
- 证据弱、画质差或模型无法稳定判断：`uncertain`，进人工复核。

第三层是二次定位。用本地 LocateAnything 在 ROI 里重新找一次异常区域，然后和 metadata 的框计算 IoU、中心偏移、面积比。它能给出一个候选替代框，适合发现框过窄、偏移、标到背景等问题；它对提示词和 ROI 很敏感，整图裸跑容易给出很大的框，所以更适合作为召回工具。

第四层是 VLM 语义复核。旧模式是把原图叠上红色 `object.box`，让 Qwen 判断红框是否框住了目标异常；新模式推荐使用 crop 证据图，让 Qwen 同时判断框内/框外。两种模式最终都收敛到 `good`、`over_inclusive`、`under_inclusive`、`wrong_target`、`uncertain`，方便后续统计。

## 当前测试结果

### 水渍训练集 bbox 全量 triage

对用户桌面这批训练集 metadata 做了 SJTU Qwen 全量复核，输入是三联证据图，模式是 `training_triage`。这批结果的目标不是最终评测集打分，而是训练集 scale 前筛选：

- `auto_accept_positive`：高纯度正例训练种子，可以直接训练。
- `rework_bbox_positive_candidate`：图中是正例或可重框正例，但当前 bbox 需要扩、缩或移动；保 TP 资产，修框后再训练。
- `label_conflict_review`：弱标签目录和视觉证据冲突，尤其是负例目录里出现地面水；人工复核，不自动入训练。
- `reject_from_positive_training`：无可用地面水、错类或硬负例，不进入正例训练。

v3 到 v4 的关键 prompt 迭代是：不要把“当前 bbox 框错但 A/C 图里确实有地面水”的正例直接 reject，而是归到 `rework_bbox_positive_candidate`，`rework_type=move_bbox`。同时保留硬约束：弱负例目录即使视觉像水，也不能 auto-accept 或 rework 进正例，必须进 label conflict。

全量 v4 结果：

| 指标 | v3 | v4 | 变化 |
|---|---:|---:|---:|
| 总 bbox | 278 | 278 | 0 |
| API error / parse error | 0 / 0 | 0 / 0 | 0 |
| 弱正例总数 | 202 | 202 | 0 |
| 高纯 clean seed | 71 | 58 | -13 |
| 正例 bbox 返工池 | 66 | 102 | +36 |
| 可恢复正例池 clean + rework | 137 | 160 | +23 |
| 正例 reject | 65 | 42 | -23 |
| 负例自动进入正例 | 0 | 0 | 0 |
| 负例冲突复核池 | 35 | 36 | +1 |

v4 的正例返工池里，`move_bbox=34`、`expand_bbox=64`、`shrink_bbox=3`。这说明原来的主要漏召回来自两类：bbox 框小截断水渍，以及 bbox 完全框偏但图片里有可重画的地面水。

当前结论：如果后续要 scale，不能只看 `auto_accept_positive` 的 58 张；那是高精度种子，不是全部 TP。应把 `auto_accept_positive + rework_bbox_positive_candidate = 160/202` 作为可恢复正例资产池，其中返工池必须修 bbox 后再用于训练。

结果文件：

- `data/qc/training_triage_v4_full_sjtu_qwen_20260701/training_triage_v4_summary.md`
- `data/qc/training_triage_v4_full_sjtu_qwen_20260701/training_triage_v4_manifest.csv`
- `data/qc/training_triage_v4_full_sjtu_qwen_20260701/training_triage_v3_v4_comparison.csv`

### v5.6 语义/几何拆分全量结果

v5.6 不再把整图语义、当前框质量和训练动作塞进一次判断。它先用无框整图做 label-blind 语义扫描，再用 bbox crop、上下文和四条边界放大图独立判断几何，最后由代码真值表融合。完整 prompt 和枚举见 `docs/bbox_qc_prompt_v5.md`。

2026-07-12 使用 SJTU `qwen` 完成 278 个 bbox 全量运行，严格完整性检查通过：278 个唯一对象、0 API error、0 缺失动作、0 缺失边界检查、全部为 `v5.6`。

| 指标 | v4 | v5.6 | 变化 |
|---|---:|---:|---:|
| 弱正例总数 | 202 | 202 | 0 |
| 高纯 clean seed | 58 | 24 | -34 |
| 正例 bbox 返工池 | 102 | 159 | +57 |
| 可恢复正例池 clean + rework | 160 (79.2%) | 183 (90.6%) | +23 |
| 正例人工复核 | 0 | 6 | +6 |
| 正例保留且不直接拒绝 | 160 (79.2%) | 189 (93.6%) | +29 |
| 正例 reject | 42 | 13 | -29 |
| 弱负例总数 | 76 | 76 | 0 |
| 负例自动进入 clean/rework | 0 | 0 | 0 |

逐对象迁移显示，v4 拒绝的弱正例中有 31 个被 v5.6 救回到 clean/rework；同时有 2 个旧版未拒绝样本被新版拒绝，净减少 29 个正例 reject。31 个救回样本中，大多数是“当前框落在设备、文字或无关背景，但整图另一处存在地面水”，应保留图片并 `move_bbox`，不能把图片本身当负例删掉。

这里的策略明确偏向 **保 TP 资产，但不保脏框**：clean seed 变少不是召回下降，而是 45 个 v4 clean 被四边界和背景占比规则收紧到 rework/manual/reject。修框后可使用 159 个 rework；6 个 manual 和 13 个 reject 在人工确认前都不能进入正例训练。

24 个 v5.6 clean 只能视为“主模型拟 clean”，不能仅凭这一次判断直接训练。全量 contact sheet 人工复查发现，光滑环氧地坪反光和框边连续水膜仍有争议样本。v5.7 因此增加独立 clean verifier：只对拟 clean 调用 `qwen3.6-27b` 反方复核，必须再次确认框内目标、四边不截断和背景不过量；任何异议或 API 失败都 fail-closed 到 rework/manual。最终可直接训练数以 verifier 通过数为准。

v5.7 对 24 个拟 clean 全量复核后的结果：

| clean gate 结果 | 数量 |
|---|---:|
| 最终 `auto_accept_positive` | 14 |
| 降级为 `under_inclusive / expand_bbox` | 7 |
| 降级为 `over_inclusive / shrink_bbox` | 3 |
| verifier/API error | 0 |

将这 24 条替换回 278 条全量清单后，有效 v5.7 结果为：14 个直接训练 clean、169 个修框后训练、6 个弱正例人工复核、13 个弱正例拒绝；可恢复正例仍为 `183/202 = 90.6%`，负例进入 clean/rework 仍为 `0/76`。因此 v5.7 没有牺牲保 TP 目标，只把 10 个可能污染训练的框从 clean 移到了返工池。

10 个降级样本中，7 个由带像素标签的六图主几何阶段直接发现，另外 3 个（`sample_baoshan_00045`、`sample_baoshan_00099`、`sample_a2_a1_00318`）由独立 verifier 否决。这说明两个改动都产生了实际增益，不是只增加 prompt 字数。

独立稳定性审计另抽取 69 个高风险样本，用 `qwen3.6-27b` 从头重跑 v5.6 两阶段：69/69 最终动作一致、31/31 个 v4 reject 救回样本仍被保留、12/12 个负例控制无 clean/rework 泄漏、API error 为 0。该结果说明路由稳定，但不是 human gold accuracy；最终 clean 仍采用更严格的 v5.7 gate。

### Mask 几何证据边界

六个目录的 278 个 mask 均存在、非空且尺寸正确，但 278/278 个 object bbox 都与 mask 非零像素的外接矩形完全相等，mask 像素也 100% 位于 bbox 内。结合 `final_bbox_source=image_difference_connected_components`，这说明 bbox 本来就是从 mask component 推出来的。

因此 `mask-bbox IoU=1` 只能证明生成链路内部自洽，不能作为独立 bbox 准确率，更不能证明框里是水、框外没有连续水体。mask 可用于给人工指出“生成图发生变化的位置”，但 v5.7 不把 mask 喂给盲语义阶段，也不允许 mask 自动放行，避免把生成器自身的定位结果循环论证成质检真值。

全量交付物：

- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/training_triage_summary.md`
- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/training_triage_manifest.csv`
- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/training_triage_transitions.csv`
- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/bbox_qc_bad_cases.md`
- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/training_triage_gallery.html`
- `data/qc/training_triage_v5_6_full_sjtu_qwen_20260712/training_triage_integrity.json`
- `data/qc/training_triage_v5_6_qwen36_audit_20260712/cross_model_audit.md`
- `data/qc/training_triage_v5_7_clean_gate_sjtu_20260712/training_triage_summary.md`
- `data/qc/training_triage_v5_7_effective_sjtu_20260712/training_triage_summary.md`
- `data/qc/training_triage_v5_7_effective_sjtu_20260712/training_triage_gallery.html`
- `data/qc/training_triage_v5_7_effective_sjtu_20260712/bbox_mask_geometry.md`

严格复算命令：

```bash
python scripts/summarize_bbox_triage_runs.py \
  --run-root data/qc/training_triage_v5_6_full_sjtu_qwen_20260712 \
  --baseline-root data/qc/training_triage_v4_full_sjtu_qwen_20260701 \
  --output-dir data/qc/training_triage_v5_6_full_sjtu_qwen_20260712 \
  --expected-total 278 \
  --expected-revision v5.6 \
  --strict
```

把 v5.7 clean gate 替换回全量并生成最终总表：

```bash
python scripts/summarize_bbox_triage_runs.py \
  --run-root data/qc/training_triage_v5_6_full_sjtu_qwen_20260712 \
  --replacement-root data/qc/training_triage_v5_7_clean_gate_sjtu_20260712 \
  --baseline-root data/qc/training_triage_v4_full_sjtu_qwen_20260701 \
  --output-dir data/qc/training_triage_v5_7_effective_sjtu_20260712 \
  --expected-total 278 \
  --strict
```

v5.7 推荐在线调用方式：

```bash
python scripts/run_anomaly_bbox_qc.py \
  --metadata-dir "/path/to/images_clear_water/metadata" \
  --asset-base-dir "/path/to/assets" \
  --output-dir "data/qc/training_triage_v5_7" \
  --review-mode training_triage \
  --triage-prompt-version v5 \
  --model qwen \
  --clean-verifier-model qwen3.6-27b \
  --workers 12 \
  --api-retries 4 \
  --retry-backoff-seconds 8 \
  --max-tokens 1200
```

### 旧输出样例 bbox 规则测试

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

主流程建议用 `规则层召回 + crop 证据 + Qwen 复核判定`：

1. 先跑规则层，把越界、极小框、crop 不覆盖、框和 selected grid 脱节的样本筛出来。
2. 对全量或规则层可疑样本生成三联 crop 证据图，先目视确认提示词是否符合带教口径。
3. 调用 Qwen 的 `crop_evidence` 模式，让模型分别判断框内、框外和 containment。
4. Qwen 判 `wrong_target`、`under_inclusive` 时进入返工清单。
5. Qwen 判 `over_inclusive`、`uncertain` 时进入人工复核。
6. LocateAnything 作为辅助：只在 Qwen 不确定或者需要给生成方一个建议替代框时调用。

这样做的原因很简单：规则层便宜、稳定，适合大批量扫；crop 证据层把“框内是否有水、框外是否漏掉水”拆开，容易给带教解释；Qwen 对语义判断更直接，适合确认框是否标错目标；LocateAnything 能给替代框，但单独使用时容易框大，适合做证据补充。

## 面向带教的框架图

```text
metadata JSON
  |
  |-- 规则检查：box 合法性 / grid 对齐 / expanded edit region / crop / mask
  |
  |-- 证据构造：A 整图红框 + B 框内 crop + C 周边 context
  |
  |-- VLM 判断：
  |     inside_has_target_anomaly?
  |     outside_has_same_anomaly?
  |     outside_extension?
  |     label_match?
  |
  |-- 裁决：
        good -> 放行
        wrong_target / under_inclusive -> 返工
        over_inclusive / uncertain / 规则异常 -> 人工复核
```

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

Mac 本机先只生成 crop 证据预览，不调用模型：

```bash
python scripts/run_anomaly_bbox_qc.py \
  --metadata-dir "/Users/ansel/Desktop/自动化标注/第一波metadata/images_clear_water/metadata" \
  --asset-base-dir "/Users/ansel/Desktop/自动化标注/第一波二筛" \
  --output-dir "data/qc/bbox_crop_evidence_preview" \
  --review-mode crop_evidence \
  --build-evidence-only \
  --limit 12 \
  --overwrite
```

Mac 本机调用 Qwen 做 crop-evidence 复核。不要把真实 key 写进仓库，优先使用本机环境变量或 Keychain helper：

```bash
python scripts/run_anomaly_bbox_qc.py \
  --metadata-dir "/Users/ansel/Desktop/自动化标注/第一波metadata/images_clear_water/metadata" \
  --asset-base-dir "/Users/ansel/Desktop/自动化标注/第一波二筛" \
  --output-dir "data/qc/bbox_crop_evidence_qwen" \
  --review-mode crop_evidence \
  --model "${QWEN_GEOMETRY_MODEL:-qwen}" \
  --interval-seconds 6.8 \
  --limit 20
```

Mac 本机调用 SJTU Qwen 做训练集 triage。24 workers 压测可跑通，但会出现服务端长尾排队；稳定批跑建议 12-16 workers：

```bash
eval "$(sjtu-ai env qwen)"
export SJTU_API_KEY="$OPENAI_API_KEY"
export SJTU_API_BASE_URL="$OPENAI_API_BASE"
export SJTU_API_DEFAULT_MODEL="$SJTU_MODEL"

python scripts/run_anomaly_bbox_qc.py \
  --metadata-dir "/Users/ansel/Desktop/自动化标注/第一波metadata/images_clear_water/metadata" \
  --asset-base-dir "/Users/ansel/Desktop/自动化标注/第一波二筛" \
  --output-dir "data/qc/training_triage_v4_full_sjtu_qwen_20260701/first_clear_water" \
  --review-mode training_triage \
  --model qwen \
  --workers 16 \
  --api-retries 4 \
  --retry-backoff-seconds 8 \
  --max-tokens 900 \
  --overwrite
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
