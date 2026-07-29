# BBox QC Prompt v5.7

## 目标

v5 用于训练集 scale 前的全量筛选，目标按优先级排序：

1. 不丢弃图中真实存在、但当前 bbox 框错的地面清水 TP 资产。
2. 只有语义明确、主几何模型判 clean、且独立 clean verifier 再次确认的正例进入 `auto_accept_positive`。
3. weak negative 目录中的视觉正例只进入 `label_conflict_review`，不得自动进入正例训练或返工训练池。
4. 阴影、均匀镜面反光、文字、设备表面、墙面-only 水迹和彩色污渍不能作为地面清水正例。
5. 证据不足时显式 `manual_review`，不能靠模型猜测补全。

## 为什么从 v4 拆开

v4 把原图语义、bbox 几何、weak label 对齐和训练动作放在同一个长 prompt 中，并把 A/B/C 三图横向拼接后压缩到 1280 像素。实测会漏掉远离当前 bbox 的地面水，例如当前框落在设备上、真实水迹位于整图另一侧时，模型可能只看框附近并直接拒绝。

v5.7 将流程拆成四个职责单一的部分：

1. 原图盲语义扫描：只看无框整图，不接收 sample id、目录、metadata、weak label 或 bbox。
2. bbox 几何判断：只看独立的 bbox crop 和上下文 crop，不接收 weak label，也不输出训练动作。
3. 确定性融合：由代码依据两个阶段的枚举输出和 weak label 生成最终动作。
4. clean seed 反方复核：只复核主流程拟 `auto_accept_positive` 的少量样本；任一否决证据都降级到 rework/manual。

当前 synthetic mask 的非零外接矩形与 object bbox 在 278/278 个对象上完全相等，因为 bbox 就来自 diff connected component。为避免循环论证，mask 不进入盲语义和 clean 自动放行条件；它只可作为生成变化位置的辅助人工证据。

## 阶段一：原图盲语义扫描

输入是一张无标注、无红框的原始整图。模型必须先检查图像是否可复核，再按左上到右下扫描所有地面区域，并对每个可疑区域尝试使用阴影、普通反光、文字、墙面水等替代解释。

正例证据：

- `irregular_wet_dry_boundary`：不规则湿干边界。
- `pooled_liquid`：可见积液或水体。
- `wet_texture_change`：局部湿润变暗并形成连续形状。
- `droplet_or_flow_trail`：水滴、流痕或从设备底部延伸到地面。
- `reflection_with_liquid_boundary`：高光同时伴随液体边界或纹理变化。
- `floor_equipment_junction`：目标位于地面和设备底座交界。

这里的 floor 必须是可行走、与房间或走道连续的地面。机柜顶、托盘、桥架、设备平台、金属盖板和桌面即使水平、即使有真实液体，也属于 `equipment_surface`，不是本任务正例。

不足以单独判正：

- 均匀发亮的环氧地坪。
- 与灯具排列一致的规则高光。
- 阴影、透视线、地砖缝、文字和时间戳。
- 仅位于竖直墙面或设备表面的水迹。
- 彩色液体、固体污渍、人物或杂物。

输出：

```json
{
  "semantic_verdict": "clear_positive",
  "candidate_location": "bottom_right",
  "candidate_surface": "floor",
  "positive_evidence": [
    "irregular_wet_dry_boundary",
    "reflection_with_liquid_boundary"
  ],
  "hard_negative_type": "none",
  "reason": "一句话描述实际看见的证据和已排除的主要混淆项"
}
```

`semantic_verdict` 只能是：

- `clear_positive`：正证据强，主要替代解释明显更弱。
- `probable_positive`：有局部正证据，但反光或阴影仍是合理解释。
- `hard_negative`：整图可复核，且没有地面清水候选。
- `uncertain`：地面不可见、画质差或证据无法稳定区分。

`candidate_surface` 只能是 `floor`、`floor_equipment_junction`、`wall`、`equipment_surface`、`other`、`none`、`uncertain`。解析层会把落在 `wall` 或 `equipment_surface` 的视觉液体 veto 为本任务 hard negative，防止模型在自然语言理由里看懂表面、却仍输出正例。

## 阶段二：bbox 几何判断

模型依次接收六张独立图片：

1. B：当前 bbox 内部原始像素，整张图都属于框内。
2. C：bbox 周边上下文，红框表示当前 bbox。
3. LEFT EDGE：左边界放大条带，红色竖线是 bbox 左边界。
4. TOP EDGE：上边界放大条带，红色横线是 bbox 上边界。
5. RIGHT EDGE：右边界放大条带，红色竖线是 bbox 右边界。
6. BOTTOM EDGE：下边界放大条带，红色横线是 bbox 下边界。

六张图的像素内都写入了 `B INSIDE BBOX`、`C CONTEXT + RED BBOX`、`LEFT/TOP/RIGHT/BOTTOM EDGE` 标签，避免多图模型只靠消息顺序猜图义。

模型不知道 weak label、整图语义结果和最终训练策略，只负责判断当前框。

输出：

```json
{
  "inside_target_status": "clear_target",
  "target_crosses_left_edge": false,
  "target_crosses_top_edge": false,
  "target_crosses_right_edge": false,
  "target_crosses_bottom_edge": false,
  "target_area_fraction": "50_to_75_percent",
  "background_excess": false,
  "outside_relation": "none",
  "box_quality": "good",
  "target_surface": "floor",
  "dominant_confounder": "none",
  "reason": "一句话说明框内目标、框边界和主要混淆项"
}
```

关键枚举：

- `inside_target_status`：`clear_target`、`probable_target`、`no_target`、`uncertain`。
- `outside_relation`：`continuous_main_target`、`minor_same_target_fringe`、`separate_target`、`none`、`uncertain`。
- `box_quality`：`good`、`under_inclusive`、`over_inclusive`、`wrong_target`、`uncertain`。
- `target_surface`：`floor`、`floor_equipment_junction`、`wall`、`equipment_surface`、`other`、`none`、`uncertain`。

四条 `target_crosses_*_edge` 必须根据对应边界放大条带逐边输出。任一边为 `true` 时，解析层会强制 `outside_relation=continuous_main_target` 和 `box_quality=under_inclusive`；任一字段缺失时，几何结果强制 `uncertain`，不得自动放行。

`background_excess` 也必须输出。它表示框内正常地面、设备、桌椅、人物或其他无关背景是否明显过多。为 `true` 且没有穿框边缘时，解析层强制 `over_inclusive`；如果同时存在穿框边缘，说明当前框既漏目标又含大量背景，返工动作改为 `move_bbox`，即重新画框。

`target_area_fraction` 必须先把水体/湿痕视作前景，估算其占整个 bbox 的面积档位：`none`、`lt_10_percent`、`10_to_25_percent`、`25_to_50_percent`、`50_to_75_percent`、`gt_75_percent`、`uncertain`。低于 25% 时解析层直接认定背景过量；25% 到 50% 且存在设备、桌椅、箱子或大块正常地面时也判背景过量。字段缺失或不确定时不得自动放行。

几何口径：

- `good`：框内是 `clear_target`，主体在地面或地面-设备交界，主要目标已覆盖，背景不过多；少量弱边缘可以是 `minor_same_target_fringe`。
- `under_inclusive`：同一片主要水体或核心水渍被框边界明显截断。
- `over_inclusive`：框内有明确目标，但正常地面、设备或背景过多。
- `wrong_target`：框内没有目标，或主体是阴影、普通反光、文字、设备、墙面-only 等。C 图框外另有水时，当前框仍是 `wrong_target`。
- `uncertain`：框内只到 `probable_target`，或画质不足。

## 阶段四：clean seed 独立反方复核

主流程输出 `auto_accept_positive` 只表示“拟 clean”。若配置 `--clean-verifier-model`，才会额外调用独立模型；rework、manual、conflict 和 reject 不增加该调用，因此成本只随拟 clean 数增长。

verifier 接收七张图：完整原图红框、B、C、LEFT、TOP、RIGHT、BOTTOM。它使用 `clean_verifier_prompt_v5()`，默认假设当前框不可直接训练，主动寻找四类否决证据：

- 同一主要目标穿过任一 bbox 边界。
- 框内目标占比不足或无关背景过多。
- 框内是普通反光、阴影、设备面或其他错目标。
- 水与反光、主要延伸与弱尾部无法稳定区分。

只有 verifier 再次返回 `clear_target + good + floor/floor_equipment_junction + 四边不穿线 + background_excess=false + 有效目标面积` 时，最终动作才保留 `auto_accept_positive`。否则按 verifier 的几何结论自动降级：under/over/wrong 分别进入 expand/shrink/move 返工；不确定或 verifier 请求失败进入 manual，不能 fail-open。

## 确定性融合

| Weak label | Semantic | Geometry | Final action | Rework |
|---|---|---|---|---|
| positive | clear positive | clean good bbox + verifier pass | `auto_accept_positive` | `none` |
| positive | clear/probable positive | under-inclusive | `rework_bbox_positive_candidate` | `expand_bbox` |
| positive | clear/probable positive | over-inclusive | `rework_bbox_positive_candidate` | `shrink_bbox` |
| positive | clear/probable positive | wrong target | `rework_bbox_positive_candidate` | `move_bbox` |
| positive | probable positive | good bbox | `manual_review` | `uncertain` |
| positive | hard negative | any stable geometry | `reject_from_positive_training` | `relabeled_negative_or_other` |
| negative | clear/probable positive | any stable geometry | `label_conflict_review` | 保留几何建议但禁止训练 |
| negative | hard negative | any stable geometry | `reject_from_positive_training` | `relabeled_negative_or_other` |
| any | semantic or geometry uncertain | any | `manual_review` | `uncertain` |

例外保护：若原图语义阶段输出 `hard_negative`，但 bbox crop 阶段输出地面上的 `clear_target` 或 `probable_target`，两个独立阶段存在强冲突，必须进入 `manual_review`，不得直接 reject。这样可以保住整图缩放后漏掉的小目标。

自动放行必须同时满足：

- weak label 为 `positive_floor_clear_water`。
- semantic 为 `clear_positive`。
- bbox 内为 `clear_target`。
- `box_quality=good`。
- `outside_relation` 为 `none` 或 `minor_same_target_fringe`。
- `target_surface` 为 `floor` 或 `floor_equipment_junction`。
- 已配置 clean verifier 时，`clean_verifier_passed=true`；verifier 超时或解析失败必须 fail-closed 到 manual。

## 公开工具中的可迁移做法

- Cleanlab Object Detection / ObjectLab 将错误拆成漏标、错类和定位错误。v5 对应拆成图像级候选、框内目标和几何质量，而不是用一个 yes/no 覆盖所有问题。
  - https://docs.cleanlab.ai/stable/tutorials/object_detection.html
  - https://arxiv.org/abs/2309.00832
- FiftyOne 的 mistakenness 同时区分标签错误、定位错误、possible missing 和 possible spurious。v5 的 `semantic_verdict`、`box_quality`、`move_bbox` 和 hard negative 分层采用同类思路。
  - https://docs.voxel51.com/getting_started/object_detection/03_finding_mistakes.html
- CVAT consensus 使用独立重复标注和 IoU 匹配。v5 challenge set 使用 Qwen 与 Qwen3.6 独立复核最终动作，不把同一模型的一次输出当成 gold。
  - https://docs.cvat.ai/docs/manual/advanced/analytics-and-monitoring/consensus/
- Datumaro 的 detection validator 强调全数据集分布、异常尺寸和类别不平衡。v5 汇总器按波次、目录、动作和 box quality 分层报告，避免只看总数。
  - https://open-edge-platform.github.io/datumaro/latest/docs/command-reference/validate.html
- TIDE 将分类、定位、背景和漏检错误分开分析。v5 同样把语义 FN 与 bbox localization error 分开统计。
  - https://arxiv.org/abs/2008.08115

## 代码入口

- 完整原图 prompt：`semantic_prompt_v5()`。
- 完整 bbox prompt：`geometry_prompt_v5()`。
- clean 反方 prompt：`clean_verifier_prompt_v5()`。
- 确定性真值表：`fuse_training_triage_v5()`。
- clean 否决融合：`apply_clean_verifier_v5()`。
- 全量运行：`scripts/run_anomaly_bbox_qc.py --review-mode training_triage --triage-prompt-version v5`。
- 汇总比较：`scripts/summarize_bbox_triage_runs.py`。
