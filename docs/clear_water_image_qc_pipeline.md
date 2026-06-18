# 漏水图片质量质检流程

这套流程服务于 clear_water 正样本筛选。输入是第一波或第二波二筛目录，目录里包含人工预分的 `clear_water`、`wall_water`、`other` 三类。运行时模型只看图片像素，文件夹名只用于离线统计和阈值校准。

## 数据入口

第一波目录：

```text
images_clear_water
images_wall_water
images_others
```

第二波目录：

```text
images_floor_clear_water
images_wall_water
images_others
```

脚本会把每张图展开成一行记录，核心字段如下：

```text
file
file_name
folder
weak_label
prompt_version
model
selected_clear_water
selection_reason
can_see_image
usable
image_quality
floor_clear_water_visible
evidence_strength
liquid_location
hard_negative_type
decision
confidence
evidence_summary
latency_s
total_tokens
```

`weak_label` 来自文件夹，只做回放统计。最终放行条件来自模型字段和视觉校准分数。

## 运行层次

第一层是 Qwen 多模态初筛。它负责从混合图片池里找出可用的地面透明水样本，同时拦住墙面渗水、普通反光、干地面、有色液体和不可读图片。

第二层是严格复核。它只处理上一层的候选样本，要求同时满足可见、可用、地面水、无硬负样本、证据强度和置信度阈值。这里会生成 calibration 表，用来记录不同阈值下的通过量和弱负样本泄漏情况。

第三层是视觉校准器。它用人工预分文件夹训练一个轻量图像特征模型，运行时不读取文件夹名，只使用图片纹理、亮度、颜色和局部统计特征给候选样本打分。阈值选择采用保守策略：先找候选集中弱负样本的最高分，再把通过阈值设在它之上，同时限制通过率上限。

最后一层是交付包。脚本把自动通过、拒绝或待复核样本复制到独立文件夹，并输出 manifest、summary 和 contact sheet，人工复核不用只看 CSV。

## 推荐命令

先设置 H20/Qwen 环境变量：

```powershell
$env:QWEN397B_API_KEY="你的 key"
$env:QWEN_GEOMETRY_MODEL="aios-smart-eye-vlm"
```

第一波初筛：

```powershell
python scripts/run_clear_water_filter.py run `
  --root "C:\Users\19310\Desktop\万国数据实习\异常检测\第一波二筛" `
  --preset first_wave `
  --resume `
  --interval-seconds 6.8
```

第二波初筛：

```powershell
python scripts/run_clear_water_filter.py run `
  --root "C:\Users\19310\Desktop\万国数据实习\异常检测\第二波二筛" `
  --preset second_wave `
  --resume `
  --interval-seconds 6.8
```

严格复核上一层候选样本：

```powershell
python scripts/run_clear_water_strict_verifier.py `
  --input-csv "C:\path\to\second_wave_v3_recall_balanced_latest.csv" `
  --only-selected `
  --auto-zero-negative `
  --target-max-rate 0.65 `
  --workers 4 `
  --resume
```

视觉校准：

```powershell
python scripts/run_clear_water_visual_calibrator.py `
  --full-csv "C:\path\to\second_wave_full_pool.csv" `
  --candidate-csv "C:\path\to\second_wave_strict_latest.csv" `
  --target-denominator 728 `
  --target-max-rate 0.65 `
  --require-vlm-accept `
  --build-pack
```

打包自动通过图片：

```powershell
python scripts/run_clear_water_filter.py build-pack `
  --csv "C:\path\to\second_wave_clear_water_v5_visual_calibrated_latest.csv" `
  --selected-only
```

## 判定口径

自动通过样本需要满足这些条件：

```text
can_see_image=true
usable=true
floor_clear_water_visible=true
liquid_location=floor 或 both
hard_negative_type=none
decision=accept_clear_water
confidence 达到阈值
floor_water_cues 数量达到阈值
visual_clear_water_score 高于弱负样本最高分
```

需要复核或拒绝的主要原因：

```text
wall_water_only
reflection_only
dry_floor
colored_liquid
unreadable
not_industrial
unclear
low_confidence
too_few_cues
visual_score_below_threshold
```

## 输出

每一层都会输出：

```text
*_latest.jsonl
*_latest.csv
*_latest_selected.csv
*_latest_rejected.csv
*_latest_summary.md
```

视觉校准层额外输出：

```text
*_latest_metadata.json
visual_calibrated_selected_pack_<N>
visual_calibrated_selected_pack_<N>.zip
```

这些文件可以直接作为批次质检记录。给带教汇报时，重点看 `selected_rate`、`weak_negative_selected`、`latency_p50`、`latency_p95` 和 `selection_reason_counts`。
