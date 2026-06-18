# H20/VLM 语义质检 Pilot 方案

规则层已经能检查文件、字段、box、crop、mask、classification 的一致性。H20/VLM 语义质检用于补上规则无法判断的问题：

```text
红框是否真的框到目标
框是否漏头/漏脚/漏主体
是否框错对象
标签是否和图像语义一致
生成异常是否真实自然
```

## 1. 运行前提

需要在公司 VPN / 可访问 H20 的网络环境下运行，并在当前 PowerShell 会话设置：

```powershell
$env:QWEN397B_API_KEY="你的 key"
$env:QWEN_GEOMETRY_MODEL="aios-smart-eye-vlm"
```

如果服务地址和配置里的默认值不同，再设置：

```powershell
$env:QWEN_GEOMETRY_API_URL="https://deepseek.gds-services.com/vllm-qwen35b/v1"
```

不要把 key 写入仓库文件。

## 2. 小样本 Pilot

先跑 20 个样本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_vlm_qc_pilot.ps1 `
  -MetadataDir "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata" `
  -AssetBaseDir "C:\Users\19310\Desktop\万国数据实习\异常检测\outputs" `
  -OutputDir "data/qc" `
  -Limit 20
```

脚本实际会调用：

```text
python scripts/run_qc_agent.py --enable-vlm --limit 20
```

## 3. 期望输出

输出仍然是：

```text
data/qc/<qc_run_id>_report.json
data/qc/<qc_run_id>_manual_review_queue.csv
```

每个 VLM 语义问题会出现在 object issue 中，典型 issue：

```text
vlm_qc_target_mismatch
vlm_qc_needs_human_review
vlm_qc_parse_error
vlm_qc_error
```

VLM 返回格式：

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

## 4. Pilot 评估方式

建议人工看 20 到 50 个样本，对比：

| 项 | 判断 |
| --- | --- |
| VLM 认为 passed 的样本 | 人工是否认可 |
| VLM 认为 needs_human_review 的样本 | 是否真的值得人工看 |
| VLM 认为 failed 的样本 | 是否误杀 |
| VLM JSON 是否稳定 | 是否出现自然语言/格式错误 |
| reason 是否有用 | 是否能辅助返工 |

## 5. 全量启用条件

满足以下条件再考虑全量跑：

```text
VLM 输出 JSON 稳定
误杀率可接受
高风险样本召回率可接受
单样本耗时和成本可接受
人工复核队列规模可控
```

## 6. 当前状态

当前仓库已经有 VLM 语义质检代码路径和 pilot 脚本。

当前已完成：

```text
规则层 metadata 全量 QC
图片资产全量 QC
VLM JSON parser 和代码路径测试
```

当前未完成：

```text
H20/VLM 语义层全量实际调用
```

原因：

```text
当前运行环境没有设置 QWEN397B_API_KEY / QWEN_GEOMETRY_MODEL / QWEN_GEOMETRY_API_URL，
且需要公司 VPN/H20 网络可达。
```
