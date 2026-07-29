# QC Agent 测试证据

测试日期：2026-06-07。

## 1. 单元测试

命令：

```powershell
python -m unittest discover -s tests
```

结果：

```text
Ran 38 tests in 7.850s
OK
```

覆盖重点：

- metadata contract validator。
- crop review。
- direct pipeline JSON retry。
- QC agent 字段快照和 `field_path`。
- asset QC crop 文件名回连。
- generated crop 从 metadata 回连。

## 2. 编译检查

命令：

```powershell
python -m compileall autolabel scripts tests
```

结果：通过。

## 3. Metadata 全量规则质检

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

输入范围：

```text
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata
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

输出：

```text
data/qc/qc_20260607T161507_0800_report.json
data/qc/qc_20260607T161507_0800_manual_review_queue.csv
```

说明：

- `manual_queue=15` 来自 5% 通过样本抽样，不代表失败。
- 本次没有发现 metadata 字段断链、缺文件、框越界、crop 异常、mask 异常。
- 这次关闭了 VLM，因此结论只覆盖规则层全量通过，暂不覆盖语义层全量通过。

## 4. 图片资产全量质检

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

输入范围：

```text
image_sequence/image_sequence
outputs/outputs/generated_images
outputs/outputs/masks
outputs/outputs/grid_previews
crops/crops
outputs/outputs/crops
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

输出：

```text
data/qc/asset_qc_20260607T161551_0800_report.json
data/qc/asset_qc_20260607T161551_0800_assets.csv
```

说明：

- 图片、crop、mask、grid preview 均可读。
- 直接标注 crop 和生成分支 crop 均能回连 manifest 或 metadata。
- 没有发现坏图、极端小图、极端比例 crop。

## 5. 尚未验证部分

尚未做 H20/VLM 语义层全量调用。

未验证项：

```text
红框是否真的框到目标
标签和图像是否语义一致
生成异常是否真实自然
人体框是否漏头/漏脚
```

后续验证建议：

1. 先选 20 到 50 个样本做 VLM pilot。
2. 和人工结果对比。
3. 确认误判类型和提示词稳定性。
4. 再决定是否全量启用 VLM 语义复核。

已补充 pilot 脚本：

```text
scripts/run_vlm_qc_pilot.ps1
```

当前环境执行 1 条样本 pilot 时，脚本在请求模型前失败并提示：

```text
QWEN397B_API_KEY is not set. Set it in the current PowerShell session before running VLM QC.
```

这说明语义层代码入口已经具备；当前会话缺 H20 环境变量和公司网络条件。设置 key 并连上公司 VPN 后，可用该脚本从 20 到 50 个样本 pilot 开始验证。
