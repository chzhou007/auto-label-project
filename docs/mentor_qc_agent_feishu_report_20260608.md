# 自动标注质检 Agent 汇报方案

汇报对象：数据生成与自动化标注项目组
汇报日期：2026-06-08
负责范围：标注结果质检、字段追溯、风险样本筛选、人工抽检设计

## 1. 先给结论

我现在把这个项目理解成三段工作：前面先产出数据和标注结果，中间用自动化规则做交付前检查，最后只把少量需要人看的样本送进人工复核。

目前本地已经看到两批数据：

| 批次 | 数据内容 | 已验证结果 | 还缺什么 |
|---|---|---|---|
| YOLO 输出框批次 | 原始帧和人体 crop | 1710 张原始帧、1231 张 crop 都能打开，合计 2941 个资源通过检查 | 还没有 YOLO 原始 bbox 坐标表，所以暂时只能确认资源层质量 |
| VLM 生成异常批次 | generated image、metadata、crop、mask、grid preview、logs | 209 条 metadata 规则检查通过；目录检查发现 sample_000162 缺 metadata 和 crop | 需要补齐这个断链样本，并做 H20/VLM 语义 pilot |

当前能报的指标要分开讲：

| 指标 | 当前结果 |
|---|---:|
| YOLO 资源层通过率 | 100% |
| VLM metadata 规则通过率 | 100% |
| VLM 生成批成套完整率 | 209/210，约 99.52% |
| YOLO bbox 准确率 | 暂时不能报，缺 bbox 坐标 |
| VLM 语义准确率 | 暂时不能报，H20/VLM pilot 还没跑完 |

我建议第一版先按这个流程推进：

```text
规则全量检查
风险样本交给 VLM 复核
少量人工抽检校准
```

这样做的好处是：文件缺失、字段缺失、bbox 越界这类硬错误可以全量拦住；画面语义这类规则看不出来的问题再交给 VLM；人工只看高风险样本和抽检样本。

## 2. 两批数据怎么处理

### 2.1 YOLO 输出框批次

本地路径：

```text
C:\Users\19310\Desktop\万国数据实习\异常检测\image_sequence\image_sequence
C:\Users\19310\Desktop\万国数据实习\异常检测\crops\crops
```

本地检查结果：

| 检查项 | 数量 | 结果 |
|---|---:|---|
| 原始帧 | 1710 | 可读 |
| crop | 1231 | 可读，尺寸正常，文件名可以回溯到原始帧 |
| 合计资源 | 2941 | 通过 |

报告文件：

```text
data/qc/original_detection_pool/asset_qc_20260608T125427_0800_report.json
```

这批数据现在能查：

- 图片是否存在。
- 图片是否能打开。
- crop 是否存在。
- crop 是否能打开。
- crop 尺寸是否异常。
- crop 能不能回溯到原始帧。

这批数据现在还查不了：

- bbox 是否越界。
- bbox 是否过松或过紧。
- 同一张图里是否有重复框。
- YOLO 置信度是否异常。
- 框内是否真的有人。

原因是本地只拿到了原图和 crop，没有拿到 YOLO 输出的 bbox 坐标表。下一步需要双泰这边补出检测结果文件，至少要有 image_id、object_id、bbox 坐标、confidence、model_version、crop_uri 这些字段。

这批数据当前结论可以这样说：资源层已经检查过，文件都能用；框的位置质量还没有证据支撑，需要等 bbox 坐标补齐后再评估。

### 2.2 VLM 生成异常批次

本地路径：

```text
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\generated_images
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\metadata
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\crops
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\masks
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\grid_previews
C:\Users\19310\Desktop\万国数据实习\异常检测\outputs\outputs\logs
```

本地检查结果：

| 检查项 | 数量 | 结果 |
|---|---:|---|
| generated images | 210 | 可读 |
| metadata | 209 | 规则检查通过 |
| objects | 209 | 规则检查通过 |
| crops | 209 | 可读 |
| masks | 210 | 可读 |
| grid previews | 210 | 可读 |

报告文件：

```text
data/qc/generated_metadata_pool/qc_20260608T125451_0800_report.json
data/qc/generated_metadata_pool/asset_qc_20260608T125420_0800_report.json
```

这批数据已经发现一个明确问题：

```text
sample_000162 有 generated image、mask、grid preview
sample_000162 缺 metadata
sample_000162 缺 crop
```

这说明只看 metadata 内部会漏问题。209 条 metadata 本身都合规，所以 metadata 规则通过率是 100%；但生成图一共有 210 张，目录之间没有完全配齐，所以成套完整率是 209/210，约 99.52%。

这批数据当前结论可以这样说：字段规则已经能跑通，也确实抓到了一个断链样本；后续要补 sample_000162，并抽样做画面语义复核。

## 3. 质检具体查什么

我会把质检拆成五类问题，避免把所有问题都混成一个准确率。

| 问题类型 | 例子 | 自动化方式 |
|---|---|---|
| 文件完整性 | 原图缺失、metadata 缺失、crop 缺失、mask 缺失 | 目录和文件检查 |
| 字段合法性 | sample_id 为空、枚举值非法、bbox 格式错 | schema 和字段规则 |
| 几何合法性 | bbox 越界、面积过小、crop 没盖住 bbox、mask 尺寸不一致 | 坐标规则 |
| 语义正确性 | 框内目标不存在、异常类别和画面不一致 | VLM 复核加人工抽检 |
| 训练可用性 | 生成图不真实、异常太弱、样本噪声明显 | VLM 复核加人工校准 |

前三类适合全量自动化。后两类要看图像语义，规则只能给风险提示，最终要靠 VLM pilot 和少量人工样本校准。

## 4. 字段清单

这部分的目标是把问题追到具体字段。带教如果问某个样本为什么失败，报告里要能定位到批次、样本、对象、字段和证据文件。

### 4.1 批次级字段

| 字段 | 用途 |
|---|---|
| batch_id | 批次 ID |
| batch_type | yolo_detection 或 vlm_generation |
| batch_owner | 负责人，比如双泰、民君 |
| source_dataset | 原始数据集或目录 |
| expected_sample_count | 预期样本数 |
| actual_sample_count | 实际样本数 |
| qc_policy_id | 质检规则编号 |
| batch_status | passed / blocked / needs_review |
| created_time | 批次创建时间 |
| qc_time | 批次质检时间 |

现在最需要补的是 batch_type 和 batch_owner。补上以后，后面就能按人、按批次、按数据来源拆统计。

### 4.2 样本级字段

| 字段 | 用途 |
|---|---|
| sample_id | 样本主键 |
| image_id | 图片 ID |
| image_uri | 原图或生成图路径 |
| source_type | cctv / manual_upload / generated |
| width | 图片宽度 |
| height | 图片高度 |
| room_type | 场景类型 |
| task_group | 任务组 |
| inspection_content | 巡检内容 |
| generation_prompt | 生成提示词 |
| generation_model | 生成模型 |
| collection_batch | 采集批次 |
| camera_id | 摄像头 |
| capture_time | 采集时间 |
| metadata_uri | metadata 路径 |
| grid_preview_uri | grid preview 路径 |
| log_uri | 生成日志路径 |

这些字段解决来源问题。只要样本有问题，就能知道它来自哪批、哪个场景、哪条生成记录或哪张原始帧。

### 4.3 对象级字段

| 字段 | 用途 |
|---|---|
| object_id | 单个框或异常对象 ID |
| object_type | person / leakage_area |
| bbox_format | xyxy / xywh |
| x1 | bbox 左上角 x |
| y1 | bbox 左上角 y |
| x2 | bbox 右下角 x |
| y2 | bbox 右下角 y |
| confidence | 模型置信度 |
| geometry_source | yolo / synthetic_generator / human |
| geometry_model.model_name | 框来源模型 |
| geometry_model.model_version | 模型版本 |
| mask_uri | mask 路径 |
| crop_uri | crop 路径 |
| crop_box | crop 坐标 |
| crop_expand_ratio | crop 扩边比例 |

YOLO 批现在主要缺这组 bbox 字段，所以目前只能查资源层。VLM 批有 metadata，可以继续查 bbox、crop、mask 之间是否对得上。

### 4.4 标签字段

| 字段 | 用途 |
|---|---|
| label_key | 标签维度 |
| label_value | 标签值 |
| confidence | 标签置信度 |
| classifier_type | rule / vlm / human |
| classifier_name | 分类器名称 |
| classifier_version | 分类器版本 |
| prompt_version | prompt 版本 |
| raw_response_uri | 原始模型响应 |

这组字段用于语义复核。比如样本标成 diesel_leak，后面就要检查画面里有没有柴油泄漏迹象，label_value 和 inspection_content 是否一致。

### 4.5 QC 输出字段

| 字段 | 用途 |
|---|---|
| qc_run_id | 质检任务 ID |
| qc_time | 质检时间 |
| qc_agent_version | 质检版本 |
| status | passed / failed / needs_human_review |
| risk_score | 风险分 |
| issue_code | 问题类型 |
| issue_message | 问题说明 |
| severity | error / warning / info |
| field_path | 具体字段路径 |
| evidence_uri | 证据路径 |
| manual_review_reason | 进入人工原因 |
| reviewer | rule / vlm / human |

field_path 是这里最关键的字段。它可以把问题定位到类似下面的位置：

```text
objects[0].box.x1
objects[0].crop.crop_uri
objects[0].classification.multi_labels[0].label_value
image_asset.image_uri
```

有了 field_path，报告就能说明哪个字段失败、为什么失败、证据在哪里。

## 5. 几种技术方案怎么选

### 5.1 全人工复核

做法：每个框、每张生成图都由人工看一遍。

优点：

- 标准明确时，人工判断最稳。
- 适合做第一批金标样本。

问题：

- 成本最高。
- 数据量一大，交付速度会被人工拖住。
- 不同人标准不一致时，还会引入新的分歧。

适合场景：

- 第一批金标样本。
- 新场景或新类别上线前。
- 高风险样本最终确认。

结论：可以用来校准，不能作为日常主流程。

### 5.2 规则质检

做法：检查文件、字段、bbox、crop、mask、目录完整性。

优点：

- 对硬错误很准。
- 速度快，成本低。
- 同一批数据反复跑，结果稳定。

问题：

- 看不懂画面语义。
- 框内目标是否真的存在、异常类别是否合理，这类问题不能只靠规则解决。

本地实测延迟：

| 数据池 | 检查量 | 延迟量级 |
|---|---:|---:|
| YOLO 资源池 | 2941 assets | 约 10 秒 |
| VLM 资源池 | 839 assets | 约 4 秒 |
| VLM metadata | 209 samples | 约 7 秒 |

适合场景：

- 每批数据交付前必跑。
- CI 检查。
- 文件和字段硬门槛。

结论：规则检查要放在第一层，所有样本都先过这一层。

### 5.3 全量 VLM 质检

做法：每个框或每个生成样本都发给 H20/VLM 复核。

优点：

- 能判断框内目标是否存在。
- 能判断异常类别和画面是否匹配。
- 能发现生成图不自然、异常不明显等问题。

问题：

- 延迟和费用高于规则层。
- 结果受 prompt、画质、框绘制方式、场景复杂度影响。
- 没有人工金标时，很难直接证明 VLM 判断有多准。

当前状态：

- 本地 shell 没有设置 `QWEN397B_API_KEY`，所以 H20/VLM 全量语义复核还没跑完。
- 这部分要单独做 pilot，不能直接把 VLM 语义准确率写进汇报结论。

适合场景：

- 中高风险样本。
- 规则层出现 warning 的样本。
- label 容易混淆的样本。
- 随机抽检样本。

结论：第一版不做无差别全量 VLM。先让规则筛掉硬问题，再把风险样本交给 VLM。

### 5.4 模型一致性反检

做法：用另一个检测模型或分类模型重新预测，再和原始标注比较。

优点：

- 对 YOLO 批很有价值。
- 可以发现漏框、重复框、明显偏框、分类冲突。

问题：

- 需要一套相对可信的复检模型。
- 还需要人工金标来判断复检模型本身靠不靠谱。

适合场景：

- YOLO 批规模变大以后。
- 已经有一批人工金标以后。

结论：这是第二阶段能力。当前先补 YOLO bbox 字段。

### 5.5 数据质量工具

可参考工具：

- CleanVision：查模糊图、低信息量图、异常尺寸图。
- fastdup：查重复图、近重复图、离群图。
- FiftyOne：做可视化复核和问题样本排序。
- Datumaro：做格式转换、数据校验和导出。

优点：

- 对图像质量、重复样本、离群样本有帮助。
- 适合离线批量分析。

问题：

- 不能直接判断 bbox 是否准确。
- 引入完整平台会增加部署和维护成本。

结论：先不引入大平台。把去重、离群检测、可视化抽检这些能力逐步接进现有 QC Agent。

### 5.6 混合分层方案

做法：

```text
规则检查全部样本
风险样本进入 VLM
VLM 拿不准的样本进入人工
随机样本进入人工抽检
人工结果回写到规则和 prompt
```

优点：

- 硬错误由规则层处理。
- 语义问题由 VLM 先看一遍。
- 人工只看少量样本。
- 指标可以拆开汇报，不会把文件错误、字段错误、语义错误混在一起。

问题：

- 需要把字段追溯和 review queue 做好。
- 需要第一批人工样本校准阈值和 prompt。

结论：这条路线最适合当前阶段。先保证字段、证据、复核队列都能追得上，再逐步提高 VLM 的覆盖范围。

## 6. 推荐落地流程

### L0 资源和目录检查

检查内容：

- 原图是否存在。
- 原图是否可读。
- crop 是否存在。
- crop 是否可读。
- mask 是否存在。
- generated image、metadata、crop、mask、grid preview 是否配齐。

当前已经抓到：

```text
sample_000162 缺 metadata 和 crop
```

### L1 metadata 契约检查

检查内容：

- JSON 是否合法。
- 是否满足 AutoLabelSample 结构。
- 必填字段是否存在。
- 枚举值是否合法。
- workflow 和 export 状态是否合理。

### L2 bbox / crop / mask 几何检查

检查内容：

- bbox 是否越界。
- bbox 面积是否过小。
- bbox 长宽比是否异常。
- 同一张图是否有重复框。
- crop_box 是否覆盖 object.box。
- mask 尺寸是否和原图一致。

YOLO 批要等 bbox 坐标补齐后才能完整跑这一层。

### L3 VLM 语义复核

VLM 输入：

- 原图或生成图。
- 带框图。
- crop。
- object_type。
- label 信息。
- inspection_content。

VLM 输出：

```text
target_visible
box_quality
label_match
visual_realism
needs_human_review
issue_flags
reason
```

触发条件：

- 规则层出现 warning。
- bbox 贴边、小目标、长宽比异常。
- 生成异常不明显。
- label 容易混淆。
- 随机抽检样本。

### L4 人工复核队列

人工主要看：

- 规则层 error。
- VLM 返回 needs_human_review。
- VLM 输出解析失败。
- VLM 判断不稳定。
- 随机抽检样本。

人工结果用于三件事：

- 估算剩余错误率。
- 调整规则阈值。
- 改 VLM prompt。

## 7. 准确性和通过率怎么报

我建议不要只报一个准确率。当前更适合拆成四个指标：

| 指标 | 含义 | 当前结果 |
|---|---|---|
| 规则通过率 | 文件、字段、几何检查通过比例 | YOLO 资源层 100%；VLM metadata 100% |
| 成套完整率 | generated image、metadata、crop、mask、grid preview 是否配齐 | VLM 批 209/210，约 99.52% |
| VLM 语义通过率 | VLM 复核通过比例 | 待 H20/VLM pilot |
| 人工抽检错误率 | 人工在抽检样本里发现的问题比例 | 待人工复核 |

目前能确认的只有规则层和目录层结果。YOLO bbox 准确率、VLM 语义准确率、训练收益都还不能确认，因为对应证据还没补齐。

第一轮人工抽检建议 50 到 60 条，覆盖两批数据、不同框大小、不同异常类型、不同场景。抽检结果出来以后，再看规则层有没有漏放，VLM 有没有误判。

## 8. 延迟怎么设计

规则层已经本地实测，都是秒级，可以作为每批数据的默认检查。

VLM 层要单独记录延迟和解析结果：

```text
vlm_request_start_time
vlm_request_end_time
vlm_latency_ms
vlm_model_name
vlm_prompt_version
vlm_parse_status
vlm_retry_count
vlm_decision
```

pilot 建议先跑 20 条样本：

1. 记录 p50 和 p95 延迟。
2. 人工复核 VLM 判断。
3. 统计 VLM 误放、误杀、不确定比例。
4. 如果延迟或误判率偏高，就缩小 VLM 调用范围，只看中高风险样本。

人工复核建议按队列控制：

- 首批校准 50 到 60 条。
- 稳定后常规抽检 5% 到 7%。
- 高风险批次抽检 10% 到 20%。

## 9. 参考论文和开源项目

### 9.1 CVAT Quality Control / Consensus

参考：

- https://docs.cvat.ai/docs/qa-analytics/quality-control/
- https://docs.cvat.ai/docs/qa-analytics/consensus/

可借鉴：

- validation set 和 ground truth 可以用来评估标注质量。
- review mode 可以把冲突样本单独拎出来。
- consensus 适合高价值任务，但人工成本比较高。

在本项目里的取法：

- 借鉴验证集、冲突队列和质量指标。
- 日常流程暂时不做多人共识，避免人工量上升。

### 9.2 Label Studio ML Backend

参考：

- https://labelstud.io/guide/ml.html

可借鉴：

- 模型可以输出预标注。
- 人工可以接收、修改、拒绝模型结果。
- 后续可以把 QC 后样本导入 Label Studio。

在本项目里的取法：

- 当前先做交付前 QC。
- 质检通过后再导出到 Label Studio。
- 后续可以把 QC Agent 或 VLM 复核封装成 Label Studio backend。

### 9.3 Cleanlab / Confident Learning

参考：

- https://arxiv.org/abs/1911.00068
- https://arxiv.org/abs/2103.14749
- https://github.com/cleanlab/cleanlab

可借鉴：

- 根据模型输出概率排序可疑 label。
- 适合后续有训练模型以后做数据清洗。

在本项目里的取法：

- 当前还没有稳定金标集和下游模型概率，先不直接接入。
- 第二阶段可以借鉴可疑样本排序方法。

### 9.4 TIDE

参考：

- https://arxiv.org/abs/2008.08115

可借鉴：

- 检测错误可以拆成分类错误、定位错误、重复框、背景误检、漏检。
- 比只看 mAP 更适合解释检测模型问题。

在本项目里的取法：

- 先借鉴错误类型。
- YOLO bbox 和人工金标补齐后，再接近似评估。

### 9.5 CleanVision / fastdup / FiftyOne / Datumaro

参考：

- https://github.com/cleanlab/cleanvision
- https://visual-layer.github.io/fastdup/
- https://docs.voxel51.com/tutorials/detection_mistakes.html
- https://docs.openvino.ai/2024/documentation/openvino-ecosystem/datumaro.html

可借鉴：

- CleanVision 查图像质量。
- fastdup 查重复图和离群图。
- FiftyOne 做可视化复核。
- Datumaro 做格式转换和数据校验。

在本项目里的取法：

- 第一版先保持轻量。
- 后续按需要接去重、离群检测和可视化抽检。
- 导出格式逐步对齐 Label Studio、YOLO、COCO、Datumaro。

## 10. 下一步计划

### 第一阶段：把两批数据纳入同一个质检口径

任务：

1. 给两批数据补 batch_id、batch_type、batch_owner。
2. YOLO 批补原始 bbox 坐标文件。
3. 处理 VLM 批 sample_000162 断链问题。
4. 每批单独输出 QC report。
5. 生成人工 review queue。

### 第二阶段：H20/VLM 语义 pilot

任务：

1. 配置 H20 环境变量。
2. 抽 20 条样本。
3. 让 VLM 返回结构化判断。
4. 人工复核 VLM 结果。
5. 统计误放、误杀、不确定比例和延迟。

指标：

```text
vlm_parse_success_rate
vlm_pass_rate
vlm_needs_human_review_rate
vlm_false_accept_rate
vlm_false_reject_rate
vlm_latency_p50
vlm_latency_p95
```

### 第三阶段：人工金标校准

任务：

1. 人工复核 50 到 60 条样本。
2. 标记错误类型。
3. 统计剩余错误率。
4. 把高频问题改成规则或 prompt。

### 第四阶段：接到训练数据使用环节

任务：

1. QC 结果写回 metadata。
2. 训练时过滤 failed 和 needs_review。
3. 对训练后错误样本做反查。
4. 输出批次级质量报表。

## 11. 明天汇报口径

可以这样讲：

当前我把数据分成两批看。第一批是 YOLO 检测框和 crop，本地已经检查了 2941 个资源，全部可读，也能回溯到原始帧。这个结论只覆盖资源层。要评价 YOLO 框的位置质量，还需要补原始 bbox 坐标文件。

第二批是 VLM 生成异常数据。这批有 metadata、generated image、crop、mask、grid preview 和 logs，所以可以检查字段、bbox、crop、mask、label、workflow 之间是否对得上。当前 209 条 metadata 全部通过规则检查，但目录完整性检查发现 sample_000162 缺 metadata 和 crop。所以这批数据的 metadata 规则通过率是 100%，成套完整率是 99.52%。

后续我建议按规则全量检查、VLM 风险复核、人工抽检校准推进。规则层先拦文件、字段、bbox、crop、mask 这些硬错误；VLM 处理规则看不出的语义问题；人工主要看 VLM 拿不准的样本和抽检样本。这样可以减少人工二次质检量，也能保留每个问题的证据和字段位置。
