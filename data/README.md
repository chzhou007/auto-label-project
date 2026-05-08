# 数据目录说明

- `raw/images/`：未标注原始图片。
- `raw/videos/`：未标注原始视频。
- `staging/image_sequence/`：预处理后的待处理图片序列和 manifest。
- `processed/crops/`：根据目标框裁剪得到的小图。
- `processed/masks/`：分割模型或生成分支产生的 mask。
- `processed/metadata/`：每张大图对应的 `AutoLabelSample` JSON。
- `exports/labelstudio/`：Label Studio 导入 JSON。
- `qc/`：抽检批次、人工复核、返工记录。
