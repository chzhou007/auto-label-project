# BBox 质检 Agent 部署与交付说明

## 1. 最终使用方式

面向非开发用户时，推荐使用 Web 端，不要让用户直接运行 Python 命令。

```text
打开网页
  ↓
上传一个 ZIP 数据包
  ↓
选择“完整质检”或“仅规则＋证据图”
  ↓
查看中文统计、逐样本表和证据图
  ↓
下载结果 ZIP
```

Web 入口为 `scripts/run_qc_web.py`。命令行入口仍保留，用于开发调试和大批次后台任务。

## 2. 输入 ZIP 规范

最稳定的输入是一个保留目录关系的 ZIP。系统会自动定位 `metadata/*.json`，并根据 metadata 中的 URI 查找图片、crop 和 mask。

```text
batch.zip
└── i2i_outputs/
    ├── metadata/                 # 必需：AutoLabelSample JSON
    │   ├── sample_001.json
    │   └── sample_002.json
    ├── generated_images/         # 必需：metadata 引用的图片
    │   ├── sample_001.png
    │   └── sample_002.png
    └── debug/                   # 建议保留
        ├── masks/
        └── crops/
```

可以直接将 I2I 产出的 `i2i_outputs` 目录压缩成 ZIP。Windows metadata 中的 `C:\...` 路径不需要手工修改，服务会按文件后缀路径在 ZIP 中重新定位。

### 安全限制

- 只接受 `.zip`。
- 拒绝 `../` 路径穿越、绝对路径和符号链接。
- 默认 ZIP 上限 2 GB，解压后上限 8 GB，文件数上限 30,000。
- 默认任务完成后删除解压的原始输入，保留结果包。

## 3. 输出内容

用户最终下载一个 ZIP，主要内容为：

| 输出 | 用途 |
|---|---|
| `bbox_qc/training_triage_summary.md` | 总 bbox、通过、修框、人工复核、拒绝数量 |
| `bbox_qc/training_triage_manifest.csv` | 逐样本最终动作，可交给数据同学后处理 |
| `bbox_qc/training_triage_gallery.html` | 可在浏览器打开的证据画廊 |
| `bbox_qc/evidence_panels/` | 整图＋框内 crop＋框外 context 三联图 |
| `bbox_qc/bbox_qc_results.jsonl` | 完整机器可读结果 |
| `bbox_qc/bbox_mask_geometry.md` | mask/bbox 一致性与证据独立性说明 |
| `rule_qc/` | metadata、图片、crop、mask 规则检查 |
| `run.log` | 运行日志，不包含 API key |

`auto_accept_positive` 只表示 bbox 语义和几何通过，不表示生成水渍真实性已通过。若要交付最终训练集，还需要增加生成图真实性 gate。

## 4. Docker 部署（推荐）

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 编辑 `.env`，至少填写：

```text
SJTU_API_BASE_URL=https://your-model-gateway.example.com/v1
SJTU_API_KEY=...
QC_WEB_USERNAME=...
QC_WEB_PASSWORD=...
QC_REQUIRE_AUTH=true
```

3. 启动：

```bash
docker compose up --build -d
```

4. 查看健康状态：

```bash
docker compose ps
docker compose logs -f qc-web
```

5. 打开：

```text
http://<服务器 IP>:7860
```

停止服务：

```bash
docker compose down
```

## 5. 本机启动

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_qc_web.py
```

默认地址为 `http://127.0.0.1:7860`。若不希望局域网其他机器访问，启动前设置：

```bash
export GRADIO_SERVER_NAME=127.0.0.1
```

## 6. 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SJTU_API_BASE_URL` | 无 | OpenAI-compatible `/v1` 地址 |
| `SJTU_API_KEY` | 无 | 模型服务密钥，只由服务端保持 |
| `QC_PRIMARY_MODEL` | `qwen` | 整图语义与 bbox 几何主模型 |
| `QC_CLEAN_VERIFIER_MODEL` | `qwen3.6-27b` | clean seed 独立反方复核模型 |
| `QC_WEB_USERNAME/PASSWORD` | 无 | Web 登录账号与密码；共享部署时建议必填 |
| `QC_REQUIRE_AUTH` | Docker 中为 `true` | 未配用户名/密码时拒绝启动 |
| `QC_WORK_ROOT` | `data/web_runs` | 服务端结果目录 |
| `QC_MAX_UPLOAD_MB` | `2048` | ZIP 上传上限 |
| `QC_MAX_EXTRACTED_MB` | `8192` | ZIP 解压后上限 |
| `QC_MAX_ARCHIVE_FILES` | `30000` | ZIP 内最大文件数 |
| `QC_KEEP_INPUTS` | `false` | 是否保留解压后原始输入 |

## 7. 交付边界

- Web 层负责上传、进度、预览和下载；判定仍使用同一个 `QcService` 和质检编排逻辑。
- 完整模式会向配置的模型端点发送图片，Web 端必须勾选知情确认。
- 本工具不将 API key 写入输入 ZIP、结果 ZIP 或运行日志。
- Docker 默认使用名为 `qc_runs` 的 volume 保留结果；长期部署需由管理员定期清理。
- 当前输出是质检分流结果，不自动修改原 metadata 或重画 bbox。
