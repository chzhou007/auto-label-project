from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from autolabel.qc_service import (  # noqa: E402
    EVIDENCE_MODE,
    FULL_MODE,
    QCServiceError,
    model_endpoint,
    run_qc_bundle,
)


CSS = """
.hero {padding: 8px 4px 2px 4px;}
.hero h1 {font-size: 1.8rem !important; margin-bottom: .25rem !important;}
.muted {color: #6b7280;}
#run-button {min-height: 48px; font-weight: 700;}
"""


def _log_tail(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else "...\n" + text[-limit:]


def run_from_web(
    archive_path: str | None,
    mode: str,
    confirm_external: bool,
    limit_value: float | None,
    workers_value: str,
) -> Iterator[tuple[Any, Any, Any, Any, Any, Any]]:
    empty_table: list[list[Any]] = []
    if not archive_path:
        yield "❌ 请先上传 ZIP 数据包。", "", empty_table, [], None, ""
        return
    if mode == FULL_MODE and not confirm_external:
        yield (
            "❌ 完整质检会将图片发送到管理员配置的模型服务，请先勾选知情确认。",
            "",
            empty_table,
            [],
            None,
            "",
        )
        return

    progress_lines: list[str] = []

    def progress(message: str) -> None:
        progress_lines.append(message)

    yield "⏳ 任务已启动，正在校验数据包…", "", empty_table, [], None, ""
    try:
        result = run_qc_bundle(
            archive_path,
            work_root=os.getenv("QC_WORK_ROOT", str(ROOT / "data" / "web_runs")),
            mode=mode,
            allow_external_model=confirm_external,
            model=os.getenv("QC_PRIMARY_MODEL") or None,
            clean_verifier_model=os.getenv("QC_CLEAN_VERIFIER_MODEL") or None,
            workers=int(workers_value),
            limit=int(limit_value) if limit_value and int(limit_value) > 0 else None,
            progress=progress,
        )
    except QCServiceError as exc:
        yield (
            f"❌ {exc}",
            "",
            empty_table,
            [],
            None,
            "\n".join(progress_lines),
        )
        return
    except Exception as exc:  # pragma: no cover - final UI safety net
        yield (
            f"❌ 未预期错误：{exc}",
            "",
            empty_table,
            [],
            None,
            "\n".join(progress_lines),
        )
        return

    yield (
        f"✅ 任务完成：`{result.run_id}`",
        result.summary_markdown,
        result.table_rows,
        result.gallery_items,
        str(result.archive_path),
        _log_tail(result.log_text),
    )


def build_app() -> gr.Blocks:
    endpoint = model_endpoint()
    with gr.Blocks(title="BBox 质检 Agent") as demo:
        gr.Markdown(
            """
            # BBox 质检 Agent
            上传自动化标注输出 ZIP，生成框内/框外证据，并将样本分流为通过、修框、人工复核或拒绝。
            """,
            elem_classes="hero",
        )
        with gr.Row():
            archive = gr.File(
                label="1. 上传 ZIP 数据包",
                file_types=[".zip"],
                type="filepath",
                scale=2,
            )
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=[
                        ("完整质检", FULL_MODE),
                        ("仅规则＋证据图（不调模型）", EVIDENCE_MODE),
                    ],
                    value=FULL_MODE,
                    label="2. 运行模式",
                )
                confirm_external = gr.Checkbox(
                    label=f"我确认数据可发送到模型服务（{endpoint}）",
                    value=False,
                )
        with gr.Accordion("高级选项", open=False):
            with gr.Row():
                limit_value = gr.Number(
                    value=0,
                    precision=0,
                    minimum=0,
                    label="最多处理 metadata 数",
                    info="0 表示全部。",
                )
                workers_value = gr.Dropdown(
                    choices=["1", "2", "4"],
                    value="1",
                    label="并发 worker",
                    info="网关限流不确定时建议 1。",
                )
        run_button = gr.Button("开始质检", variant="primary", elem_id="run-button")
        status = gr.Markdown("等待上传…")
        summary = gr.Markdown()
        table = gr.Dataframe(
            headers=["样本", "最终动作", "框质量", "修改建议", "理由"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
            label="逐样本结果",
        )
        gallery = gr.Gallery(
            label="优先查看：人工复核→修框→拒绝→通过",
            columns=3,
            object_fit="contain",
            height="auto",
        )
        result_file = gr.File(label="下载完整结果 ZIP", interactive=False)
        with gr.Accordion("运行日志", open=False):
            log_box = gr.Textbox(lines=16, max_lines=24, interactive=False, show_label=False)

        run_button.click(
            run_from_web,
            inputs=[archive, mode, confirm_external, limit_value, workers_value],
            outputs=[status, summary, table, gallery, result_file, log_box],
            api_name="run_qc",
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    work_root = Path(os.getenv("QC_WORK_ROOT", str(ROOT / "data" / "web_runs"))).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    username = os.getenv("QC_WEB_USERNAME")
    password = os.getenv("QC_WEB_PASSWORD")
    require_auth = os.getenv("QC_REQUIRE_AUTH", "false").strip().lower() in {"1", "true", "yes"}
    if require_auth and not (username and password):
        raise SystemExit(
            "QC_REQUIRE_AUTH=true, but QC_WEB_USERNAME/QC_WEB_PASSWORD are not both configured."
        )
    auth = (username, password) if username and password else None
    demo = build_app().queue(default_concurrency_limit=1, max_size=8)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        auth=auth,
        allowed_paths=[str(work_root)],
        max_file_size=f"{os.getenv('QC_MAX_UPLOAD_MB', '2048')}mb",
        show_error=True,
        css=CSS,
    )


if __name__ == "__main__":
    main()
