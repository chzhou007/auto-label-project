from __future__ import annotations

import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
FULL_MODE = "v5_7"
EVIDENCE_MODE = "evidence_only"
VALID_MODES = {FULL_MODE, EVIDENCE_MODE}

ACTION_LABELS = {
    "auto_accept_positive": "BBox 通过",
    "rework_bbox_positive_candidate": "修框后可用",
    "manual_review": "人工复核",
    "label_conflict_review": "标签冲突复核",
    "reject_from_positive_training": "拒绝进入正例训练",
}
BOX_QUALITY_LABELS = {
    "good": "框正确",
    "over_inclusive": "框太大",
    "under_inclusive": "框太小",
    "wrong_target": "框错目标",
    "uncertain": "不确定",
}
REWORK_LABELS = {
    "none": "无需修改",
    "expand_bbox": "扩大框",
    "shrink_bbox": "缩小框",
    "move_bbox": "移动/重画框",
    "relabeled_negative_or_other": "改为负例或其他类",
    "uncertain": "人工决定",
}


class QCServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class BundleInfo:
    extract_root: Path
    metadata_dir: Path
    asset_root: Path
    metadata_count: int
    object_count: int


@dataclass(frozen=True)
class QCJobResult:
    run_id: str
    run_dir: Path
    archive_path: Path
    summary_markdown: str
    table_headers: list[str]
    table_rows: list[list[Any]]
    gallery_items: list[tuple[str, str]]
    log_text: str


ProgressCallback = Callable[[str], None]


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def model_endpoint() -> str:
    value = (
        os.getenv("QWEN_GEOMETRY_API_URL")
        or os.getenv("SJTU_API_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    )
    if not value:
        return "未配置"
    parsed = urlsplit(value)
    return parsed.netloc or value


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_archive_bytes: int | None = None,
    max_extracted_bytes: int | None = None,
    max_files: int | None = None,
) -> Path:
    archive = Path(archive_path)
    destination_path = Path(destination)
    if archive.suffix.lower() != ".zip":
        raise QCServiceError("请上传 .zip 数据包。")
    if not archive.is_file():
        raise QCServiceError("上传的 ZIP 文件不存在。")

    archive_limit = max_archive_bytes or _env_int("QC_MAX_UPLOAD_MB", 2048) * 1024 * 1024
    extracted_limit = max_extracted_bytes or _env_int("QC_MAX_EXTRACTED_MB", 8192) * 1024 * 1024
    file_limit = max_files or _env_int("QC_MAX_ARCHIVE_FILES", 30000)
    if archive.stat().st_size > archive_limit:
        raise QCServiceError(
            f"ZIP 超过限制：{archive.stat().st_size / 1024 / 1024:.1f} MB > "
            f"{archive_limit / 1024 / 1024:.0f} MB。"
        )

    destination_path.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination_path.resolve()
    with zipfile.ZipFile(archive) as source:
        members = [item for item in source.infolist() if not item.is_dir()]
        if len(members) > file_limit:
            raise QCServiceError(f"ZIP 文件数过多：{len(members)} > {file_limit}。")
        total_size = sum(item.file_size for item in members)
        if total_size > extracted_limit:
            raise QCServiceError(
                f"ZIP 解压后过大：{total_size / 1024 / 1024:.1f} MB > "
                f"{extracted_limit / 1024 / 1024:.0f} MB。"
            )

        for info in source.infolist():
            if info.filename.startswith("__MACOSX/") or info.filename.endswith("/.DS_Store"):
                continue
            if _is_zip_symlink(info):
                raise QCServiceError(f"ZIP 不允许包含符号链接：{info.filename}")
            normalized = PurePosixPath(info.filename.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise QCServiceError(f"ZIP 包含不安全路径：{info.filename}")
            if not normalized.parts:
                continue
            target = destination_path.joinpath(*normalized.parts)
            target_resolved = target.resolve()
            if target_resolved != destination_resolved and destination_resolved not in target_resolved.parents:
                raise QCServiceError(f"ZIP 路径越界：{info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as input_handle, target.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    return destination_path


def _read_sample(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("image_asset"), dict) or not isinstance(value.get("objects"), list):
        return None
    return value


def discover_bundle(extract_root: str | Path) -> BundleInfo:
    root = Path(extract_root)
    candidates: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    for path in root.rglob("*.json"):
        sample = _read_sample(path)
        if sample is None:
            continue
        candidates.setdefault(path.parent, []).append((path, sample))
    if not candidates:
        raise QCServiceError(
            "数据包中没有找到 AutoLabelSample metadata JSON；"
            "请确保 ZIP 内包含 metadata/*.json。"
        )

    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            len(item[1]),
            item[0].name.lower() == "metadata",
            "i2i_outputs" in str(item[0]).lower(),
            -len(item[0].parts),
        ),
        reverse=True,
    )
    metadata_dir, samples = ranked[0]
    object_count = sum(
        1
        for _, sample in samples
        for obj in sample.get("objects") or []
        if isinstance(obj, dict)
    )
    if object_count == 0:
        raise QCServiceError("metadata 中没有可质检的 objects[]。")
    return BundleInfo(
        extract_root=root,
        metadata_dir=metadata_dir,
        asset_root=metadata_dir.parent,
        metadata_count=len(samples),
        object_count=object_count,
    )


def _service_env() -> dict[str, str]:
    env = dict(os.environ)
    api_key = env.get("QWEN397B_API_KEY") or env.get("SJTU_API_KEY") or env.get("OPENAI_API_KEY")
    base_url = (
        env.get("QWEN_GEOMETRY_API_URL")
        or env.get("SJTU_API_BASE_URL")
        or env.get("OPENAI_API_BASE")
        or env.get("OPENAI_BASE_URL")
    )
    if api_key:
        env.setdefault("SJTU_API_KEY", api_key)
    if base_url:
        env.setdefault("SJTU_API_BASE_URL", base_url)
    return env


def _run_command(
    args: list[str],
    *,
    env: dict[str, str],
    log_handle: Any,
    accepted_codes: set[int] | None = None,
    progress: ProgressCallback | None = None,
) -> int:
    accepted = accepted_codes or {0}
    log_handle.write(f"$ {' '.join(args)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log_handle.write(line)
        log_handle.flush()
        if progress and line.strip():
            progress(line.strip())
    process.stdout.close()
    return_code = process.wait()
    if return_code not in accepted:
        raise QCServiceError(f"质检子进程失败（exit={return_code}），请查看运行日志。")
    return return_code


def _selected_object_count(bundle: BundleInfo, limit: int | None) -> int:
    paths = sorted(bundle.metadata_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    return sum(
        1
        for path in paths
        for obj in ((_read_sample(path) or {}).get("objects") or [])
        if isinstance(obj, dict)
    )


def _manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        return []
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_web_outputs(
    *,
    mode: str,
    bundle: BundleInfo,
    bbox_dir: Path,
) -> tuple[str, list[str], list[list[Any]], list[tuple[str, str]]]:
    headers = ["样本", "最终动作", "框质量", "修改建议", "理由"]
    if mode == EVIDENCE_MODE:
        results_path = bbox_dir / "bbox_qc_results.jsonl"
        records = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        gallery = [
            (str(Path(record["evidence_panel_path"]).resolve()), str(record.get("sample_id") or ""))
            for record in records[:24]
            if record.get("evidence_panel_path") and Path(record["evidence_panel_path"]).is_file()
        ]
        summary = (
            "## 证据生成完成\n\n"
            f"- metadata：{bundle.metadata_count} 份\n"
            f"- bbox 对象：{len(records)} 个\n"
            "- 本模式未调用 VLM，不包含通过/修框/拒绝结论。\n"
        )
        rows = [[record.get("sample_id"), "仅生成证据", "-", "-", ""] for record in records]
        return summary, headers, rows, gallery

    manifest_path = bbox_dir / "training_triage_manifest.csv"
    records = _manifest_rows(manifest_path)
    action_counts = Counter(record.get("training_action") or "missing" for record in records)
    clean = action_counts["auto_accept_positive"]
    rework = action_counts["rework_bbox_positive_candidate"]
    manual = action_counts["manual_review"] + action_counts["label_conflict_review"]
    reject = action_counts["reject_from_positive_training"]
    total = len(records)
    recoverable = clean + rework
    summary = "\n".join(
        [
            "## 质检完成",
            "",
            f"- 总 bbox：**{total}**",
            f"- BBox 通过：**{clean}**",
            f"- 修框后可用：**{rework}**",
            f"- 人工复核：**{manual}**",
            f"- 拒绝：**{reject}**",
            f"- 可恢复正例：**{recoverable}/{total} ({(100 * recoverable / total if total else 0):.1f}%)**",
            "",
            "> `BBox 通过` 只代表框的语义与几何通过，不代表生成图的清水真实性已通过。",
        ]
    )
    table_rows = [
        [
            record.get("sample_id") or "",
            ACTION_LABELS.get(record.get("training_action") or "", record.get("training_action") or ""),
            BOX_QUALITY_LABELS.get(record.get("box_quality") or "", record.get("box_quality") or ""),
            REWORK_LABELS.get(record.get("rework_type") or "", record.get("rework_type") or ""),
            record.get("reason") or "",
        ]
        for record in records
    ]
    priority = {
        "manual_review": 0,
        "label_conflict_review": 0,
        "rework_bbox_positive_candidate": 1,
        "reject_from_positive_training": 2,
        "auto_accept_positive": 3,
    }
    gallery: list[tuple[str, str]] = []
    for record in sorted(records, key=lambda item: priority.get(item.get("training_action") or "", 9)):
        panel = Path(record.get("evidence_panel_path") or "")
        if panel.is_file():
            action = ACTION_LABELS.get(record.get("training_action") or "", "")
            quality = BOX_QUALITY_LABELS.get(record.get("box_quality") or "", "")
            gallery.append((str(panel.resolve()), f"{record.get('sample_id')} | {action} | {quality}"))
        if len(gallery) >= 24:
            break
    return summary, headers, table_rows, gallery


def _write_result_readme(path: Path, *, mode: str) -> None:
    text = [
        "自动化标注质检结果包",
        "",
        "bbox_qc/training_triage_summary.md: 中英文总结（完整质检模式）",
        "bbox_qc/training_triage_manifest.csv: 逐样本最终分流清单",
        "bbox_qc/training_triage_gallery.html: 可视化证据画廊",
        "bbox_qc/evidence_panels/: 整图＋框内＋框外三联证据",
        "rule_qc/: metadata/图片/crop/mask 规则质检",
        "run.log: 运行日志（不包含 API key）",
        "",
        f"运行模式: {mode}",
        "注意: BBox 通过不等于生成图真实性通过。",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def _package_results(results_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path in sorted(results_dir.rglob("*")):
            if not path.is_file() or "problem_pack" in path.parts:
                continue
            target.write(path, path.relative_to(results_dir))


def _keep_inputs() -> bool:
    return os.getenv("QC_KEEP_INPUTS", "false").strip().lower() in {"1", "true", "yes"}


def run_qc_bundle(
    archive_path: str | Path,
    *,
    work_root: str | Path,
    mode: str = FULL_MODE,
    allow_external_model: bool = False,
    model: str | None = None,
    clean_verifier_model: str | None = None,
    workers: int = 1,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> QCJobResult:
    run_id = f"qc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    input_dir = Path(work_root).expanduser().resolve() / run_id / "input"
    try:
        return _run_qc_bundle_impl(
            archive_path,
            work_root=work_root,
            run_id=run_id,
            mode=mode,
            allow_external_model=allow_external_model,
            model=model,
            clean_verifier_model=clean_verifier_model,
            workers=workers,
            limit=limit,
            progress=progress,
        )
    finally:
        if not _keep_inputs():
            shutil.rmtree(input_dir, ignore_errors=True)


def _run_qc_bundle_impl(
    archive_path: str | Path,
    *,
    work_root: str | Path,
    run_id: str,
    mode: str,
    allow_external_model: bool,
    model: str | None,
    clean_verifier_model: str | None,
    workers: int,
    limit: int | None,
    progress: ProgressCallback | None,
) -> QCJobResult:
    if mode not in VALID_MODES:
        raise QCServiceError(f"不支持的运行模式：{mode}")
    if mode == FULL_MODE and not allow_external_model:
        raise QCServiceError("完整质检会向管理员配置的模型服务发送图片，请先勾选知情确认。")

    env = _service_env()
    if mode == FULL_MODE:
        if not env.get("SJTU_API_KEY"):
            raise QCServiceError("服务端未配置 API key，请联系管理员设置 SJTU_API_KEY。")
        if not env.get("SJTU_API_BASE_URL"):
            raise QCServiceError("服务端未配置模型地址，请设置 SJTU_API_BASE_URL。")

    safe_workers = min(8, max(1, int(workers or 1)))
    safe_limit = int(limit) if limit and int(limit) > 0 else None
    selected_model = model or env.get("QWEN_GEOMETRY_MODEL") or env.get("SJTU_API_DEFAULT_MODEL") or env.get("OPENAI_MODEL") or "qwen"
    selected_verifier = clean_verifier_model or env.get("QC_CLEAN_VERIFIER_MODEL") or "qwen3.6-27b"
    run_dir = Path(work_root).expanduser().resolve() / run_id
    input_dir = run_dir / "input"
    results_dir = run_dir / "results"
    rule_dir = results_dir / "rule_qc"
    bbox_dir = results_dir / "bbox_qc"
    log_path = results_dir / "run.log"
    results_dir.mkdir(parents=True, exist_ok=False)

    if progress:
        progress("正在安全解压 ZIP…")
    safe_extract_zip(archive_path, input_dir)
    bundle = discover_bundle(input_dir)
    expected_objects = _selected_object_count(bundle, safe_limit)
    if expected_objects == 0:
        raise QCServiceError("当前 limit 下没有可处理 bbox。")

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "metadata_dir": str(bundle.metadata_dir),
                    "asset_root": str(bundle.asset_root),
                    "metadata_count": bundle.metadata_count,
                    "object_count": bundle.object_count,
                    "expected_objects": expected_objects,
                    "model": selected_model if mode == FULL_MODE else None,
                    "clean_verifier_model": selected_verifier if mode == FULL_MODE else None,
                    "endpoint": model_endpoint() if mode == FULL_MODE else None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if progress:
            progress(f"已找到 {bundle.metadata_count} 份 metadata，{expected_objects} 个 bbox。")

        rule_args = [
            sys.executable,
            str(ROOT / "scripts" / "run_qc_agent.py"),
            "--config",
            str(ROOT / "configs" / "autolabel.yaml"),
            "--metadata-dir",
            str(bundle.metadata_dir),
            "--asset-base-dir",
            str(bundle.asset_root),
            "--output-dir",
            str(rule_dir),
            "--sampling-ratio",
            "0",
        ]
        if safe_limit is not None:
            rule_args.extend(["--limit", str(safe_limit)])
        if progress:
            progress("正在运行 metadata/资产规则质检…")
        _run_command(
            rule_args,
            env=env,
            log_handle=log_handle,
            accepted_codes={0, 1},
            progress=progress,
        )

        bbox_args = [
            sys.executable,
            str(ROOT / "scripts" / "run_anomaly_bbox_qc.py"),
            "--metadata-dir",
            str(bundle.metadata_dir),
            "--asset-base-dir",
            str(bundle.asset_root),
            "--output-dir",
            str(bbox_dir),
            "--review-mode",
            "training_triage",
            "--triage-prompt-version",
            "v5",
            "--workers",
            str(safe_workers),
            "--api-retries",
            "4",
            "--retry-backoff-seconds",
            "8",
            "--max-tokens",
            "1200",
            "--overwrite",
        ]
        if safe_limit is not None:
            bbox_args.extend(["--limit", str(safe_limit)])
        if mode == EVIDENCE_MODE:
            bbox_args.append("--build-evidence-only")
        else:
            bbox_args.extend(
                [
                    "--model",
                    selected_model,
                    "--clean-verifier-model",
                    selected_verifier,
                ]
            )
        if progress:
            progress("正在生成三联证据…" if mode == EVIDENCE_MODE else "正在运行语义＋几何质检…")
        _run_command(bbox_args, env=env, log_handle=log_handle, progress=progress)

        if mode == FULL_MODE:
            summary_args = [
                sys.executable,
                str(ROOT / "scripts" / "summarize_bbox_triage_runs.py"),
                "--run-root",
                str(bbox_dir),
                "--output-dir",
                str(bbox_dir),
                "--expected-total",
                str(expected_objects),
                "--expected-revision",
                "v5.7",
                "--strict",
            ]
            if progress:
                progress("正在严格校验结果完整性…")
            _run_command(summary_args, env=env, log_handle=log_handle, progress=progress)

        audit_args = [
            sys.executable,
            str(ROOT / "scripts" / "audit_bbox_mask_geometry.py"),
            "--metadata-dir",
            str(bundle.metadata_dir),
            "--asset-base-dir",
            str(bundle.asset_root),
            "--output-dir",
            str(bbox_dir),
        ]
        _run_command(audit_args, env=env, log_handle=log_handle, progress=progress)

    _write_result_readme(results_dir / "README.txt", mode=mode)
    summary, headers, rows, gallery = _build_web_outputs(mode=mode, bundle=bundle, bbox_dir=bbox_dir)
    archive_output = run_dir / f"{run_id}_results.zip"
    _package_results(results_dir, archive_output)
    return QCJobResult(
        run_id=run_id,
        run_dir=run_dir,
        archive_path=archive_output,
        summary_markdown=summary,
        table_headers=headers,
        table_rows=rows,
        gallery_items=gallery,
        log_text=log_path.read_text(encoding="utf-8"),
    )
