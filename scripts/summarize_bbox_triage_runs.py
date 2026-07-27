from __future__ import annotations

import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


POSITIVE_PREFIX = "positive"
NEGATIVE_PREFIX = "negative"
RECOVERABLE_ACTIONS = {"auto_accept_positive", "rework_bbox_positive_candidate"}


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("sample_id") or ""), str(record.get("object_id") or "")


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    direct_path = root / "bbox_qc_results.jsonl"
    paths = ([direct_path] if direct_path.is_file() else []) + sorted(
        root.glob("*/bbox_qc_results.jsonl")
    )
    for path in paths:
        run_name = path.parent.name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_run"] = run_name
                records.append(record)
    return records


def index_records(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {record_key(record): record for record in records}


def merge_replacements(
    records: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_keys = {record_key(record) for record in records}
    replacement_index = index_records(replacements)
    unknown = sorted(set(replacement_index) - base_keys)
    if unknown:
        raise ValueError(f"replacement keys missing from base run: {len(unknown)}")
    merged: list[dict[str, Any]] = []
    for record in records:
        replacement = replacement_index.get(record_key(record))
        if replacement is None:
            merged.append(record)
            continue
        updated = dict(replacement)
        updated["_run"] = record.get("_run")
        updated["replaced_triage_prompt_revision"] = record.get(
            "triage_prompt_revision"
        )
        merged.append(updated)
    return merged


def validate_records(
    records: list[dict[str, Any]],
    *,
    expected_total: int | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    keys = [record_key(record) for record in records]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        issues.append(f"duplicate sample/object keys: {len(duplicate_keys)}")
    if expected_total is not None and len(records) != expected_total:
        issues.append(f"expected {expected_total} records, found {len(records)}")

    api_errors = [record_key(record) for record in records if record.get("error")]
    if api_errors:
        issues.append(f"API/runtime errors: {len(api_errors)}")
    missing_actions = [
        record_key(record) for record in records if not record.get("training_action")
    ]
    if missing_actions:
        issues.append(f"missing training_action: {len(missing_actions)}")

    revisions = Counter(
        str(record.get("triage_prompt_revision") or "missing") for record in records
    )
    if expected_revision is not None and revisions != Counter(
        {expected_revision: len(records)}
    ):
        issues.append(f"expected revision {expected_revision}, found {dict(revisions)}")

    v5_records = [
        record
        for record in records
        if str(record.get("triage_prompt_revision") or "").startswith("v5")
    ]
    incomplete_edges = [
        record_key(record)
        for record in v5_records
        if record.get("edge_checks_complete") is not True
    ]
    if incomplete_edges:
        issues.append(f"incomplete v5 edge checks: {len(incomplete_edges)}")
    missing_area = [
        record_key(record)
        for record in v5_records
        if record.get("target_area_fraction") in {None, "", "uncertain"}
    ]
    if missing_area:
        issues.append(f"missing/uncertain v5 target area: {len(missing_area)}")
    verifier_errors = [
        record_key(record) for record in records if record.get("clean_verifier_error")
    ]
    if verifier_errors:
        issues.append(f"clean verifier errors: {len(verifier_errors)}")
    unverified_auto = [
        record_key(record)
        for record in records
        if record.get("training_action") == "auto_accept_positive"
        and record.get("clean_verifier_model")
        and record.get("clean_verifier_passed") is not True
    ]
    if unverified_auto:
        issues.append(
            f"auto-accept records without verifier pass: {len(unverified_auto)}"
        )

    return {
        "valid": not issues,
        "record_count": len(records),
        "unique_key_count": len(set(keys)),
        "api_error_count": len(api_errors),
        "missing_action_count": len(missing_actions),
        "incomplete_edge_check_count": len(incomplete_edges),
        "missing_target_area_count": len(missing_area),
        "clean_verifier_error_count": len(verifier_errors),
        "unverified_auto_accept_count": len(unverified_auto),
        "revision_counts": dict(revisions),
        "issues": issues,
    }


def is_positive(record: dict[str, Any]) -> bool:
    return str(record.get("weak_label_hint") or "").lower().startswith(POSITIVE_PREFIX)


def is_negative(record: dict[str, Any]) -> bool:
    return str(record.get("weak_label_hint") or "").lower().startswith(NEGATIVE_PREFIX)


def revision_label(records: list[dict[str, Any]]) -> str:
    revisions = sorted(
        {str(record.get("triage_prompt_revision") or "unknown") for record in records}
    )
    if "v5.7" in revisions and len(revisions) > 1:
        return "effective v5.7"
    return "+".join(revisions)


def metric_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    positives = [record for record in records if is_positive(record)]
    negatives = [record for record in records if is_negative(record)]

    def action_count(rows: list[dict[str, Any]], action: str) -> int:
        return sum(record.get("training_action") == action for record in rows)

    positive_auto = action_count(positives, "auto_accept_positive")
    positive_rework = action_count(positives, "rework_bbox_positive_candidate")
    positive_manual = action_count(positives, "manual_review")
    positive_conflict = action_count(positives, "label_conflict_review")
    positive_reject = action_count(positives, "reject_from_positive_training")
    negative_auto = action_count(negatives, "auto_accept_positive")
    negative_rework = action_count(negatives, "rework_bbox_positive_candidate")
    negative_manual = action_count(negatives, "manual_review")
    negative_conflict = action_count(negatives, "label_conflict_review")
    negative_reject = action_count(negatives, "reject_from_positive_training")
    return {
        "total": len(records),
        "api_errors": sum(bool(record.get("error")) for record in records),
        "weak_positive_total": len(positives),
        "clean_seed_positive": positive_auto,
        "positive_rework": positive_rework,
        "recoverable_positive_pool": positive_auto + positive_rework,
        "positive_manual_review": positive_manual,
        "positive_label_conflict": positive_conflict,
        "positive_preserved_non_reject": len(positives) - positive_reject,
        "positive_reject": positive_reject,
        "weak_negative_total": len(negatives),
        "negative_auto_accept_leak": negative_auto,
        "negative_rework_leak": negative_rework,
        "negative_manual_review": negative_manual,
        "negative_conflict_review": negative_conflict,
        "negative_reject": negative_reject,
    }


def rate(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "n/a"


def write_manifest(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "_run",
        "sample_id",
        "object_id",
        "weak_label_hint",
        "decision",
        "semantic_verdict",
        "semantic_candidate_location",
        "semantic_candidate_surface",
        "inside_target_status",
        "outside_relation",
        "box_quality",
        "target_surface",
        "dominant_confounder",
        "truncated_edges",
        "edge_checks_complete",
        "background_excess",
        "target_area_fraction",
        "stage_disagreement",
        "primary_box_quality",
        "clean_verifier_model",
        "clean_verifier_passed",
        "clean_verifier_box_quality",
        "clean_verifier_rework_type",
        "clean_verifier_reason",
        "clean_verifier_error",
        "image_has_positive_candidate",
        "training_action",
        "rework_type",
        "bbox_usable_for_training",
        "label_conflict",
        "issue_flags",
        "reason",
        "image_path",
        "metadata_path",
        "evidence_panel_path",
        "error",
        "triage_prompt_revision",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in fields}
            row["issue_flags"] = ";".join(record.get("issue_flags") or [])
            row["truncated_edges"] = ";".join(record.get("truncated_edges") or [])
            writer.writerow(row)


def per_run_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for run_name in sorted({record["_run"] for record in records}):
        subset = [record for record in records if record["_run"] == run_name]
        counts = Counter(
            record.get("training_action") or "missing" for record in subset
        )
        rows.append(
            [
                run_name,
                len(subset),
                counts["auto_accept_positive"],
                counts["rework_bbox_positive_candidate"],
                counts["manual_review"],
                counts["label_conflict_review"],
                counts["reject_from_positive_training"],
                sum(bool(record.get("error")) for record in subset),
            ]
        )
    return rows


def write_summary(
    records: list[dict[str, Any]],
    metrics: dict[str, int],
    path: Path,
    transition_rows: list[dict[str, Any]] | None = None,
) -> None:
    current_label = revision_label(records)
    pos_total = metrics["weak_positive_total"]
    neg_total = metrics["weak_negative_total"]
    action_counts = Counter(
        record.get("training_action") or "missing" for record in records
    )
    semantic_counts = Counter(
        record.get("semantic_verdict") or "missing" for record in records
    )
    surface_counts = Counter(
        record.get("semantic_candidate_surface") or "missing" for record in records
    )
    quality_counts = Counter(
        record.get("box_quality") or "missing" for record in records
    )
    rework_counts = Counter(
        record.get("rework_type") or "missing"
        for record in records
        if record.get("training_action")
        in {"rework_bbox_positive_candidate", "label_conflict_review"}
    )
    lines = [
        "# Training triage run summary",
        "",
        "## Core metrics",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Total bbox objects | {metrics['total']} | 100.0% |",
        f"| API errors | {metrics['api_errors']} | {rate(metrics['api_errors'], metrics['total'])} |",
        f"| Weak positive total | {pos_total} | 100.0% |",
        f"| Clean positive seed | {metrics['clean_seed_positive']} | {rate(metrics['clean_seed_positive'], pos_total)} |",
        f"| Positive bbox rework pool | {metrics['positive_rework']} | {rate(metrics['positive_rework'], pos_total)} |",
        f"| Recoverable positive pool | {metrics['recoverable_positive_pool']} | {rate(metrics['recoverable_positive_pool'], pos_total)} |",
        f"| Positive manual-review pool | {metrics['positive_manual_review']} | {rate(metrics['positive_manual_review'], pos_total)} |",
        f"| Positive preserved, non-reject | {metrics['positive_preserved_non_reject']} | {rate(metrics['positive_preserved_non_reject'], pos_total)} |",
        f"| Positive rejects | {metrics['positive_reject']} | {rate(metrics['positive_reject'], pos_total)} |",
        f"| Weak negative total | {neg_total} | 100.0% |",
        f"| Negative auto-accept leakage | {metrics['negative_auto_accept_leak']} | {rate(metrics['negative_auto_accept_leak'], neg_total)} |",
        f"| Negative rework leakage | {metrics['negative_rework_leak']} | {rate(metrics['negative_rework_leak'], neg_total)} |",
        f"| Negative label-conflict review | {metrics['negative_conflict_review']} | {rate(metrics['negative_conflict_review'], neg_total)} |",
        f"| Negative manual review | {metrics['negative_manual_review']} | {rate(metrics['negative_manual_review'], neg_total)} |",
        "",
        "## Distributions",
        "",
        f"- training_action: `{json.dumps(action_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- semantic_verdict: `{json.dumps(semantic_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- semantic_candidate_surface: `{json.dumps(surface_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- box_quality: `{json.dumps(quality_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- rework_type: `{json.dumps(rework_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Per directory",
        "",
        "| Run | Total | Clean | Rework | Manual | Conflict | Reject | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_run_rows(records):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.extend(
        [
            "",
            "## Weak-label proxy confusion matrices",
            "",
            "Direct-trainable positive means only `auto_accept_positive`:",
            "",
            "| Weak truth / Prediction | Direct positive | Not direct positive |",
            "|---|---:|---:|",
            f"| Weak positive | {metrics['clean_seed_positive']} | {pos_total - metrics['clean_seed_positive']} |",
            f"| Weak negative | {metrics['negative_auto_accept_leak']} | {neg_total - metrics['negative_auto_accept_leak']} |",
            "",
            "Recoverable positive means `auto_accept_positive + rework_bbox_positive_candidate`:",
            "",
            "| Weak truth / Prediction | Recoverable positive | Not recoverable positive |",
            "|---|---:|---:|",
            f"| Weak positive | {metrics['recoverable_positive_pool']} | {pos_total - metrics['recoverable_positive_pool']} |",
            f"| Weak negative | {metrics['negative_auto_accept_leak'] + metrics['negative_rework_leak']} | {neg_total - metrics['negative_auto_accept_leak'] - metrics['negative_rework_leak']} |",
            "",
            "## Interpretation boundary",
            "",
            "- Directory labels are weak training labels, not a human-adjudicated gold set.",
            "- `auto_accept_positive` measures a high-precision clean seed, not total TP recall.",
            "- `auto_accept_positive + rework_bbox_positive_candidate` is the recoverable positive pool.",
            "- `manual_review` preserves uncertain assets but does not make them trainable.",
            "- Negative-folder visual positives remain conflict cases and never enter positive training automatically.",
        ]
    )
    if transition_rows is not None:
        positive_rows = [
            row
            for row in transition_rows
            if str(row.get("weak_label_hint")).startswith("positive")
        ]
        negative_rows = [
            row
            for row in transition_rows
            if str(row.get("weak_label_hint")).startswith("negative")
        ]
        rescued = [
            row
            for row in positive_rows
            if row.get("transition") == "rescued_from_reject"
        ]
        newly_rejected = [
            row for row in positive_rows if row.get("transition") == "newly_rejected"
        ]
        tightened = [
            row
            for row in positive_rows
            if row.get("baseline_action") == "auto_accept_positive"
            and row.get("current_action") != "auto_accept_positive"
        ]
        promoted = [
            row
            for row in positive_rows
            if row.get("baseline_action") != "auto_accept_positive"
            and row.get("current_action") == "auto_accept_positive"
        ]
        lines.extend(
            [
                "",
                f"## v4 to {current_label} decision migration",
                "",
                f"- Rescued weak positives from v4 reject into clean/rework: **{len(rescued)}**.",
                f"- Newly rejected weak positives: **{len(newly_rejected)}**.",
                f"- v4 clean seeds tightened to rework/manual/reject: **{len(tightened)}**.",
                f"- v4 non-clean cases promoted to clean: **{len(promoted)}**.",
                "",
                "### Weak-positive action matrix",
                "",
                f"| v4 action | {current_label} action | Count |",
                "|---|---|---:|",
            ]
        )
        for (old_action, new_action), count in sorted(
            Counter(
                (row["baseline_action"], row["current_action"]) for row in positive_rows
            ).items()
        ):
            lines.append(f"| {old_action} | {new_action} | {count} |")
        lines.extend(
            [
                "",
                "### Weak-negative action matrix",
                "",
                f"| v4 action | {current_label} action | Count |",
                "|---|---|---:|",
            ]
        )
        for (old_action, new_action), count in sorted(
            Counter(
                (row["baseline_action"], row["current_action"]) for row in negative_rows
            ).items()
        ):
            lines.append(f"| {old_action} | {new_action} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison(
    current: dict[str, int], baseline: dict[str, int], path: Path
) -> None:
    keys = [
        "clean_seed_positive",
        "positive_rework",
        "recoverable_positive_pool",
        "positive_manual_review",
        "positive_preserved_non_reject",
        "positive_reject",
        "negative_auto_accept_leak",
        "negative_rework_leak",
        "negative_conflict_review",
        "negative_manual_review",
        "api_errors",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "baseline", "current", "delta"])
        for key in keys:
            old = baseline.get(key, 0)
            new = current.get(key, 0)
            writer.writerow([key, old, new, new - old])


def transition_name(old_action: str, new_action: str) -> str:
    if old_action == new_action:
        return "unchanged"
    if (
        old_action == "reject_from_positive_training"
        and new_action in RECOVERABLE_ACTIONS
    ):
        return "rescued_from_reject"
    if (
        old_action != "reject_from_positive_training"
        and new_action == "reject_from_positive_training"
    ):
        return "newly_rejected"
    if (
        old_action == "auto_accept_positive"
        and new_action == "rework_bbox_positive_candidate"
    ):
        return "tightened_clean_to_rework"
    if old_action == "auto_accept_positive" and new_action == "manual_review":
        return "tightened_clean_to_manual"
    if old_action != "auto_accept_positive" and new_action == "auto_accept_positive":
        return "promoted_to_clean"
    return "changed_review_route"


def build_transition_rows(
    current_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    current = index_records(current_records)
    baseline = index_records(baseline_records)
    issues: list[str] = []
    missing_current = sorted(set(baseline) - set(current))
    missing_baseline = sorted(set(current) - set(baseline))
    if missing_current:
        issues.append(f"baseline-only sample/object keys: {len(missing_current)}")
    if missing_baseline:
        issues.append(f"current-only sample/object keys: {len(missing_baseline)}")

    rows: list[dict[str, Any]] = []
    for key in sorted(set(current) & set(baseline)):
        new = current[key]
        old = baseline[key]
        old_action = str(old.get("training_action") or "missing")
        new_action = str(new.get("training_action") or "missing")
        rows.append(
            {
                "_run": new.get("_run"),
                "sample_id": new.get("sample_id"),
                "object_id": new.get("object_id"),
                "weak_label_hint": new.get("weak_label_hint"),
                "baseline_action": old_action,
                "current_action": new_action,
                "transition": transition_name(old_action, new_action),
                "current_revision": new.get("triage_prompt_revision"),
                "baseline_quality": old.get("box_quality"),
                "current_quality": new.get("box_quality"),
                "current_rework": new.get("rework_type"),
                "semantic_verdict": new.get("semantic_verdict"),
                "semantic_candidate_surface": new.get("semantic_candidate_surface"),
                "stage_disagreement": new.get("stage_disagreement"),
                "reason": new.get("reason"),
                "image_path": new.get("image_path"),
                "overlay_path": new.get("overlay_path"),
                "evidence_panel_path": new.get("evidence_panel_path"),
            }
        )
    return rows, issues


def write_transitions(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "_run",
        "sample_id",
        "object_id",
        "weak_label_hint",
        "baseline_action",
        "current_action",
        "transition",
        "current_revision",
        "baseline_quality",
        "current_quality",
        "current_rework",
        "semantic_verdict",
        "semantic_candidate_surface",
        "stage_disagreement",
        "reason",
        "image_path",
        "overlay_path",
        "evidence_panel_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_cell(value: Any, limit: int = 150) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text.replace("|", "\\|")


def markdown_asset_link(label: str, value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return f"[{label}]({path})"


def write_bad_case_report(rows: list[dict[str, Any]], path: Path) -> None:
    revisions = sorted({str(row.get("current_revision") or "unknown") for row in rows})
    current_label = (
        "effective v5.7"
        if "v5.7" in revisions and len(revisions) > 1
        else "+".join(revisions)
    )
    sections = [
        (
            "Rescued v4 rejects",
            [row for row in rows if row["transition"] == "rescued_from_reject"],
        ),
        (
            "Newly rejected weak positives",
            [
                row
                for row in rows
                if row["transition"] == "newly_rejected"
                and str(row["weak_label_hint"]).startswith("positive")
            ],
        ),
        (
            "Current weak-positive manual review",
            [
                row
                for row in rows
                if str(row["weak_label_hint"]).startswith("positive")
                and row["current_action"] == "manual_review"
            ],
        ),
        (
            "v4 clean seed tightened",
            [
                row
                for row in rows
                if row["baseline_action"] == "auto_accept_positive"
                and row["current_action"] != "auto_accept_positive"
            ],
        ),
    ]
    lines = [
        f"# BBox QC v4 to {current_label} bad-case audit",
        "",
        "This report is an audit queue, not gold-label adjudication. Weak folder labels remain proxy truth.",
    ]
    for title, selected in sections:
        lines.extend(
            [
                "",
                f"## {title} ({len(selected)})",
                "",
                f"| Sample | v4 -> {current_label} | Quality / rework | Semantic | Evidence | Reason |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in selected:
            evidence = " / ".join(
                part
                for part in (
                    markdown_asset_link("original", row.get("image_path")),
                    markdown_asset_link("overlay", row.get("overlay_path")),
                    markdown_asset_link("six-view", row.get("evidence_panel_path")),
                )
                if part
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_cell(row.get("sample_id")),
                        markdown_cell(
                            f"{row['baseline_action']} -> {row['current_action']}", 90
                        ),
                        markdown_cell(
                            f"{row.get('current_quality')} / {row.get('current_rework')}",
                            70,
                        ),
                        markdown_cell(
                            f"{row.get('semantic_verdict')} / {row.get('semantic_candidate_surface')}",
                            70,
                        ),
                        evidence,
                        markdown_cell(row.get("reason")),
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def path_uri(value: Any, *, relative_to: Path | None = None) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if relative_to is not None:
        try:
            path.relative_to(Path.cwd().resolve())
            return Path(os.path.relpath(path, relative_to.resolve())).as_posix()
        except ValueError:
            pass
    return path.as_uri()


def write_gallery(
    records: list[dict[str, Any]],
    metrics: dict[str, int],
    path: Path,
    transition_rows: list[dict[str, Any]] | None = None,
) -> None:
    gallery_revision_label = revision_label(records)
    transitions = {
        (str(row.get("sample_id") or ""), str(row.get("object_id") or "")): str(
            row.get("transition") or ""
        )
        for row in transition_rows or []
    }

    def options(field: str, values: list[str]) -> str:
        labels = {
            "_run": "全部批次",
            "training_action": "全部动作",
            "box_quality": "全部框质量",
            "transition": "全部迁移",
        }
        items = [f'<option value="">{html.escape(labels[field])}</option>']
        items.extend(
            f'<option value="{html.escape(value)}">{html.escape(value)}</option>'
            for value in values
        )
        return "".join(items)

    cards: list[str] = []
    for record in sorted(
        records, key=lambda item: (str(item.get("_run")), record_key(item))
    ):
        key = record_key(record)
        transition = transitions.get(key, "")
        action = str(record.get("training_action") or "missing")
        quality = str(record.get("box_quality") or "missing")
        run_name = str(record.get("_run") or "missing")
        original = path_uri(record.get("image_path"), relative_to=path.parent)
        overlay = path_uri(record.get("overlay_path"), relative_to=path.parent)
        evidence = path_uri(record.get("evidence_panel_path"), relative_to=path.parent)
        initial = overlay or original or evidence
        search_text = " ".join(
            str(record.get(field) or "")
            for field in (
                "sample_id",
                "object_id",
                "reason",
                "semantic_verdict",
                "rework_type",
            )
        ).lower()
        cards.append(
            f"""<article class="case" data-run="{html.escape(run_name)}" data-action="{html.escape(action)}"
              data-quality="{html.escape(quality)}" data-transition="{html.escape(transition)}"
              data-search="{html.escape(search_text)}">
              <div class="media"><img loading="lazy" src="{html.escape(initial)}" alt="{html.escape(str(record.get('sample_id')))} bbox"></div>
              <div class="media-modes" role="group" aria-label="图像视图">
                <button type="button" data-src="{html.escape(overlay)}">框</button>
                <button type="button" data-src="{html.escape(original)}">原图</button>
                <button type="button" data-src="{html.escape(evidence)}">六图</button>
              </div>
              <div class="case-head"><strong>{html.escape(str(record.get('sample_id')))}</strong><span class="status">{html.escape(action)}</span></div>
              <dl>
                <div><dt>批次</dt><dd>{html.escape(run_name)}</dd></div>
                <div><dt>框</dt><dd>{html.escape(quality)} / {html.escape(str(record.get('rework_type') or 'none'))}</dd></div>
                <div><dt>语义</dt><dd>{html.escape(str(record.get('semantic_verdict') or 'missing'))} / {html.escape(str(record.get('semantic_candidate_surface') or 'missing'))}</dd></div>
                <div><dt>迁移</dt><dd>{html.escape(transition or 'n/a')}</dd></div>
              </dl>
              <p>{html.escape(markdown_cell(record.get('reason'), 260))}</p>
            </article>"""
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BBox QC {html.escape(gallery_revision_label)} 全量画廊</title>
<style>
:root{{--ink:#172027;--muted:#66717a;--line:#d8dde1;--paper:#f5f6f4;--panel:#fff;--green:#19734b;--amber:#9a5a00;--red:#a33131;--blue:#27658a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line)}}
.top{{max-width:1680px;margin:auto;padding:14px 18px 12px}} h1{{font-size:20px;margin:0 0 10px}} .metrics{{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin-bottom:10px}} .metrics b{{color:var(--ink)}}
.filters{{display:grid;grid-template-columns:minmax(180px,1.4fr) repeat(4,minmax(145px,1fr));gap:8px}} input,select{{width:100%;height:36px;border:1px solid #bfc7cc;border-radius:6px;background:#fff;padding:0 10px;color:var(--ink)}}
main{{max-width:1680px;margin:auto;padding:16px 18px 30px}} #grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(315px,1fr));gap:12px}} .case{{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:7px;overflow:hidden}}
.media{{aspect-ratio:16/10;background:#e7e9e7;display:grid;place-items:center;border-bottom:1px solid var(--line)}} .media img{{width:100%;height:100%;object-fit:contain}} .media-modes{{display:grid;grid-template-columns:repeat(3,1fr);height:32px;border-bottom:1px solid var(--line)}}
.media-modes button{{border:0;border-right:1px solid var(--line);background:#f8f9f8;color:#354149;cursor:pointer}} .media-modes button:last-child{{border-right:0}} .media-modes button:hover{{background:#e8f0f3}}
.case-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;padding:11px 12px 8px}} .case-head strong{{overflow-wrap:anywhere}} .status{{font-size:11px;padding:2px 5px;border:1px solid #aab5bb;border-radius:4px;color:#30414a;max-width:54%;overflow-wrap:anywhere}}
dl{{margin:0;padding:0 12px;display:grid;gap:4px}} dl div{{display:grid;grid-template-columns:42px minmax(0,1fr);gap:6px}} dt{{color:var(--muted)}} dd{{margin:0;overflow-wrap:anywhere}} p{{margin:9px 12px 12px;color:#46525a;min-height:60px;overflow-wrap:anywhere}} .hidden{{display:none}} #visible-count{{color:var(--blue)}}
@media(max-width:800px){{.filters{{grid-template-columns:1fr 1fr}} .filters input{{grid-column:1/-1}} #grid{{grid-template-columns:1fr}} header{{position:static}}}}
</style>
</head>
<body>
<header><div class="top">
  <h1>BBox QC {html.escape(gallery_revision_label)} 全量画廊</h1>
  <div class="metrics"><span>对象 <b>{metrics['total']}</b></span><span>当前显示 <b id="visible-count">{metrics['total']}</b></span><span>clean <b>{metrics['clean_seed_positive']}</b></span><span>可恢复正例 <b>{metrics['recoverable_positive_pool']}</b></span><span>负例自动泄漏 <b>{metrics['negative_auto_accept_leak'] + metrics['negative_rework_leak']}</b></span></div>
  <div class="filters">
    <input id="search" type="search" placeholder="sample id / reason" aria-label="搜索">
    <select id="run">{options('_run', sorted({str(r.get('_run')) for r in records}))}</select>
    <select id="action">{options('training_action', sorted({str(r.get('training_action')) for r in records}))}</select>
    <select id="quality">{options('box_quality', sorted({str(r.get('box_quality')) for r in records}))}</select>
    <select id="transition">{options('transition', sorted({value for value in transitions.values() if value}))}</select>
  </div>
</div></header>
<main><section id="grid">{''.join(cards)}</section></main>
<script>
const controls = ['search','run','action','quality','transition'].map(id => document.getElementById(id));
const cases = [...document.querySelectorAll('.case')];
function applyFilters() {{
  const query = document.getElementById('search').value.trim().toLowerCase();
  const values = Object.fromEntries(['run','action','quality','transition'].map(id => [id, document.getElementById(id).value]));
  let visible = 0;
  for (const item of cases) {{
    const show = (!query || item.dataset.search.includes(query)) && Object.entries(values).every(([key,value]) => !value || item.dataset[key] === value);
    item.classList.toggle('hidden', !show); if (show) visible += 1;
  }}
  document.getElementById('visible-count').textContent = visible;
}}
for (const control of controls) control.addEventListener('input', applyFilters);
document.addEventListener('click', event => {{
  const button = event.target.closest('.media-modes button'); if (!button || !button.dataset.src) return;
  button.closest('.case').querySelector('.media img').src = button.dataset.src;
}});
</script>
</body></html>"""
    path.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize and compare bbox training-triage JSONL runs."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-root", default=None)
    parser.add_argument(
        "--replacement-root",
        default=None,
        help="Optional subset run whose matching sample/object records replace the base run before summarization.",
    )
    parser.add_argument("--expected-total", type=int, default=None)
    parser.add_argument("--expected-revision", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when integrity validation fails.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(run_root)
    if not records:
        raise SystemExit(f"No */bbox_qc_results.jsonl files found under {run_root}")
    replacement_count = 0
    if args.replacement_root:
        replacements = load_records(Path(args.replacement_root))
        replacement_count = len(replacements)
        records = merge_replacements(records, replacements)
    metrics = metric_counts(records)
    integrity = validate_records(
        records,
        expected_total=args.expected_total,
        expected_revision=args.expected_revision,
    )
    integrity["replacement_record_count"] = replacement_count
    transition_rows: list[dict[str, Any]] | None = None
    baseline_records: list[dict[str, Any]] = []
    if args.baseline_root:
        baseline_records = load_records(Path(args.baseline_root))
        transition_rows, comparison_issues = build_transition_rows(
            records, baseline_records
        )
        integrity["issues"].extend(comparison_issues)
        integrity["valid"] = not integrity["issues"]

    write_manifest(records, output_dir / "training_triage_manifest.csv")
    write_summary(
        records,
        metrics,
        output_dir / "training_triage_summary.md",
        transition_rows=transition_rows,
    )
    if baseline_records:
        write_comparison(
            metrics,
            metric_counts(baseline_records),
            output_dir / "training_triage_comparison.csv",
        )
    if transition_rows is not None:
        write_transitions(
            transition_rows, output_dir / "training_triage_transitions.csv"
        )
        write_bad_case_report(transition_rows, output_dir / "bbox_qc_bad_cases.md")
    write_gallery(
        records,
        metrics,
        output_dir / "training_triage_gallery.html",
        transition_rows=transition_rows,
    )
    (output_dir / "training_triage_integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "training_triage_summary.md")
    if args.strict and not integrity["valid"]:
        for issue in integrity["issues"]:
            print(f"integrity error: {issue}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
