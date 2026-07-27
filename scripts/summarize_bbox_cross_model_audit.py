from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any


RECOVERABLE_ACTIONS = {"auto_accept_positive", "rework_bbox_positive_candidate"}


def key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("sample_id") or ""), str(record.get("object_id") or "")


def load_records(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*/bbox_qc_results.jsonl")):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_run"] = path.parent.name
            record_key = key(record)
            if record_key in records:
                raise ValueError(f"duplicate sample/object key: {record_key}")
            records[record_key] = record
    return records


def action_route(action: Any) -> str:
    value = str(action or "missing")
    if value == "auto_accept_positive":
        return "clean"
    if value == "rework_bbox_positive_candidate":
        return "rework"
    if value in {"manual_review", "label_conflict_review"}:
        return "review"
    if value == "reject_from_positive_training":
        return "reject"
    return "missing"


def build_rows(
    primary: dict[tuple[str, str], dict[str, Any]],
    audit: dict[tuple[str, str], dict[str, Any]],
    selection: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons = {
        (str(item.get("sample_id") or ""), str(item.get("object_id") or "")): item.get(
            "selection_reasons"
        )
        or []
        for item in selection
    }
    rows: list[dict[str, Any]] = []
    for record_key in sorted(audit):
        audit_record = audit[record_key]
        primary_record = primary.get(record_key, {})
        primary_action = primary_record.get("training_action")
        audit_action = audit_record.get("training_action")
        rows.append(
            {
                "run": audit_record.get("_run"),
                "sample_id": audit_record.get("sample_id"),
                "object_id": audit_record.get("object_id"),
                "weak_label_hint": audit_record.get("weak_label_hint"),
                "selection_reasons": ";".join(reasons.get(record_key, [])),
                "primary_action": primary_action,
                "audit_action": audit_action,
                "exact_action_agreement": primary_action == audit_action,
                "primary_route": action_route(primary_action),
                "audit_route": action_route(audit_action),
                "route_agreement": action_route(primary_action)
                == action_route(audit_action),
                "primary_quality": primary_record.get("box_quality"),
                "audit_quality": audit_record.get("box_quality"),
                "primary_rework": primary_record.get("rework_type"),
                "audit_rework": audit_record.get("rework_type"),
                "audit_semantic": audit_record.get("semantic_verdict"),
                "audit_error": audit_record.get("error"),
                "primary_reason": primary_record.get("reason"),
                "audit_reason": audit_record.get("reason"),
                "image_path": audit_record.get("image_path"),
                "overlay_path": audit_record.get("overlay_path"),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rate(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}%" if total else "n/a"


def write_summary(
    rows: list[dict[str, Any]], selection: list[dict[str, Any]], path: Path
) -> None:
    exact = sum(row["exact_action_agreement"] for row in rows)
    route = sum(row["route_agreement"] for row in rows)
    errors = sum(bool(row["audit_error"]) for row in rows)
    negative_leaks = sum(
        str(row["weak_label_hint"]).startswith("negative")
        and row["audit_action"] in RECOVERABLE_ACTIONS
        for row in rows
    )
    selected_reasons = sorted(
        {reason for item in selection for reason in item.get("selection_reasons") or []}
    )
    lines = [
        "# BBox QC cross-model audit",
        "",
        "## Core metrics",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Audited objects | {len(rows)} | 100.0% |",
        f"| Audit API/runtime errors | {errors} | {rate(errors, len(rows))} |",
        f"| Exact action agreement | {exact} | {rate(exact, len(rows))} |",
        f"| Coarse route agreement | {route} | {rate(route, len(rows))} |",
        f"| Negative controls entering clean/rework | {negative_leaks} | {rate(negative_leaks, sum(str(row['weak_label_hint']).startswith('negative') for row in rows))} |",
        "",
        "## Selection strata",
        "",
        "| Stratum | Total | Audit clean | Audit recoverable | Audit review | Audit reject | Exact agree |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for reason in selected_reasons:
        subset = [
            row for row in rows if reason in str(row["selection_reasons"]).split(";")
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    reason,
                    str(len(subset)),
                    str(
                        sum(
                            row["audit_action"] == "auto_accept_positive"
                            for row in subset
                        )
                    ),
                    str(
                        sum(
                            row["audit_action"] in RECOVERABLE_ACTIONS for row in subset
                        )
                    ),
                    str(
                        sum(
                            row["audit_action"]
                            in {"manual_review", "label_conflict_review"}
                            for row in subset
                        )
                    ),
                    str(
                        sum(
                            row["audit_action"] == "reject_from_positive_training"
                            for row in subset
                        )
                    ),
                    str(sum(row["exact_action_agreement"] for row in subset)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Action matrix",
            "",
            "| Primary action | Audit action | Count |",
            "|---|---|---:|",
        ]
    )
    for (primary_action, audit_action), count in sorted(
        Counter(
            (str(row["primary_action"]), str(row["audit_action"])) for row in rows
        ).items()
    ):
        lines.append(f"| {primary_action} | {audit_action} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- This is independent model agreement on a risk-stratified subset, not human gold-label accuracy.",
            "- Agreement supports stability; disagreement identifies review or prompt-hardening targets.",
            "- Weak directory labels are proxy truth and must not be reported as final precision/recall ground truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a risk-stratified cross-model bbox QC audit."
    )
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-total", type=int, default=None)
    args = parser.parse_args()

    primary = load_records(Path(args.primary_root))
    audit = load_records(Path(args.audit_root))
    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    if args.expected_total is not None and len(audit) != args.expected_total:
        raise SystemExit(
            f"expected {args.expected_total} audit records, found {len(audit)}"
        )
    rows = build_rows(primary, audit, selection)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "cross_model_audit.csv")
    write_summary(rows, selection, output_dir / "cross_model_audit.md")
    print(output_dir / "cross_model_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
