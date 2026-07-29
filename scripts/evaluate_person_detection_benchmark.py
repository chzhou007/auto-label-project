from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def box_iou(left: list[int | float], right: list[int | float]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    x1 = max(lx1, rx1)
    y1 = max(ly1, ry1)
    x2 = min(lx2, rx2)
    y2 = min(ly2, ry2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(gt_boxes: list[dict[str, Any]], pred_boxes: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    candidates = []
    for gt_idx, gt in enumerate(gt_boxes):
        for pred_idx, pred in enumerate(pred_boxes):
            candidates.append((box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"]), gt_idx, pred_idx))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    ious = []
    for iou, gt_idx, pred_idx in candidates:
        if iou < iou_threshold:
            break
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        ious.append(iou)

    tp = len(ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_iou_sum": sum(ious),
        "matched_iou_count": len(ious),
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def finalize_counts(counts: dict[str, float]) -> dict[str, float]:
    tp = counts.get("tp", 0.0)
    fp = counts.get("fp", 0.0)
    fn = counts.get("fn", 0.0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "gt_boxes": int(counts.get("gt_boxes", 0)),
        "pred_boxes": int(counts.get("pred_boxes", 0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": safe_div(counts.get("matched_iou_sum", 0.0), counts.get("matched_iou_count", 0.0)),
    }


def evaluate_benchmark(benchmark_path: Path, iou_thresholds: list[float], include_pseudo: bool = False) -> dict[str, Any]:
    samples = read_json(benchmark_path).get("samples") or []
    eligible = [sample for sample in samples if include_pseudo or sample.get("source") == "annotation"]
    results: dict[str, Any] = {
        "benchmark": str(benchmark_path),
        "include_pseudo": include_pseudo,
        "sample_count": len(eligible),
        "thresholds": {},
    }

    for threshold in iou_thresholds:
        overall = defaultdict(float)
        by_project: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
        for sample in eligible:
            gt_boxes = sample.get("boxes") or []
            pred_boxes = sample.get("prediction_boxes") or []
            matched = match_boxes(gt_boxes, pred_boxes, threshold)
            project = str(sample.get("project_name") or "unknown")
            for bucket in (overall, by_project[project]):
                bucket["gt_boxes"] += len(gt_boxes)
                bucket["pred_boxes"] += len(pred_boxes)
                for key, value in matched.items():
                    bucket[key] += value

        results["thresholds"][f"{threshold:.2f}"] = {
            "overall": finalize_counts(overall),
            "by_project": {project: finalize_counts(counts) for project, counts in sorted(by_project.items())},
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Label Studio predictions against person detection benchmark annotations.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--iou-thresholds", default="0.5,0.75")
    parser.add_argument("--include-pseudo", action="store_true", help="Also evaluate pseudo-GT samples. Do not use for final reporting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = [float(item.strip()) for item in args.iou_thresholds.split(",") if item.strip()]
    results = evaluate_benchmark(args.benchmark, thresholds, include_pseudo=args.include_pseudo)
    if args.output:
        write_json(args.output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
