from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_benchmark_report_schema(tmp_path) -> None:
    benchmark = importlib.import_module("autolabel.modules.generation.benchmark")
    assert hasattr(benchmark, "summarize_localizer_results"), (
        "benchmark.py 需要提供 summarize_localizer_results(results) 用于汇总 localizer 横向评测"
    )
    results = [
        {"sample_id": "s1", "anomaly_type": "water_leak", "localizer": "rgb_diff", "success": True, "bbox_iou": 0.62, "precision": 0.70, "recall": 0.80, "mask_iou": 0.58, "elapsed_ms": 18, "reason": None, "fallback_used": False, "passes_quality": True, "manual_accept": True},
        {"sample_id": "s1", "anomaly_type": "water_leak", "localizer": "pgcd_lpips", "success": True, "bbox_iou": 0.76, "precision": 0.82, "recall": 0.84, "mask_iou": 0.72, "elapsed_ms": 55, "reason": None, "fallback_used": False, "passes_quality": True, "manual_accept": True},
        {"sample_id": "s2", "anomaly_type": "oil_leak", "localizer": "rgb_diff", "success": False, "bbox_iou": 0.0, "precision": 0.0, "recall": 0.0, "mask_iou": 0.0, "elapsed_ms": 20, "reason": "global_change", "fallback_used": False, "passes_quality": False, "manual_accept": False},
        {"sample_id": "s2", "anomaly_type": "oil_leak", "localizer": "pgcd_lpips", "success": False, "bbox_iou": 0.0, "precision": 0.0, "recall": 0.0, "mask_iou": 0.0, "elapsed_ms": 60, "reason": "no_component", "fallback_used": True, "passes_quality": False, "manual_accept": False},
    ]
    summary = benchmark.summarize_localizer_results(results)
    for name in ["rgb_diff", "pgcd_lpips"]:
        assert name in summary
        row = summary[name]
        for key in ["sample_count", "success_rate", "fallback_rate", "quality_pass_rate", "mean_bbox_iou", "mean_precision", "mean_recall", "mean_mask_iou", "avg_elapsed_ms", "manual_accept_rate", "failure_reason_distribution"]:
            assert key in row
        assert 0.0 <= row["success_rate"] <= 1.0
        assert 0.0 <= row["fallback_rate"] <= 1.0
        assert 0.0 <= row["quality_pass_rate"] <= 1.0
        assert 0.0 <= row["mean_bbox_iou"] <= 1.0
        assert 0.0 <= row["manual_accept_rate"] <= 1.0
    report_path = Path(tmp_path) / "benchmark_summary.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == summary


def test_benchmark_failure_summary_includes_anomaly_type_aggregates(tmp_path) -> None:
    benchmark = importlib.import_module("autolabel.modules.generation.benchmark")
    results = [
        {"sample_id": "s1", "object_id": "o1", "anomaly_type": "water_leak", "localizer": "rgb_diff", "success": True, "bbox_iou": 0.62, "precision": 0.70, "recall": 0.80, "mask_iou": 0.58, "elapsed_ms": 18, "reason": None, "fallback_used": False, "passes_quality": True},
        {"sample_id": "s1", "object_id": "o1", "anomaly_type": "water_leak", "localizer": "pgcd_lpips", "success": False, "bbox_iou": 0.10, "precision": 0.12, "recall": 0.14, "mask_iou": 0.09, "elapsed_ms": 50, "reason": "no_component", "fallback_used": False, "passes_quality": False},
        {"sample_id": "s2", "object_id": "o2", "anomaly_type": "oil_leak", "localizer": "rgb_diff", "success": False, "bbox_iou": 0.0, "precision": 0.0, "recall": 0.0, "mask_iou": 0.0, "elapsed_ms": 20, "reason": "global_change", "fallback_used": False, "passes_quality": False},
        {"sample_id": "s2", "object_id": "o2", "anomaly_type": "oil_leak", "localizer": "pgcd_lpips", "success": False, "bbox_iou": 0.0, "precision": 0.0, "recall": 0.0, "mask_iou": 0.0, "elapsed_ms": 60, "reason": "quality_failed:background_preservation_low", "fallback_used": True, "passes_quality": False},
    ]

    outputs = benchmark.write_localizer_benchmark_reports(
        results,
        tmp_path,
        stem="localizer_benchmark_generation",
    )

    failure_summary = json.loads(Path(outputs["failure_summary_json"]).read_text(encoding="utf-8"))
    assert set(failure_summary) == {
        "by_localizer",
        "by_anomaly_type",
        "by_anomaly_type_and_localizer",
        "failure_reason_distribution",
    }
    assert failure_summary["by_anomaly_type"]["water_leak"]["sample_count"] == 2
    assert failure_summary["by_anomaly_type"]["oil_leak"]["sample_count"] == 2
    assert failure_summary["by_anomaly_type_and_localizer"]["water_leak"]["rgb_diff"]["success_rate"] == 1.0
    assert failure_summary["by_anomaly_type_and_localizer"]["oil_leak"]["pgcd_lpips"]["fallback_rate"] == 1.0
    assert failure_summary["by_localizer"]["rgb_diff"]["manual_accept_rate"] is None
    assert failure_summary["failure_reason_distribution"]["by_localizer"]["pgcd_lpips"] == {
        "no_component": 1,
        "quality_failed:background_preservation_low": 1,
    }
    assert failure_summary["failure_reason_distribution"]["by_anomaly_type"]["oil_leak"] == {
        "global_change": 1,
        "quality_failed:background_preservation_low": 1,
    }
    assert failure_summary["failure_reason_distribution"]["by_anomaly_type_and_localizer"]["water_leak"]["pgcd_lpips"] == {
        "no_component": 1,
    }
