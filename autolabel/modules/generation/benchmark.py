from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from autolabel.utils import now_iso_shanghai, write_json


class BenchmarkRecorder:
    def __init__(self, enabled: bool, output_dir: str | Path, config: dict[str, Any]) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.config = dict(config)
        self._lock = threading.Lock()
        self.stage_durations: dict[str, list[float]] = defaultdict(list)
        self.task_durations: list[float] = []
        self.counters: Counter[str] = Counter()
        self.failure_reasons: Counter[str] = Counter()
        self.anomaly_total: Counter[str] = Counter()
        self.anomaly_success: Counter[str] = Counter()
        self.grid_rank_total: Counter[int] = Counter()
        self.grid_rank_success: Counter[int] = Counter()
        self.topk_attempt_counts: list[int] = []
        self.success_candidate_ranks: list[int] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_stage(name, time.perf_counter() - start)

    def record_stage(self, name: str, duration_seconds: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.stage_durations[name].append(float(duration_seconds))

    def incr(self, name: str, value: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.counters[name] += int(value)

    def record_task(
        self,
        anomaly_type: str,
        success: bool,
        duration_seconds: float,
        failure_reason: str | None = None,
        attempted_candidates: int = 0,
        success_candidate_rank: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.task_durations.append(float(duration_seconds))
            self.anomaly_total[anomaly_type] += 1
            if success:
                self.anomaly_success[anomaly_type] += 1
            elif failure_reason:
                self.failure_reasons[failure_reason] += 1
            if attempted_candidates:
                self.topk_attempt_counts.append(int(attempted_candidates))
            if success_candidate_rank is not None:
                self.success_candidate_ranks.append(int(success_candidate_rank))

    def record_grid_rank(self, rank: int, success: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.grid_rank_total[int(rank)] += 1
            if success:
                self.grid_rank_success[int(rank)] += 1

    def write_summary(self) -> Path | None:
        if not self.enabled:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = now_iso_shanghai().replace(":", "").replace("-", "").replace("+", "_")
        path = self.output_dir / f"run_summary_{timestamp}.json"
        with self._lock:
            total = len(self.task_durations)
            success = sum(self.anomaly_success.values())
            failed = max(0, total - success)
            elapsed = sum(self.task_durations)
            stage_names = [
                "load_image",
                "grid_preview",
                "vlm_selection",
                "wan_submit",
                "wan_poll",
                "download",
                "generated_size_check",
                "diff_localization",
                "quality",
                "crop",
                "metadata_validation",
                "metadata_write",
            ]
            for stage in list(self.stage_durations):
                if stage not in stage_names:
                    stage_names.append(stage)
            counter_names = [
                "qwen_grid_calls",
                "qwen_review_calls",
                "qwen_fallback_count",
                "qwen_cache_hits",
                "wan_generation_calls",
                "wan_failures",
                "fine_grid_calls",
            ]
            counters = {name: self.counters.get(name, 0) for name in counter_names}
            counters.update(dict(self.counters))
            summary = {
                "created_at": now_iso_shanghai(),
                "total_tasks": total,
                "success_count": success,
                "failure_count": failed,
                "success_rate": success / total if total else 0.0,
                "avg_task_seconds": _mean(self.task_durations),
                "p50_task_seconds": _percentile(self.task_durations, 50),
                "p90_task_seconds": _percentile(self.task_durations, 90),
                "p95_task_seconds": _percentile(self.task_durations, 95),
                "samples_per_hour": (success / elapsed * 3600.0) if elapsed > 0 else 0.0,
                "stage_seconds": {stage: _describe(self.stage_durations.get(stage, [])) for stage in stage_names},
                "counters": counters,
                "failure_reasons": dict(self.failure_reasons),
                "anomaly_type_success": {
                    anomaly: {
                        "total": self.anomaly_total[anomaly],
                        "success": self.anomaly_success[anomaly],
                        "success_rate": self.anomaly_success[anomaly] / self.anomaly_total[anomaly],
                    }
                    for anomaly in sorted(self.anomaly_total)
                    if self.anomaly_total[anomaly]
                },
                "topk_avg_attempt_count": _mean(self.topk_attempt_counts),
                "topk_success_candidate_rank_avg": _mean(self.success_candidate_ranks),
                "topk_success_candidate_ranks": list(self.success_candidate_ranks),
                "grid_rank_success": {
                    str(rank): {
                        "attempted": self.grid_rank_total[rank],
                        "success": self.grid_rank_success[rank],
                        "success_rate": self.grid_rank_success[rank] / self.grid_rank_total[rank],
                    }
                    for rank in sorted(self.grid_rank_total)
                    if self.grid_rank_total[rank]
                },
                "config": self.config,
            }
        write_json(path, summary)
        return path


def _mean(values: list[float] | list[int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def _describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": float(sum(values)),
        "avg": _mean(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
    }
