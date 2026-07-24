from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ("background", "line", "road")
PALETTE = ((0, 0, 0), (220, 20, 60), (128, 64, 128))


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _window_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> int:
    return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _positions(start: int, stop: int, stride: int) -> list[int]:
    if stop < start:
        return []
    values = list(range(start, stop + 1, stride))
    if not values or values[-1] != stop:
        values.append(stop)
    return values


def _stable_tiebreak(sample_id: str, bbox: tuple[int, int, int, int]) -> float:
    payload = f"{sample_id}:{bbox[0]}:{bbox[1]}:{bbox[2]}:{bbox[3]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") / 2**32


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = True
    kernel = np.ones((5, 5), dtype=np.uint8)
    return cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)


def _distance_to_non_road(road_mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return np.zeros(road_mask.shape, dtype=np.float32)
    return cv2.distanceTransform(road_mask.astype(np.uint8), cv2.DIST_L2, 3)


def select_floor_box(
    labels: np.ndarray,
    sample_id: str,
    box_size: int = 200,
    road_class_id: int = 2,
    line_class_id: int = 1,
    road_probability: np.ndarray | None = None,
    road_coverage_min: float = 0.95,
    line_coverage_max: float = 0.05,
    stride: int | None = None,
) -> dict[str, Any]:
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation label map, got shape={labels.shape}")
    height, width = labels.shape
    box_size = int(box_size)
    if box_size <= 0:
        raise ValueError("box_size must be positive")
    if width < box_size + 2 or height < box_size + 2:
        return {
            "status": "skipped",
            "reason": "no_visible_floor_region",
            "detail": "image is smaller than the required edit box",
        }

    raw_road = labels == int(road_class_id)
    road = _remove_small_components(raw_road, max(64, box_size * box_size // 100))
    line = labels == int(line_class_id)
    floor_area_ratio = float(road.mean()) if road.size else 0.0
    if not road.any():
        return {
            "status": "skipped",
            "reason": "no_visible_floor_region",
            "detail": "segmentation contains no road pixels",
            "floor_area_ratio": floor_area_ratio,
        }

    road_integral = _integral(road)
    line_integral = _integral(line)
    probability_integral = _integral(
        np.clip(road_probability, 0.0, 1.0) * 1000000
        if road_probability is not None
        else road.astype(np.float32) * 1000000
    )
    boundary_distance = _distance_to_non_road(road)
    area = box_size * box_size
    stride = max(8, box_size // 20) if stride is None else max(1, int(stride))
    xs = _positions(1, width - box_size - 1, stride)
    ys = _positions(1, height - box_size - 1, stride)

    best: tuple[float, tuple[int, int, int, int], dict[str, float]] | None = None
    for y1 in ys:
        for x1 in xs:
            x2 = x1 + box_size
            y2 = y1 + box_size
            road_ratio = _window_sum(road_integral, x1, y1, x2, y2) / area
            if road_ratio < float(road_coverage_min):
                continue
            line_ratio = _window_sum(line_integral, x1, y1, x2, y2) / area
            if line_ratio > float(line_coverage_max):
                continue
            center_x = x1 + box_size // 2
            center_y = y1 + box_size // 2
            if not road[center_y, center_x]:
                continue
            mean_probability = (
                _window_sum(probability_integral, x1, y1, x2, y2) / area / 1000000.0
            )
            boundary_distance_score = 1.0 - min(
                float(boundary_distance[center_y, center_x]) / max(1.0, box_size * 1.5),
                1.0,
            )
            lower_image_score = center_y / max(1, height)
            edge_clearance = min(x1, y1, width - x2, height - y2)
            edge_score = min(edge_clearance / max(1.0, box_size), 1.0)
            bbox = (x1, y1, x2, y2)
            score = (
                0.50 * road_ratio
                + 0.25 * mean_probability
                + 0.12 * boundary_distance_score
                + 0.08 * lower_image_score
                + 0.05 * edge_score
                + 1e-7 * _stable_tiebreak(sample_id, bbox)
            )
            metrics = {
                "road_coverage_ratio": road_ratio,
                "line_coverage_ratio": line_ratio,
                "mean_road_probability": mean_probability,
                "floor_area_ratio": floor_area_ratio,
                "selection_score": score,
            }
            if best is None or score > best[0]:
                best = (score, bbox, metrics)

    if best is None:
        return {
            "status": "skipped",
            "reason": "no_visible_floor_region",
            "detail": f"no {box_size}x{box_size} box meets road/line coverage thresholds",
            "floor_area_ratio": floor_area_ratio,
        }
    return {
        "status": "selected",
        "bbox": list(best[1]),
        **best[2],
    }


def _resize_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if labels.shape == (height, width):
        return labels
    return np.asarray(
        Image.fromarray(labels.astype(np.uint8), mode="L").resize((width, height), Image.Resampling.NEAREST)
    )


def _resize_probability(probability: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    if probability is None:
        return None
    width, height = size
    if probability.shape == (height, width):
        return probability
    image = Image.fromarray(probability.astype(np.float32), mode="F")
    return np.asarray(image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)


def _save_artifacts(
    image_path: Path,
    labels: np.ndarray,
    selection: dict[str, Any],
    road_class_id: int,
    mask_path: Path,
    overlay_path: Path,
) -> None:
    road = labels == int(road_class_id)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(road, 255, 0).astype(np.uint8), mode="L").save(mask_path)

    with Image.open(image_path) as source:
        overlay = source.convert("RGB")
    tint = Image.new("RGB", overlay.size, (0, 180, 170))
    alpha = Image.fromarray(np.where(road, 90, 0).astype(np.uint8), mode="L")
    overlay = Image.composite(tint, overlay, alpha)
    if selection.get("status") == "selected":
        ImageDraw.Draw(overlay).rectangle(tuple(selection["bbox"]), outline=(255, 215, 0), width=4)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(overlay_path, quality=92)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(config: str, checkpoint: str, device: str):
    from mmseg.apis import init_model

    model = init_model(config, checkpoint, device=device)
    model.dataset_meta = {"classes": CLASSES, "palette": PALETTE}
    return model


def _infer(model, image_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    from mmseg.apis import inference_model

    result = inference_model(model, str(image_path))
    labels = result.pred_sem_seg.data.detach().cpu().numpy().squeeze().astype(np.uint8)
    road_probability = None
    logits = getattr(result, "seg_logits", None)
    if logits is not None:
        values = logits.data.detach().float().cpu().numpy()
        values -= values.max(axis=0, keepdims=True)
        exp_values = np.exp(values)
        probabilities = exp_values / np.maximum(exp_values.sum(axis=0, keepdims=True), 1e-12)
        road_probability = probabilities[2].astype(np.float32)
    return labels, road_probability


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch MMSeg floor selector worker")
    parser.add_argument("--requests-jsonl", required=False)
    parser.add_argument("--results-jsonl", required=False)
    parser.add_argument("--config", required=False)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="segformer_mit-b0-roadline1000")
    parser.add_argument("--road-class-id", type=int, default=2)
    parser.add_argument("--line-class-id", type=int, default=1)
    parser.add_argument("--box-size", type=int, default=200)
    parser.add_argument("--road-coverage-min", type=float, default=0.95)
    parser.add_argument("--line-coverage-max", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if not args.dry_run:
        if not args.config or not Path(args.config).is_file():
            raise FileNotFoundError(f"MMSeg config not found: {args.config}")
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"MMSeg checkpoint not found: {args.checkpoint}")
        model = _load_model(str(Path(args.config).resolve()), str(checkpoint_path), args.device)
        checkpoint_digest = _checkpoint_digest(checkpoint_path)
    else:
        model = None
        checkpoint_digest = "dry_run"

    if args.check_only:
        num_classes = int(getattr(model.decode_head, "num_classes", 0)) if model is not None else 3
        if num_classes != 3:
            raise RuntimeError(f"Expected a three-class floor model, got num_classes={num_classes}")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "model_name": args.model_name,
                    "num_classes": num_classes,
                    "classes": CLASSES,
                    "checkpoint_digest": checkpoint_digest,
                    "device": args.device,
                }
            )
        )
        return 0
    if not args.requests_jsonl or not args.results_jsonl:
        parser.error("--requests-jsonl and --results-jsonl are required unless --check-only is used")

    results: list[dict[str, Any]] = []
    for request in _read_jsonl(Path(args.requests_jsonl)):
        sample_id = str(request["sample_id"])
        image_path = Path(str(request["image_path"]))
        try:
            with Image.open(image_path) as source:
                size = source.size
            if args.dry_run:
                labels = np.full((size[1], size[0]), args.road_class_id, dtype=np.uint8)
                road_probability = np.ones(labels.shape, dtype=np.float32)
            else:
                labels, road_probability = _infer(model, image_path)
                labels = _resize_labels(labels, size)
                road_probability = _resize_probability(road_probability, size)
            selection = select_floor_box(
                labels,
                sample_id,
                box_size=int(request.get("box_size") or args.box_size),
                road_class_id=args.road_class_id,
                line_class_id=args.line_class_id,
                road_probability=road_probability,
                road_coverage_min=args.road_coverage_min,
                line_coverage_max=args.line_coverage_max,
            )
            mask_path = Path(str(request["mask_path"]))
            overlay_path = Path(str(request["overlay_path"]))
            _save_artifacts(
                image_path,
                labels,
                selection,
                args.road_class_id,
                mask_path,
                overlay_path,
            )
            results.append(
                {
                    "sample_id": sample_id,
                    "selection_backend": "mmseg_floor_selector",
                    "segmentation_model": args.model_name,
                    "checkpoint_digest": checkpoint_digest,
                    "floor_class_id": args.road_class_id,
                    "line_class_id": args.line_class_id,
                    "floor_mask_path": str(mask_path.resolve()),
                    "floor_overlay_path": str(overlay_path.resolve()),
                    **selection,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "failed",
                    "reason": "floor_segmentation_failed",
                    "error": str(exc),
                }
            )
    _write_jsonl(Path(args.results_jsonl), results)
    return 0 if all(row.get("status") != "failed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
