from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


POSITIVE_FOLDERS = {"images_floor_clear_water", "images_clear_water"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def weak_positive(row: dict[str, Any]) -> bool:
    return str(row.get("folder", "")) in POSITIVE_FOLDERS


def group_key(row: dict[str, Any]) -> str:
    stem = Path(str(row.get("file_name") or row.get("file", ""))).stem
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else stem


def image_features(path: Path, max_side: int = 192) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    gray = arr.mean(axis=2)
    feats: list[float] = []

    for data in (arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], gray):
        feats.extend(
            [
                float(data.mean()),
                float(data.std()),
                float(np.percentile(data, 10)),
                float(np.percentile(data, 50)),
                float(np.percentile(data, 90)),
            ]
        )

    for channel in (arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], gray):
        hist, _ = np.histogram(channel, bins=12, range=(0, 1), density=True)
        feats.extend(hist.astype(float).tolist())

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    feats.extend(
        [
            float(grad.mean()),
            float(grad.std()),
            float(np.percentile(grad, 90)),
            float(np.percentile(grad, 99)),
        ]
    )

    regions = [
        arr,
        arr[h // 2 :, :, :],
        arr[int(h * 0.65) :, :, :],
        arr[:, : w // 2, :],
        arr[:, w // 2 :, :],
    ]
    for region in regions:
        if region.size == 0:
            continue
        region_gray = region.mean(axis=2)
        feats.extend([float(region_gray.mean()), float(region_gray.std())])
        for channel_index in range(3):
            channel = region[:, :, channel_index]
            feats.extend([float(channel.mean()), float(channel.std())])

    for yy in range(4):
        for xx in range(4):
            y0, y1 = int(h * yy / 4), int(h * (yy + 1) / 4)
            x0, x1 = int(w * xx / 4), int(w * (xx + 1) / 4)
            cell = arr[y0:y1, x0:x1, :]
            if cell.size == 0:
                feats.extend([0.0] * 5)
                continue
            cell_gray = cell.mean(axis=2)
            feats.extend(
                [
                    float(cell_gray.mean()),
                    float(cell_gray.std()),
                    float(cell[:, :, 0].mean()),
                    float(cell[:, :, 1].mean()),
                    float(cell[:, :, 2].mean()),
                ]
            )

    return np.asarray(feats, dtype=np.float32)


def load_or_build_features(rows: list[dict[str, Any]], cache_path: Path, max_side: int) -> tuple[np.ndarray, list[str]]:
    files = [str(Path(str(row["file"])).resolve()) for row in rows]
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        cached_files = cache["files"].astype(str).tolist()
        if cached_files == files:
            return cache["features"].astype(np.float32), files

    features = []
    kept_files = []
    for index, file in enumerate(files, start=1):
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(file)
        features.append(image_features(path, max_side=max_side))
        kept_files.append(file)
        if index % 100 == 0:
            print(f"features {index}/{len(files)}", flush=True)
    matrix = np.vstack(features).astype(np.float32)
    np.savez_compressed(cache_path, features=matrix, files=np.asarray(kept_files, dtype=object))
    return matrix, kept_files


def make_model(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> dict[str, Any]:
    unique_groups = len(set(groups.tolist()))
    folds = max(2, min(folds, unique_groups))
    probs = np.zeros(len(y), dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model(seed + fold)
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        print(f"cv fold {fold}/{folds} done", flush=True)

    max_neg = float(probs[y == 0].max()) if np.any(y == 0) else 1.0
    zero_neg = probs > max_neg
    return {
        "cv_auc": float(roc_auc_score(y, probs)),
        "cv_ap": float(average_precision_score(y, probs)),
        "cv_zero_negative_threshold": max_neg,
        "cv_zero_negative_selected": int(zero_neg.sum()),
        "cv_zero_negative_positive": int((zero_neg & (y == 1)).sum()),
        "cv_prob_min": float(probs.min()),
        "cv_prob_max": float(probs.max()),
    }


def basic_vlm_accept(row: dict[str, Any]) -> bool:
    return (
        str(row.get("decision", "")).strip().lower() == "accept_clear_water"
        and parse_bool(row.get("can_see_image"))
        and parse_bool(row.get("usable"))
        and parse_bool(row.get("floor_clear_water_visible"))
        and str(row.get("hard_negative_type", "")).strip().lower() == "none"
    )


def choose_threshold(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    require_vlm_accept: bool,
    target_denominator: int,
    target_max_rate: float,
) -> tuple[float, dict[str, Any]]:
    mask = np.ones(len(rows), dtype=bool)
    if require_vlm_accept:
        mask &= np.asarray([basic_vlm_accept(row) for row in rows], dtype=bool)
    negatives = np.asarray([not weak_positive(row) for row in rows], dtype=bool)
    negative_scores = scores[mask & negatives]
    threshold = float(negative_scores.max() + 1e-12) if negative_scores.size else 1.0
    selected = mask & (scores > threshold)
    max_selected = int(target_denominator * target_max_rate)
    if selected.sum() > max_selected:
        sorted_scores = np.sort(scores[mask])[::-1]
        threshold = float(sorted_scores[max_selected - 1] + 1e-12) if max_selected > 0 else 1.0
        selected = mask & (scores > threshold)
    positives = np.asarray([weak_positive(row) for row in rows], dtype=bool)
    return threshold, {
        "selected": int(selected.sum()),
        "weak_positive_selected": int((selected & positives).sum()),
        "weak_negative_selected": int((selected & negatives).sum()),
        "selected_rate": float(selected.sum() / target_denominator) if target_denominator else 0.0,
        "weak_precision": float((selected & positives).sum() / selected.sum()) if selected.sum() else 0.0,
        "weak_recall_on_candidate_positives": float((selected & positives).sum() / positives.sum()) if positives.sum() else 0.0,
    }


def materialize_file(source: Path, dest: Path, *, mode: str, copy_fallback: bool) -> tuple[str, str]:
    if mode == "manifest-only":
        return "", "manifest-only"
    if dest.exists():
        dest.unlink()
    try:
        if mode == "hardlink":
            os.link(source, dest)
        elif mode == "symlink":
            os.symlink(source, dest)
        elif mode == "copy":
            shutil.copy2(source, dest)
        else:
            raise ValueError(f"unknown file mode: {mode}")
        return str(dest), mode
    except OSError:
        if copy_fallback and mode != "copy":
            shutil.copy2(source, dest)
            return str(dest), "copy_fallback"
        return "", f"{mode}_failed_manifest_only"


def build_pack(
    selected_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    file_mode: str,
    copy_fallback: bool,
) -> Path:
    pack_dir = out_dir / f"visual_calibrated_selected_pack_{len(selected_rows)}"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        source = Path(str(row["file"]))
        target = pack_dir / source.name
        if target.exists():
            target = pack_dir / f"{source.stem}_{abs(hash(str(source.parent))) % 100000}{source.suffix}"
        pack_file, pack_file_mode = materialize_file(
            source,
            target,
            mode=file_mode,
            copy_fallback=copy_fallback,
        )
        manifest_rows.append(
            {
                **row,
                "source_file": str(source),
                "pack_file": pack_file,
                "pack_file_mode": pack_file_mode,
            }
        )
    write_csv(pack_dir / "manifest.csv", manifest_rows)
    return pack_dir


def write_summary(path: Path, rows: list[dict[str, Any]], cv: dict[str, Any], threshold: float, stats: dict[str, Any]) -> None:
    selected_rows = [row for row in rows if parse_bool(row.get("visual_calibrated_selected"))]
    folder_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        folder = str(row.get("folder", ""))
        folder_counts[folder]["total"] += 1
        if parse_bool(row.get("visual_calibrated_selected")):
            folder_counts[folder]["selected"] += 1
    reason_counts = Counter(str(row.get("visual_calibration_reason")) for row in rows)

    lines = [
        "# visual calibrated clear_water filter summary",
        "",
        f"total_candidates: {len(rows)}",
        f"selected_clear_water: {len(selected_rows)} ({stats['selected_rate']:.2%} of full denominator)",
        f"weak_negative_selected: {stats['weak_negative_selected']}",
        f"weak_precision_by_folder_proxy: {stats['weak_precision']:.2%}",
        f"weak_recall_on_candidate_clear_water: {stats['weak_recall_on_candidate_positives']:.2%}",
        f"chosen_visual_threshold: {threshold:.6f}",
        "",
        "## Cross-validation on weak labels",
        f"cv_auc: {cv['cv_auc']:.4f}",
        f"cv_average_precision: {cv['cv_ap']:.4f}",
        f"cv_zero_negative_selected: {cv['cv_zero_negative_selected']}",
        f"cv_zero_negative_positive: {cv['cv_zero_negative_positive']}",
        "",
        "## By folder, audit only",
        "| folder | selected | total | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for folder, counter in sorted(folder_counts.items()):
        total = counter["total"]
        selected = counter["selected"]
        lines.append(f"| {folder} | {selected} | {total} | {selected / total if total else 0:.2%} |")
    lines.extend(["", "## Reason counts"])
    for reason, count in reason_counts.most_common():
        lines.append(f"- {reason}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    full_csv = Path(args.full_csv).resolve()
    candidate_csv = Path(args.candidate_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else candidate_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(full_csv)
    candidate_rows = read_csv(candidate_csv)

    X, _ = load_or_build_features(
        train_rows,
        out_dir / f"{args.prefix}_train_features.npz",
        max_side=args.max_side,
    )
    y = np.asarray([1 if weak_positive(row) else 0 for row in train_rows], dtype=np.int64)
    groups = np.asarray([group_key(row) for row in train_rows])
    cv = cross_validate(X, y, groups, folds=args.cv_folds, seed=args.seed)

    model = make_model(args.seed)
    model.fit(X, y)

    X_candidate, _ = load_or_build_features(
        candidate_rows,
        out_dir / f"{args.prefix}_candidate_features.npz",
        max_side=args.max_side,
    )
    scores = model.predict_proba(X_candidate)[:, 1]
    threshold, stats = choose_threshold(
        candidate_rows,
        scores,
        require_vlm_accept=args.require_vlm_accept,
        target_denominator=args.target_denominator,
        target_max_rate=args.target_max_rate,
    )

    selected_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    for row, score in zip(candidate_rows, scores):
        row_out: dict[str, Any] = dict(row)
        row_out["visual_clear_water_score"] = f"{float(score):.8f}"
        row_out["visual_threshold"] = f"{threshold:.8f}"
        if args.require_vlm_accept and not basic_vlm_accept(row):
            selected = False
            reason = "vlm_basic_reject"
        elif float(score) > threshold:
            selected = True
            reason = "visual_score_above_zero_negative_threshold"
        else:
            selected = False
            reason = "visual_score_below_threshold"
        row_out["visual_calibrated_selected"] = selected
        row_out["visual_calibration_reason"] = reason
        output_rows.append(row_out)
        if selected:
            selected_rows.append(row_out)

    csv_path = out_dir / f"{args.prefix}.csv"
    selected_csv = out_dir / f"{args.prefix}_selected.csv"
    rejected_csv = out_dir / f"{args.prefix}_rejected.csv"
    summary_path = out_dir / f"{args.prefix}_summary.md"
    metadata_path = out_dir / f"{args.prefix}_metadata.json"
    write_csv(csv_path, output_rows)
    write_csv(selected_csv, selected_rows)
    write_csv(rejected_csv, [row for row in output_rows if not parse_bool(row.get("visual_calibrated_selected"))])
    pack_dir = (
        build_pack(
            selected_rows,
            out_dir,
            file_mode=args.pack_file_mode,
            copy_fallback=args.copy_fallback,
        )
        if args.build_pack
        else None
    )
    write_summary(summary_path, output_rows, cv, threshold, stats)
    metadata_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "stats": stats,
                "cv": cv,
                "pack_dir": str(pack_dir) if pack_dir else None,
                "target_denominator": args.target_denominator,
                "target_max_rate": args.target_max_rate,
                "require_vlm_accept": args.require_vlm_accept,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for latest, source in {
        out_dir / f"{args.prefix}_latest.csv": csv_path,
        out_dir / f"{args.prefix}_latest_selected.csv": selected_csv,
        out_dir / f"{args.prefix}_latest_rejected.csv": rejected_csv,
        out_dir / f"{args.prefix}_latest_summary.md": summary_path,
        out_dir / f"{args.prefix}_latest_metadata.json": metadata_path,
    }.items():
        shutil.copy2(source, latest)
    print(f"CSV={csv_path}")
    print(f"SELECTED_CSV={selected_csv}")
    print(f"SUMMARY={summary_path}")
    if pack_dir:
        print(f"PACK_DIR={pack_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-only calibration layer for clear_water QC.")
    parser.add_argument("--full-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prefix", default="second_wave_clear_water_v5_visual_calibrated")
    parser.add_argument("--target-denominator", type=int, default=728)
    parser.add_argument("--target-max-rate", type=float, default=0.65)
    parser.add_argument("--max-side", type=int, default=192)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--require-vlm-accept", action="store_true", default=True)
    parser.add_argument("--no-require-vlm-accept", dest="require_vlm_accept", action="store_false")
    parser.add_argument("--build-pack", action="store_true")
    parser.add_argument(
        "--pack-file-mode",
        choices=["hardlink", "symlink", "copy", "manifest-only"],
        default="hardlink",
        help="How to expose selected images in the pack. hardlink avoids duplicate disk usage on the same drive.",
    )
    parser.add_argument(
        "--copy-fallback",
        action="store_true",
        help="Copy source images only if hardlink/symlink creation fails. Default keeps manifest only on failure.",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
