from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from grid import bbox_to_box_dict


def _clip_roi(roi_bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi_bbox
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def localize_change_bbox(
    original_image_path: str,
    edited_image_path: str,
    roi_bbox: tuple[int, int, int, int],
    anomaly_type: str,
    mask_output_path: str,
) -> dict:
    original = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    edited = cv2.imread(edited_image_path, cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"failed to read original image: {original_image_path}")
    if edited is None:
        raise ValueError(f"failed to read edited image: {edited_image_path}")
    if edited.shape[:2] != original.shape[:2]:
        edited = cv2.resize(edited, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)

    height, width = original.shape[:2]
    x1, y1, x2, y2 = _clip_roi(roi_bbox, width, height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty roi_bbox")

    ori_roi = original[y1:y2, x1:x2]
    edt_roi = edited[y1:y2, x1:x2]

    lab_ori = cv2.cvtColor(ori_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_edt = cv2.cvtColor(edt_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    color_diff = np.linalg.norm(lab_ori - lab_edt, axis=2)
    color_diff = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    gray_ori = cv2.cvtColor(ori_roi, cv2.COLOR_BGR2GRAY)
    gray_edt = cv2.cvtColor(edt_roi, cv2.COLOR_BGR2GRAY)
    struct_diff = cv2.absdiff(gray_ori, gray_edt)

    merged = cv2.addWeighted(color_diff, 0.65, struct_diff, 0.35, 0)
    merged = cv2.GaussianBlur(merged, (5, 5), 0)
    _, otsu = cv2.threshold(merged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fixed_threshold = 12 if anomaly_type == "water_leak" else 18 if anomaly_type in {"diesel_leak", "coolant_leak"} else 14
    _, fixed = cv2.threshold(merged, fixed_threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(otsu, fixed)

    if anomaly_type == "coolant_leak":
        hsv = cv2.cvtColor(edt_roi, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([35, 20, 40]), np.array([95, 255, 255]))
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(green, fixed))
    elif anomaly_type == "water_leak":
        hsv = cv2.cvtColor(edt_roi, cv2.COLOR_BGR2HSV)
        low_saturation = cv2.inRange(hsv, np.array([0, 0, 70]), np.array([180, 80, 255]))
        _, subtle_structure = cv2.threshold(struct_diff, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(low_saturation, subtle_structure))
    elif anomaly_type == "oil_leak":
        gray_drop = cv2.subtract(gray_ori, gray_edt)
        _, dark = cv2.threshold(gray_drop, 12, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(mask, dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = width * height
    min_area = max(16, int(image_area * 0.00005))
    max_area = int(image_area * 0.20)
    keep = np.zeros(mask.shape, dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    total_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if area > max_area:
            continue
        lx = int(stats[label, cv2.CC_STAT_LEFT])
        ly = int(stats[label, cv2.CC_STAT_TOP])
        lw = int(stats[label, cv2.CC_STAT_WIDTH])
        lh = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((lx, ly, lx + lw, ly + lh))
        keep[labels == label] = 255
        total_area += area

    if not boxes:
        raise ValueError("diff localization failed: no valid changed connected components")
    if total_area > max_area:
        raise ValueError(f"diff localization failed: changed area too large ({total_area})")

    bx1 = min(b[0] for b in boxes) + x1
    by1 = min(b[1] for b in boxes) + y1
    bx2 = max(b[2] for b in boxes) + x1
    by2 = max(b[3] for b in boxes) + y1
    if bx2 - bx1 < 8 or by2 - by1 < 8:
        raise ValueError(f"diff localization failed: bbox too small {(bx1, by1, bx2, by2)}")

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = keep
    Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(mask_output_path, full_mask)

    return {
        "bbox": bbox_to_box_dict((bx1, by1, bx2, by2)),
        "area": int(total_area),
        "mask_uri": str(mask_output_path),
        "status": "ok",
        "diff_method": "lab_difference_grayscale_difference_morphology_connected_components",
    }
