from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.qc_agent import (  # noqa: E402
    build_overlay_data_url,
    parse_vlm_qc_payload,
    resolve_local_uri,
)
from autolabel.adapters.crop_reviewer import parse_review_payload  # noqa: E402
from autolabel.utils import read_json  # noqa: E402


CROP_EVIDENCE_BOOL_KEYS = {
    "inside_has_target_anomaly",
    "outside_has_same_anomaly",
    "outside_extension",
}
CROP_EVIDENCE_TEXT_KEYS = {
    "containment_quality",
    "target_location",
    "outside_evidence_summary",
}
TRAINING_BOOL_KEYS = {
    "image_has_positive_candidate",
    "visual_positive_class",
    "bbox_usable_for_training",
    "weak_label_consistent",
    "label_conflict",
}
TRAINING_TEXT_KEYS = {
    "training_action",
    "rework_type",
    "training_priority",
    "visual_class",
    "positive_candidate_location",
    "positive_candidate_relation_to_bbox",
    "hard_negative_type",
}
CONTAINMENT_VALUES = {
    "tight_contained",
    "loose_but_ok",
    "over_inclusive",
    "under_inclusive",
    "wrong_target",
    "uncertain",
}
TRAINING_ACTION_VALUES = {
    "auto_accept_positive",
    "rework_bbox_positive_candidate",
    "manual_review",
    "label_conflict_review",
    "reject_from_positive_training",
}
REWORK_TYPE_VALUES = {
    "none",
    "expand_bbox",
    "shrink_bbox",
    "move_bbox",
    "relabeled_negative_or_other",
    "uncertain",
}
TRAINING_PRIORITY_VALUES = {"high", "medium", "low", "reject"}
REVIEW_MODES = {"overlay", "crop_evidence", "training_triage"}
TRIAGE_PROMPT_VERSIONS = {"v4", "v5"}
TRIAGE_PROMPT_REVISIONS = {"v4": "v4", "v5": "v5.7"}
BOX_QUALITY_VALUES = {"good", "over_inclusive", "under_inclusive", "wrong_target", "uncertain"}
SEMANTIC_VERDICT_VALUES = {"clear_positive", "probable_positive", "hard_negative", "uncertain"}
SEMANTIC_LOCATION_VALUES = {
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
    "multiple",
    "none",
    "uncertain",
}
SEMANTIC_HARD_NEGATIVE_VALUES = {
    "none",
    "dry_floor",
    "shadow",
    "uniform_specular_reflection",
    "text_or_overlay",
    "wall_only_water",
    "equipment_surface",
    "colored_liquid_or_stain",
    "person_or_clutter",
    "floor_not_visible",
    "low_image_quality",
    "other",
    "uncertain",
}
SEMANTIC_SURFACE_VALUES = {
    "floor",
    "floor_equipment_junction",
    "wall",
    "equipment_surface",
    "other",
    "none",
    "uncertain",
}
INSIDE_TARGET_STATUS_VALUES = {"clear_target", "probable_target", "no_target", "uncertain"}
OUTSIDE_RELATION_VALUES = {
    "continuous_main_target",
    "minor_same_target_fringe",
    "separate_target",
    "none",
    "uncertain",
}
TARGET_SURFACE_VALUES = {
    "floor",
    "floor_equipment_junction",
    "wall",
    "equipment_surface",
    "other",
    "none",
    "uncertain",
}
GEOMETRY_CONFOUNDER_VALUES = {
    "none",
    "shadow",
    "uniform_specular_reflection",
    "text_or_overlay",
    "wall_only_water",
    "equipment_surface",
    "colored_liquid_or_stain",
    "person_or_clutter",
    "dry_floor",
    "low_image_quality",
    "other",
    "uncertain",
}
GEOMETRY_EDGE_KEYS = (
    "target_crosses_left_edge",
    "target_crosses_top_edge",
    "target_crosses_right_edge",
    "target_crosses_bottom_edge",
)
TARGET_AREA_FRACTION_VALUES = {
    "none",
    "lt_10_percent",
    "10_to_25_percent",
    "25_to_50_percent",
    "50_to_75_percent",
    "gt_75_percent",
    "uncertain",
}


def box_numbers(box: Any) -> tuple[int, int, int, int] | None:
    try:
        if isinstance(box, dict):
            return int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def box_iou(left: Any, right: Any) -> float:
    left_values = box_numbers(left)
    right_values = box_numbers(right)
    if left_values is None or right_values is None:
        return 0.0
    ax1, ay1, ax2, ay2 = left_values
    bx1, by1, bx2, by2 = right_values
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    left_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    right_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def box_contains(outer: Any, inner: Any, tolerance: int = 0) -> bool:
    outer_values = box_numbers(outer)
    inner_values = box_numbers(inner)
    if outer_values is None or inner_values is None:
        return False
    ox1, oy1, ox2, oy2 = outer_values
    ix1, iy1, ix2, iy2 = inner_values
    return ox1 - tolerance <= ix1 and oy1 - tolerance <= iy1 and ox2 + tolerance >= ix2 and oy2 + tolerance >= iy2


def parse_boolish(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是", "有"}:
        return True
    if text in {"false", "0", "no", "n", "否", "无"}:
        return False
    return default


def normalize_containment_quality(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    return text if text in CONTAINMENT_VALUES else "uncertain"


def _normalized_choice(value: Any, allowed: set[str], default: str | None = None) -> str | None:
    if value in (None, ""):
        return default
    text = str(value).strip().lower()
    return text if text in allowed else default


def add_crop_evidence_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    raw = parsed.get("raw")
    if not isinstance(raw, dict):
        return parsed

    for key in CROP_EVIDENCE_BOOL_KEYS:
        parsed[key] = parse_boolish(raw.get(key))
    for key in CROP_EVIDENCE_TEXT_KEYS:
        value = raw.get(key)
        parsed[key] = str(value).strip() if value not in (None, "") else None
    parsed["containment_quality"] = normalize_containment_quality(parsed.get("containment_quality"))

    inside = parsed.get("inside_has_target_anomaly")
    if inside is not None:
        parsed["visible_target"] = inside
    if parsed.get("outside_extension") and parsed.get("box_quality") == "good":
        parsed["box_quality"] = "under_inclusive"

    for key in TRAINING_BOOL_KEYS:
        parsed[key] = parse_boolish(raw.get(key))
    for key in TRAINING_TEXT_KEYS:
        value = raw.get(key)
        parsed[key] = str(value).strip() if value not in (None, "") else None
    parsed["training_action"] = _normalized_choice(
        parsed.get("training_action"),
        TRAINING_ACTION_VALUES,
        default="manual_review",
    )
    parsed["rework_type"] = _normalized_choice(parsed.get("rework_type"), REWORK_TYPE_VALUES, default="uncertain")
    parsed["training_priority"] = _normalized_choice(
        parsed.get("training_priority"),
        TRAINING_PRIORITY_VALUES,
        default="medium",
    )
    return parsed


def append_issue_flag(parsed: dict[str, Any], flag: str) -> None:
    flags = parsed.get("issue_flags") if isinstance(parsed.get("issue_flags"), list) else []
    if flag not in flags:
        flags.append(flag)
    parsed["issue_flags"] = flags


def apply_training_triage_guard(parsed: dict[str, Any], weak_label_hint: str) -> dict[str, Any]:
    if not parsed.get("training_action"):
        return parsed
    weak = weak_label_hint.lower()
    from_negative_bucket = weak.startswith("negative")
    from_positive_bucket = weak.startswith("positive")
    action = parsed.get("training_action")
    image_has_candidate = parsed.get("image_has_positive_candidate") is True
    visual_positive = parsed.get("visual_positive_class") is True
    inside_has_target = parsed.get("inside_has_target_anomaly") is True
    bbox_usable = parsed.get("bbox_usable_for_training")
    box_quality = parsed.get("box_quality")
    outside_extension = parsed.get("outside_extension") is True
    has_positive_candidate = image_has_candidate or visual_positive

    if from_negative_bucket and (
        action in {"auto_accept_positive", "rework_bbox_positive_candidate"}
        or has_positive_candidate
    ):
        parsed["training_action"] = "label_conflict_review"
        parsed["label_conflict"] = True
        parsed["needs_human_review"] = True
        parsed["training_priority"] = "medium"
        append_issue_flag(parsed, "weak_label_conflict")

    if from_positive_bucket and action == "auto_accept_positive":
        if not inside_has_target or bbox_usable is False or box_quality != "good" or outside_extension:
            parsed["training_action"] = "rework_bbox_positive_candidate"
            parsed["bbox_usable_for_training"] = False
            parsed["needs_human_review"] = True
            parsed["training_priority"] = "high"
            if outside_extension:
                parsed["rework_type"] = "expand_bbox"
            elif not inside_has_target:
                parsed["rework_type"] = "move_bbox"
            elif parsed.get("rework_type") in (None, "none"):
                parsed["rework_type"] = "uncertain"
            append_issue_flag(parsed, "auto_accept_downgraded_to_bbox_rework")
        else:
            parsed["label_conflict"] = False
            parsed["needs_human_review"] = False
            parsed["training_priority"] = parsed.get("training_priority") or "high"

    if from_positive_bucket and action == "reject_from_positive_training" and has_positive_candidate:
        parsed["training_action"] = "rework_bbox_positive_candidate"
        parsed["bbox_usable_for_training"] = False
        parsed["label_conflict"] = False
        parsed["needs_human_review"] = True
        parsed["training_priority"] = "high"
        if not inside_has_target:
            parsed["rework_type"] = "move_bbox"
        elif outside_extension:
            parsed["rework_type"] = "expand_bbox"
        elif parsed.get("rework_type") in (None, "none"):
            parsed["rework_type"] = "uncertain"
        append_issue_flag(parsed, "positive_candidate_needs_bbox_rework")

    if from_positive_bucket and parsed.get("training_action") == "auto_accept_positive":
        parsed["label_conflict"] = False
        parsed["needs_human_review"] = False
        parsed["training_priority"] = parsed.get("training_priority") or "high"
    return parsed


def _normalized_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    result: list[str] = []
    for item in items:
        text = str(item).strip().lower()
        if text and text not in result:
            result.append(text)
    return result


def parse_semantic_v5_payload(text: str) -> dict[str, Any]:
    raw = parse_review_payload(text)
    verdict = _normalized_choice(raw.get("semantic_verdict"), SEMANTIC_VERDICT_VALUES, default="uncertain")
    candidate_location = _normalized_choice(
        raw.get("candidate_location"),
        SEMANTIC_LOCATION_VALUES,
        default="uncertain",
    )
    hard_negative_type = _normalized_choice(
        raw.get("hard_negative_type"),
        SEMANTIC_HARD_NEGATIVE_VALUES,
        default="uncertain",
    )
    candidate_surface = _normalized_choice(
        raw.get("candidate_surface"),
        SEMANTIC_SURFACE_VALUES,
        default="uncertain",
    )
    if verdict in {"clear_positive", "probable_positive"}:
        if candidate_surface in {"wall", "equipment_surface", "other", "none"}:
            verdict = "hard_negative"
            hard_negative_type = (
                "wall_only_water"
                if candidate_surface == "wall"
                else "equipment_surface"
                if candidate_surface == "equipment_surface"
                else "other"
            )
        elif candidate_surface == "uncertain":
            verdict = "uncertain"
            hard_negative_type = "uncertain"
        else:
            hard_negative_type = "none"
    elif verdict == "hard_negative" and hard_negative_type == "none":
        hard_negative_type = "other"
    return {
        "semantic_verdict": verdict,
        "candidate_location": candidate_location,
        "candidate_surface": candidate_surface,
        "positive_evidence": _normalized_string_list(raw.get("positive_evidence")),
        "hard_negative_type": hard_negative_type,
        "semantic_reason": str(raw.get("reason") or "").strip(),
        "semantic_raw": raw,
        "semantic_raw_response_text": text,
    }


def parse_geometry_v5_payload(text: str) -> dict[str, Any]:
    raw = parse_review_payload(text)
    inside_status = _normalized_choice(
        raw.get("inside_target_status"),
        INSIDE_TARGET_STATUS_VALUES,
        default="uncertain",
    )
    outside_relation = _normalized_choice(
        raw.get("outside_relation"),
        OUTSIDE_RELATION_VALUES,
        default="uncertain",
    )
    box_quality = _normalized_choice(raw.get("box_quality"), BOX_QUALITY_VALUES, default="uncertain")
    target_surface = _normalized_choice(raw.get("target_surface"), TARGET_SURFACE_VALUES, default="uncertain")
    confounder = _normalized_choice(
        raw.get("dominant_confounder"),
        GEOMETRY_CONFOUNDER_VALUES,
        default="uncertain",
    )
    edge_checks = {key: parse_boolish(raw.get(key)) for key in GEOMETRY_EDGE_KEYS}
    edge_checks_complete = all(value is not None for value in edge_checks.values())
    background_excess = parse_boolish(raw.get("background_excess"))
    target_area_fraction = _normalized_choice(
        raw.get("target_area_fraction"),
        TARGET_AREA_FRACTION_VALUES,
        default="uncertain",
    )
    truncated_edges = [
        key.removeprefix("target_crosses_").removesuffix("_edge")
        for key, value in edge_checks.items()
        if value is True
    ]
    if target_area_fraction in {"lt_10_percent", "10_to_25_percent"}:
        background_excess = True
    elif (
        target_area_fraction == "25_to_50_percent"
        and confounder not in {"none", "uncertain"}
    ):
        background_excess = True

    if inside_status == "no_target" and box_quality != "uncertain":
        box_quality = "wrong_target"
    if inside_status == "uncertain":
        box_quality = "uncertain"
    if outside_relation == "continuous_main_target" and inside_status in {"clear_target", "probable_target"}:
        box_quality = "under_inclusive"
    if truncated_edges and inside_status in {"clear_target", "probable_target"}:
        outside_relation = "continuous_main_target"
        box_quality = "under_inclusive"
    if (
        background_excess is True
        and not truncated_edges
        and inside_status in {"clear_target", "probable_target"}
    ):
        box_quality = "over_inclusive"
    if inside_status in {"clear_target", "probable_target"} and target_surface in {
        "wall",
        "equipment_surface",
        "other",
        "none",
    }:
        box_quality = "wrong_target"
    if inside_status in {"clear_target", "probable_target"} and target_surface == "uncertain":
        box_quality = "uncertain"
    if (
        not edge_checks_complete
        or background_excess is None
        or target_area_fraction == "uncertain"
        or (inside_status in {"clear_target", "probable_target"} and target_area_fraction == "none")
    ):
        outside_relation = "uncertain"
        box_quality = "uncertain"
    return {
        "inside_target_status": inside_status,
        "outside_relation": outside_relation,
        "box_quality": box_quality,
        "target_surface": target_surface,
        "dominant_confounder": confounder,
        **edge_checks,
        "edge_checks_complete": edge_checks_complete,
        "truncated_edges": truncated_edges,
        "background_excess": background_excess,
        "target_area_fraction": target_area_fraction,
        "geometry_reason": str(raw.get("reason") or "").strip(),
        "geometry_raw": raw,
        "geometry_raw_response_text": text,
    }


def _v5_rework_type(box_quality: str, outside_relation: str, has_candidate: bool) -> str:
    if box_quality == "wrong_target" and has_candidate:
        return "move_bbox"
    if box_quality == "wrong_target":
        return "relabeled_negative_or_other"
    if box_quality == "under_inclusive" or outside_relation == "continuous_main_target":
        return "expand_bbox"
    if box_quality == "over_inclusive":
        return "shrink_bbox"
    if box_quality == "uncertain":
        return "uncertain"
    return "none"


def fuse_training_triage_v5(
    semantic: dict[str, Any],
    geometry: dict[str, Any],
    weak_label_hint: str,
) -> dict[str, Any]:
    weak = weak_label_hint.lower()
    from_negative_bucket = weak.startswith("negative")
    from_positive_bucket = weak.startswith("positive")
    semantic_verdict = semantic.get("semantic_verdict") or "uncertain"
    has_candidate = semantic_verdict in {"clear_positive", "probable_positive"}
    clear_semantic = semantic_verdict == "clear_positive"
    probable_semantic = semantic_verdict == "probable_positive"
    semantic_uncertain = semantic_verdict == "uncertain"
    inside_status = geometry.get("inside_target_status") or "uncertain"
    clear_inside = inside_status == "clear_target"
    probable_inside = inside_status == "probable_target"
    box_quality = geometry.get("box_quality") or "uncertain"
    outside_relation = geometry.get("outside_relation") or "uncertain"
    target_surface = geometry.get("target_surface") or "uncertain"
    geometry_uncertain = (
        inside_status == "uncertain"
        or box_quality == "uncertain"
        or outside_relation == "uncertain"
    )
    good_surface = target_surface in {"floor", "floor_equipment_junction"}
    clean_bbox = (
        clear_inside
        and box_quality == "good"
        and outside_relation in {"none", "minor_same_target_fringe"}
        and good_surface
    )
    rework_type = _v5_rework_type(box_quality, outside_relation, has_candidate)
    if geometry.get("background_excess") is True and geometry.get("truncated_edges"):
        rework_type = "move_bbox"
    stage_disagreement = (
        semantic_verdict == "hard_negative"
        and inside_status in {"clear_target", "probable_target"}
        and good_surface
    )

    if semantic_uncertain or geometry_uncertain or stage_disagreement:
        action = "manual_review"
    elif from_negative_bucket and has_candidate:
        action = "label_conflict_review"
    elif from_negative_bucket:
        action = "reject_from_positive_training"
    elif from_positive_bucket and clear_semantic and clean_bbox:
        action = "auto_accept_positive"
    elif from_positive_bucket and has_candidate and box_quality in {
        "under_inclusive",
        "over_inclusive",
        "wrong_target",
    }:
        action = "rework_bbox_positive_candidate"
    elif from_positive_bucket and has_candidate and (probable_semantic or probable_inside):
        action = "manual_review"
    elif from_positive_bucket and has_candidate:
        action = "manual_review"
    elif from_positive_bucket:
        action = "reject_from_positive_training"
    elif has_candidate:
        action = "label_conflict_review"
    else:
        action = "reject_from_positive_training"

    if action == "auto_accept_positive":
        rework_type = "none"
    elif action == "manual_review" and rework_type == "none":
        rework_type = "uncertain"

    label_conflict = (from_negative_bucket and has_candidate) or (
        from_positive_bucket and semantic_verdict == "hard_negative"
    )
    weak_label_consistent = (from_positive_bucket and has_candidate) or (
        from_negative_bucket and semantic_verdict == "hard_negative"
    )
    issue_flags: list[str] = []
    if probable_semantic:
        issue_flags.append("semantic_probable_positive")
    if semantic_uncertain:
        issue_flags.append("semantic_uncertain")
    if geometry_uncertain:
        issue_flags.append("bbox_geometry_uncertain")
    if stage_disagreement:
        issue_flags.append("semantic_bbox_stage_disagreement")
    if geometry.get("truncated_edges"):
        issue_flags.append("target_crosses_bbox_edges")
    if label_conflict:
        issue_flags.append("weak_label_conflict")
    if box_quality not in {"good", "uncertain"}:
        issue_flags.append(f"bbox_{box_quality}")
    confounder = geometry.get("dominant_confounder")
    if confounder not in {None, "", "none", "uncertain"}:
        issue_flags.append(f"confounder_{confounder}")
    hard_negative_type = semantic.get("hard_negative_type") or "uncertain"
    if semantic_verdict == "hard_negative" and hard_negative_type not in {"none", "uncertain"}:
        issue_flags.append(f"hard_negative_{hard_negative_type}")

    if clear_inside or probable_inside:
        candidate_relation = "contained" if box_quality == "good" else "overlapping"
    elif has_candidate:
        candidate_relation = "outside_bbox"
    else:
        candidate_relation = "none"

    semantic_reason = semantic.get("semantic_reason") or ""
    geometry_reason = geometry.get("geometry_reason") or ""
    reason_parts = [part for part in (f"semantic: {semantic_reason}" if semantic_reason else "", f"bbox: {geometry_reason}" if geometry_reason else "") if part]
    return {
        "visible_target": clear_inside or probable_inside,
        "inside_has_target_anomaly": clear_inside or probable_inside,
        "outside_has_same_anomaly": outside_relation in {"continuous_main_target", "minor_same_target_fringe"},
        "outside_extension": outside_relation == "continuous_main_target",
        "box_quality": box_quality,
        "containment_quality": (
            "tight_contained"
            if box_quality == "good" and outside_relation == "none"
            else "loose_but_ok"
            if box_quality == "good"
            else box_quality
        ),
        "label_match": weak_label_consistent,
        "target_location": target_surface,
        "image_has_positive_candidate": has_candidate,
        "visual_positive_class": has_candidate,
        "visual_class": "floor_clear_water" if has_candidate else hard_negative_type,
        "positive_candidate_location": semantic.get("candidate_location"),
        "semantic_candidate_surface": semantic.get("candidate_surface"),
        "positive_candidate_relation_to_bbox": candidate_relation,
        "hard_negative_type": hard_negative_type,
        "bbox_usable_for_training": action == "auto_accept_positive",
        "weak_label_consistent": weak_label_consistent,
        "label_conflict": label_conflict,
        "stage_disagreement": stage_disagreement,
        "training_action": action,
        "training_priority": (
            "high"
            if action in {"auto_accept_positive", "rework_bbox_positive_candidate"}
            else "medium"
            if action in {"manual_review", "label_conflict_review"}
            else "reject"
        ),
        "rework_type": rework_type,
        "needs_human_review": action in {
            "rework_bbox_positive_candidate",
            "manual_review",
            "label_conflict_review",
        },
        "issue_flags": issue_flags,
        "outside_evidence_summary": geometry_reason,
        "reason": " | ".join(reason_parts),
        **semantic,
        **geometry,
    }


def apply_clean_verifier_v5(
    primary: dict[str, Any],
    verifier: dict[str, Any] | None,
    *,
    verifier_model: str,
    verifier_error: str | None = None,
) -> dict[str, Any]:
    result = dict(primary)
    result["clean_verifier_model"] = verifier_model
    result["clean_verifier_error"] = verifier_error
    result["primary_box_quality"] = primary.get("box_quality")
    if verifier is None:
        result.update(
            {
                "clean_verifier_passed": False,
                "clean_verifier_box_quality": "uncertain",
                "clean_verifier_rework_type": "uncertain",
                "clean_verifier_reason": "Verifier unavailable; clean candidate downgraded to manual review.",
                "training_action": "manual_review",
                "training_priority": "medium",
                "rework_type": "uncertain",
                "bbox_usable_for_training": False,
                "needs_human_review": True,
                "issue_flags": list(primary.get("issue_flags") or []) + ["clean_verifier_unavailable"],
            }
        )
        return result

    clean = (
        verifier.get("inside_target_status") == "clear_target"
        and verifier.get("box_quality") == "good"
        and verifier.get("outside_relation") in {"none", "minor_same_target_fringe"}
        and verifier.get("target_surface") in {"floor", "floor_equipment_junction"}
        and verifier.get("edge_checks_complete") is True
        and not verifier.get("truncated_edges")
        and verifier.get("background_excess") is False
        and verifier.get("target_area_fraction")
        not in {None, "", "none", "lt_10_percent", "10_to_25_percent", "uncertain"}
    )
    verifier_quality = str(verifier.get("box_quality") or "uncertain")
    rework_type = _v5_rework_type(
        verifier_quality,
        str(verifier.get("outside_relation") or "uncertain"),
        bool(primary.get("image_has_positive_candidate")),
    )
    if verifier.get("background_excess") is True and verifier.get("truncated_edges"):
        rework_type = "move_bbox"
    result.update(
        {
            "clean_verifier_passed": clean,
            "clean_verifier_box_quality": verifier_quality,
            "clean_verifier_rework_type": "none" if clean else rework_type,
            "clean_verifier_reason": verifier.get("geometry_reason"),
            "clean_verifier_raw_response_text": verifier.get("geometry_raw_response_text"),
        }
    )
    if clean:
        return result

    result["training_action"] = (
        "rework_bbox_positive_candidate"
        if verifier_quality in {"under_inclusive", "over_inclusive", "wrong_target"}
        and primary.get("image_has_positive_candidate") is True
        else "manual_review"
    )
    result["training_priority"] = "high" if result["training_action"] == "rework_bbox_positive_candidate" else "medium"
    result["rework_type"] = rework_type if result["training_action"] == "rework_bbox_positive_candidate" else "uncertain"
    result["bbox_usable_for_training"] = False
    result["needs_human_review"] = True
    result["issue_flags"] = list(primary.get("issue_flags") or []) + ["clean_verifier_veto"]
    result["reason"] = " | ".join(
        part
        for part in (
            str(primary.get("reason") or ""),
            f"clean verifier: {verifier.get('geometry_reason')}" if verifier.get("geometry_reason") else "",
        )
        if part
    )
    for key in (
        "inside_target_status",
        "outside_relation",
        "box_quality",
        "target_surface",
        "dominant_confounder",
        "edge_checks_complete",
        "truncated_edges",
        "background_excess",
        "target_area_fraction",
    ):
        result[key] = verifier.get(key)
    result["geometry_reason"] = result.get("reason")
    return result


def semantic_prompt_v5() -> str:
    return """你是工业摄像头训练集的独立语义质检员。你只会看到一张无标注、无红框的原始整图。

目标：判断整张图任何位置是否存在可用于后续重画 bbox 的“地面清水/透明水膜/湿痕/积水”候选。不要猜测目录标签，也不要假设画面中央或最显眼物体就是目标。

正例范围：
- 水或近透明液体位于可行走、与房间/走道连续的地面，或地面与设备底座交界，或从设备底部流到地面。
- 支持证据包括：不规则且闭合/半闭合的湿干边界、局部湿润变暗并与周围干地面形成连续形状、水滴/流痕/积液、带边界或纹理变化的液体反光。
- 单独出现一块亮斑不够；镜面反光只有同时伴随液体边界、湿干纹理变化或水滴/流痕时才是正证据。

非目标或不足以判正：
- 均匀发亮的环氧地坪、与灯具排列一致的规则高光、阴影、透视线、地砖缝、文字和时间戳。
- 仅在竖直墙面/设备表面的水迹，设备金属反光，彩色液体或固体污渍，人物和杂物。
- 水平放置不等于地面：机柜顶、托盘、桥架、设备平台、金属盖板和桌面都属于 equipment_surface；即使上面真的有液体，也不是本任务的地面清水正例。

固定检查顺序：
1. 先判断图像是否足以复核：地面是否可见、是否严重模糊/过曝/欠曝/遮挡。
2. 按左上到右下扫描整张图的地面区域，不得在发现红字、设备或第一处可疑反光后停止。
3. 对每个可疑区域，与邻近干地面比较边界、纹理和反光形态；再尝试用阴影、普通反光、文字或墙面水解释它。
4. 只有视觉证据强且替代解释明显更弱时，输出 clear_positive。
5. 有局部正证据但仍可能是反光/阴影时，输出 probable_positive；整图可复核且没有候选时才输出 hard_negative；无法稳定判断时输出 uncertain。

candidate_location 使用九宫格位置；多处候选填 multiple，无候选填 none。
candidate_surface 只能是 floor、floor_equipment_junction、wall、equipment_surface、other、none、uncertain。必须先确认候选是否位于可行走地面，不能因为表面水平就填 floor。
positive_evidence 只从这些值中选择：irregular_wet_dry_boundary、pooled_liquid、wet_texture_change、droplet_or_flow_trail、reflection_with_liquid_boundary、floor_equipment_junction。没有则 []。
hard_negative_type 只从这些值中选择：none、dry_floor、shadow、uniform_specular_reflection、text_or_overlay、wall_only_water、equipment_surface、colored_liquid_or_stain、person_or_clutter、floor_not_visible、low_image_quality、other、uncertain。

只输出合法 JSON object，不要 markdown，不要额外解释：
{
  "semantic_verdict": "clear_positive",
  "candidate_location": "bottom_right",
  "candidate_surface": "floor",
  "positive_evidence": ["irregular_wet_dry_boundary", "reflection_with_liquid_boundary"],
  "hard_negative_type": "none",
  "reason": "一句话描述实际看见的证据和已排除的主要混淆项"
}
"""


def geometry_prompt_v5() -> str:
    return """你是工业视觉 bbox 几何质检员。你会依次看到六张图：
- 第 1 张：B，当前 bbox 内部的原始像素裁剪；整张 B 都属于框内。
- 第 2 张：C，bbox 周边上下文；红框表示当前 bbox，红框外只用于判断同一目标是否被截断。
- 第 3 张：LEFT EDGE，bbox 左边界放大条带；红色竖线是左边界。
- 第 4 张：TOP EDGE，bbox 上边界放大条带；红色横线是上边界。
- 第 5 张：RIGHT EDGE，bbox 右边界放大条带；红色竖线是右边界。
- 第 6 张：BOTTOM EDGE，bbox 下边界放大条带；红色横线是下边界。

你不知道目录标签，也不知道整图其他位置是否有目标。只判断当前 bbox 的内容和几何质量，不输出训练动作。

目标是位于可行走、与房间/走道连续的地面或地面-设备交界处的清水、透明/近透明水膜、湿痕或积水。机柜顶、托盘、桥架、设备平台、金属盖板和桌面即使水平、即使有真实液体，也必须判为 equipment_surface 和 wrong_target。仅有均匀地坪高光、灯光反射、阴影、文字、设备表面、墙面-only 水迹、彩色污渍时，不算框内目标。

固定判断顺序：
1. inside_target_status：先判断 B 内是否有目标。clear_target=有明确液体证据；probable_target=有局部证据但可能是反光/阴影；no_target=可复核但没有目标；uncertain=画质不足。
2. target_surface：目标主体所在表面。只有 floor 或 floor_equipment_junction 可作为本任务目标。
3. outside_relation：只比较 C 红框边缘。
   - continuous_main_target：同一片主要水体/核心水渍被红框明显截断。
   - minor_same_target_fringe：只有少量弱边缘、轻微亮面或不影响学习的尾部在框外；这种情况允许 good。
   - separate_target：框外另有不连续目标，不代表当前框需要扩张。
   - none：无同一目标延伸。
4. 必须依次查看第 3-6 张边界放大图。对每条红线分别判断“同一片主要目标是否从线的一侧穿过红线并继续到另一侧”，输出四个布尔字段。不得只看 C 图的总体印象。只要任一边为 true，outside_relation 必须是 continuous_main_target，box_quality 必须是 under_inclusive。远处不连续的另一滩水不能把边字段设为 true。
5. target_area_fraction：先在 B 图中只把目标水体/湿痕当作前景，估算它占整个 bbox 的面积档位：none、lt_10_percent、10_to_25_percent、25_to_50_percent、50_to_75_percent、gt_75_percent、uncertain。不要把桌椅、箱子、设备或普通地面算进目标面积。
6. dominant_confounder：即使框内有清晰目标，也要填写占比最大的无关内容；只有无关内容极少时才能填 none。桌椅/箱子/杂物用 person_or_clutter，设备用 equipment_surface，大块正常地面用 dry_floor。
7. background_excess：判断当前 bbox 中是否包含明显过多的正常地面、设备、桌椅、人物或其他背景，以至于训练会学习非目标特征。target_area_fraction 小于 25% 时必须为 true；25% 到 50% 且存在明显无关物体时也必须为 true。不要把包围不规则目标所需的少量边缘留白判为 true。
8. box_quality：
   - good：B 内有 clear_target，主体位于目标表面，主要清晰目标已覆盖，背景不过多；outside_relation 可为 none 或 minor_same_target_fringe。
   - under_inclusive：框内有目标，但 continuous_main_target，或核心水体明显被截断。
   - over_inclusive：框内有明确目标，但正常地面/设备/背景明显过多，训练会学脏。
   - wrong_target：框内无目标，或主体是阴影、普通反光、文字、设备、墙面-only 等。即使 C 框外另有目标，当前框仍是 wrong_target。
   - uncertain：inside_target_status=probable_target/uncertain，或证据不足以稳定区分。

dominant_confounder 只从这些值中选择：none、shadow、uniform_specular_reflection、text_or_overlay、wall_only_water、equipment_surface、colored_liquid_or_stain、person_or_clutter、dry_floor、low_image_quality、other、uncertain。

只输出合法 JSON object，不要 markdown，不要额外解释：
{
  "inside_target_status": "clear_target",
  "target_crosses_left_edge": false,
  "target_crosses_top_edge": false,
  "target_crosses_right_edge": false,
  "target_crosses_bottom_edge": false,
  "target_area_fraction": "50_to_75_percent",
  "background_excess": false,
  "outside_relation": "none",
  "box_quality": "good",
  "target_surface": "floor",
  "dominant_confounder": "none",
  "reason": "一句话说明框内目标、框边界和主要混淆项"
}
"""


def clean_verifier_prompt_v5() -> str:
    return """你是 bbox clean seed 的独立反方复核员。主模型已经声称这个框可以直接训练，但你不能沿用该结论；你的任务是主动寻找否决证据。

你会依次看到七张图：
1. FULL OVERLAY：完整原图和红框。
2. B INSIDE：框内原始像素。
3. C CONTEXT：框周边上下文和红框。
4-7. LEFT、TOP、RIGHT、BOTTOM EDGE：四条边界放大图，红线是对应 bbox 边界。

只有同时满足以下条件才可输出 good：
- B 中明确是真实地面清水/透明水膜/湿痕，而不是普通环氧地坪高光、灯光倒影、阴影、文字、设备面或墙面水迹。
- FULL OVERLAY 和 C 中同一片主要水体没有越出红框；必须检查红框四边，不得只看框内最亮部分。
- 四张 EDGE 图中同一目标均未穿过红线。任一边穿线必须输出对应 true、continuous_main_target、under_inclusive。
- 框内目标面积足够，设备、桌椅、人物和正常地面不过多。

否决优先级：
1. 同一主要目标延伸到框外 -> under_inclusive。
2. 框内有目标但背景过多 -> over_inclusive。
3. 框内不是目标，哪怕框外有水 -> wrong_target。
4. 目标与普通反光无法稳定区分，或边界连续性看不清 -> uncertain；禁止勉强输出 good。

机柜顶、托盘、设备平台、金属盖板和桌面属于 equipment_surface，不是地面目标。minor_same_target_fringe 只允许极少、微弱且不影响学习的尾部；明显水带、核心水体或大面积湿亮区域在框外时不能使用该值。

只输出合法 JSON object，不要 markdown，不要额外解释：
{
  "inside_target_status": "clear_target",
  "target_crosses_left_edge": false,
  "target_crosses_top_edge": false,
  "target_crosses_right_edge": false,
  "target_crosses_bottom_edge": false,
  "target_area_fraction": "50_to_75_percent",
  "background_excess": false,
  "outside_relation": "none",
  "box_quality": "good",
  "target_surface": "floor",
  "dominant_confounder": "none",
  "reason": "一句话明确说明是否找到 clean seed 否决证据"
}
"""


def prompt_for(sample: dict[str, Any], obj: dict[str, Any], *, review_mode: str = "overlay") -> str:
    labels = obj.get("classification", {}).get("multi_labels") or []
    label_text = ", ".join(
        f"{item.get('label_key')}={item.get('label_value')}"
        for item in labels
        if isinstance(item, dict) and item.get("label_key")
    ) or "no classification labels"
    scene = sample.get("image_asset", {}).get("scene_context") or {}
    weak_label_hint = sample.get("_qc_weak_label_hint") or "unknown"
    if review_mode == "training_triage":
        return f"""你是工业视觉训练集质检员。输入图是一张三联证据图：
A = 原始整图，红框是待复核 bbox；
B = bbox 内部原始像素裁剪图，只代表“框内”；
C = bbox 周边放大上下文，红框内是 bbox，红框外是邻近区域，用来判断水渍/污渍是否延伸到框外。

任务目标：
这不是最终评测集打分，而是训练集 scale 前的自动筛选。目标有两层：
1. 产出可以直接训练的高纯度 bbox 正例种子集。
2. 尽量保住可修正的 TP 资产：如果图片里确实有地面清水/水渍正例，但当前 bbox 框错、框偏、框小或框大，不要直接丢弃；应进入 rework_bbox_positive_candidate，后续重画 bbox 后再训练。

必须避免把错框、错类、阴影、文字、设备反光、墙面非目标异常直接放进正例训练。

弱标签信息来自目录或 metadata，只能作为参考，不能盲信：
- sample_id: {sample.get("sample_id")}
- object_id: {obj.get("object_id")}
- object_type: {obj.get("object_type")}
- weak_label_hint: {weak_label_hint}
- labels: {label_text}
- inspection_content: {scene.get("inspection_content")}
- task_group: {scene.get("task_group")}

视觉正例定义：
目标正例是 clear_water / floor water leak。你必须同时判断两个层级：
- bbox-level：当前红框是否已经可直接作为训练 bbox。
- image-level：即使红框错了，A 图或 C 图里是否存在可返工重框的地面清水/水渍候选。

具体定义：
- 红框内必须有真实可见的地面清水、透明/近透明水膜、湿痕、积液、局部水反光或不规则水渍边界。
- 目标最好在 floor / floor-equipment junction / equipment base on floor；只有墙面渗水、竖直墙面水迹、文字、设备面板、阴影、普通反光、干地面、彩色液体不能自动当成正例训练。
- 如果图中确实有地面水，但 bbox 没兜完整或完全框偏，不能直接入训练；应进入 rework_bbox_positive_candidate，并用 rework_type 标出 expand_bbox / shrink_bbox / move_bbox。
- 不要求 bbox 框住整片地面的所有轻微反光、泛湿亮面或远处弱水迹；只要红框包含主要、清晰、可学习的核心水渍/水膜区域，且框外没有被截断的主要连续水体，可以视为 good 或 loose_but_ok。

bbox 质量口径：
1. inside_has_target_anomaly：B 图或 C 图红框内是否有目标水渍/湿痕/液膜。
2. outside_extension：C 图红框外是否有同一片目标水渍连续延伸。
3. image_has_positive_candidate：A 图或 C 图里是否存在可重框的地面清水/水渍候选；它可以在红框外。
4. box_quality 只能是 good、over_inclusive、under_inclusive、wrong_target、uncertain：
   - good：框内有目标，主要清晰目标基本都在框内，外部无明显主要连续延伸，背景不过多。
   - under_inclusive：框内有目标，但主要连续水体/水渍核心明显被截断，或者主目标大部分在框外。不要因为普通地面反光、弱湿亮面或远处无关水迹就判 under_inclusive。
   - over_inclusive：框内有目标，但正常区域/设备/背景过多，训练会学脏。
   - wrong_target：框内没有目标水渍，或框住文字、阴影、设备、普通反光、墙面非目标等。注意：wrong_target 仍可能是 image-level 正例候选，只要 A/C 中另有地面水可重框。
   - uncertain：画质或证据不足，无法稳定判断。

训练动作规则：
- auto_accept_positive：给干净地面清水训练正例；要求 inside_has_target_anomaly=true、bbox_usable_for_training=true、visual_positive_class=true、target_location=floor 或 floor_equipment_junction、box_quality=good、outside_extension=false、label_match=true。允许 containment_quality=loose_but_ok。
- rework_bbox_positive_candidate：视觉上是正例或 image-level 正例候选，但 bbox 需要扩/缩/移动。典型情况：
  a. under_inclusive -> rework_type=expand_bbox；
  b. over_inclusive -> rework_type=shrink_bbox；
  c. wrong_target 但 A/C 里有明显地面水候选 -> rework_type=move_bbox；
  这类是 TP 资产候选，必须保留，但 bbox_usable_for_training=false，不能直接入训练。
- label_conflict_review：weak_label_hint 是 wall_water/others/negative，但视觉上像干净地面水正例；或 weak_label_hint 是 clear_water/floor_clear_water 但视觉明显不一致。不要自动入训练，交人工复核。
- reject_from_positive_training：图中没有可用地面清水候选，或只有非目标：墙面-only、文字/阴影/设备/普通反光、彩色液体、干地面、人员/杂物等。
- manual_review：证据弱、画质差、无法稳定判断。

硬性约束：
- 如果 weak_label_hint 以 negative 开头，即使视觉上是地面水且 bbox 很好，也绝不能输出 training_action=auto_accept_positive 或 rework_bbox_positive_candidate；必须输出 label_conflict_review。
- 只有 weak_label_hint 是 positive_floor_clear_water 且视觉和 bbox 都干净时，才允许 auto_accept_positive。
- 如果 weak_label_hint 是 positive_floor_clear_water，当前 bbox 框偏但 A/C 中存在地面水候选，应输出 rework_bbox_positive_candidate，不要输出 reject_from_positive_training。

输出要求：
只输出合法 JSON object，不要 markdown，不要解释文字。字段必须完整。

输出格式：
{{
  "inside_has_target_anomaly": true,
  "outside_has_same_anomaly": false,
  "outside_extension": false,
  "visible_target": true,
  "box_quality": "good",
  "containment_quality": "tight_contained",
  "label_match": true,
  "target_location": "floor",
  "image_has_positive_candidate": true,
  "visual_positive_class": true,
  "visual_class": "floor_clear_water",
  "positive_candidate_location": "inside_bbox",
  "positive_candidate_relation_to_bbox": "contained",
  "hard_negative_type": "none",
  "bbox_usable_for_training": true,
  "weak_label_consistent": true,
  "label_conflict": false,
  "training_action": "auto_accept_positive",
  "training_priority": "high",
  "rework_type": "none",
  "needs_human_review": false,
  "issue_flags": [],
  "outside_evidence_summary": "",
  "reason": ""
}}
"""

    if review_mode == "crop_evidence":
        return f"""你是工业异常检测标注质检员。输入图是一张三联证据图：
A = 原始整图，红框是待复核 bbox；
B = bbox 内部原始像素裁剪图，只代表“框内”；
C = bbox 周边放大上下文，红框内是 bbox，红框外是邻近区域，用来判断水渍/污渍是否延伸到框外。

只质检这个异常 bbox 是否兜住目标异常。目标异常必须与 labels / inspection_content 对应，例如漏水、清水水膜、湿痕、积液、油渍、冷却液或对应液体痕迹。阀门、螺栓、管道、接头、设备边缘、地面空白、阴影、正常反光、文字或网格线不能算目标异常。

样本信息：
- sample_id: {sample.get("sample_id")}
- object_id: {obj.get("object_id")}
- object_type: {obj.get("object_type")}
- labels: {label_text}
- inspection_content: {scene.get("inspection_content")}
- task_group: {scene.get("task_group")}

判定口径：
1. inside_has_target_anomaly 表示 B 图或 C 图红框内是否有目标异常痕迹；只看到正常设备/地面时为 false。
2. outside_has_same_anomaly 表示 C 图红框外是否能看到同一种异常痕迹。
3. outside_extension 只在红框外异常与红框内异常连续、明显属于同一片水渍/污渍/液膜时为 true；远处无关异常不要算。
4. label_match 表示红框内目标是否与 labels / inspection_content 一致。
5. containment_quality 从 tight_contained、loose_but_ok、over_inclusive、under_inclusive、wrong_target、uncertain 中选一个。
6. box_quality 只能从 good、over_inclusive、under_inclusive、wrong_target、uncertain 中选一个，并与 containment_quality 对齐：
   - good：框内有目标异常，主要异常基本在框内，外部无明显连续延伸。
   - over_inclusive：框内有目标异常，但背景、设备或正常区域明显过多。
   - under_inclusive：框内有目标异常，但同一片异常明显延伸到框外，或主异常只被框住一小部分。
   - wrong_target：框内没有目标异常，或框住的是正常设备/阴影/普通反光/空白。
   - uncertain：画质差、异常太弱或裁剪证据不足。
7. wrong_target、under_inclusive、over_inclusive、uncertain 都 needs_human_review=true。
8. 只输出合法 JSON object，不要 markdown，不要解释文字。

输出格式：
{{
  "inside_has_target_anomaly": true,
  "outside_has_same_anomaly": false,
  "outside_extension": false,
  "visible_target": true,
  "box_quality": "good",
  "containment_quality": "tight_contained",
  "label_match": true,
  "needs_human_review": false,
  "issue_flags": [],
  "target_location": "floor",
  "outside_evidence_summary": "",
  "reason": ""
}}
"""

    return f"""你是工业异常检测标注质检员。图片上的红色矩形是待复核的异常标注框。

只质检“异常区域框”本身。红框必须框住图中真实可见的异常痕迹，例如漏油、漏水、湿痕、液膜、积液、污渍、反光水迹、异常颜色或异常形态。阀门、螺栓、管道、接头、设备边缘、地面空白、阴影和正常反光都不能当成异常目标。

样本信息：
- sample_id: {sample.get("sample_id")}
- object_id: {obj.get("object_id")}
- object_type: {obj.get("object_type")}
- labels: {label_text}
- inspection_content: {scene.get("inspection_content")}
- task_group: {scene.get("task_group")}

判定要求：
1. visible_target 只表示红框内是否能看到与 labels / inspection_content 对应的可见异常痕迹；只看到设备部件时必须为 false。
2. label_match 表示红框内的异常痕迹是否和标签一致。比如 diesel_leak 应该有柴油/油渍/湿润油膜特征，coolant_leak 应该有冷却液或对应液体痕迹。
3. box_quality 只能从 good、over_inclusive、under_inclusive、wrong_target、uncertain 中选一个。
4. good：红框基本覆盖主要异常区域，允许少量边缘留白。
5. over_inclusive：红框包含异常，但背景、设备或正常区域明显过多。
6. under_inclusive：红框只框住异常的一小部分，或者图中主异常明显延伸到红框外。
7. wrong_target：红框框到了设备、螺栓、管道、阴影、文字、正常反光、空白区域，或框内没有可见异常。
8. uncertain：画质差、异常太弱、遮挡严重，无法稳定判断。
9. 如果图中有明显异常但红框没有覆盖主要异常，按 under_inclusive 或 wrong_target 处理，并设置 needs_human_review=true。
10. wrong_target 直接 needs_human_review=true；under_inclusive、over_inclusive、uncertain 也需要复核。
11. 只输出合法 JSON object，不要 markdown，不要解释文字。

输出格式：
{{
  "visible_target": true,
  "box_quality": "good",
  "label_match": true,
  "needs_human_review": false,
  "issue_flags": [],
  "reason": ""
}}
"""


def object_context(obj: dict[str, Any]) -> dict[str, Any]:
    detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
    params = detail.get("generation_params") if isinstance(detail.get("generation_params"), dict) else {}
    return {
        "box": obj.get("box"),
        "selected_grid": params.get("selected_grid"),
        "grid_bbox": params.get("grid_bbox"),
        "expanded_edit_bbox": params.get("expanded_edit_bbox"),
        "final_bbox_source": params.get("final_bbox_source"),
        "localization_pipeline": params.get("localization_pipeline"),
    }


def weak_label_hint_from_path(metadata_path: Path) -> str:
    text = str(metadata_path).lower()
    name = metadata_path.name.lower()
    if "images_floor_clear_water" in text or "images_clear_water" in text or name.startswith(("pos_good__", "pos_under__", "pos_wrong__")):
        return "positive_floor_clear_water"
    if "images_wall_water" in text or name.startswith(("neg_under__", "neg_wrong__")):
        return "negative_wall_water_or_hard_negative"
    if "images_others" in text or name.startswith("neg_floor_conflict_good__"):
        return "negative_other_or_label_conflict_candidate"
    return "unknown"


def weak_label_hint_from_sample(metadata_path: Path, sample: dict[str, Any]) -> str:
    path_hint = weak_label_hint_from_path(metadata_path)
    if path_hint != "unknown":
        return path_hint

    scene = (sample.get("image_asset") or {}).get("scene_context") or {}
    label_values = {
        str(item.get("label_value") or "").strip().lower()
        for obj in sample.get("objects") or []
        if isinstance(obj, dict)
        for item in (obj.get("classification") or {}).get("multi_labels") or []
        if isinstance(item, dict)
    }
    inspection_content = str(scene.get("inspection_content") or "").strip().lower()
    positive_values = {
        "clear_water",
        "floor_clear_water",
        "floor_water_leak",
        "water_leak",
    }
    if inspection_content in positive_values or label_values.intersection(positive_values):
        return "positive_floor_clear_water"
    return "unknown"


def resolve_image_for_qc(
    image_uri: str | None,
    metadata_path: Path,
    asset_base_dirs: list[Path],
) -> tuple[Path | None, bool, list[str]]:
    image_path, image_exists, candidates = resolve_local_uri(image_uri, metadata_path, asset_base_dirs)
    if image_exists or not image_uri:
        return image_path, image_exists, candidates

    basename = PureWindowsPath(str(image_uri)).name if "\\" in str(image_uri) else Path(str(image_uri)).name
    fallback_candidates: list[str] = []
    if not basename:
        return image_path, image_exists, candidates
    for root in asset_base_dirs:
        if root.is_file():
            continue
        direct = root / basename
        fallback_candidates.append(str(direct))
        if direct.exists():
            return direct, True, candidates + fallback_candidates
        if root.exists():
            for match in sorted(root.rglob(basename)):
                fallback_candidates.append(str(match))
                if match.is_file():
                    return match, True, candidates + fallback_candidates
    return image_path, image_exists, candidates + fallback_candidates


def rule_flags(obj: dict[str, Any], *, min_grid_iou: float, tolerance_px: int) -> dict[str, Any]:
    ctx = object_context(obj)
    flags: list[str] = []
    evidence: dict[str, Any] = {}
    box = ctx["box"]
    grid_bbox = ctx["grid_bbox"]
    expanded_edit_bbox = ctx["expanded_edit_bbox"]

    if box_numbers(box) is None:
        flags.append("invalid_box")
    if expanded_edit_bbox is not None and not box_contains(expanded_edit_bbox, box, tolerance=tolerance_px):
        flags.append("box_outside_expanded_edit_region")
        evidence["expanded_edit_bbox"] = expanded_edit_bbox
    if grid_bbox is not None:
        grid_iou = box_iou(box, grid_bbox)
        evidence["grid_iou"] = round(grid_iou, 4)
        if grid_iou < min_grid_iou:
            flags.append("box_disconnected_from_selected_grid")
            evidence["grid_bbox"] = grid_bbox
            evidence["selected_grid"] = ctx["selected_grid"]
    return {"rule_flags": flags, "rule_evidence": evidence}


def clip_box_values(values: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = values
    clipped = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    cx1, cy1, cx2, cy2 = clipped
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return clipped


def expand_box_values(
    values: tuple[int, int, int, int],
    width: int,
    height: int,
    *,
    ratio: float,
    min_pad: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = values
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(int(round(box_w * ratio)), min_pad)
    pad_y = max(int(round(box_h * ratio)), min_pad)
    return clip_box_values((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _fit_panel(image: Any, target_height: int) -> Any:
    if image.height == target_height:
        return image
    scale = target_height / float(image.height)
    width = max(1, round(image.width * scale))
    return image.resize((width, target_height))


def _add_panel_label(image: Any, label: str) -> Any:
    from PIL import Image, ImageDraw

    bar_h = 32
    output = Image.new("RGB", (image.width, image.height + bar_h), (246, 246, 246))
    output.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, image.width - 1, bar_h - 1), fill=(30, 30, 30))
    draw.text((10, 10), label, fill=(255, 255, 255))
    return output


def _encode_jpeg_data_url(image: Any, *, max_side: int | None, jpeg_quality: int) -> str:
    if max_side is not None and max_side > 0 and max(image.size) > max_side:
        from PIL import Image

        scale = max_side / float(max(image.size))
        resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(resized, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_clean_image_data_url(
    image_path: Path,
    *,
    max_side: int | None,
    jpeg_quality: int = 92,
) -> str:
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return _encode_jpeg_data_url(image, max_side=max_side, jpeg_quality=jpeg_quality)


def build_geometry_evidence_urls(
    image_path: Path,
    box: Any,
    *,
    context_expand_ratio: float,
    context_min_pad: int,
    max_side: int | None,
    jpeg_quality: int = 92,
) -> list[str]:
    from PIL import Image, ImageDraw

    values = box_numbers(box)
    if values is None:
        raise ValueError(f"invalid box: {box}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    clipped = clip_box_values(values, image.width, image.height)
    if clipped is None:
        raise ValueError(f"box outside image or empty after clipping: {box}")
    context_box = expand_box_values(
        clipped,
        image.width,
        image.height,
        ratio=context_expand_ratio,
        min_pad=context_min_pad,
    ) or clipped
    x1, y1, x2, y2 = clipped
    cx1, cy1, cx2, cy2 = context_box

    inside = image.crop((x1, y1, x2, y2))
    context = image.crop((cx1, cy1, cx2, cy2))
    draw = ImageDraw.Draw(context)
    line_width = max(3, round(min(context.size) / 140))
    draw.rectangle(
        (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
        outline=(255, 0, 0),
        width=line_width,
    )

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = max(context_min_pad, round(box_w * 0.35))
    pad_y = max(context_min_pad, round(box_h * 0.35))

    def edge_view(edge: str) -> Any:
        if edge == "left":
            raw_box = (x1 - pad_x, y1 - pad_y, x1 + pad_x, y2 + pad_y)
        elif edge == "top":
            raw_box = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y1 + pad_y)
        elif edge == "right":
            raw_box = (x2 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
        else:
            raw_box = (x1 - pad_x, y2 - pad_y, x2 + pad_x, y2 + pad_y)
        clipped_edge = clip_box_values(raw_box, image.width, image.height) or clipped
        ex1, ey1, ex2, ey2 = clipped_edge
        view = image.crop((ex1, ey1, ex2, ey2))
        view_draw = ImageDraw.Draw(view)
        edge_line_width = max(4, round(min(view.size) / 100))
        if edge in {"left", "right"}:
            boundary_x = (x1 if edge == "left" else x2) - ex1
            view_draw.line((boundary_x, 0, boundary_x, view.height), fill=(255, 0, 0), width=edge_line_width)
        else:
            boundary_y = (y1 if edge == "top" else y2) - ey1
            view_draw.line((0, boundary_y, view.width, boundary_y), fill=(255, 0, 0), width=edge_line_width)
        return view

    views = [
        _add_panel_label(inside, "B INSIDE BBOX"),
        _add_panel_label(context, "C CONTEXT + RED BBOX"),
        _add_panel_label(edge_view("left"), "LEFT EDGE"),
        _add_panel_label(edge_view("top"), "TOP EDGE"),
        _add_panel_label(edge_view("right"), "RIGHT EDGE"),
        _add_panel_label(edge_view("bottom"), "BOTTOM EDGE"),
    ]
    return [
        _encode_jpeg_data_url(view, max_side=max_side, jpeg_quality=jpeg_quality)
        for view in views
    ]


def build_crop_evidence_image(
    image_path: Path,
    box: Any,
    *,
    context_expand_ratio: float = 0.8,
    context_min_pad: int = 48,
    panel_height: int = 460,
) -> tuple[Any, dict[str, Any]]:
    from PIL import Image, ImageDraw

    values = box_numbers(box)
    if values is None:
        raise ValueError(f"invalid box: {box}")

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    width, height = image.size
    clipped = clip_box_values(values, width, height)
    if clipped is None:
        raise ValueError(f"box outside image or empty after clipping: {box}")
    x1, y1, x2, y2 = clipped
    context_box = expand_box_values(
        clipped,
        width,
        height,
        ratio=context_expand_ratio,
        min_pad=context_min_pad,
    )
    if context_box is None:
        context_box = clipped

    full = image.copy()
    full_draw = ImageDraw.Draw(full)
    line_width = max(3, round(min(full.size) / 180))
    for offset in range(line_width):
        full_draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 0, 0))

    inside = image.crop((x1, y1, x2, y2))
    inside_draw = ImageDraw.Draw(inside)
    inside_draw.rectangle((0, 0, inside.width - 1, inside.height - 1), outline=(255, 0, 0), width=max(3, line_width))

    cx1, cy1, cx2, cy2 = context_box
    context = image.crop((cx1, cy1, cx2, cy2))
    context_draw = ImageDraw.Draw(context)
    rx1, ry1, rx2, ry2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    context_draw.rectangle((rx1, ry1, rx2, ry2), outline=(255, 0, 0), width=max(3, line_width))

    panels = [
        _add_panel_label(_fit_panel(full, panel_height), "A full image + bbox"),
        _add_panel_label(_fit_panel(inside, panel_height), "B inside bbox crop"),
        _add_panel_label(_fit_panel(context, panel_height), "C context: outside red box"),
    ]
    gap = 10
    sheet_w = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    sheet_h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap

    return sheet, {
        "bbox_clipped": [x1, y1, x2, y2],
        "context_box": [cx1, cy1, cx2, cy2],
        "context_expand_ratio": context_expand_ratio,
        "context_min_pad": context_min_pad,
        "panel_size": list(sheet.size),
    }


def save_crop_evidence_panel(
    image_path: Path,
    box: Any,
    output_path: Path,
    *,
    context_expand_ratio: float,
    context_min_pad: int,
    request_image_max_side: int | None,
    jpeg_quality: int = 90,
) -> tuple[str, dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet, metadata = build_crop_evidence_image(
        image_path,
        box,
        context_expand_ratio=context_expand_ratio,
        context_min_pad=context_min_pad,
    )
    sheet.save(output_path, quality=92)
    image_url = _encode_jpeg_data_url(sheet, max_side=request_image_max_side, jpeg_quality=jpeg_quality)
    return image_url, metadata


def save_overlay(image_path: Path, box: Any, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    values = box_numbers(box)
    if values is not None:
        x1, y1, x2, y2 = values
        line_width = max(3, round(min(image.size) / 180))
        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=(255, 0, 0))
    image.save(output_path, quality=92)


def pack_record(record: dict[str, Any], pack_dir: Path) -> None:
    status = record["decision"]
    quality = record.get("box_quality") or "unknown"
    if status in {"passed", "evidence_only"}:
        return
    if status == "failed":
        bucket = f"failed_{quality}"
    elif record.get("rule_flags"):
        bucket = "review_rule_suspect"
    else:
        bucket = f"review_{quality}"
    sample_id = record["sample_id"]
    object_id = record["object_id"]
    target_dir = pack_dir / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    overlay_src = Path(record["overlay_path"])
    image_src = Path(record["image_path"])
    shutil.copy2(overlay_src, target_dir / f"{sample_id}_{object_id}_overlay.jpg")
    if image_src.exists():
        shutil.copy2(image_src, target_dir / f"{sample_id}_{object_id}{image_src.suffix.lower()}")


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "object_id": record.get("object_id"),
        "decision": record.get("decision"),
        "box_quality": record.get("box_quality"),
        "visible_target": record.get("visible_target"),
        "label_match": record.get("label_match"),
        "needs_human_review": record.get("needs_human_review"),
        "inside_has_target_anomaly": record.get("inside_has_target_anomaly"),
        "outside_has_same_anomaly": record.get("outside_has_same_anomaly"),
        "outside_extension": record.get("outside_extension"),
        "containment_quality": record.get("containment_quality"),
        "target_location": record.get("target_location"),
        "outside_evidence_summary": record.get("outside_evidence_summary"),
        "image_has_positive_candidate": record.get("image_has_positive_candidate"),
        "visual_positive_class": record.get("visual_positive_class"),
        "visual_class": record.get("visual_class"),
        "positive_candidate_location": record.get("positive_candidate_location"),
        "positive_candidate_relation_to_bbox": record.get("positive_candidate_relation_to_bbox"),
        "hard_negative_type": record.get("hard_negative_type"),
        "bbox_usable_for_training": record.get("bbox_usable_for_training"),
        "weak_label_consistent": record.get("weak_label_consistent"),
        "label_conflict": record.get("label_conflict"),
        "training_action": record.get("training_action"),
        "training_priority": record.get("training_priority"),
        "rework_type": record.get("rework_type"),
        "semantic_verdict": record.get("semantic_verdict"),
        "semantic_candidate_location": record.get("semantic_candidate_location"),
        "semantic_candidate_surface": record.get("semantic_candidate_surface"),
        "semantic_positive_evidence": ";".join(record.get("semantic_positive_evidence") or []),
        "semantic_reason": record.get("semantic_reason"),
        "inside_target_status": record.get("inside_target_status"),
        "outside_relation": record.get("outside_relation"),
        "target_surface": record.get("target_surface"),
        "dominant_confounder": record.get("dominant_confounder"),
        "geometry_reason": record.get("geometry_reason"),
        "truncated_edges": ";".join(record.get("truncated_edges") or []),
        "edge_checks_complete": record.get("edge_checks_complete"),
        "background_excess": record.get("background_excess"),
        "target_area_fraction": record.get("target_area_fraction"),
        "stage_disagreement": record.get("stage_disagreement"),
        "primary_box_quality": record.get("primary_box_quality"),
        "clean_verifier_passed": record.get("clean_verifier_passed"),
        "clean_verifier_box_quality": record.get("clean_verifier_box_quality"),
        "clean_verifier_rework_type": record.get("clean_verifier_rework_type"),
        "clean_verifier_reason": record.get("clean_verifier_reason"),
        "clean_verifier_error": record.get("clean_verifier_error"),
        "issue_flags": ";".join(record.get("issue_flags") or []),
        "rule_flags": ";".join(record.get("rule_flags") or []),
        "reason": record.get("reason"),
        "inspection_content": record.get("inspection_content"),
        "labels": record.get("labels"),
        "weak_label_hint": record.get("weak_label_hint"),
        "selected_grid": record.get("selected_grid"),
        "grid_iou": record.get("rule_evidence", {}).get("grid_iou"),
        "box": json.dumps(record.get("box"), ensure_ascii=False),
        "grid_bbox": json.dumps(record.get("grid_bbox"), ensure_ascii=False),
        "expanded_edit_bbox": json.dumps(record.get("expanded_edit_bbox"), ensure_ascii=False),
        "image_path": record.get("image_path"),
        "metadata_path": record.get("metadata_path"),
        "overlay_path": record.get("overlay_path"),
        "evidence_panel_path": record.get("evidence_panel_path"),
        "review_mode": record.get("review_mode"),
        "triage_prompt_version": record.get("triage_prompt_version"),
        "triage_prompt_revision": record.get("triage_prompt_revision"),
        "semantic_model": record.get("semantic_model"),
        "geometry_model": record.get("geometry_model"),
        "clean_verifier_model": record.get("clean_verifier_model"),
        "error": record.get("error"),
        "latency_seconds": record.get("latency_seconds"),
    }


def load_done(results_path: Path) -> set[str]:
    done: set[str] = set()
    if not results_path.exists():
        return done
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("decision") == "evidence_only":
                continue
            done.add(f"{record.get('sample_id')}::{record.get('object_id')}")
    return done


def call_vlm_json_text(
    client: Any,
    *,
    model: str,
    image_urls: list[str],
    prompt: str,
    max_tokens: int,
    system_message: str,
) -> str:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    ]
    content.append({"type": "text", "text": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def call_vlm(client: Any, *, model: str, image_url: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    text = call_vlm_json_text(
        client,
        model=model,
        image_urls=[image_url],
        prompt=prompt,
        max_tokens=max_tokens,
        system_message="You are a strict JSON API for industrial anomaly bounding-box quality control. Return JSON only.",
    )
    parsed = parse_vlm_qc_payload(text)
    parsed = add_crop_evidence_fields(parsed)
    parsed["raw_response_text"] = text
    return parsed


def call_vlm_json_text_with_retries(
    client_factory: Any,
    *,
    model: str,
    image_urls: list[str],
    prompt: str,
    max_tokens: int,
    retries: int,
    retry_backoff_seconds: float,
    system_message: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return call_vlm_json_text(
                client_factory(),
                model=model,
                image_urls=image_urls,
                prompt=prompt,
                max_tokens=max_tokens,
                system_message=system_message,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(max(0.0, retry_backoff_seconds) * (attempt + 1))
    assert last_error is not None
    raise last_error


def call_vlm_with_retries(
    client_factory: Any,
    *,
    model: str,
    image_url: str,
    prompt: str,
    max_tokens: int,
    retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(0, retries) + 1):
        try:
            return call_vlm(
                client_factory(),
                model=model,
                image_url=image_url,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(max(0.0, retry_backoff_seconds) * (attempt + 1))
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable VLM QC for anomaly bounding boxes.")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--asset-base-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=(
            os.getenv("QWEN_GEOMETRY_MODEL")
            or os.getenv("SJTU_API_DEFAULT_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "qwen"
        ),
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help="Optional image-level semantic model for training_triage v5; defaults to --model.",
    )
    parser.add_argument(
        "--geometry-model",
        default=None,
        help="Optional bbox geometry model for training_triage v5; defaults to --model.",
    )
    parser.add_argument(
        "--clean-verifier-model",
        default=None,
        help="Optional independent model that vetoes provisional auto-accept decisions only.",
    )
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("QWEN_GEOMETRY_API_URL")
            or os.getenv("SJTU_API_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        ),
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.getenv("QWEN397B_API_KEY")
            or os.getenv("SJTU_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--request-image-max-side", type=int, default=1280)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent VLM workers. Use >1 for fast smoke/full runs.")
    parser.add_argument("--api-retries", type=int, default=2, help="Retries for transient API/runtime failures per object.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=5.0, help="Linear retry backoff base seconds.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-grid-iou", type=float, default=0.01)
    parser.add_argument("--region-tolerance-px", type=int, default=2)
    parser.add_argument(
        "--review-mode",
        choices=sorted(REVIEW_MODES),
        default="overlay",
        help="overlay sends a full-image red-box view; crop_evidence sends full/inside/context panels.",
    )
    parser.add_argument(
        "--triage-prompt-version",
        choices=sorted(TRIAGE_PROMPT_VERSIONS),
        default="v5",
        help="v4 uses one three-panel prompt; v5 uses blind full-image semantics plus independent bbox geometry.",
    )
    parser.add_argument(
        "--build-evidence-only",
        action="store_true",
        help="Build overlays/crop evidence panels and reports without calling the VLM API.",
    )
    parser.add_argument(
        "--context-expand-ratio",
        type=float,
        default=0.8,
        help="Expansion ratio around bbox for crop_evidence context panel.",
    )
    parser.add_argument(
        "--context-min-pad",
        type=int,
        default=48,
        help="Minimum pixel padding around bbox for crop_evidence context panel.",
    )
    args = parser.parse_args()
    semantic_model = args.semantic_model or args.model
    geometry_model = args.geometry_model or args.model
    clean_verifier_model = args.clean_verifier_model

    if not args.build_evidence_only and not args.api_key:
        raise SystemExit("Missing API key. Set QWEN397B_API_KEY or SJTU_API_KEY.")
    if not args.build_evidence_only and not args.base_url:
        raise SystemExit("Missing base URL. Set QWEN_GEOMETRY_API_URL or SJTU_API_BASE_URL.")

    OpenAI = None
    if not args.build_evidence_only:
        from openai import OpenAI

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    evidence_dir = output_dir / "evidence_panels"
    pack_dir = output_dir / "problem_pack"
    results_path = output_dir / "bbox_qc_results.jsonl"
    csv_path = output_dir / "bbox_qc_results.csv"
    summary_path = output_dir / "summary.md"

    metadata_paths = sorted(Path(args.metadata_dir).glob("*.json"))
    if args.limit is not None:
        metadata_paths = metadata_paths[: max(0, args.limit)]

    done = set() if args.overwrite else load_done(results_path)
    asset_base_dirs = [Path(path) for path in args.asset_base_dir]

    tasks: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        sample = read_json(metadata_path)
        image_uri = (sample.get("image_asset") or {}).get("image_uri")
        image_path, image_exists, _ = resolve_image_for_qc(image_uri, metadata_path, asset_base_dirs)
        objects = sample.get("objects") if isinstance(sample.get("objects"), list) else []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            sample_id = sample.get("sample_id") or metadata_path.stem
            object_id = obj.get("object_id") or "object"
            key = f"{sample_id}::{object_id}"
            if key in done:
                continue
            tasks.append(
                {
                    "metadata_path": metadata_path,
                    "sample": sample,
                    "image_uri": image_uri,
                    "image_path": image_path,
                    "image_exists": image_exists,
                    "obj": obj,
                    "sample_id": sample_id,
                    "object_id": object_id,
                }
            )

    def client_factory() -> Any:
        if OpenAI is None:
            return None
        return OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout_seconds)

    def process_task(task: dict[str, Any]) -> dict[str, Any]:
        metadata_path = task["metadata_path"]
        sample = task["sample"]
        obj = task["obj"]
        sample_id = task["sample_id"]
        object_id = task["object_id"]
        image_uri = task["image_uri"]
        image_path = task["image_path"]
        image_exists = task["image_exists"]
        labels = obj.get("classification", {}).get("multi_labels") or []
        label_text = ";".join(
            f"{item.get('label_key')}={item.get('label_value')}"
            for item in labels
            if isinstance(item, dict) and item.get("label_key")
        )
        scene = sample.get("image_asset", {}).get("scene_context") or {}
        ctx = object_context(obj)
        rules = rule_flags(obj, min_grid_iou=args.min_grid_iou, tolerance_px=args.region_tolerance_px)
        overlay_path = overlays_dir / f"{sample_id}_{object_id}_overlay.jpg"
        evidence_panel_path = evidence_dir / f"{sample_id}_{object_id}_crop_evidence.jpg"
        weak_label_hint = weak_label_hint_from_sample(metadata_path, sample)
        base_record = {
            "sample_id": sample_id,
            "object_id": object_id,
            "metadata_path": str(metadata_path),
            "weak_label_hint": weak_label_hint,
            "image_path": str(image_path) if image_path else "",
            "overlay_path": str(overlay_path),
            "evidence_panel_path": str(evidence_panel_path)
            if args.review_mode in {"crop_evidence", "training_triage"}
            else "",
            "review_mode": args.review_mode,
            "triage_prompt_version": args.triage_prompt_version if args.review_mode == "training_triage" else "",
            "triage_prompt_revision": (
                TRIAGE_PROMPT_REVISIONS[args.triage_prompt_version]
                if args.review_mode == "training_triage"
                else ""
            ),
            "semantic_model": semantic_model if args.review_mode == "training_triage" else "",
            "geometry_model": geometry_model if args.review_mode == "training_triage" else "",
            "clean_verifier_model": clean_verifier_model if args.review_mode == "training_triage" else "",
            "inspection_content": scene.get("inspection_content"),
            "labels": label_text,
            "box": ctx["box"],
            "selected_grid": ctx["selected_grid"],
            "grid_bbox": ctx["grid_bbox"],
            "expanded_edit_bbox": ctx["expanded_edit_bbox"],
            "final_bbox_source": ctx["final_bbox_source"],
            "localization_pipeline": ctx["localization_pipeline"],
            **rules,
        }

        started = time.perf_counter()
        try:
            if image_path is None or not image_exists:
                raise FileNotFoundError(f"image not found: {image_uri}")
            save_overlay(image_path, ctx["box"], overlay_path)
            if args.review_mode in {"crop_evidence", "training_triage"}:
                image_url, evidence_metadata = save_crop_evidence_panel(
                    image_path,
                    ctx["box"],
                    evidence_panel_path,
                    context_expand_ratio=args.context_expand_ratio,
                    context_min_pad=args.context_min_pad,
                    request_image_max_side=args.request_image_max_side,
                )
                base_record["evidence_panel"] = evidence_metadata
            else:
                image_url = build_overlay_data_url(image_path, ctx["box"], max_side=args.request_image_max_side)
            if args.build_evidence_only:
                record = {
                    **base_record,
                    "decision": "evidence_only",
                    "visible_target": None,
                    "box_quality": None,
                    "label_match": None,
                    "needs_human_review": None,
                    "issue_flags": [],
                    "reason": "Evidence assets generated; VLM call skipped.",
                }
                record["latency_seconds"] = round(time.perf_counter() - started, 3)
                return record

            if args.review_mode == "training_triage" and args.triage_prompt_version == "v5":
                semantic_image_url = build_clean_image_data_url(
                    image_path,
                    max_side=args.request_image_max_side,
                )
                geometry_image_urls = build_geometry_evidence_urls(
                    image_path,
                    ctx["box"],
                    context_expand_ratio=args.context_expand_ratio,
                    context_min_pad=args.context_min_pad,
                    max_side=args.request_image_max_side,
                )
                semantic_text = call_vlm_json_text_with_retries(
                    client_factory,
                    model=semantic_model,
                    image_urls=[semantic_image_url],
                    prompt=semantic_prompt_v5(),
                    max_tokens=args.max_tokens,
                    retries=args.api_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    system_message=(
                        "You are a label-blind industrial image semantic quality reviewer. "
                        "Inspect the entire image and return one JSON object only."
                    ),
                )
                geometry_text = call_vlm_json_text_with_retries(
                    client_factory,
                    model=geometry_model,
                    image_urls=geometry_image_urls,
                    prompt=geometry_prompt_v5(),
                    max_tokens=args.max_tokens,
                    retries=args.api_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    system_message=(
                        "You are an independent industrial bounding-box geometry reviewer. "
                        "Do not infer dataset labels or training actions. Return one JSON object only."
                    ),
                )
                semantic_parsed = parse_semantic_v5_payload(semantic_text)
                geometry_parsed = parse_geometry_v5_payload(geometry_text)
                parsed = fuse_training_triage_v5(semantic_parsed, geometry_parsed, weak_label_hint)
                clean_verifier_parsed: dict[str, Any] | None = None
                if clean_verifier_model and parsed.get("training_action") == "auto_accept_positive":
                    try:
                        full_overlay_url = build_overlay_data_url(
                            image_path,
                            ctx["box"],
                            max_side=args.request_image_max_side,
                        )
                        clean_verifier_text = call_vlm_json_text_with_retries(
                            client_factory,
                            model=clean_verifier_model,
                            image_urls=[full_overlay_url, *geometry_image_urls],
                            prompt=clean_verifier_prompt_v5(),
                            max_tokens=args.max_tokens,
                            retries=args.api_retries,
                            retry_backoff_seconds=args.retry_backoff_seconds,
                            system_message=(
                                "You are an adversarial clean-seed bounding-box verifier. "
                                "Veto any box that is not clearly safe for direct training. Return JSON only."
                            ),
                        )
                        clean_verifier_parsed = parse_geometry_v5_payload(clean_verifier_text)
                        parsed = apply_clean_verifier_v5(
                            parsed,
                            clean_verifier_parsed,
                            verifier_model=clean_verifier_model,
                        )
                    except Exception as exc:
                        parsed = apply_clean_verifier_v5(
                            parsed,
                            None,
                            verifier_model=clean_verifier_model,
                            verifier_error=repr(exc),
                        )
                parsed["raw_response_text"] = json.dumps(
                    {
                        "semantic": semantic_parsed.get("semantic_raw"),
                        "geometry": geometry_parsed.get("geometry_raw"),
                        "clean_verifier": (
                            clean_verifier_parsed.get("geometry_raw")
                            if clean_verifier_parsed is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                )
            else:
                parsed = call_vlm_with_retries(
                    client_factory,
                    model=args.model,
                    image_url=image_url,
                    prompt=prompt_for(
                        {
                            **sample,
                            "_qc_weak_label_hint": weak_label_hint,
                        },
                        obj,
                        review_mode=args.review_mode,
                    ),
                    max_tokens=args.max_tokens,
                    retries=args.api_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                if args.review_mode == "training_triage":
                    parsed = apply_training_triage_guard(parsed, weak_label_hint)
            quality = parsed.get("box_quality")
            visible = bool(parsed.get("visible_target"))
            label_match = bool(parsed.get("label_match"))
            needs_review = bool(parsed.get("needs_human_review"))
            if args.review_mode == "training_triage":
                training_action = parsed.get("training_action")
                if training_action == "auto_accept_positive":
                    decision = "passed"
                elif training_action == "reject_from_positive_training":
                    decision = "failed"
                else:
                    decision = "needs_human_review"
            elif not visible or not label_match or quality == "wrong_target":
                decision = "failed"
            elif needs_review or quality in {"over_inclusive", "under_inclusive", "uncertain"} or rules["rule_flags"]:
                decision = "needs_human_review"
            else:
                decision = "passed"
            record = {
                **base_record,
                "decision": decision,
                "visible_target": visible,
                "box_quality": quality,
                "label_match": label_match,
                "needs_human_review": needs_review,
                "inside_has_target_anomaly": parsed.get("inside_has_target_anomaly"),
                "outside_has_same_anomaly": parsed.get("outside_has_same_anomaly"),
                "outside_extension": parsed.get("outside_extension"),
                "containment_quality": parsed.get("containment_quality"),
                "target_location": parsed.get("target_location"),
                "outside_evidence_summary": parsed.get("outside_evidence_summary"),
                "image_has_positive_candidate": parsed.get("image_has_positive_candidate"),
                "visual_positive_class": parsed.get("visual_positive_class"),
                "visual_class": parsed.get("visual_class"),
                "positive_candidate_location": parsed.get("positive_candidate_location"),
                "positive_candidate_relation_to_bbox": parsed.get("positive_candidate_relation_to_bbox"),
                "hard_negative_type": parsed.get("hard_negative_type"),
                "bbox_usable_for_training": parsed.get("bbox_usable_for_training"),
                "weak_label_consistent": parsed.get("weak_label_consistent"),
                "label_conflict": parsed.get("label_conflict"),
                "training_action": parsed.get("training_action"),
                "training_priority": parsed.get("training_priority"),
                "rework_type": parsed.get("rework_type"),
                "issue_flags": parsed.get("issue_flags", []),
                "reason": parsed.get("reason"),
                "raw_response_text": parsed.get("raw_response_text"),
                "semantic_verdict": parsed.get("semantic_verdict"),
                "semantic_candidate_location": parsed.get("candidate_location"),
                "semantic_candidate_surface": parsed.get("candidate_surface"),
                "semantic_positive_evidence": parsed.get("positive_evidence"),
                "semantic_reason": parsed.get("semantic_reason"),
                "semantic_raw_response_text": parsed.get("semantic_raw_response_text"),
                "inside_target_status": parsed.get("inside_target_status"),
                "outside_relation": parsed.get("outside_relation"),
                "target_surface": parsed.get("target_surface"),
                "dominant_confounder": parsed.get("dominant_confounder"),
                "geometry_reason": parsed.get("geometry_reason"),
                "geometry_raw_response_text": parsed.get("geometry_raw_response_text"),
                "truncated_edges": parsed.get("truncated_edges"),
                "edge_checks_complete": parsed.get("edge_checks_complete"),
                "background_excess": parsed.get("background_excess"),
                "target_area_fraction": parsed.get("target_area_fraction"),
                "stage_disagreement": parsed.get("stage_disagreement"),
                "primary_box_quality": parsed.get("primary_box_quality"),
                "clean_verifier_passed": parsed.get("clean_verifier_passed"),
                "clean_verifier_box_quality": parsed.get("clean_verifier_box_quality"),
                "clean_verifier_rework_type": parsed.get("clean_verifier_rework_type"),
                "clean_verifier_reason": parsed.get("clean_verifier_reason"),
                "clean_verifier_raw_response_text": parsed.get("clean_verifier_raw_response_text"),
                "clean_verifier_error": parsed.get("clean_verifier_error"),
            }
        except Exception as exc:
            record = {
                **base_record,
                "decision": "needs_human_review",
                "visible_target": None,
                "box_quality": "uncertain",
                "label_match": None,
                "needs_human_review": True,
                "inside_has_target_anomaly": None,
                "outside_has_same_anomaly": None,
                "outside_extension": None,
                "containment_quality": None,
                "target_location": None,
                "outside_evidence_summary": None,
                "image_has_positive_candidate": None,
                "visual_positive_class": None,
                "visual_class": None,
                "positive_candidate_location": None,
                "positive_candidate_relation_to_bbox": None,
                "hard_negative_type": None,
                "bbox_usable_for_training": None,
                "weak_label_consistent": None,
                "label_conflict": None,
                "training_action": None,
                "training_priority": None,
                "rework_type": None,
                "clean_verifier_passed": None,
                "clean_verifier_box_quality": None,
                "clean_verifier_rework_type": None,
                "clean_verifier_reason": None,
                "clean_verifier_raw_response_text": None,
                "clean_verifier_error": None,
                "issue_flags": ["api_or_runtime_error"],
                "reason": "QC call failed; route to manual review",
                "error": repr(exc),
            }
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        return record

    def write_record(record: dict[str, Any], results_handle: Any, index: int) -> None:
        results_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_handle.flush()
        pack_record(record, pack_dir)
        print(
            f"{index:04d} {record['sample_id']} {record['object_id']} "
            f"{record['decision']} {record.get('box_quality')} {record.get('latency_seconds')}s",
            flush=True,
        )

    new_records = 0
    workers = max(1, int(args.workers or 1))
    with results_path.open("a", encoding="utf-8") as results_handle:
        if workers == 1:
            for task in tasks:
                record = process_task(task)
                new_records += 1
                write_record(record, results_handle, new_records)
                if args.interval_seconds > 0:
                    time.sleep(args.interval_seconds)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                for future in as_completed(futures):
                    record = future.result()
                    new_records += 1
                    write_record(record, results_handle, new_records)

    records = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(flatten_record(records[0]).keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record(record))

    counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    for record in records:
        counts[record["decision"]] = counts.get(record["decision"], 0) + 1
        quality = record.get("box_quality") or "unknown"
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    summary_path.write_text(
        "\n".join(
            [
                "# Anomaly BBox QC Summary",
                "",
                f"- total_records: {len(records)}",
                f"- decision_counts: {json.dumps(counts, ensure_ascii=False)}",
                f"- box_quality_counts: {json.dumps(quality_counts, ensure_ascii=False)}",
                f"- review_mode: {args.review_mode}",
                f"- triage_prompt_version: {args.triage_prompt_version if args.review_mode == 'training_triage' else ''}",
                f"- triage_prompt_revision: {TRIAGE_PROMPT_REVISIONS[args.triage_prompt_version] if args.review_mode == 'training_triage' else ''}",
                f"- semantic_model: {semantic_model if args.review_mode == 'training_triage' else ''}",
                f"- geometry_model: {geometry_model if args.review_mode == 'training_triage' else ''}",
                f"- clean_verifier_model: {clean_verifier_model if args.review_mode == 'training_triage' else ''}",
                f"- build_evidence_only: {args.build_evidence_only}",
                f"- results_jsonl: {results_path}",
                f"- results_csv: {csv_path}",
                f"- overlays: {overlays_dir}",
                f"- evidence_panels: {evidence_dir if args.review_mode in {'crop_evidence', 'training_triage'} else ''}",
                f"- problem_pack: {pack_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
