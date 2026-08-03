from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

BBox = tuple[int, int, int, int]


def bbox_iou(a: BBox | list[int], b: BBox | list[int]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def assert_bbox_valid(bbox: BBox | list[int], image_size: tuple[int, int]) -> None:
    w, h = image_size
    assert len(bbox) == 4
    x1, y1, x2, y2 = bbox
    assert 0 <= x1 < x2 <= w
    assert 0 <= y1 < y2 <= h


def result_to_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return result
    out: dict[str, Any] = {}
    for name in ("success", "final_bbox", "mask_path", "method", "metrics", "reason", "fallback_used"):
        if hasattr(result, name):
            out[name] = getattr(result, name)
    return out


def make_base_image(size: tuple[int, int] = (256, 192)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, (132, 132, 132))
    draw = ImageDraw.Draw(img)
    # 轻微地面纹理，避免纯色导致算法过拟合。
    for y in range(0, h, 16):
        color = 126 if (y // 16) % 2 == 0 else 138
        draw.line([(0, y), (w, y)], fill=(color, color, color), width=1)
    for x in range(0, w, 32):
        draw.line([(x, 0), (x, h)], fill=(120, 120, 120), width=1)
    return img


def add_water_patch(img: Image.Image, bbox: BBox, fill=(190, 205, 210), alpha: int = 120) -> Image.Image:
    out = img.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse(bbox, fill=(*fill, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return Image.alpha_composite(out, overlay).convert("RGB")


def add_reflection(img: Image.Image, bbox: BBox) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(bbox, fill=(195, 195, 195))
    return out.filter(ImageFilter.GaussianBlur(radius=1))


def add_global_shift(img: Image.Image, delta: int = 52) -> Image.Image:
    array = np.asarray(img.convert("RGB"), dtype=np.int16)
    shifted = np.clip(array + delta, 0, 255).astype(np.uint8)
    return Image.fromarray(shifted, mode="RGB")


def save_pair(tmp_path: Path, original: Image.Image, generated: Image.Image) -> tuple[Path, Path]:
    original_path = tmp_path / "original.jpg"
    generated_path = tmp_path / "generated.jpg"
    original.save(original_path)
    generated.save(generated_path)
    return original_path, generated_path


@pytest.fixture()
def simple_change_pair(tmp_path: Path) -> dict[str, Any]:
    base = make_base_image()
    prompt_box: BBox = (80, 80, 170, 150)
    gt_bbox: BBox = (98, 96, 152, 132)
    generated = add_water_patch(base, gt_bbox)
    original_path, generated_path = save_pair(tmp_path, base, generated)
    return {
        "original_path": original_path,
        "generated_path": generated_path,
        "prompt_box": prompt_box,
        "gt_bbox": gt_bbox,
        "image_size": base.size,
        "output_dir": tmp_path,
    }


@pytest.fixture()
def no_change_pair(tmp_path: Path) -> dict[str, Any]:
    base = make_base_image()
    original_path, generated_path = save_pair(tmp_path, base, base.copy())
    return {
        "original_path": original_path,
        "generated_path": generated_path,
        "prompt_box": (80, 80, 170, 150),
        "gt_bbox": None,
        "image_size": base.size,
        "output_dir": tmp_path,
    }


@pytest.fixture()
def prompt_prior_pair(tmp_path: Path) -> dict[str, Any]:
    base = make_base_image()
    prompt_box: BBox = (82, 80, 172, 150)
    gt_bbox: BBox = (102, 96, 150, 132)
    generated = add_water_patch(base, gt_bbox, alpha=105)
    # prompt_box 外部制造一个面积更大的反光变化，用来验证 prior 不被大面积扰动带偏。
    generated = add_reflection(generated, (10, 12, 74, 62))
    original_path, generated_path = save_pair(tmp_path, base, generated)
    return {
        "original_path": original_path,
        "generated_path": generated_path,
        "prompt_box": prompt_box,
        "gt_bbox": gt_bbox,
        "outside_change_bbox": (10, 12, 74, 62),
        "image_size": base.size,
        "output_dir": tmp_path,
    }


@pytest.fixture()
def resized_generated_pair(tmp_path: Path) -> dict[str, Any]:
    base = make_base_image(size=(256, 192))
    prompt_box: BBox = (80, 80, 170, 150)
    gt_bbox: BBox = (98, 96, 152, 132)
    generated = add_water_patch(base, gt_bbox)
    generated = generated.resize((320, 240))
    original_path, generated_path = save_pair(tmp_path, base, generated)
    return {
        "original_path": original_path,
        "generated_path": generated_path,
        "prompt_box": prompt_box,
        "gt_bbox": gt_bbox,
        "image_size": base.size,
        "output_dir": tmp_path,
    }


@pytest.fixture()
def global_change_pair(tmp_path: Path) -> dict[str, Any]:
    base = make_base_image()
    generated = add_global_shift(base, delta=56)
    original_path, generated_path = save_pair(tmp_path, base, generated)
    return {
        "original_path": original_path,
        "generated_path": generated_path,
        "prompt_box": (80, 80, 170, 150),
        "gt_bbox": None,
        "image_size": base.size,
        "output_dir": tmp_path,
    }


class FakeHeatmapBackend:
    """用于 PGCD 单测的可注入 heatmap backend，避免测试依赖真实 LPIPS 模型。"""

    def __init__(self, hot_boxes: list[BBox], image_size: tuple[int, int] = (256, 192), value: float = 1.0):
        self.hot_boxes = hot_boxes
        self.image_size = image_size
        self.value = value

    def compute(self, original: Any, generated: Any, **kwargs: Any) -> np.ndarray:
        w, h = self.image_size
        heatmap = np.zeros((h, w), dtype=np.float32)
        for x1, y1, x2, y2 in self.hot_boxes:
            heatmap[y1:y2, x1:x2] = self.value
        return heatmap


class ZeroHeatmapBackend:
    def compute(self, original: Any, generated: Any, **kwargs: Any) -> np.ndarray:
        if hasattr(original, "size"):
            w, h = original.size
        else:
            h, w = 192, 256
        return np.zeros((h, w), dtype=np.float32)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
