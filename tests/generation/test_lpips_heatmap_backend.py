from __future__ import annotations

import importlib
from pathlib import Path

from .conftest import result_to_dict


def test_pgcd_metrics_record_heatmap_backend(simple_change_pair) -> None:
    registry = importlib.import_module("autolabel.modules.generation.localizers.registry")
    localizer = registry.create_localizer("pgcd_lpips", min_component_area=100, debug=True)
    result = localizer.localize(
        original_image_path=str(simple_change_pair["original_path"]),
        generated_image_path=str(simple_change_pair["generated_path"]),
        prompt_box=list(simple_change_pair["prompt_box"]),
        anomaly_type="water_leak",
        mask_output_path=str(Path(simple_change_pair["output_dir"]) / "lpips_backend_mask.png"),
        debug_dir=str(simple_change_pair["output_dir"]),
    )
    data = result_to_dict(result)
    metrics = data["metrics"]
    assert "pgcd_heatmap_backend" in metrics
    assert metrics["pgcd_heatmap_backend"] in {"lpips", "rgb_diff_fallback"}
    if metrics["pgcd_heatmap_backend"] == "lpips":
        assert metrics["pgcd_lpips_backbone"] == "alex"
        assert metrics["pgcd_lpips_input_max_side"] == 768
    else:
        assert metrics.get("pgcd_heatmap_backend_reason")
