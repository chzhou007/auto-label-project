from __future__ import annotations

import importlib
from pathlib import Path

from autolabel.model_config import build_generation_extra_cli_args


ROOT = Path(__file__).resolve().parents[2]


def test_build_generation_extra_cli_args_supports_multiple_sidecars() -> None:
    args = build_generation_extra_cli_args(
        {
            "primary": "pgcd_lpips",
            "fallback": "rgb_diff",
            "sidecar_eval": ["pgcd_lpips", "pgcd_lpips_sam2"],
            "sam2": {"enabled": False, "model": "tiny"},
        }
    )
    assert "--localizer-sidecar-eval" in args
    index = args.index("--localizer-sidecar-eval")
    assert args[index + 1] == "pgcd_lpips,pgcd_lpips_sam2"


def test_project_declares_lpips_runtime_dependencies() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dependency in ["torch", "torchvision", "lpips"]:
        assert dependency in requirements
        assert dependency in pyproject


def test_generation_cli_parser_supports_localizer_flags() -> None:
    main = importlib.import_module("autolabel.modules.generation.main")
    parser = main.build_arg_parser()

    args = parser.parse_args(
        [
            "--benchmark",
            "--localizer",
            "pgcd_lpips_sam2",
            "--localizer-fallback",
            "rgb_diff",
            "--localizer-debug",
            "--localizer-sidecar-eval",
            "pgcd_lpips,rgb_diff",
            "--pgcd-threshold",
            "adaptive",
            "--pgcd-min-component-area",
            "123",
            "--pgcd-max-global-change-ratio",
            "0.42",
            "--pgcd-prompt-prior-weight-area",
            "0.31",
            "--pgcd-prompt-prior-weight-lpips",
            "0.41",
            "--pgcd-prompt-prior-weight-iou",
            "0.21",
            "--pgcd-prompt-prior-weight-distance",
            "0.11",
            "--sam2-enabled",
            "--sam2-model",
            "small",
        ]
    )

    assert args.benchmark is True
    assert args.localizer == "pgcd_lpips_sam2"
    assert args.localizer_fallback == "rgb_diff"
    assert args.localizer_debug is True
    assert args.localizer_sidecar_eval == "pgcd_lpips,rgb_diff"
    assert args.pgcd_threshold == "adaptive"
    assert args.pgcd_min_component_area == 123
    assert args.pgcd_max_global_change_ratio == 0.42
    assert args.pgcd_prompt_prior_weight_area == 0.31
    assert args.pgcd_prompt_prior_weight_lpips == 0.41
    assert args.pgcd_prompt_prior_weight_iou == 0.21
    assert args.pgcd_prompt_prior_weight_distance == 0.11
    assert args.sam2_enabled is True
    assert args.sam2_model == "small"


def test_generation_cli_overrides_update_nested_localizer_config() -> None:
    main = importlib.import_module("autolabel.modules.generation.main")
    parser = main.build_arg_parser()
    args = parser.parse_args(
        [
            "--benchmark",
            "--localizer",
            "pgcd_lpips",
            "--localizer-fallback",
            "rgb_diff",
            "--localizer-debug",
            "--localizer-sidecar-eval",
            "pgcd_lpips_sam2",
            "--pgcd-threshold",
            "fixed",
            "--pgcd-min-component-area",
            "456",
            "--pgcd-max-global-change-ratio",
            "0.25",
            "--pgcd-prompt-prior-weight-area",
            "0.3",
            "--pgcd-prompt-prior-weight-lpips",
            "0.4",
            "--pgcd-prompt-prior-weight-iou",
            "0.2",
            "--pgcd-prompt-prior-weight-distance",
            "0.1",
            "--sam2-enabled",
            "--sam2-model",
            "tiny",
        ]
    )

    updated = main.apply_generation_cli_overrides(
        {
            "modules": {
                "generation": {
                    "localizer": {
                        "primary": "rgb_diff",
                        "fallback": "none",
                        "debug": False,
                        "sidecar_eval": None,
                        "pgcd": {
                            "threshold": "otsu",
                            "min_component_area": 100,
                            "max_global_change_ratio": 1.0,
                            "prompt_prior_weights": {
                                "area": 1.0,
                                "lpips": 0.0,
                                "iou": 0.0,
                                "distance": 0.0,
                            },
                        },
                        "sam2": {"enabled": False, "model": "base"},
                    }
                }
            }
        },
        args,
    )

    localizer_cfg = updated["modules"]["generation"]["localizer"]
    assert localizer_cfg["benchmark"] is True
    assert localizer_cfg["primary"] == "pgcd_lpips"
    assert localizer_cfg["fallback"] == "rgb_diff"
    assert localizer_cfg["debug"] is True
    assert localizer_cfg["sidecar_eval"] == "pgcd_lpips_sam2"
    assert localizer_cfg["pgcd"]["threshold"] == "fixed"
    assert localizer_cfg["pgcd"]["min_component_area"] == 456
    assert localizer_cfg["pgcd"]["max_global_change_ratio"] == 0.25
    assert localizer_cfg["pgcd"]["prompt_prior_weights"] == {
        "area": 0.3,
        "lpips": 0.4,
        "iou": 0.2,
        "distance": 0.1,
    }
    assert localizer_cfg["sam2"] == {"enabled": True, "model": "tiny"}
