from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ...config_loader import load_config
from ...orchestrator import apply_generation_run_overrides, configure_processed_root, run_generation_branch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AutoLabel generation branch with roadmap localizer postprocess.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--processed-root", default=None)
    parser.add_argument("--generation-image-model-key", default=None)
    parser.add_argument("--generation-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--no-ingest-generated", action="store_true")
    parser.add_argument("--localizer", choices=["rgb_diff", "pgcd_lpips", "pgcd_lpips_sam2"], default=None)
    parser.add_argument("--localizer-fallback", default=None)
    parser.add_argument("--localizer-debug", action="store_true")
    parser.add_argument("--localizer-sidecar-eval", default=None)
    parser.add_argument("--pgcd-threshold", choices=["otsu", "adaptive", "fixed"], default=None)
    parser.add_argument("--pgcd-min-component-area", type=int, default=None)
    parser.add_argument("--pgcd-max-global-change-ratio", type=float, default=None)
    parser.add_argument("--pgcd-prompt-prior-weight-area", type=float, default=None)
    parser.add_argument("--pgcd-prompt-prior-weight-lpips", type=float, default=None)
    parser.add_argument("--pgcd-prompt-prior-weight-iou", type=float, default=None)
    parser.add_argument("--pgcd-prompt-prior-weight-distance", type=float, default=None)
    parser.add_argument("--sam2-enabled", action="store_true")
    parser.add_argument("--sam2-model", default=None)
    return parser


def apply_generation_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    resolved = deepcopy(config)
    modules = resolved.setdefault("modules", {})
    generation_module = modules.setdefault("generation", {})
    localizer_cfg = generation_module.setdefault("localizer", {})
    pgcd_cfg = localizer_cfg.setdefault("pgcd", {})
    sam2_cfg = localizer_cfg.setdefault("sam2", {})
    prompt_prior_weights = pgcd_cfg.setdefault("prompt_prior_weights", {})

    if args.localizer is not None:
        localizer_cfg["primary"] = args.localizer
    if args.localizer_fallback is not None:
        localizer_cfg["fallback"] = args.localizer_fallback
    if args.localizer_debug:
        localizer_cfg["debug"] = True
    if args.localizer_sidecar_eval is not None:
        localizer_cfg["sidecar_eval"] = args.localizer_sidecar_eval
    if args.benchmark:
        localizer_cfg["benchmark"] = True

    if args.pgcd_threshold is not None:
        pgcd_cfg["threshold"] = args.pgcd_threshold
    if args.pgcd_min_component_area is not None:
        pgcd_cfg["min_component_area"] = int(args.pgcd_min_component_area)
    if args.pgcd_max_global_change_ratio is not None:
        pgcd_cfg["max_global_change_ratio"] = float(args.pgcd_max_global_change_ratio)
    if args.pgcd_prompt_prior_weight_area is not None:
        prompt_prior_weights["area"] = float(args.pgcd_prompt_prior_weight_area)
    if args.pgcd_prompt_prior_weight_lpips is not None:
        prompt_prior_weights["lpips"] = float(args.pgcd_prompt_prior_weight_lpips)
    if args.pgcd_prompt_prior_weight_iou is not None:
        prompt_prior_weights["iou"] = float(args.pgcd_prompt_prior_weight_iou)
    if args.pgcd_prompt_prior_weight_distance is not None:
        prompt_prior_weights["distance"] = float(args.pgcd_prompt_prior_weight_distance)

    if args.sam2_enabled:
        sam2_cfg["enabled"] = True
    if args.sam2_model is not None:
        sam2_cfg["model"] = args.sam2_model

    return resolved


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = apply_generation_cli_overrides(load_config(args.config), args)
    configure_processed_root(config, args.processed_root)
    apply_generation_run_overrides(
        config,
        image_model_key=args.generation_image_model_key,
        workers=args.generation_workers,
    )
    return run_generation_branch(
        config,
        tasks_csv=args.tasks,
        image_root=args.image_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        limit=args.limit,
        ingest_metadata=not args.no_ingest_generated,
        preflight=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
