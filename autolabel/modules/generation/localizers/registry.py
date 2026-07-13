from __future__ import annotations

from typing import Any

from .pgcd_lpips import PGCDLPIPSLocalizer
from .pgcd_lpips_sam2 import PGCDLPIPSSAM2Localizer
from .rgb_diff import RGBDiffLocalizer


def create_localizer(name: str, **kwargs: Any):
    normalized = name.strip().lower()
    if normalized == "rgb_diff":
        return RGBDiffLocalizer(**kwargs)
    if normalized == "pgcd_lpips":
        return PGCDLPIPSLocalizer(**kwargs)
    if normalized == "pgcd_lpips_sam2":
        return PGCDLPIPSSAM2Localizer(**kwargs)
    raise ValueError(f"Unknown localizer: {name}")
