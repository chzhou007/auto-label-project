from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ...utils import now_iso_shanghai


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self.handle.close()

    def log(self, task_id: str, stage: str, status: str, **kwargs: Any) -> None:
        payload = {"time": now_iso_shanghai(), "task_id": task_id, "stage": stage, "status": status, **kwargs}
        with self._lock:
            self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.handle.flush()
