from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import requests


DEFAULT_PERSON_KEYWORDS = ("人员", "人体", "手机", "打架", "翻越")
PERSON_LABEL_ALIASES = {
    "person",
    "Person",
    "falling",
    "other",
    "wearing_helmet",
    "no_helmet",
    "wearing_reflective_vest",
    "no_reflective_vest",
    "using_phone",
    "not_using_phone",
    "smoking",
    "not_smoking",
    "sleeping",
    "not_sleeping",
    "climbing_over_railing",
    "not_climbing_over_railing",
    "fighting",
    "not_fighting",
    "touching_equipment",
    "not_touching_equipment",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ls_rect_to_xyxy(value: dict[str, Any], width: int, height: int) -> list[int]:
    x = float(value["x"])
    y = float(value["y"])
    w = float(value["width"])
    h = float(value["height"])
    x1 = round(width * x / 100.0)
    y1 = round(height * y / 100.0)
    x2 = round(width * (x + w) / 100.0)
    y2 = round(height * (y + h) / 100.0)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def iter_rectangles(task: dict[str, Any], source: str) -> list[dict[str, Any]]:
    containers = task.get(source) or []
    rectangles = []
    for container in containers:
        for result in container.get("result") or []:
            if result.get("type") != "rectanglelabels":
                continue
            value = result.get("value") or {}
            labels = value.get("rectanglelabels") or []
            label = str(labels[0]) if labels else "person"
            if label not in PERSON_LABEL_ALIASES:
                continue
            width = int(result.get("original_width") or task.get("data", {}).get("width") or 0)
            height = int(result.get("original_height") or task.get("data", {}).get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            rectangles.append(
                {
                    "id": result.get("id"),
                    "label": label,
                    "bbox_xyxy": ls_rect_to_xyxy(value, width, height),
                    "original_width": width,
                    "original_height": height,
                }
            )
    return rectangles


def task_id(project_name: str, task: dict[str, Any], index: int) -> str:
    data = task.get("data") or {}
    rel = str(data.get("relative_path") or data.get("image") or index)
    safe = "".join(ch if ch.isalnum() else "_" for ch in rel)
    return f"{project_name}_{safe}".strip("_")


def is_person_project(path: Path, keywords: tuple[str, ...]) -> bool:
    return any(keyword in path.stem for keyword in keywords)


def is_person_project_title(title: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in title for keyword in keywords)


class LabelStudioSession:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()

    def login(self) -> None:
        token_resp = self.session.post(
            f"{self.base_url}/api/token",
            json={"username": self.username, "password": self.password},
            timeout=30,
        )
        if token_resp.ok:
            token = token_resp.json().get("token")
            if token:
                self.session.headers.update({"Authorization": f"Token {token}"})
                return

        login_page = self.session.get(f"{self.base_url}/user/login", timeout=30)
        login_page.raise_for_status()
        csrf = self.session.cookies.get("csrftoken")
        login_resp = self.session.post(
            f"{self.base_url}/user/login",
            data={"email": self.username, "password": self.password, "csrfmiddlewaretoken": csrf},
            headers={"Referer": f"{self.base_url}/user/login"},
            timeout=30,
            allow_redirects=True,
        )
        login_resp.raise_for_status()
        whoami = self.session.get(f"{self.base_url}/api/current-user/whoami", timeout=30)
        whoami.raise_for_status()

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        url = f"{self.base_url}/api/projects"
        while url:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                projects.extend(payload)
                break
            projects.extend(payload.get("results") or [])
            url = payload.get("next")
        return projects

    def collect_project_tasks(self, project_id: int) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self.session.get(
                f"{self.base_url}/api/tasks",
                params={"project": project_id, "page": page, "page_size": 500},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, list):
                tasks.extend(payload)
                break
            page_tasks = payload.get("tasks") or payload.get("results") or []
            tasks.extend(page_tasks)
            total = int(payload.get("total") or payload.get("count") or len(tasks))
            if len(tasks) >= total or not page_tasks:
                break
            page += 1
        detailed_tasks = []
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                detailed_tasks.append(task)
                continue
            resp = self.session.get(f"{self.base_url}/api/tasks/{task_id}", timeout=60)
            resp.raise_for_status()
            detailed_tasks.append(resp.json())
        return detailed_tasks


def benchmark_from_project_tasks(project_tasks: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = []
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()
    box_count = 0

    for project, tasks in project_tasks:
        project_name = str(project.get("title") or f"project_{project.get('id')}")
        project_id = project.get("id")
        for index, task in enumerate(tasks, start=1):
            annotations = iter_rectangles(task, "annotations")
            predictions = iter_rectangles(task, "predictions")
            boxes = annotations if annotations else predictions
            source = "annotation" if annotations else "prediction_pseudo_gt"
            if not boxes:
                continue
            data = task.get("data") or {}
            sample = {
                "sample_id": task_id(project_name, task, index),
                "labelstudio_task_id": task.get("id"),
                "labelstudio_project_id": project_id,
                "project_name": project_name,
                "source": source,
                "image": data.get("image"),
                "relative_path": data.get("relative_path"),
                "picturelink": data.get("picturelink"),
                "inspection_content": data.get("inspection_content") or project_name,
                "camera_name": data.get("camera_name"),
                "acquisition_time": data.get("acquisition_time"),
                "width": boxes[0]["original_width"],
                "height": boxes[0]["original_height"],
                "boxes": boxes,
                "prediction_boxes": predictions,
            }
            samples.append(sample)
            source_counts[source] += 1
            project_counts[project_name] += 1
            for box in boxes:
                label_counts[box["label"]] += 1
            box_count += len(boxes)

    summary = {
        "sample_count": len(samples),
        "box_count": box_count,
        "source_counts": dict(sorted(source_counts.items())),
        "project_counts": dict(sorted(project_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }
    return samples, summary


def write_benchmark(output_dir: Path, samples: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "person_detection_benchmark.json", {"samples": samples})
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "labelstudio_project_id",
                "labelstudio_task_id",
                "project_name",
                "source",
                "image",
                "relative_path",
                "width",
                "height",
                "box_count",
                "prediction_box_count",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "labelstudio_project_id": sample.get("labelstudio_project_id"),
                    "labelstudio_task_id": sample.get("labelstudio_task_id"),
                    "project_name": sample["project_name"],
                    "source": sample["source"],
                    "image": sample["image"],
                    "relative_path": sample["relative_path"],
                    "width": sample["width"],
                    "height": sample["height"],
                    "box_count": len(sample["boxes"]),
                    "prediction_box_count": len(sample.get("prediction_boxes") or []),
                }
            )
    return summary


def build_benchmark(input_dir: Path, output_dir: Path, keywords: tuple[str, ...]) -> dict[str, Any]:
    files = sorted(path for path in input_dir.glob("*.json") if is_person_project(path, keywords))
    project_tasks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for path in files:
        project_tasks.append(({"title": path.stem, "id": None}, read_json(path)))
    samples, summary = benchmark_from_project_tasks(project_tasks)
    summary.update(
        {
            "input": "local_export",
            "input_dir": str(input_dir),
            "project_files": [path.name for path in files],
            "note": (
                "Samples with Label Studio annotations use annotations as ground truth. "
                "Samples without annotations use predictions as pseudo ground truth and must not be reported as a final mAP benchmark."
            ),
        }
    )
    return write_benchmark(output_dir, samples, summary)


def build_benchmark_from_labelstudio(
    base_url: str,
    username: str,
    password: str,
    output_dir: Path,
    keywords: tuple[str, ...],
    project_ids: set[int] | None = None,
) -> dict[str, Any]:
    client = LabelStudioSession(base_url, username, password)
    client.login()
    projects = []
    for project in client.list_projects():
        project_id = int(project["id"])
        title = str(project.get("title") or "")
        if project_ids and project_id not in project_ids:
            continue
        if not project_ids and not is_person_project_title(title, keywords):
            continue
        projects.append(project)

    project_tasks = []
    for project in sorted(projects, key=lambda item: int(item["id"])):
        tasks = client.collect_project_tasks(int(project["id"]))
        project_tasks.append((project, tasks))

    samples, summary = benchmark_from_project_tasks(project_tasks)
    summary.update(
        {
            "input": "labelstudio_api",
            "base_url": base_url,
            "project_ids": [int(project["id"]) for project in sorted(projects, key=lambda item: int(item["id"]))],
            "project_titles": [str(project.get("title")) for project in sorted(projects, key=lambda item: int(item["id"]))],
            "note": (
                "Samples with Label Studio annotations use annotations as ground truth. "
                "Samples without annotations use predictions as pseudo ground truth and must not be reported as a final mAP benchmark."
            ),
        }
    )
    return write_benchmark(output_dir, samples, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a person-related detection benchmark from Label Studio exports.")
    parser.add_argument("--input-dir", type=Path, default=Path("labelstudio_sh16_20260605_by_project"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/benchmarks/person_sh16_labelstudio_20260706"))
    parser.add_argument("--keywords", default=",".join(DEFAULT_PERSON_KEYWORDS))
    parser.add_argument("--base-url", default="")
    parser.add_argument("--username", default=os.getenv("LABEL_STUDIO_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LABEL_STUDIO_PASSWORD", ""))
    parser.add_argument("--project-ids", default="", help="Comma-separated Label Studio project ids. Overrides keyword filtering.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keywords = tuple(item.strip() for item in args.keywords.split(",") if item.strip())
    project_ids = {int(item.strip()) for item in args.project_ids.split(",") if item.strip()}
    if args.base_url:
        if not args.username or not args.password:
            raise SystemExit("--base-url requires --username/--password or LABEL_STUDIO_USERNAME/LABEL_STUDIO_PASSWORD.")
        summary = build_benchmark_from_labelstudio(
            args.base_url,
            args.username,
            args.password,
            args.output_dir,
            keywords,
            project_ids=project_ids or None,
        )
    else:
        summary = build_benchmark(args.input_dir, args.output_dir, keywords)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
