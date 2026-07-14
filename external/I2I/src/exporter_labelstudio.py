from __future__ import annotations


def autolabel_to_labelstudio(sample: dict) -> dict:
    image = sample["image_asset"]
    width = image["width"]
    height = image["height"]
    results = []
    for obj in sample["objects"]:
        box = obj["box"]
        labels = [
            item["label_value"]
            for item in obj["classification"]["multi_labels"]
            if item.get("label_key") == "anomaly_type"
        ]
        results.append(
            {
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": box["x1"] / width * 100,
                    "y": box["y1"] / height * 100,
                    "width": (box["x2"] - box["x1"]) / width * 100,
                    "height": (box["y2"] - box["y1"]) / height * 100,
                    "rectanglelabels": labels,
                },
            }
        )
    return {
        "data": {"image": image["image_uri"], "sample_id": sample["sample_id"]},
        "predictions": [{"model_version": "vlm_grid_image_edit_autolabel_pipeline_v1.0", "result": results}],
    }
