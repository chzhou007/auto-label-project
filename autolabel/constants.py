SOURCE_TYPES = {
    "generated",
    "human_staged",
    "cctv",
    "robot_dog",
    "manual_upload",
}

GEOMETRY_SOURCES = {
    "detector",
    "segmentation_model",
    "synthetic_generator",
    "human_annotation",
}

CLASSIFIER_TYPES = {
    "vlm",
    "classifier",
    "rule",
    "human",
}

WORKFLOW_STATUSES = {
    "raw",
    "boxed",
    "cropped",
    "classified",
    "sampling_selected",
    "qc_in_progress",
    "qc_completed",
    "export_ready",
    "exported",
    "discarded",
}

EXPORT_FORMATS = {
    "labelstudio",
    "coco",
    "yolo",
    "custom",
}

EXPORT_STATUSES = {
    "not_exported",
    "exported",
    "failed",
}

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
}
