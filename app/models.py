"""모델 슬롯 — 검출기/상태판정기를 이름으로 교체 가능하게 로드한다.

사용:
    from models import load_detector, load_state_classifier
    det = load_detector("yolo26n")      # yolov8n | yolo26n | yolo26s | rtdetr
    clf = load_state_classifier("yolov8n-cls")
"""
from pathlib import Path

import ultralytics.utils as uu
import ultralytics.utils.checks as chk

uu.ONLINE = False
chk.check_font = lambda *a, **k: None

from ultralytics import YOLO

ROOT = Path(r"D:\Desktop\Aside\독립심화\stealth-lampstate-detector")
WEIGHTS = {
    "yolov8n": r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt",
    "yolo26n": r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\yolo26n.pt",
    "yolo26s": r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\yolo26s.pt",
}


def load_detector(name: str, trained: bool = True):
    """trained=True 면 우리가 학습한 검출기, False 면 COCO 사전학습 모델(비교용)."""
    import os
    if trained:
        p = Path(os.environ.get("STAGE1_WEIGHTS", r"D:\Cache\stealth_w4\final2\runs\s1\weights\best.pt"))
        if p.exists():
            return YOLO(str(p)), f"stage1_trained:{p.name}"
    if name not in WEIGHTS:
        raise ValueError(f"unknown detector: {name}")
    p = Path(WEIGHTS[name])
    if not p.exists():
        raise FileNotFoundError(f"weights not found: {p}")
    return YOLO(str(p)), name


def load_state_classifier(path: str | None = None):
    p = Path(path or r"D:\Cache\stealth_w4\final2\runs\s2\weights\best.pt")
    if not p.exists():
        p = Path(r"D:\Cache\stealth_w4\stage2\runs\s2\weights\best.pt")
    return YOLO(str(p))

