"""생성이미지 폴더 기반 검출 데이터셋 구축
- 입력: D:\\Desktop\\40_개인_기타\\생성이미지 (ChatGPT 이미지 생성, 49 unique)
- 차량 후보: YOLOv8n COCO 검출
- 분류 규칙(파일럿): 박스 안 밝은 픽셀(V>200) 비율이 1.5% 이상이면 normal(램프 점등), 미만이면 stealth(소등)
- 출력: D:\\Cache\\stealth_w4\\gen_ds/{train,val}/{images,labels} + data.yaml + label_stats.json
"""
import hashlib, json, random
from pathlib import Path
import numpy as np
import cv2
import ultralytics.utils as uu
import ultralytics.utils.checks as chk
uu.ONLINE = False
chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

SRC = Path(r"D:\Desktop\40_개인_기타\생성이미지")
OUT = Path(r"D:\Cache\stealth_w4\gen_ds")
VEH = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")

def imread_unicode(p):
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

files, seen = [], set()
for f in sorted(SRC.glob("*.png")):
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    files.append(f)
print("unique images:", len(files))

det = YOLO(str(VEH))
if OUT.exists():
    import shutil; shutil.rmtree(OUT)

imgs_meta = []
for f in files:
    img = imread_unicode(f)
    if img is None:
        continue
    r = det.predict(img, imgsz=1280, conf=0.20, device=0, verbose=False, workers=0)[0]
    labels = []
    H, W = img.shape[:2]
    for b in r.boxes:
        if int(b.cls) not in (2, 3, 5, 7):
            continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2 - x1) < 20 or (y2 - y1) < 14:
            continue
        crop = img[y1:y2, x1:x2]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bright = float((g > 200).mean())
        cls = 0 if bright >= 0.015 else 1  # 0 normal, 1 stealth
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        labels.append((cls, cx, cy, bw, bh, bright))
    imgs_meta.append({"file": f, "shape": (H, W), "labels": labels})

# 장면 단위 분할(해시) 80/20
random.seed(0)
idx = list(range(len(imgs_meta)))
random.shuffle(idx)
n_val = max(2, len(idx) // 5)
val_idx = set(idx[:n_val])

stats = {"images": len(imgs_meta), "train": 0, "val": 0, "normal_boxes": 0, "stealth_boxes": 0, "images_without_boxes": 0}
for i, m in enumerate(imgs_meta):
    split = "val" if i in val_idx else "train"
    d_img = OUT / split / "images"; d_lbl = OUT / split / "labels"
    d_img.mkdir(parents=True, exist_ok=True); d_lbl.mkdir(parents=True, exist_ok=True)
    img = imread_unicode(m["file"])
    name = f"gen_{i:03d}"
    cv2.imencode(".jpg", img)[1].tofile(str(d_img / f"{name}.jpg"))
    lines = []
    for (cls, cx, cy, bw, bh, bright) in m["labels"]:
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stats["normal_boxes" if cls == 0 else "stealth_boxes"] += 1
    (d_lbl / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    stats[split] += 1
    if not lines:
        stats["images_without_boxes"] += 1

yaml = f"""path: {OUT.as_posix()}
train: train/images
val: val/images
names:
  0: normal_vehicle
  1: stealth_vehicle
"""
(OUT / "data.yaml").write_text(yaml, encoding="utf-8")
(OUT / "label_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
