"""Stage-1 검출기 학습: 합성본(ROI 편집) + 실제 프레임, 2클래스(normal/stealth)
- 합성: scene50(50) + scene112(94) 병합, 중복(같은 프레임·겹치는 박스) 제거
- 실제 프레임: 차량=normal (의사 라벨), 학습/평가 프레임 분리
"""
import json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

S50 = Path(r"D:\Cache\stealth_w4\scene50")
S112 = Path(r"D:\Cache\stealth_w4\scene112")
ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
OUT = Path(r"D:\Cache\stealth_w4\stage1")
DS = OUT / "ds"
OUT.mkdir(parents=True, exist_ok=True)

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img): cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(p))

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

items = []
for man, root in ((S50 / "scene50_manifest.json", S50), (S112 / "scene112_manifest.json", S112)):
    if not man.exists(): continue
    data = json.loads(man.read_text(encoding="utf-8"))
    for it in data["items"]:
        items.append({"scene": root / "scenes" / f"{it['id']}.png", "frame": it["frame"],
                      "box": it.get("car_box") or it.get("box"), "src": root.name})

# 같은 프레임에서 박스가 겹치는 합성본은 중복으로 제거
kept, seen = [], []
for it in items:
    dup = False
    for (fr, bx) in seen:
        if fr == it["frame"] and iou(bx, it["box"]) > 0.3:
            dup = True; break
    if dup: continue
    kept.append(it); seen.append((it["frame"], it["box"]))
print("composites total:", len(items), "kept:", len(kept))

det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
CLS = (2, 3, 5, 7)
def pseudo(img, skip=None):
    r = det.predict(img, imgsz=960, conf=0.30, device=0, verbose=False, workers=0)[0]
    H, W = img.shape[:2]; out = []
    for b in r.boxes:
        if int(b.cls) not in CLS: continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 16 or (y2-y1) < 12: continue
        if skip and iou((x1, y1, x2, y2), skip) > 0.15: continue
        out.append((x1, y1, x2, y2))
    return out, H, W

if DS.exists(): shutil.rmtree(DS)
def add(split, name, img, lines):
    (DS/split/"images").mkdir(parents=True, exist_ok=True)
    (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    wr(DS/split/"images"/f"{name}.jpg", img)
    (DS/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

n_tr = n_va = 0
for i, it in enumerate(kept):
    img = rd(it["scene"])
    if img is None: continue
    x1, y1, x2, y2 = it["box"]; H, W = img.shape[:2]
    lines = [f"1 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
    for (a, b, c, d) in pseudo(img, skip=(x1, y1, x2, y2))[0]:
        lines.append(f"0 {(a+c)/2/W:.6f} {(b+d)/2/H:.6f} {(c-a)/W:.6f} {(d-b)/H:.6f}")
    split = "val" if i % 5 == 0 else "train"
    add(split, f"comp_{i:03d}", img, lines)
    n_tr += split == "train"; n_va += split == "val"

frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
train_frames = frames[::2]
eval_frames = [f for f in frames if f not in set(train_frames)]
for i, f in enumerate(train_frames):
    img = rd(f)
    bs, H, W = pseudo(img)
    add("train", f"real_{i:03d}", img, [f"0 {(a+c)/2/W:.6f} {(b+d)/2/H:.6f} {(c-a)/W:.6f} {(d-b)/H:.6f}" for (a, b, c, d) in bs])

(DS/"data.yaml").write_text(
    f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: normal_vehicle\n  1: stealth_vehicle\n",
    encoding="utf-8")
print("stage1 dataset: comp train/val", n_tr, n_va, "real train", len(train_frames), "eval", len(eval_frames))

t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(data=str(DS/"data.yaml"), epochs=100, imgsz=640, batch=8, device=0, workers=0,
            project=str(OUT/"runs"), name="s1", exist_ok=True,
            plots=False, amp=False, verbose=False, seed=0, patience=25)
best = OUT/"runs"/"s1"/"weights"/"best.pt"
m = YOLO(str(best))
v = m.val(data=str(DS/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)

res = {"composites_total": len(items), "composites_used": len(kept), "comp_train": n_tr, "comp_val": n_va,
       "real_train": len(train_frames), "eval_frames": len(eval_frames),
       "train_seconds": round(time.time()-t0, 1),
       "val_mAP50": round(float(v.box.map50), 4), "val_mAP50_95": round(float(v.box.map), 4),
       "val_precision": round(float(v.box.mp), 4), "val_recall": round(float(v.box.mr), 4),
       "best": str(best)}

# 홀드아웃 실제 프레임: 클래스별 검출 수
by_conf = {}
for conf in (0.25, 0.40, 0.60):
    n = {0: 0, 1: 0}
    for p in eval_frames:
        r = m.predict(str(p), imgsz=960, conf=conf, device=0, verbose=False, workers=0)[0]
        for b in r.boxes: n[int(b.cls)] += 1
    by_conf[str(conf)] = {"normal": n[0], "stealth": n[1]}
res["eval_by_conf"] = by_conf
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
