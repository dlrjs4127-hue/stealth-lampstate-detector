"""ITS 배경 합성본 2클래스 학습: 정상(실제 점등 차량, COCO 의사라벨) vs 스텔스(합성 표적)
- 합성본: 표적=스텔스, 그 외 차량=정상
- 실제 프레임: 차량=정상(의사라벨)
- 평가: 학습에 안 쓴 실제 프레임에서 스텔스 후보/정상 검출 수
"""
import json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu
import ultralytics.utils.checks as chk
uu.ONLINE = False
chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ROOT = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\stealth_autonomous_20261104")
ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
OUT = Path(r"D:\Cache\stealth_w4\itsmix2")
DS = OUT / "ds"
PREV = OUT / "preview"
OUT.mkdir(parents=True, exist_ok=True); PREV.mkdir(parents=True, exist_ok=True)

def imread_u(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img): cv2.imencode(".png", img)[1].tofile(str(p))

def target_box(sd):
    bb = sd / "BBOX.json"
    if bb.exists():
        j = json.loads(bb.read_text(encoding="utf-8"))
        t = j.get("target", {})
        if t.get("bbox_xyxy"): return [int(v) for v in t["bbox_xyxy"]]
        for c in j.get("auto_candidates", []):
            if c.get("target_candidate"): return [int(v) for v in c["bbox_xyxy"]]
    return None

pos = []
for sd in sorted(ROOT.glob("batch_*/*")):
    sc = sd / "SCENE.png"
    if not (sd.is_dir() and sc.exists()): continue
    box = target_box(sd)
    if not box: continue
    if (box[2]-box[0]) < 10 or (box[3]-box[1]) < 10: continue
    pos.append({"scene": sc, "box": box})
print("composites:", len(pos))

frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
train_real = frames[::2]
eval_frames = [f for f in frames if f not in set(train_real)]

det = YOLO("yolov8n.pt")
CLS = (2, 3, 5, 7)
def pseudo_labels(img):
    r = det.predict(img, imgsz=960, conf=0.30, device=0, verbose=False, workers=0)[0]
    H, W = img.shape[:2]
    out = []
    for b in r.boxes:
        if int(b.cls) not in CLS: continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 16 or (y2-y1) < 12: continue
        out.append(((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H))
    return out

if DS.exists(): shutil.rmtree(DS)
def add(split, name, img, lines):
    (DS/split/"images").mkdir(parents=True, exist_ok=True)
    (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    imwrite_u(DS/split/"images"/f"{name}.png", img)
    (DS/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

# 합성본
for i, rec in enumerate(pos):
    split = "val" if i % 5 == 0 else "train"
    img = imread_u(rec["scene"]); H, W = img.shape[:2]
    x1, y1, x2, y2 = rec["box"]
    lines = [f"1 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
    for (cx, cy, bw, bh) in pseudo_labels(img):
        bx1, by1 = (cx-bw/2)*W, (cy-bh/2)*H
        bx2, by2 = (cx+bw/2)*W, (cy+bh/2)*H
        inter = max(0, min(bx2, x2)-max(bx1, x1)) * max(0, min(by2, y2)-max(by1, y1))
        union = (bx2-bx1)*(by2-by1) + (x2-x1)*(y2-y1) - inter
        if union > 0 and inter/union > 0.25:
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    add(split, f"comp_{i:03d}", img, lines)
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 235), 2)
    cv2.imwrite(str(PREV / f"{split}_comp_{i:03d}.jpg"), vis)

# 실제 프레임(정상 차량 의사라벨)
for i, f in enumerate(train_real):
    img = imread_u(f)
    add("train", f"real_{i:03d}", img, [f"0 {c:.6f} {y:.6f} {w:.6f} {h:.6f}" for (c, y, w, h) in pseudo_labels(img)])

(DS/"data.yaml").write_text(
    f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: normal_vehicle\n  1: stealth_vehicle\n",
    encoding="utf-8")

t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(data=str(DS/"data.yaml"), epochs=120, imgsz=640, batch=8, device=0, workers=0,
            project=str(OUT/"runs"), name="itsmix2", exist_ok=True,
            plots=False, amp=False, verbose=False, seed=0, patience=40)
train_s = time.time()-t0
best = OUT/"runs"/"itsmix2"/"weights"/"best.pt"
m = YOLO(str(best))
v = m.val(data=str(DS/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)
res = {"composites": len(pos), "train_real_frames": len(train_real), "eval_frames": len(eval_frames),
       "train_seconds": round(train_s,1),
       "val_mAP50": round(float(v.box.map50),4), "val_mAP50_95": round(float(v.box.map),4),
       "val_precision": round(float(v.box.mp),4), "val_recall": round(float(v.box.mr),4), "best": str(best)}

by_conf = {}
for conf in (0.25, 0.40, 0.60):
    n = {0:0, 1:0}; hit = 0
    for p in eval_frames:
        r = m.predict(str(p), imgsz=960, conf=conf, device=0, verbose=False, workers=0)[0]
        for b in r.boxes: n[int(b.cls)] += 1
        hit += 1 if len(r.boxes) else 0
    by_conf[str(conf)] = {"normal": n[0], "stealth": n[1], "frames_hit": hit,
                          "frame_hit_rate": round(hit/len(eval_frames),4)}

crops, saved = [], 0
for p in eval_frames:
    img = imread_u(p)
    r = m.predict(str(p), imgsz=960, conf=0.25, device=0, verbose=False, workers=0)[0]
    if not len(r.boxes): continue
    for b in r.boxes:
        x1,y1,x2,y2 = [int(t) for t in b.xyxy[0].tolist()]
        cls, sc = int(b.cls), float(b.conf)
        col = (60,60,235) if cls == 1 else (60,190,90)
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        cv2.putText(img, f"{'STEALTH' if cls==1 else 'normal'} {sc:.2f}", (x1, max(14,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if cls == 1:
            crops.append((sc, img[max(0,y1-8):y2+8, max(0,x1-8):x2+8].copy()))
    if saved < 16:
        cv2.imwrite(str(OUT/f"eval_{saved:02d}_{p.stem}.jpg"), img); saved += 1

crops.sort(key=lambda t:-t[0])
tiles=[]
for sc,c in crops[:12]:
    t=cv2.resize(c,(150,150),interpolation=cv2.INTER_NEAREST)
    cv2.putText(t,f"{sc:.2f}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1,cv2.LINE_AA)
    tiles.append(t)
if tiles:
    while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT/"montage_stealth.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))

res.update({"eval_by_conf": by_conf, "stealth_candidates_at_025": len(crops), "annotated": saved})
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
