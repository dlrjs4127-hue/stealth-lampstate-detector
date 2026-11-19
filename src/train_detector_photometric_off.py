"""실제 프레임 안에서 소등 차량을 광도 변환으로 만들어(실제 배경·실제 차량 형태) 2클래스 학습
- 각 실제 프레임에서 차량을 검출하고, 일부 차량만 램프 소등처럼 광도 변환 → stealth 라벨
- 나머지 차량은 normal 라벨, 변환하지 않은 원본 프레임도 함께 사용
- 평가: 학습에 쓰지 않은 실제 프레임 51장
"""
import json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu
import ultralytics.utils.checks as chk
uu.ONLINE = False
chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
COMP = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\stealth_autonomous_20261104")
OUT = Path(r"D:\Cache\stealth_w4\exp4")
DS = OUT / "ds"
OUT.mkdir(parents=True, exist_ok=True)

def imread_u(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def save_jpg(p, img, q=92):
    cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tofile(str(p))

def lamps_off(img, box, seed=0):
    """박스 안 램프 광원을 감쇠해 소등처럼 만든다 (픽셀 단위 광도 변환)"""
    out = img.copy()
    x1, y1, x2, y2 = box
    roi = out[y1:y2, x1:x2].astype(np.float32)
    hsv = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    # 밝은 픽셀(램프/반사)일수록 강하게 감쇠
    w = np.clip((v - 120.0) / 135.0, 0, 1)[..., None]
    roi = roi * (1.0 - 0.85 * w)
    # 전체 노출도 낮춘다
    roi = roi * 0.82
    rng = np.random.default_rng(seed)
    roi = roi + rng.normal(0, 2.0, roi.shape)
    out[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)
    return out

det = YOLO("yolov8n.pt")
CLSIDS = (2, 3, 5, 7)
def boxes(img):
    r = det.predict(img, imgsz=960, conf=0.30, device=0, verbose=False, workers=0)[0]
    out = []
    H, W = img.shape[:2]
    for b in r.boxes:
        if int(b.cls) not in CLSIDS: continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 18 or (y2-y1) < 14: continue
        out.append((x1, y1, x2, y2))
    return out, H, W

frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
train_frames = frames[::2]
eval_frames = [f for f in frames if f not in set(train_frames)]

if DS.exists(): shutil.rmtree(DS)
def add(split, name, img, lines):
    (DS/split/"images").mkdir(parents=True, exist_ok=True)
    (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    save_jpg(DS/split/"images"/f"{name}.jpg", img)
    (DS/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

def yolo_lines(items, H, W):
    return [f"{c} {((x1+x2)/2)/W:.6f} {((y1+y2)/2)/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}" for (c,x1,y1,x2,y2) in items]

rng = np.random.default_rng(1)
n_raw = n_off = n_comp = 0
for i, f in enumerate(train_frames):
    img = imread_u(f)
    bs, H, W = boxes(img)
    if not bs: continue
    # 원본: 전부 정상
    add("train", f"real_{i:03d}", img, yolo_lines([(0,*b) for b in bs], H, W)); n_raw += 1
    # 소등 변환: 2~3대를 stealth, 나머지 정상
    k = min(len(bs), max(2, len(bs)//2))
    idx = rng.choice(len(bs), size=k, replace=False)
    out = img.copy()
    for j in idx.tolist():
        out = lamps_off(out, bs[j], seed=j)
    items = [(1 if j in set(idx.tolist()) else 0, *b) for j, b in enumerate(bs)]
    add("train", f"off_{i:03d}", out, yolo_lines(items, H, W)); n_off += 1

# 합성본도 소량 포함(표적 라벨 유지)
def target_box(sd):
    bb = sd / "BBOX.json"
    if not bb.exists(): return None
    j = json.loads(bb.read_text(encoding="utf-8"))
    t = j.get("target", {})
    if t.get("bbox_xyxy"): return [int(v) for v in t["bbox_xyxy"]]
    for c in j.get("auto_candidates", []):
        if c.get("target_candidate"): return [int(v) for v in c["bbox_xyxy"]]
    return None

for i, sd in enumerate(sorted(COMP.glob("batch_*/*"))):
    sc = sd / "SCENE.png"
    if not (sd.is_dir() and sc.exists()): continue
    tb = target_box(sd)
    if not tb or (tb[2]-tb[0]) < 10: continue
    img = imread_u(sc); H, W = img.shape[:2]
    bs, _, _ = boxes(img)
    items = [(1, *tb)] + [(0, *b) for b in bs]
    split = "val" if i % 6 == 0 else "train"
    add(split, f"comp_{i:03d}", img, yolo_lines(items, H, W)); n_comp += 1

(DS/"data.yaml").write_text(
    f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: normal_vehicle\n  1: stealth_vehicle\n",
    encoding="utf-8")
counts = {s: len(list((DS/s/"images").glob("*.jpg"))) for s in ("train","val")}
print("dataset", counts, "raw", n_raw, "off", n_off, "comp", n_comp)

t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(data=str(DS/"data.yaml"), epochs=70, imgsz=640, batch=16, device=0, workers=0,
            project=str(OUT/"runs"), name="exp4", exist_ok=True,
            plots=False, amp=False, verbose=False, seed=0, patience=20)
train_s = time.time()-t0
best = OUT/"runs"/"exp4"/"weights"/"best.pt"
m = YOLO(str(best))
v = m.val(data=str(DS/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)
res = {"train_images": counts["train"], "val_images": counts["val"],
       "raw_frames": n_raw, "off_frames": n_off, "composites": n_comp,
       "train_seconds": round(train_s,1),
       "val_mAP50": round(float(v.box.map50),4), "val_mAP50_95": round(float(v.box.map),4),
       "val_precision": round(float(v.box.mp),4), "val_recall": round(float(v.box.mr),4),
       "eval_frames": len(eval_frames), "best": str(best)}

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
