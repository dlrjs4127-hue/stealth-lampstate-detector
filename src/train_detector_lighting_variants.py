"""확장 학습: 합성본 조도 변형 + 실제 OFF 크롭 혼합 → 실제 프레임 성능 확인
- 스텔스: (a) 실제 ITS 배경 합성본의 조도/노출/색온도 변형 5종, (b) 실제 차량 ON→OFF 페어의 OFF 크롭
- 정상: 실제 프레임 차량(의사라벨), 합성본 주변 점등 차량(의사라벨)
- 평가: 학습에 쓰지 않은 실제 프레임 51장 / conf 3단계
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
PAIR = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\synthesis\paired_night_aug_v1\run_20261103_v1\images")
PAIR_CSV = PAIR.parent / "AUGMENTED_PAIRS.csv"
OUT = Path(r"D:\Cache\stealth_w4\exp3")
DS = OUT / "ds"
PREV = OUT / "preview"
OUT.mkdir(parents=True, exist_ok=True); PREV.mkdir(parents=True, exist_ok=True)

def imread_u(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def imwrite_u(p, img, ext=".jpg", q=92):
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    buf.tofile(str(p))

def target_box(sd):
    bb = sd / "BBOX.json"
    if bb.exists():
        j = json.loads(bb.read_text(encoding="utf-8"))
        t = j.get("target", {})
        if t.get("bbox_xyxy"): return [int(v) for v in t["bbox_xyxy"]]
        for c in j.get("auto_candidates", []):
            if c.get("target_candidate"): return [int(v) for v in c["bbox_xyxy"]]
    return None

# ---------- 조도/노출/색온도 변형 ----------
def profile(img, name, rng):
    f = img.astype(np.float32)
    if name == "base":
        out = f
    elif name == "dark":
        out = f * 0.72
    elif name == "gamma_low":
        out = 255.0 * np.power(np.clip(f / 255.0, 0, 1), 1.45)
    elif name == "cool":
        out = f * np.array([1.12, 1.0, 0.86], dtype=np.float32)
    elif name == "warm":
        out = f * np.array([0.88, 1.0, 1.16], dtype=np.float32)
    elif name == "noisy":
        out = f * 0.85 + rng.normal(0, 7.0, f.shape)
    elif name == "blur":
        out = cv2.GaussianBlur(f, (3, 3), 0.6)
    else:
        out = f
    return np.clip(out, 0, 255).astype(np.uint8)

PROFILES = ["base", "dark", "gamma_low", "cool", "warm", "noisy", "blur"]

# ---------- 데이터 수집 ----------
pos = []
for sd in sorted(ROOT.glob("batch_*/*")):
    sc = sd / "SCENE.png"
    if not (sd.is_dir() and sc.exists()): continue
    box = target_box(sd)
    if not box or (box[2]-box[0]) < 10 or (box[3]-box[1]) < 10: continue
    pos.append({"scene": sc, "box": box})
print("composites:", len(pos))

frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
train_real = frames[::2]
eval_frames = [f for f in frames if f not in set(train_real)]

# 실제 OFF 페어 크롭
off_crops = []
if PAIR_CSV.exists():
    import csv
    for r in csv.DictReader(open(PAIR_CSV, encoding="utf-8-sig")):
        f = PAIR / Path(r["off_image_relpath"]).name
        if f.exists(): off_crops.append(f)
print("real OFF crops:", len(off_crops))

det = YOLO("yolov8n.pt")
CLSIDS = (2, 3, 5, 7)
def pseudo(img):
    r = det.predict(img, imgsz=960, conf=0.30, device=0, verbose=False, workers=0)[0]
    H, W = img.shape[:2]; out = []
    for b in r.boxes:
        if int(b.cls) not in CLSIDS: continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 16 or (y2-y1) < 12: continue
        out.append(((x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H, x1, y1, x2, y2))
    return out

if DS.exists(): shutil.rmtree(DS)
def add(split, name, img, lines):
    (DS/split/"images").mkdir(parents=True, exist_ok=True)
    (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    imwrite_u(DS/split/"images"/f"{name}.jpg", img)
    (DS/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

rng = np.random.default_rng(0)
n_comp = n_variant = n_off = 0

# 1) 합성본 + 조도 변형 (검증은 원본만)
for i, rec in enumerate(pos):
    img = imread_u(rec["scene"]); H, W = img.shape[:2]
    x1, y1, x2, y2 = rec["box"]
    others = [p for p in pseudo(img) if not (
        max(0, min(p[6], x2)-max(p[4], x1)) * max(0, min(p[7], y2)-max(p[5], y1)) /
        max(1e-6, (p[6]-p[4])*(p[7]-p[5]) + (x2-x1)*(y2-y1) -
           max(0, min(p[6], x2)-max(p[4], x1)) * max(0, min(p[7], y2)-max(p[5], y1))) > 0.25)]
    base_lines = [f"1 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
    base_lines += [f"0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f}" for p in others]
    if i % 5 == 0:
        add("val", f"comp_{i:03d}", img, base_lines); n_comp += 1
    else:
        for prof in PROFILES:
            out = profile(img, prof, rng)
            add("train", f"comp_{i:03d}_{prof}", out, base_lines)
            n_variant += 1
        n_comp += 1
    if i < 6:
        vis = img.copy(); cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 235), 2)
        cv2.imwrite(str(PREV / f"comp_{i:03d}.jpg"), vis)

# 2) 실제 OFF 크롭 (스텔스 양성)
for j, f in enumerate(off_crops):
    img = imread_u(f)
    if img is None: continue
    H, W = img.shape[:2]
    pad = 0.04
    lines = [f"1 0.5 0.5 {1-2*pad:.6f} {1-2*pad:.6f}"]
    split = "val" if j % 7 == 0 else "train"
    add(split, f"offcrop_{j:03d}", img, lines)
    n_off += 1

# 3) 실제 프레임 (정상 차량 의사라벨)
for i, f in enumerate(train_real):
    img = imread_u(f)
    lines = [f"0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f}" for p in pseudo(img)]
    add("train", f"real_{i:03d}", img, lines)

(DS/"data.yaml").write_text(
    f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: normal_vehicle\n  1: stealth_vehicle\n",
    encoding="utf-8")
counts = {s: len(list((DS/s/"images").glob("*.jpg"))) for s in ("train", "val")}
print("dataset", counts, "variants", n_variant, "off crops", n_off)

# ---------- 학습 ----------
t0 = time.time()
model = YOLO("yolov8n.pt")
model.train(data=str(DS/"data.yaml"), epochs=100, imgsz=640, batch=16, device=0, workers=0,
            project=str(OUT/"runs"), name="exp3", exist_ok=True,
            plots=False, amp=False, verbose=False, seed=0, patience=30)
train_s = time.time()-t0
best = OUT/"runs"/"exp3"/"weights"/"best.pt"
m = YOLO(str(best))
v = m.val(data=str(DS/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)
res = {"composites": len(pos), "lighting_variants": n_variant, "real_off_crops": n_off,
       "train_images": counts["train"], "val_images": counts["val"],
       "train_seconds": round(train_s,1),
       "val_mAP50": round(float(v.box.map50),4), "val_mAP50_95": round(float(v.box.map),4),
       "val_precision": round(float(v.box.mp),4), "val_recall": round(float(v.box.mr),4),
       "eval_frames": len(eval_frames)}

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

res.update({"eval_by_conf": by_conf, "stealth_candidates_at_025": len(crops), "annotated": saved, "best": str(best)})
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
