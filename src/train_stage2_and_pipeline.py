"""Stage-2 상태 판정 분류기 학습 + 2단계 파이프라인 평가
- 분류 데이터: 실제 ON/OFF 대응쌍(628쌍) + 합성 stealth 크롭 + 실제 정상 크롭
- 평가: Stage-1(검출) → 차량 크롭 → Stage-2(상태) → 소등 후보 수/지연
"""
import csv, json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

PAIR = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\synthesis\paired_night_aug_v1\run_20261103_v1")
PAIR_CSV = PAIR / "AUGMENTED_PAIRS.csv"
S50 = Path(r"D:\Cache\stealth_w4\scene50")
S112 = Path(r"D:\Cache\stealth_w4\scene112")
ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
S1 = Path(r"D:\Cache\stealth_w4\stage1\runs\s1\weights\best.pt")
OUT = Path(r"D:\Cache\stealth_w4\stage2")
DS = OUT / "ds"
OUT.mkdir(parents=True, exist_ok=True)

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img): cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tofile(str(p))

if DS.exists(): shutil.rmtree(DS)
def add(split, label, name, img):
    d = DS / split / label
    d.mkdir(parents=True, exist_ok=True)
    wr(d / f"{name}.jpg", img)

# 1) 실제 ON/OFF 대응쌍
rows = list(csv.DictReader(open(PAIR_CSV, encoding="utf-8-sig")))
n_pair = 0
for r in rows:
    on_f = PAIR / "images" / Path(r["on_image_relpath"]).name
    off_f = PAIR / "images" / Path(r["off_image_relpath"]).name
    if not (on_f.exists() and off_f.exists()): continue
    split = "val" if n_pair % 7 == 0 else "train"
    on, off = rd(on_f), rd(off_f)
    if on is None or off is None: continue
    add(split, "on", f"pair_{n_pair:04d}_on", on)
    add(split, "off", f"pair_{n_pair:04d}_off", off)
    n_pair += 1

# 2) 합성 stealth 크롭 (train 전용)
n_syn = 0
for man, root in ((S50 / "scene50_manifest.json", S50), (S112 / "scene112_manifest.json", S112)):
    if not man.exists(): continue
    data = json.loads(man.read_text(encoding="utf-8"))
    for it in data["items"]:
        img = rd(root / "scenes" / f"{it['id']}.png")
        if img is None: continue
        x1, y1, x2, y2 = it.get("car_box") or it.get("box")
        pad = int(0.12 * max(x2-x1, y2-y1))
        c = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
        if c.size == 0 or min(c.shape[:2]) < 20: continue
        add("train", "off", f"syn_{n_syn:04d}", c)
        n_syn += 1

# 3) 실제 정상 차량 크롭 (train 전용)
det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
train_frames = frames[::2]
eval_frames = [f for f in frames if f not in set(train_frames)]
n_norm = 0
for f in train_frames:
    img = rd(f)
    r = det.predict(img, imgsz=960, conf=0.35, device=0, verbose=False, workers=0)[0]
    for b in r.boxes:
        if int(b.cls) not in (2, 3, 5, 7): continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 20 or (y2-y1) < 15: continue
        pad = int(0.12 * max(x2-x1, y2-y1))
        c = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
        add("train", "on", f"real_{n_norm:04d}", c)
        n_norm += 1

print("pairs", n_pair, "syn off", n_syn, "real on", n_norm)

t0 = time.time()
clf = YOLO("yolov8n-cls.pt")
clf.train(data=str(DS), epochs=30, imgsz=160, batch=64, device=0, workers=0,
          project=str(OUT/"runs"), name="s2", exist_ok=True,
          plots=False, amp=False, verbose=False, seed=0)
best_cls = OUT/"runs"/"s2"/"weights"/"best.pt"
m2 = YOLO(str(best_cls))
v2 = m2.val(data=str(DS), imgsz=160, device=0, plots=False, verbose=False, workers=0)
names2 = m2.names

# 4) 2단계 파이프라인 평가
m1 = YOLO(str(S1))
res = {"stage2_train_seconds": round(time.time()-t0, 1),
       "stage2_top1": round(float(v2.top1), 4), "stage2_top5": round(float(v2.top5), 4),
       "stage2_classes": names2, "pairs": n_pair, "syn_off": n_syn, "real_on": n_norm,
       "stage1_weights": str(S1), "stage2_weights": str(best_cls)}

def run_pipeline(frames, tau1, tau2):
    stats = {"frames": 0, "vehicles": 0, "stealth_candidates": 0, "latency_ms": []}
    for f in frames:
        img = rd(f)
        t = time.time()
        r = m1.predict(img, imgsz=960, conf=0.25, device=0, verbose=False, workers=0)[0]
        boxes = []
        for b in r.boxes:
            if int(b.cls) not in (0, 1): continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (x2-x1) < 18 or (y2-y1) < 14: continue
            boxes.append((x1, y1, x2, y2, int(b.cls), float(b.conf)))
        for (x1, y1, x2, y2, cls, conf) in boxes:
            pad = int(0.10 * max(x2-x1, y2-y1))
            crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
            if crop.size == 0: continue
            pr = m2.predict(crop, imgsz=160, device=0, verbose=False, workers=0)[0].probs
            p_off = float(pr.data[names2.index("off") if isinstance(names2, list) else [k for k, v in names2.items() if v == "off"][0]])
            stats["vehicles"] += 1
            if (cls == 1 and conf >= tau1) or p_off >= tau2:
                stats["stealth_candidates"] += 1
        stats["frames"] += 1
        stats["latency_ms"].append((time.time()-t) * 1000)
    lat = stats["latency_ms"]
    stats["latency_mean_ms"] = round(float(np.mean(lat)), 1) if lat else None
    stats["latency_p90_ms"] = round(float(np.percentile(lat, 90)), 1) if lat else None
    del stats["latency_ms"]
    return stats

res["pipeline"] = {
    "tau1_0.25_tau2_0.5": run_pipeline(eval_frames, 0.25, 0.5),
    "tau1_0.40_tau2_0.6": run_pipeline(eval_frames, 0.40, 0.6),
    "tau1_0.60_tau2_0.8": run_pipeline(eval_frames, 0.60, 0.8),
}
res["eval_frames"] = len(eval_frames)
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
