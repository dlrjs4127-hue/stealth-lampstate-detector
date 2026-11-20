"""누수 없는 분할로 재구성: 원본 프레임 ID 기준 3분할(학습/검증/테스트)
- 고유 프레임 51개(v1/v2 중복 제거) → train 26 / val 10 / test 15
- 합성본은 해당 프레임이 속한 split으로만 배치(test 프레임 합성본은 제외)
- 실제 학습 음성 = train 프레임, 테스트 = test 프레임(완전 분리)
- Stage-1/Stage-2 학습 + 2단계 파이프라인 평가(클래스별 AP 포함)
"""
import csv, json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
S50 = Path(r"D:\Cache\stealth_w4\scene50")
S112 = Path(r"D:\Cache\stealth_w4\scene112")
PAIR = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\synthesis\paired_night_aug_v1\run_20261103_v1")
OUT = Path(r"D:\Cache\stealth_w4\final")
OUT.mkdir(parents=True, exist_ok=True)

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img, q=92): cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tofile(str(p))

# ---- 1. 고유 프레임 목록 ----
all_frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
by_name = {}
for p in all_frames:
    by_name.setdefault(p.name, []).append(p)
frames = {name: paths[0] for name, paths in by_name.items()}
names = sorted(frames)
print("unique frames:", len(names), "(raw files:", len(all_frames), ")")

rng = np.random.default_rng(2026)
perm = rng.permutation(len(names))
train_f = {names[i] for i in perm[:26]}
val_f = {names[i] for i in perm[26:36]}
test_f = {names[i] for i in perm[36:]}
assert not (train_f & val_f) and not (train_f & test_f) and not (val_f & test_f)

# ---- 2. 합성본 매니페스트 병합 + 프레임 기준 배치 ----
comps = []
for man, root in ((S50 / "scene50_manifest.json", S50), (S112 / "scene112_manifest.json", S112)):
    if not man.exists(): continue
    for it in json.loads(man.read_text(encoding="utf-8"))["items"]:
        comps.append({"scene": root / "scenes" / f"{it['id']}.png", "frame": it["frame"],
                      "box": it.get("car_box") or it.get("box")})
comps = [c for c in comps if c["frame"] in train_f or c["frame"] in val_f]   # test 프레임 합성본 제외
print("composites usable (train/val frames only):", len(comps))

det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
CLS = (2, 3, 5, 7)
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

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

# ---- 3. Stage-1 데이터셋 ----
DS1 = OUT / "ds1"
if DS1.exists(): shutil.rmtree(DS1)
def add1(split, name, img, lines):
    (DS1/split/"images").mkdir(parents=True, exist_ok=True)
    (DS1/split/"labels").mkdir(parents=True, exist_ok=True)
    wr(DS1/split/"images"/f"{name}.jpg", img)
    (DS1/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

cnt1 = {"train": {"comp": 0, "real": 0}, "val": {"comp": 0, "real": 0}}
for i, c in enumerate(comps):
    img = rd(c["scene"])
    if img is None: continue
    x1, y1, x2, y2 = c["box"]; H, W = img.shape[:2]
    lines = [f"1 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
    for (a, b, cc, d) in pseudo(img, skip=(x1, y1, x2, y2))[0]:
        lines.append(f"0 {(a+cc)/2/W:.6f} {(b+d)/2/H:.6f} {(cc-a)/W:.6f} {(d-b)/H:.6f}")
    split = "val" if c["frame"] in val_f else "train"
    add1(split, f"comp_{i:03d}", img, lines); cnt1[split]["comp"] += 1

for i, name in enumerate(sorted(train_f)):
    img = rd(frames[name])
    bs, H, W = pseudo(img)
    add1("train", f"real_{i:03d}", img, [f"0 {(a+c2)/2/W:.6f} {(b+d)/2/H:.6f} {(c2-a)/W:.6f} {(d-b)/H:.6f}" for (a, b, c2, d) in bs])
    cnt1["train"]["real"] += 1

(DS1/"data.yaml").write_text(f"path: {DS1.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: normal_vehicle\n  1: stealth_vehicle\n", encoding="utf-8")
print("stage1 counts:", cnt1)

# ---- 4. Stage-1 학습 ----
t0 = time.time()
m1 = YOLO("yolov8n.pt")
m1.train(data=str(DS1/"data.yaml"), epochs=100, imgsz=640, batch=8, device=0, workers=0,
         project=str(OUT/"runs"), name="s1", exist_ok=True,
         plots=False, amp=False, verbose=False, seed=0, patience=25)
best1 = OUT/"runs"/"s1"/"weights"/"best.pt"
v1 = YOLO(str(best1)).val(data=str(DS1/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)

# ---- 5. Stage-2 데이터셋(프레임 분리 유지) ----
DS2 = OUT / "ds2"
if DS2.exists(): shutil.rmtree(DS2)
def add2(split, label, name, img):
    d = DS2/split/label; d.mkdir(parents=True, exist_ok=True)
    wr(d/f"{name}.jpg", img, 95)

n_pair = n_syn = n_real = 0
rows = list(csv.DictReader(open(PAIR/"AUGMENTED_PAIRS.csv", encoding="utf-8-sig")))
for r in rows:
    on_f = PAIR/"images"/Path(r["on_image_relpath"]).name
    off_f = PAIR/"images"/Path(r["off_image_relpath"]).name
    if not (on_f.exists() and off_f.exists()): continue
    on, off = rd(on_f), rd(off_f)
    if on is None or off is None: continue
    split = "val" if n_pair % 7 == 0 else "train"
    add2(split, "on", f"pair_{n_pair:04d}_on", on); add2(split, "off", f"pair_{n_pair:04d}_off", off)
    n_pair += 1
for c in comps:
    img = rd(c["scene"])
    if img is None: continue
    x1, y1, x2, y2 = c["box"]; pad = int(0.12*max(x2-x1, y2-y1))
    crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
    if crop.size == 0 or min(crop.shape[:2]) < 20: continue
    add2("train", "off", f"syn_{n_syn:04d}", crop); n_syn += 1
for name in sorted(train_f):
    img = rd(frames[name])
    r = det.predict(img, imgsz=960, conf=0.35, device=0, verbose=False, workers=0)[0]
    for b in r.boxes:
        if int(b.cls) not in CLS: continue
        x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
        if (x2-x1) < 20 or (y2-y1) < 15: continue
        pad = int(0.12*max(x2-x1, y2-y1))
        crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
        add2("train", "on", f"real_{n_real:04d}", crop); n_real += 1

m2 = YOLO("yolov8n-cls.pt")
m2.train(data=str(DS2), epochs=30, imgsz=160, batch=64, device=0, workers=0,
         project=str(OUT/"runs"), name="s2", exist_ok=True,
         plots=False, amp=False, verbose=False, seed=0)
best2 = OUT/"runs"/"s2"/"weights"/"best.pt"
v2 = YOLO(str(best2)).val(data=str(DS2), imgsz=160, device=0, plots=False, verbose=False, workers=0)

# ---- 6. 테스트 프레임(완전 분리)에서 2단계 파이프라인 ----
net1, net2 = YOLO(str(best1)), YOLO(str(best2))
names2 = net2.names
off_idx = [k for k, v in names2.items() if v == "off"][0] if isinstance(names2, dict) else names2.index("off")

def run_pipeline(frame_names, tau1, tau2):
    veh = cand = 0; lat = []
    for nm in frame_names:
        img = rd(frames[nm]); t = time.time()
        r = net1.predict(img, imgsz=960, conf=0.25, device=0, verbose=False, workers=0)[0]
        for b in r.boxes:
            if int(b.cls) not in (0, 1): continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (x2-x1) < 18 or (y2-y1) < 14: continue
            pad = int(0.10*max(x2-x1, y2-y1))
            crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
            if crop.size == 0: continue
            pr = net2.predict(crop, imgsz=160, device=0, verbose=False, workers=0)[0].probs
            p_off = float(pr.data[off_idx])
            veh += 1
            if (int(b.cls) == 1 and float(b.conf) >= tau1) or p_off >= tau2:
                cand += 1
        lat.append((time.time()-t)*1000)
    return {"frames": len(frame_names), "vehicles": veh, "stealth_candidates": cand,
            "latency_mean_ms": round(float(np.mean(lat)), 1), "latency_p90_ms": round(float(np.percentile(lat, 90)), 1)}

maps = list(map(float, v1.box.maps)) if hasattr(v1.box, "maps") else []
res = {
    "split": {"train_frames": len(train_f), "val_frames": len(val_f), "test_frames": len(test_f),
              "overlap_train_val": len(train_f & val_f), "overlap_train_test": len(train_f & test_f),
              "overlap_val_test": len(val_f & test_f)},
    "stage1": {"comp_train": cnt1["train"]["comp"], "comp_val": cnt1["val"]["comp"],
               "real_train": cnt1["train"]["real"], "train_seconds": round(time.time()-t0, 1),
               "mAP50": round(float(v1.box.map50), 4), "mAP50_95": round(float(v1.box.map), 4),
               "precision": round(float(v1.box.mp), 4), "recall": round(float(v1.box.mr), 4),
               "per_class_map50": {"normal": round(maps[0], 4) if len(maps) > 0 else None,
                                   "stealth": round(maps[1], 4) if len(maps) > 1 else None}},
    "stage2": {"pairs": n_pair, "syn_off": n_syn, "real_on": n_real,
               "top1": round(float(v2.top1), 4), "top5": round(float(v2.top5), 4)},
    "pipeline_on_test_frames": {
        "tau_0.25_0.5": run_pipeline(sorted(test_f), 0.25, 0.5),
        "tau_0.40_0.6": run_pipeline(sorted(test_f), 0.40, 0.6),
    },
    "weights": {"stage1": str(best1), "stage2": str(best2)},
}
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
