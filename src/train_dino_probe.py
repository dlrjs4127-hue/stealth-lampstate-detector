"""DINOv2 동결 특징 + Linear Probe — 램프 상태 판정 (ChatGPT A안 최소 실험)

- 데이터: 실제 ON/OFF 대응쌍(628) + 합성 off 크롭. 부모 단위 분리(train/val)
- 특징: DINOv2 ViT-S/14 CLS (384d) @ 224
- 분류기: 능형 회귀(ridge) 기반 선형 판별 → 확률은 온도 스케일링 후 softmax
- 평가: val 정확도/F1, 열화(블러/JPEG35/저조도) val, ECE
- 비교: 동일 val 크롭에 대해 YOLOv8n-cls top1
"""
import csv, json, math
from pathlib import Path

import cv2
import numpy as np
import torch

import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

PAIR = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\synthesis\paired_night_aug_v1\run_20261103_v1")
OUT = Path(r"D:\Cache\stealth_w4\dino_probe")
OUT.mkdir(parents=True, exist_ok=True)
S50 = Path(r"D:\Cache\stealth_w4\scene50")
S112 = Path(r"D:\Cache\stealth_w4\scene112")
Y8CLS = Path(r"D:\Cache\stealth_w4\final2\runs\s2\weights\best.pt")

def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def to_tensor(img, size=224):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))

def degrade(img, kind):
    if kind == "clean":
        return img
    if kind == "blur":
        k = np.zeros((5, 5), np.float32); k[2, :] = 1/5
        return cv2.filter2D(img, -1, k)
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "lowlight":
        return np.clip(img.astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    return img

# ---- 데이터 ----
rows = list(csv.DictReader(open(PAIR / "AUGMENTED_PAIRS.csv", encoding="utf-8-sig")))
samples = []   # (path, label, group)
for i, r in enumerate(rows):
    for col, lab in (("on_image_relpath", 1), ("off_image_relpath", 0)):
        p = PAIR / "images" / Path(r[col]).name
        if p.exists():
            samples.append((p, lab, r["source_parent_id"], i))
syn = 0
for man, root in ((S50 / "scene50_manifest.json", S50), (S112 / "scene112_manifest.json", S112)):
    if not man.exists():
        continue
    for it in json.loads(man.read_text(encoding="utf-8"))["items"]:
        img = rd(root / "scenes" / f"{it['id']}.png")
        if img is None:
            continue
        x1, y1, x2, y2 = it.get("car_box") or it.get("box")
        pad = int(0.12 * max(x2-x1, y2-y1))
        c = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
        if c.size == 0 or min(c.shape[:2]) < 20:
            continue
        dst = OUT / f"syn_{syn:03d}.png"
        cv2.imwrite(str(dst), c)
        samples.append((dst, 0, f"syn{it['id']}", 10**6 + syn))
        syn += 1

groups = sorted({s[2] for s in samples})
rng = np.random.default_rng(0)
val_groups = set(groups[::7])
train = [s for s in samples if s[2] not in val_groups]
val = [s for s in samples if s[2] in val_groups]
print("train", len(train), "val", len(val), "syn", syn)

# ---- DINO 특징 ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).to(device).eval()

@torch.no_grad()
def feats(items, kind="clean"):
    out = []
    for p, lab, g, idx in items:
        img = rd(p)
        if img is None:
            out.append(np.zeros(384, np.float32)); continue
        x = to_tensor(degrade(img, kind)).unsqueeze(0).to(device)
        f = model(x).cpu().numpy().ravel()
        out.append(f.astype(np.float32))
    return np.stack(out)

def report(name, Xtr, ytr, Xva, yva):
    # ridge 선형 판별 (one-vs-rest 2클래스 → 로짓)
    X1 = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    lam = 1.0
    A = X1.T @ X1 + lam * np.eye(X1.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, X1.T @ ytr.astype(np.float32))
    def logits(X):
        return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ w
    s = logits(Xva)
    p = 1 / (1 + np.exp(-s))
    pred = (p >= 0.5).astype(int)
    acc = float((pred == yva).mean())
    tp = int(((pred == 1) & (yva == 1)).sum()); fp = int(((pred == 1) & (yva == 0)).sum())
    fn = int(((pred == 0) & (yva == 1)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    f1 = 2*prec*rec / max(1e-9, prec + rec)
    # ECE (10 bins)
    bins = np.linspace(0, 1, 11); ece = 0.0
    conf = np.where(pred == 1, p, 1 - p)
    for b in range(10):
        m = (conf >= bins[b]) & (conf < bins[b+1])
        if m.sum() > 0:
            ece += abs(conf[m].mean() - (pred[m] == yva[m]).mean()) * m.sum() / len(yva)
    return {"split": name, "n": int(len(yva)), "acc": round(acc, 4),
            "f1_off": round(float(f1), 4), "precision_off": round(float(prec), 4),
            "recall_off": round(float(rec), 4), "ece": round(float(ece), 4)}

ytr = np.array([s[1] for s in train]); yva = np.array([s[1] for s in val])
Xtr = feats(train); Xva = feats(val)
res = {"dino": [report("clean", Xtr, ytr, Xva, yva)],
       "dino_degraded": [], "n_train": len(train), "n_val": len(val), "syn_off": syn}
for kind in ("blur", "jpeg", "lowlight"):
    res["dino_degraded"].append(report(kind, Xtr, ytr, feats(val, kind), yva))

# ---- YOLOv8n-cls 비교(동일 val 크롭) ----
clf = YOLO(str(Y8CLS)); names = clf.names
off_idx = [k for k, v in names.items() if v == "off"][0] if isinstance(names, dict) else names.index("off")
y8 = []
for kind in ("clean", "blur", "jpeg", "lowlight"):
    correct = 0
    for p, lab, g, idx in val:
        img = rd(p)
        if img is None: continue
        pr = clf.predict(degrade(img, kind), imgsz=160, device=0, verbose=False, workers=0)[0].probs
        pred = 0 if int(pr.top1) == off_idx else 1
        correct += int(pred == lab)
    y8.append({"split": kind, "n": len(val), "acc": round(correct / max(1, len(val)), 4)})

res["yolov8n_cls"] = y8
(OUT / "dino_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
