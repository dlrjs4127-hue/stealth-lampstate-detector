"""새 야간 수집분(11/21~22)에서 ROI 100개 추출 — 크기·위치 다양화"""
import json, shutil
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ROOTS = [Path(r"D:\Cache\its_night\2026-11-21_2000"),
         Path(r"D:\Cache\its_night\2026-11-21_2230"),
         Path(r"D:\Cache\its_night\2026-11-22_0130")]
OUT = Path(r"D:\Cache\stealth_w4\roi100b")
TMP = Path(r"C:\Users\miso7\.aside\u\0\sessions\2026-11-19_ouzbWEzJKDpmO0CC\tmp\roi_in100b")
for d in (OUT, TMP):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        if f.is_file(): f.unlink()

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
frames = []
for r in ROOTS:
    frames += [p for p in sorted(r.rglob("frame_*.png"))]
print("frames:", len(frames))

rng = np.random.default_rng(31)
SIZES = [(200, 150), (180, 135), (160, 120), (220, 165)]
meta, n = [], 0
for fp in frames:
    if n >= 100: break
    img = rd(fp)
    if img is None: continue
    H, W = img.shape[:2]
    r = det.predict(img, imgsz=960, conf=0.25, device=0, verbose=False, workers=0)[0]
    bs = [[int(t) for t in b.xyxy[0].tolist()] for b in r.boxes if int(b.cls) in (2, 3, 5, 7)]
    for _ in range(24):
        rw, rh = SIZES[int(rng.integers(0, len(SIZES)))]
        x = int(rng.uniform(0.08, 0.92) * W - rw/2)
        y = int(rng.uniform(0.50, 0.90) * H - rh/2)
        x = max(0, min(W - rw - 1, x)); y = max(0, min(H - rh - 1, y))
        rb = [x, y, x + rw, y + rh]
        # 기존 차량과 겹치면 회피
        if any(not (rb[2] < b[0] or rb[0] > b[2] or rb[3] < b[1] or rb[1] > b[3]) for b in bs):
            continue
        patch = img[y:y+rh, x:x+rw]
        if patch.mean() > 110 or patch.std() < 8:
            continue
        name = f"n{n:03d}"
        cv2.imwrite(str(OUT / f"{name}.png"), patch)
        shutil.copy2(OUT / f"{name}.png", TMP / f"{name}.png")
        meta.append({"roi": name, "frame": str(fp), "camera": fp.parent.name, "box": rb,
                     "size": [rw, rh], "mean": round(float(patch.mean()), 1)})
        n += 1
        break

(OUT / "roi_meta.json").write_text(json.dumps({"count": len(meta), "items": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"count": len(meta), "cameras": len({m['camera'] for m in meta}),
                  "sizes": {str(m['size']): sum(1 for x in meta if x['size'] == m['size']) for m in meta}},
                 ensure_ascii=False)[:400])
