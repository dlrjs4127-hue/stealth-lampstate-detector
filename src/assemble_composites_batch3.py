"""새 ROI 100장 편집본 → 실제 프레임 복원 + BBOX 라벨 + 다양성 통계"""
import json
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ROI = Path(r"D:\Cache\stealth_w4\roi100b")
OUT = Path(r"D:\Cache\stealth_w4\scene100b")
SC, PV = OUT / "scenes", OUT / "preview_bbox"
for d in (SC, PV):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"): f.unlink()

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img): cv2.imencode(".png", img)[1].tofile(str(p))

meta = {m["roi"]: m for m in json.loads((ROI / "roi_meta.json").read_text(encoding="utf-8"))["items"]}
det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")

TYPES = ["sedan","SUV","minivan","hatchback","pickup","box truck","city bus","sports coupe","taxi","compact car"]
COLORS = ["black","dark navy","dark gray","white","silver","deep red"]
VIEWS = ["rear","rear three-quarter","front","front three-quarter"]
LIGHT = ["very dark","moderate street","tunnel-exit","hazy night"]

def feather(frame, patch, box, fade=8):
    x, y, x2, y2 = box
    h, w = y2 - y, x2 - x
    p = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.ones((h, w), np.float32)
    for i in range(fade):
        a = i / fade
        mask[i, :] *= a; mask[h-1-i, :] *= a; mask[:, i] *= a; mask[:, w-1-i] *= a
    mask = mask[..., None]
    out = frame.astype(np.float32).copy()
    out[y:y2, x:x2] = out[y:y2, x:x2] * (1 - mask) + p.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

made, skipped, items = 0, [], []
for i in range(100):
    name = f"n{i:03d}"
    edit = ROI / f"{name}_edit.png"
    if name not in meta or not edit.exists():
        skipped.append((name, "missing")); continue
    it = meta[name]
    frame = rd(it["frame"])
    patch = rd(edit)
    if frame is None or patch is None:
        skipped.append((name, "unreadable")); continue
    x, y, x2, y2 = it["box"]
    comp = feather(frame, patch, it["box"])

    diff = cv2.absdiff(comp, frame)
    msk = np.ones(frame.shape[:2], np.uint8); msk[y:y2, x:x2] = 0
    outside = int((diff.max(axis=2)[msk > 0] > 0).sum())

    r = det.predict(comp[y:y2, x:x2], imgsz=640, conf=0.20, device=0, verbose=False, workers=0)[0]
    best = None
    for b in r.boxes:
        if int(b.cls) not in (2, 3, 5, 7): continue
        bx1, by1, bx2, by2 = [float(t) for t in b.xyxy[0].tolist()]
        area = (bx2-bx1)*(by2-by1)
        if best is None or area > best[4]:
            best = (bx1, by1, bx2, by2, area)
    if best is None:
        skipped.append((name, "no_car")); continue
    gx1, gy1 = int(x + best[0]), int(y + best[1]); gx2, gy2 = int(x + best[2]), int(y + best[3])
    if (gx2-gx1) < 12 or (gy2-gy1) < 10:
        skipped.append((name, "too_small")); continue

    sid = f"s{i:03d}"
    wr(SC / f"{sid}.png", comp)
    vis = comp.copy()
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (40, 40, 235), 2)
    cv2.putText(vis, "stealth_vehicle", (gx1, max(14, gy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 235), 1, cv2.LINE_AA)
    wr(PV / f"{sid}_bbox.png", vis)
    items.append({
        "id": sid, "roi": name, "camera": it["camera"], "frame": it["frame"],
        "roi_box": it["box"], "car_box": [gx1, gy1, gx2, gy2],
        "width_px": gx2 - gx1, "outside_changed_px": outside,
        "type": TYPES[i % len(TYPES)], "color": COLORS[(i // 10) % len(COLORS)],
        "view": VIEWS[(i // 3) % len(VIEWS)], "light": LIGHT[(i // 7) % len(LIGHT)],
    })
    made += 1

(OUT / "scene100b_manifest.json").write_text(json.dumps({"made": made, "skipped": skipped, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
cameras = len({x["camera"] for x in items})
widths = sorted(x["width_px"] for x in items)
stats = {
    "made": made, "skipped": len(skipped),
    "skip_reasons": {k: sum(1 for _, r2 in skipped if r2 == k) for k in {r2 for _, r2 in skipped}},
    "cameras": cameras, "unique_frames": len({x["frame"] for x in items}),
    "outside_changed_max": max((x["outside_changed_px"] for x in items), default=0),
    "width": {"min": widths[0] if widths else None, "median": widths[len(widths)//2] if widths else None,
              "max": widths[-1] if widths else None},
    "types": {t: sum(1 for x in items if x["type"] == t) for t in TYPES},
    "colors": {c: sum(1 for x in items if x["color"] == c) for c in COLORS},
    "views": {v: sum(1 for x in items if x["view"] == v) for v in VIEWS},
}
(OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
