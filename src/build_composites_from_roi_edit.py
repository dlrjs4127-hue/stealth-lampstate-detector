"""112개 편집 ROI를 실제 프레임에 복원 → 합성본 112장 + YOLO 라벨 + BBOX 프리뷰"""
import json
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
ROI = Path(r"D:\Cache\stealth_w4\roi120")
OUT = Path(r"D:\Cache\stealth_w4\scene112")
SC, PV = OUT / "scenes", OUT / "preview_bbox"
for d in (SC, PV):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"): f.unlink()

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img): cv2.imencode(".png", img)[1].tofile(str(p))

meta = {m["roi"]: m for m in json.loads((ROI / "roi_meta.json").read_text(encoding="utf-8"))["items"]}
det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")

def feather_paste(frame, patch, box, fade=8):
    x, y, x2, y2 = box
    h, w = y2 - y, x2 - x
    p = cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.ones((h, w), dtype=np.float32)
    f = max(1, fade)
    for i in range(f):
        a = i / f
        mask[i, :] *= a; mask[h-1-i, :] *= a; mask[:, i] *= a; mask[:, w-1-i] *= a
    mask = mask[..., None]
    out = frame.copy().astype(np.float32)
    out[y:y2, x:x2] = out[y:y2, x:x2] * (1 - mask) + p.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

edits = sorted(ROI.glob("*_edit.png"))
made, skipped, items = 0, [], []
for ef in edits:
    stem = ef.stem.replace("_edit", "")          # roi_07 or roi_07_v2
    base = stem.split("_v2")[0]
    if base not in meta:
        skipped.append((stem, "no_meta")); continue
    it = meta[base]
    fr = next((p for p in ITS.rglob(it["frame"])), None)
    if fr is None:
        skipped.append((stem, "no_frame")); continue
    frame, patch = rd(fr), rd(ef)
    x, y, x2, y2 = it["box"]
    comp = feather_paste(frame, patch, it["box"])

    diff = cv2.absdiff(comp, frame)
    mask_out = np.ones(frame.shape[:2], dtype=np.uint8); mask_out[y:y2, x:x2] = 0
    outside_changed = int((diff.max(axis=2)[mask_out > 0] > 0).sum())

    r = det.predict(comp[y:y2, x:x2], imgsz=640, conf=0.20, device=0, verbose=False, workers=0)[0]
    best = None
    for b in r.boxes:
        if int(b.cls) not in (2, 3, 5, 7): continue
        bx1, by1, bx2, by2 = [float(t) for t in b.xyxy[0].tolist()]
        area = (bx2-bx1)*(by2-by1)
        if best is None or area > best[4]:
            best = (bx1, by1, bx2, by2, area)
    if best is None:
        skipped.append((stem, "no_car")); continue
    gx1, gy1 = int(x + best[0]), int(y + best[1]); gx2, gy2 = int(x + best[2]), int(y + best[3])
    if (gx2-gx1) < 12 or (gy2-gy1) < 10:
        skipped.append((stem, "too_small")); continue

    name = f"stealth_{made:03d}"
    wr(SC / f"{name}.png", comp)
    vis = comp.copy()
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (40, 40, 235), 2)
    cv2.putText(vis, "stealth_vehicle", (gx1, max(14, gy1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 235), 1, cv2.LINE_AA)
    wr(PV / f"{name}_bbox.png", vis)
    items.append({"id": name, "source_edit": ef.name, "roi": base, "variant": "_v2" in stem,
                  "frame": it["frame"], "roi_box": it["box"], "car_box": [gx1, gy1, gx2, gy2],
                  "outside_changed_px": outside_changed})
    made += 1

(OUT / "scene112_manifest.json").write_text(json.dumps({"made": made, "skipped": skipped, "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({"made": made, "skipped": len(skipped),
                  "reasons": {k: sum(1 for _, r in skipped if r == k) for k in {r for _, r in skipped}},
                  "variants": sum(1 for i in items if i["variant"]),
                  "outside_changed_max": max((i["outside_changed_px"] for i in items), default=0),
                  "unique_frames": len({i["frame"] for i in items})}, ensure_ascii=False, indent=2))
