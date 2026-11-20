"""ON/OFF 균형 대응쌍 생성 (계획: 36조건 × 3대 = 108쌍)

원리
- 기존 ROI 편집 합성본(OFF)에서 시작해, **같은 배경·같은 차량·같은 위치**에
  램프 광원만 물리적으로 합성해 ON 버전을 만든다. → 램프 상태 외 차이는 0에 가깝다.
- 조건: 거리(3) × 조도(3) × 시점(4) = 36셀 × 3쌍 = 108쌍
- 저장: pairs/<id>_off.png, pairs/<id>_on.png + pairs_manifest.json
"""
import json, math
from pathlib import Path

import cv2
import numpy as np

S50 = Path(r"D:\Cache\stealth_w4\scene50")
S112 = Path(r"D:\Cache\stealth_w4\scene112")
SMORE = Path(r"D:\Cache\stealth_w4\scene_more")
OUT = Path(r"D:\Cache\stealth_w4\pairs108")
PAIR = OUT / "pairs"
PAIR.mkdir(parents=True, exist_ok=True)
for f in PAIR.glob("*"):
    f.unlink()

def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

items = []
for man, root in ((S50 / "scene50_manifest.json", S50), (S112 / "scene112_manifest.json", S112),
                  (SMORE / "scene_more_manifest.json", SMORE)):
    if not man.exists():
        continue
    for it in json.loads(man.read_text(encoding="utf-8"))["items"]:
        box = it.get("car_box") or it.get("box")
        items.append({"scene": root / "scenes" / f"{it['id']}.png", "frame": it["frame"],
                      "box": box, "src": root.name})
print("composites:", len(items))

def dist_bin(w):
    return 0 if w >= 80 else (1 if w >= 40 else 2)      # 근/중/원

def illum_bin(mean):
    return 0 if mean < 55 else (1 if mean < 85 else 2)  # 저/중/고

def view_bin(box):
    w, h = box[2]-box[0], box[3]-box[1]
    ar = w / max(1, h)
    return 0 if ar >= 1.7 else (1 if ar >= 1.25 else (2 if ar >= 0.95 else 3))  # 후면/사선/측면/정면 근사

def add_lamps(img, box, strength=1.0):
    """차량 하단 좌·우에 램프 광원을 합성해 ON 상태를 만든다(도로 반사 포함)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    out = img.astype(np.float32).copy()
    H, W = out.shape[:2]
    cx, cy = (x1 + x2) / 2, y1 + h * 0.72
    offsets = [(-0.28 * w, 0), (0.28 * w, 0)]
    r = max(2.0, 0.11 * w)
    for dx, dy in offsets:
        lx, ly = int(cx + dx), int(cy + dy)
        if not (0 <= lx < W and 0 <= ly < H):
            continue
        # 광원(웜톤) + 블룸
        tmp = np.zeros((H, W), np.float32)
        cv2.circle(tmp, (lx, ly), int(max(2, r)), 1.0, -1)
        bloom = cv2.GaussianBlur(tmp, (0, 0), max(2.0, r * 1.6))
        warm = np.array([60, 120, 255], np.float32) * strength      # BGR: 붉은 후미등
        out += bloom[..., None] * warm * 0.85
        # 도로 반사(아래로 번짐)
        refl = np.zeros((H, W), np.float32)
        cv2.ellipse(refl, (lx, min(H-1, y2 + int(0.5*h))), (int(max(3, r*1.2)), int(max(3, 0.35*h))), 0, 0, 360, 0.6, -1)
        refl = cv2.GaussianBlur(refl, (0, 0), max(2.0, r))
        out += refl[..., None] * warm * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)

# 조건 셀별 후보 수집
cells = {}
for it in items:
    img = rd(it["scene"])
    if img is None:
        continue
    x1, y1, x2, y2 = it["box"]
    key = (dist_bin(x2 - x1), illum_bin(float(img.mean())), view_bin(it["box"]))
    cells.setdefault(key, []).append(it)

rng = np.random.default_rng(20261120)
made = []
for cell, lst in sorted(cells.items()):
    rng.shuffle(lst)
    for k in range(3):
        if not lst:
            break
        it = lst.pop()
        img = rd(it["scene"])
        if img is None:
            continue
        pid = f"d{cell[0]}i{cell[1]}v{cell[2]}_{k}"
        off = img
        on = add_lamps(img, it["box"], strength=1.0 + 0.15 * k)
        cv2.imwrite(str(PAIR / f"{pid}_off.png"), off)
        cv2.imwrite(str(PAIR / f"{pid}_on.png"), on)
        made.append({"id": pid, "cell": cell, "frame": it["frame"], "src": it["src"],
                     "box": it["box"], "width_px": it["box"][2] - it["box"][0],
                     "frame_mean": round(float(img.mean()), 1)})

(OUT / "pairs_manifest.json").write_text(json.dumps({"count": len(made), "items": made}, ensure_ascii=False, indent=1), encoding="utf-8")
cells_filled = len({tuple(m["cell"]) for m in made})
print(json.dumps({"pairs": len(made), "cells_filled": cells_filled,
                  "dist": {d: sum(1 for m in made if m["cell"][0] == d) for d in (0,1,2)},
                  "illum": {i: sum(1 for m in made if m["cell"][1] == i) for i in (0,1,2)},
                  "view": {v: sum(1 for m in made if m["cell"][2] == v) for v in (0,1,2,3)}},
                 ensure_ascii=False, indent=2))
