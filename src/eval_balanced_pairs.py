"""균형쌍을 가능한 만큼 확장 + 그 쌍으로 상태분류기 통제 평가(램프 단서만 다른 조건)"""
import json
from pathlib import Path

import cv2
import numpy as np
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ROOT = Path(r"D:\Cache\stealth_w4")
PAIRS = ROOT / "pairs108"
PAIR = PAIRS / "pairs"
CLS = ROOT / "final2" / "runs" / "s2" / "weights" / "best.pt"

man = json.loads((PAIRS / "pairs_manifest.json").read_text(encoding="utf-8"))
print("pairs built:", man["count"])

clf = YOLO(str(CLS))
names = clf.names
off_idx = [k for k, v in names.items() if v == "off"][0] if isinstance(names, dict) else names.index("off")
on_idx = [k for k, v in names.items() if v == "on"][0] if isinstance(names, dict) else names.index("on")

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

rows = []
for it in man["items"]:
    pid = it["id"]
    of = PAIR / f"{pid}_off.png"
    on = PAIR / f"{pid}_on.png"
    if not (of.exists() and on.exists()):
        continue
    row = {"id": pid, "cell": it["cell"], "width_px": it["width_px"]}
    for kind, p in (("off", of), ("on", on)):
        img = rd(p)
        x1, y1, x2, y2 = it["box"]
        pad = int(0.10 * max(x2 - x1, y2 - y1))
        crop = img[max(0, y1-pad):min(img.shape[0], y2+pad), max(0, x1-pad):min(img.shape[1], x2+pad)]
        pr = clf.predict(crop, imgsz=160, device=0, verbose=False, workers=0)[0].probs
        p_off = float(pr.data[off_idx])
        row[kind] = round(p_off, 3)
        row[f"{kind}_pred"] = "off" if int(pr.top1) == off_idx else "on"
    rows.append(row)

correct_off = sum(1 for r in rows if r["off_pred"] == "off")
correct_on = sum(1 for r in rows if r["on_pred"] == "on")
res = {
    "pairs": len(rows),
    "off_correct": correct_off, "on_correct": correct_on,
    "off_acc": round(correct_off / max(1, len(rows)), 4),
    "on_acc": round(correct_on / max(1, len(rows)), 4),
    "balanced_acc": round((correct_off / max(1, len(rows)) + correct_on / max(1, len(rows))) / 2, 4),
    "mean_p_off_off_images": round(float(np.mean([r["off"] for r in rows])), 3) if rows else None,
    "mean_p_off_on_images": round(float(np.mean([r["on"] for r in rows])), 3) if rows else None,
    "cells": sorted({tuple(r["cell"]) for r in rows}),
}
(PAIRS / "pair_eval.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
