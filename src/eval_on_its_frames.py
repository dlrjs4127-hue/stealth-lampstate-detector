"""W4 파일럿: 실제 ITS 야간 CCTV에 A/B 모델 적용 + 소등 후보 크롭 몽타주"""
import json
from pathlib import Path
import cv2
import numpy as np
import ultralytics.utils as uu
import ultralytics.utils.checks as chk
uu.ONLINE = False
chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

SRC = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
VEH = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
OUT = Path(r"D:\Cache\stealth_w4\its_pilot")
OUT.mkdir(parents=True, exist_ok=True)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
frames = [p for p in sorted(SRC.rglob("*.jpg")) if "context_frames" in str(p)]
print("frames", len(frames))

det = YOLO(str(VEH))
models = {
    "A_gpt_only": Path(r"D:\Cache\stealth_w4\runs\pilot_A_gpt_only\weights\best.pt"),
    "B_gpt_plus_pair": Path(r"D:\Cache\stealth_w4\runs\pilot_B_gpt_plus_pair\weights\best.pt"),
}
summary = {}
crops_for_montage = []
for name, w in models.items():
    m = YOLO(str(w))
    names = m.names
    veh = flagged = 0
    saved = 0
    confs = []
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        r = det.predict(str(p), imgsz=960, conf=0.15, device=0, verbose=False)[0]
        drew = False
        for b in r.boxes:
            cid = int(b.cls)
            if cid not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (x2 - x1) < 24 or (y2 - y1) < 24:
                continue
            pad = int(0.08 * max(x2 - x1, y2 - y1))
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            pr = m.predict(crop, imgsz=160, device=0, verbose=False)[0].probs
            state = names[int(pr.top1)]
            score = float(pr.data[int(pr.top1)])
            veh += 1
            if state == "stealth":
                flagged += 1
                confs.append(score)
                crops_for_montage.append((name, score, crop.copy()))
                color = (60, 60, 235)
            else:
                color = (60, 190, 90)
            if name == "B_gpt_plus_pair" and saved < 20:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                lbl = ("OFF? " if state == "stealth" else "ON ") + f"{score:.2f}"
                cv2.putText(img, lbl, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                drew = True
        if name == "B_gpt_plus_pair" and drew and saved < 20:
            cv2.imwrite(str(OUT / f"its_B_{saved:02d}.jpg"), img)
            saved += 1
    summary[name] = {"vehicles": veh, "stealth_flags": flagged,
                     "flag_rate": round(flagged / veh, 4) if veh else 0,
                     "mean_flag_conf": round(float(np.mean(confs)), 3) if confs else None,
                     "annotated": saved}

# B 모델 소등 후보 상위 크롭 몽타주 (내 육안 검증용)
b_crops = sorted([c for c in crops_for_montage if c[0] == "B_gpt_plus_pair"], key=lambda t: -t[1])[:16]
tiles = []
for _, sc, c in b_crops:
    t = cv2.resize(c, (128, 128), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, f"{sc:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
if tiles:
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles) - len(tiles) % 4, 4)]
    cv2.imwrite(str(OUT / "montage_B_flags.jpg"), np.vstack(rows))

Path(OUT / "its_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
