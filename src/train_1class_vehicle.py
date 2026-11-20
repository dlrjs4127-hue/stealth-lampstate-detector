"""1클래스 vehicle 검출기 학습 (YOLO26n) — 최종계획 2주차 작업의 선행 실행
- 기존 final2(2클래스) 이미지/라벨을 1클래스(vehicle)로 변환
- 오늘 20:00 ITS 야간 수집 36프레임을 추가(train 카메라 프레임과 중복 없음 확인 후 train에만)
- 평가: mAP50, Recall (1클래스)
"""
import json, shutil, time
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

SRC = Path(r"D:\Cache\stealth_w4\final2\ds1")
NIGHT = Path(r"D:\Cache\its_night\2026-11-20_2000")
OUT = Path(r"D:\Cache\stealth_w4\y26_1cls")
DS = OUT / "ds"
OUT.mkdir(parents=True, exist_ok=True)

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def wr(p, img): cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(p))

if DS.exists(): shutil.rmtree(DS)
def add(split, name, img, lines):
    (DS/split/"images").mkdir(parents=True, exist_ok=True)
    (DS/split/"labels").mkdir(parents=True, exist_ok=True)
    wr(DS/split/"images"/f"{name}.jpg", img)
    (DS/split/"labels"/f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

# 1) 기존 2클래스 → 1클래스 변환
n_tr = n_va = 0
for split in ("train", "val"):
    for lp in sorted((SRC/split/"labels").glob("*.txt")):
        ip = SRC/split/"images"/(lp.stem + ".jpg")
        if not ip.exists(): continue
        lines = []
        for ln in lp.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if len(parts) == 5:
                lines.append("0 " + " ".join(parts[1:]))
        add(split, lp.stem, rd(ip), lines)
        n_tr += split == "train"; n_va += split == "val"

# 2) 오늘 야간 수집 프레임 추가 (COCO 의사 라벨, train만)
det = YOLO(r"D:\Desktop\Aside\stealth_vehicle_ai\yolov8n.pt")
n_new = 0
if NIGHT.exists():
    for p in sorted(NIGHT.rglob("frame_*.png")):
        img = rd(p)
        if img is None: continue
        H, W = img.shape[:2]
        r = det.predict(img, imgsz=960, conf=0.30, device=0, verbose=False, workers=0)[0]
        lines = []
        for b in r.boxes:
            if int(b.cls) not in (2, 3, 5, 7): continue
            x1, y1, x2, y2 = [int(t) for t in b.xyxy[0].tolist()]
            if (x2-x1) < 16 or (y2-y1) < 12: continue
            lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        add("train", f"night_{n_new:03d}", img, lines)
        n_new += 1

(DS/"data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: train/images\nval: val/images\nnames:\n  0: vehicle\n", encoding="utf-8")
print("1cls dataset: train", n_tr, "+ nights", n_new, "val", n_va)

t0 = time.time()
m = YOLO("yolo26n.pt")
m.train(data=str(DS/"data.yaml"), epochs=100, imgsz=640, batch=8, device=0, workers=0,
        project=str(OUT/"runs"), name="s1_1cls", exist_ok=True, plots=False, amp=False,
        verbose=False, seed=0, patience=25)
best = OUT/"runs"/"s1_1cls"/"weights"/"best.pt"
v = YOLO(str(best)).val(data=str(DS/"data.yaml"), imgsz=640, device=0, plots=False, verbose=False, workers=0)
res = {"train_images": n_tr + n_new, "val_images": n_va, "night_added": n_new,
       "train_seconds": round(time.time()-t0, 1),
       "mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
       "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
       "weights": str(best)}
(OUT/"metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
