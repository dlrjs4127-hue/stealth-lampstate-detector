"""실시간 환경 열화 시험: 흔들림(roll)·모션블러·저화질 압축·노이즈에서 성능 유지 확인
- clean 대비 검출 차량 수/소등 후보/지연을 비교
"""
import json
from pathlib import Path
import numpy as np, cv2
import ultralytics.utils as uu, ultralytics.utils.checks as chk
uu.ONLINE = False; chk.check_font = lambda *a, **k: None
from ultralytics import YOLO

ITS = Path(r"D:\Desktop\Aside\stealth_vehicle_ai\REBUILD_V4\data\its_night_review")
FIN = Path(r"D:\Cache\stealth_w4\final")
net1 = YOLO(str(FIN / "runs" / "s1" / "weights" / "best.pt"))
net2 = YOLO(str(FIN / "runs" / "s2" / "weights" / "best.pt"))
names2 = net2.names
off_idx = [k for k, v in names2.items() if v == "off"][0] if isinstance(names2, dict) else names2.index("off")

def rd(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

frames = [p for p in sorted(ITS.rglob("*.jpg")) if "context_frames" in str(p)]
by_name = {}
for p in frames:
    by_name.setdefault(p.name, p)
names = sorted(by_name)
rng = np.random.default_rng(2026)
perm = rng.permutation(len(names))
test_names = [names[i] for i in perm[36:]]      # 최종 학습과 동일한 test 15프레임

def degrade(img, kind):
    h, w = img.shape[:2]
    if kind == "clean":
        return img
    if kind == "blur":                     # 흔들림/모션블러
        k = np.zeros((9, 9), dtype=np.float32); k[4, :] = 1/9
        return cv2.filter2D(img, -1, k)
    if kind == "shake":                    # 카메라 흔들림(회전 + 이동)
        M = cv2.getRotationMatrix2D((w/2, h/2), 1.8, 1.0)
        M[0, 2] += 4; M[1, 2] += 3
        out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return out
    if kind == "jpeg":                     # 저화질 스트리밍
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "noise":                    # 야간 고감도 노이즈
        return np.clip(img.astype(np.float32) + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    return img

def run(kind):
    veh = cand = 0; lat = []
    for nm in test_names:
        img = degrade(rd(by_name[nm]), kind)
        t = cv2.getTickCount()
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
            if (int(b.cls) == 1 and float(b.conf) >= 0.25) or p_off >= 0.5:
                cand += 1
        lat.append((cv2.getTickCount()-t)/cv2.getTickFrequency()*1000)
    return {"frames": len(test_names), "vehicles": veh, "stealth_candidates": cand,
            "latency_mean_ms": round(float(np.mean(lat)), 1), "latency_p90_ms": round(float(np.percentile(lat, 90)), 1)}

res = {}
for kind in ("clean", "blur", "shake", "jpeg", "noise"):
    res[kind] = run(kind)
base = res["clean"]["vehicles"] or 1
for k in res:
    res[k]["vehicle_retention_vs_clean"] = round(res[k]["vehicles"]/base, 3)

out = {"test_frames": len(test_names), "results": res}
Path(FIN / "robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
