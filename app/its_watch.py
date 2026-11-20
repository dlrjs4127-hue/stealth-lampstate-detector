"""ITS 실시간 감시 프로그램 (CLI)

기능
 1) ITS Open API에서 카메라 목록/영상 수집 → 프레임 추출 (--mode collect)
 2) 프레임 폴더에 대해 검출 → 상태판정 → 이벤트 판정 → 알림/리포트 (--mode run)
 3) 검출기 교체 가능(--detector yolo26n 등), 상태판정기 교체 가능(--state-weights)

실행 예
    python app/its_watch.py --mode run --frames D:\\Cache\\its_night\\2026-11-20_2000 --detector yolo26n
    python app/its_watch.py --mode collect --clips 12 --frames 3
"""
import argparse, base64, json, math, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import EventConfig, EventJudge
from models import load_detector, load_state_classifier

OUT_ROOT = Path(r"D:\Cache\its_watch")
VEHICLE_CLASSES = (0, 1)          # 0=normal_vehicle, 1=stealth_vehicle (학습 모델 기준)
COCO_VEHICLES = (2, 3, 5, 7)


def rd(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def entropy_of(probs_data):
    p = np.asarray(probs_data, dtype=np.float64)
    p = p / max(1e-9, p.sum())
    n = len(p)
    h = -float((p * np.log(np.clip(p, 1e-9, 1))).sum())
    return h / math.log(n) if n > 1 else 0.0


def overlay_guard(img, box, min_std=6.0, bright_frac_max=0.12, top_frac=0.25):
    """자막·표지판 같은 오버레이와 무특징(그림자) 영역을 걸러낸다."""
    x1, y1, x2, y2 = box
    H = img.shape[0]
    if (y1 + y2) / 2 < H * top_frac:          # 상단 자막 영역
        return "top_overlay"
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return "empty"
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(g.std()) < min_std:              # 특징 없는 어두운 면
        return "flat_dark"
    if float((g > 200).mean()) > bright_frac_max:  # 흰 글자·표지 과다
        return "bright_text"
    return None
def collect(clips: int, frames: int):
    """ITS Open API에서 클립을 받아 프레임을 추출한다(수집 스크립트 호출)."""
    script = Path(__file__).resolve().parent.parent / "src" / "collect_its_night.py"
    cmd = [sys.executable, str(script), "--clips", str(clips), "--frames", str(frames)]
    subprocess.run(cmd, check=False)


def run(frames_dir: Path, detector: str, state_weights: str | None, out_dir: Path,
        conf: float, imgsz: int):
    det, det_name = load_detector(detector, trained=True)
    dn = set(det.names.values()) if isinstance(det.names, dict) else set(det.names)
    OUR_MODEL = dn.issubset({"vehicle", "normal_vehicle", "stealth_vehicle"})
    clf = load_state_classifier(state_weights)
    names = clf.names
    off_idx = [k for k, v in names.items() if v == "off"][0] if isinstance(names, dict) else names.index("off")

    judge = EventJudge(EventConfig())
    frames = sorted([p for p in frames_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg")])
    if not frames:
        print(json.dumps({"error": f"no frames in {frames_dir}"})); return

    ann_dir = out_dir / "annotated"; ann_dir.mkdir(parents=True, exist_ok=True)
    latencies = []
    frame_rows = []
    skipped = {}

    for fp in frames:
        img = rd(fp)
        if img is None:
            continue
        cam = fp.parent.name
        t0 = time.time()
        r = det.predict(img, imgsz=imgsz, conf=conf, device=0, verbose=False, workers=0)[0]
        nobs = 0
        for b in r.boxes:
            cid = int(b.cls)
            if OUR_MODEL:
                if cid not in VEHICLE_CLASSES:
                    continue
            elif cid not in COCO_VEHICLES:
                continue
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1
            if w < 12 or h < 10:
                continue
            skip = overlay_guard(img, (x1, y1, x2, y2))
            if skip:
                skipped[skip] = skipped.get(skip, 0) + 1
                continue
            pad = int(0.10 * max(w, h))
            crop = img[max(0, y1 - pad):min(img.shape[0], y2 + pad),
                       max(0, x1 - pad):min(img.shape[1], x2 + pad)]
            if crop.size == 0:
                continue
            pr = clf.predict(crop, imgsz=160, device=0, verbose=False, workers=0)[0].probs
            p_off = float(pr.data[off_idx])
            ent = entropy_of(pr.data.tolist())
            judge.observe(f"{cam}/{fp.name}", (x1, y1, x2, y2), p_off, ent, w)
            nobs += 1
            # 주석
            col = (60, 60, 235) if p_off >= 0.5 else (60, 190, 90)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"{'OFF?' if p_off >= 0.5 else 'ON'} {p_off:.2f}", (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        frame_rows.append({"camera": cam, "frame": fp.name, "observations": nobs, "latency_ms": round(dt, 1)})
        cv2.imwrite(str(ann_dir / f"{cam}__{fp.name}"), img)

    summ = judge.summary()
    result = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "frames_dir": str(frames_dir),
        "detector": det_name,
        "frames": len(frame_rows),
        "observations": sum(f["observations"] for f in frame_rows),
        "latency_mean_ms": round(float(np.mean(latencies)), 1),
        "latency_p90_ms": round(float(np.percentile(latencies, 90)), 1),
        "events": summ,
        "skipped_guards": skipped,
        "config": {"conf": conf, "imgsz": imgsz, "enter": judge.cfg.enter, "exit": judge.cfg.exit,
                   "ewma_alpha": judge.cfg.ewma_alpha, "min_width_px": judge.cfg.min_width_px},
        "frames_detail": frame_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in judge.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 간단 HTML 리포트
    rows = "\n".join(
        f"<tr><td>{f['camera']}</td><td>{f['frame']}</td><td>{f['observations']}</td><td>{f['latency_ms']}</td></tr>"
        for f in frame_rows[:80]
    )
    ev = "\n".join(
        f"<tr><td>{e['event']}</td><td>{e['track_id']}</td><td>{e['frame']}</td><td>{e['score']}</td></tr>"
        for e in judge.events[:80]
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>ITS 감시 결과</title>
<style>body{{font-family:system-ui,'Malgun Gothic';margin:24px;color:#17252a}}table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #d5e3db;padding:4px 8px;font-size:12px}}th{{background:#173f35;color:#fff}}
.k{{display:inline-block;margin-right:18px}}.v{{font-weight:700}}</style>
<h1>ITS 야간 스텔스 감시 결과</h1>
<p><span class="k">검출기 <span class="v">{det_name}</span></span>
<span class="k">프레임 <span class="v">{result['frames']}</span></span>
<span class="k">관측 <span class="v">{result['observations']}</span></span>
<span class="k">확정 이벤트 <span class="v">{summ['confirmed_events']}</span></span>
<span class="k">UNKNOWN 트랙 <span class="v">{summ['unknown_tracks']}</span></span>
<span class="k">지연 평균/p90 <span class="v">{result['latency_mean_ms']} / {result['latency_p90_ms']} ms</span></span></p>
<h2>이벤트</h2><table><tr><th>이벤트</th><th>트랙</th><th>프레임</th><th>점수</th></tr>{ev or '<tr><td colspan=4>없음</td></tr>'}</table>
<h2>프레임 상세(상위 80)</h2><table><tr><th>카메라</th><th>프레임</th><th>관측</th><th>지연(ms)</th></tr>{rows}</table>
"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "frames_detail"}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["collect", "run"], default="run")
    ap.add_argument("--frames", type=str, default="")
    ap.add_argument("--detector", type=str, default="yolov8n")
    ap.add_argument("--state-weights", type=str, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--nframes", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    a = ap.parse_args()

    if a.mode == "collect":
        collect(a.clips, a.nframes)
        return

    if not a.frames:
        print(json.dumps({"error": "--frames 필요"})); sys.exit(2)
    frames_dir = Path(a.frames)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.out) if a.out else OUT_ROOT / f"{stamp}_{a.detector}"
    run(frames_dir, a.detector, a.state_weights, out_dir, a.conf, a.imgsz)


if __name__ == "__main__":
    main()


