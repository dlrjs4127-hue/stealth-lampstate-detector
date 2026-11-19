"""평가 실행 스크립트 — docs/EVALUATION_PLAN.md 의 지표를 그대로 계산한다.

라벨 CSV: frame,vehicle_id,x1,y1,x2,y2,state      (state: on | off | unknown)
예측 CSV: frame,x1,y1,x2,y2,state,score           (모델 추론 결과를 이 형식으로 저장)

사용 예
    python src/evaluate_predictions.py --labels labels.csv --preds preds.csv
    python src/evaluate_predictions.py --labels labels.csv --preds preds.csv --iou 0.3 --conf 0.4

출력: 콘솔 요약 + results/eval_report.json
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def read_csv(path, kind):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            rows.append({
                "frame": r["frame"],
                "box": box,
                "state": (r.get("state") or "").strip().lower(),
                "score": float(r.get("score", 1.0) or 1.0),
                "id": r.get("vehicle_id", ""),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--iou", type=float, default=0.3, help="매칭 IoU (기본 0.3: 원거리 소형 차량 고려)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="results/eval_report.json")
    args = ap.parse_args()

    labels = [r for r in read_csv(args.labels, "labels") if r["state"] in ("on", "off")]
    preds = [r for r in read_csv(args.preds, "preds") if r["score"] >= args.conf]

    gt_by_frame = defaultdict(list)
    for r in labels:
        gt_by_frame[r["frame"]].append(r)
    pred_by_frame = defaultdict(list)
    for r in preds:
        pred_by_frame[r["frame"]].append(r)

    frames = sorted(set(gt_by_frame) | set(pred_by_frame))
    used_pred = set()
    tp = fp = fn = 0
    state_conf = defaultdict(int)  # (gt_state, pred_state) -> count
    fp_per_frame = {}

    for fi, frame in enumerate(frames):
        gts = gt_by_frame.get(frame, [])
        prs = sorted(pred_by_frame.get(frame, []), key=lambda x: -x["score"])
        matched_gt = set()
        frame_fp = 0
        for pi, pr in enumerate(prs):
            best, best_iou = None, 0.0
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                v = iou(pr["box"], gt["box"])
                if v > best_iou:
                    best, best_iou = gi, v
            if best is not None and best_iou >= args.iou:
                matched_gt.add(best)
                tp += 1
                state_conf[(gts[best]["state"], pr["state"] or "unknown")] += 1
            else:
                fp += 1
                frame_fp += 1
        fn += len(gts) - len(matched_gt)
        fp_per_frame[frame] = frame_fp

    n_off = sum(1 for r in labels if r["state"] == "off")
    n_on = sum(1 for r in labels if r["state"] == "on")
    off_hit = state_conf[("off", "off")]
    off_pred = sum(v for (g, p), v in state_conf.items() if p == "off")
    on_hit = state_conf[("on", "on")]
    on_pred = sum(v for (g, p), v in state_conf.items() if p == "on")

    def safe(a, b):
        return a / b if b else 0.0

    recall_off = safe(off_hit, n_off)
    prec_off = safe(off_hit, off_pred)
    recall_on = safe(on_hit, n_on)
    prec_on = safe(on_hit, on_pred)
    f1_off = safe(2 * prec_off * recall_off, prec_off + recall_off)
    lo, hi = wilson(off_hit, n_off)

    report = {
        "frames": len(frames),
        "gt": {"on": n_on, "off": n_off},
        "detection": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(safe(tp, tp + fp), 4),
            "recall": round(safe(tp, tp + fn), 4),
            "f1": round(safe(2 * tp, 2 * tp + fp + fn), 4),
            "iou_threshold": args.iou,
        },
        "state": {
            "recall_off": round(recall_off, 4),
            "precision_off": round(prec_off, 4),
            "f1_off": round(f1_off, 4),
            "recall_on": round(recall_on, 4),
            "precision_on": round(prec_on, 4),
            "balanced_accuracy": round((recall_off + recall_on) / 2, 4),
            "recall_off_wilson95": [round(lo, 4), round(hi, 4)],
            "confusion": {f"gt_{g}->pred_{p}": v for (g, p), v in sorted(state_conf.items())},
        },
        "ops": {
            "fp_per_frame": round(safe(fp, len(frames)), 4),
            "conf_threshold": args.conf,
        },
        "judgement": {},
    }
    # 계획서 판정 기준 적용
    j = report["judgement"]
    j["pilot_recall_ok"] = recall_off >= 0.50
    j["pilot_fp_ok"] = report["ops"]["fp_per_frame"] <= 1.0
    j["final_recall_ok"] = recall_off >= 0.70
    j["final_precision_ok"] = prec_off >= 0.60
    j["final_map50_ok"] = "별도 측정(m.val mAP50 >= 0.60)"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"프레임 {report['frames']} · GT on/off = {n_on}/{n_off}")
    print(f"검출: P {report['detection']['precision']} R {report['detection']['recall']} F1 {report['detection']['f1']} (IoU {args.iou})")
    print(f"상태: 소등 재현율 {report['state']['recall_off']} {report['state']['recall_off_wilson95']} "
          f"정밀도 {report['state']['precision_off']} F1 {report['state']['f1_off']}")
    print(f"운영: FP/frame {report['ops']['fp_per_frame']} (conf {args.conf})")
    print(f"판정: 파일럿 재현율 {j['pilot_recall_ok']} · 파일럿 FP {j['pilot_fp_ok']} · "
          f"최종 재현율 {j['final_recall_ok']} · 최종 정밀도 {j['final_precision_ok']}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
