"""ITS Open API 야간 CCTV 수집기
- 같은 도로(경부선 등)와 주변 카메라를 순환하며 짧은 클립을 받아 프레임을 추출한다.
- 야간(20:00~05:00)에 실행하면 조명이 켜진 실제 야간 장면을 확보할 수 있다.
- 저장: D:\\Cache\\its_night\\<YYYY-MM-DD>\\<camera>\\frame_XX.png + manifest.json
- 중복 제거: 클립 sha256 + 프레임 sha256
사용:
    set ITS_API_KEY 또는 config/secrets.local.json 의 its_api_key
    python src/collect_its_night.py --clips 12 --frames 3
"""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(r"D:\Desktop\Aside\stealth_vehicle_ai")
SECRETS = ROOT / "config" / "secrets.local.json"
OUT_ROOT = Path(r"D:\Cache\its_night")
API = "https://openapi.its.go.kr:9443/cctvInfo"
REGIONS = [
    ("경부선", 126.90, 127.15, 37.40, 37.55),
    ("수도권", 126.80, 127.20, 37.35, 37.60),
    ("영동선", 127.20, 127.60, 37.35, 37.60),
]


def api_key() -> str:
    k = os.environ.get("ITS_API_KEY", "").strip()
    if k:
        return k
    if SECRETS.exists():
        return str(json.loads(SECRETS.read_text(encoding="utf-8")).get("its_api_key", "")).strip()
    return ""


def cameras(key: str, name_filter: str, bbox) -> list:
    minx, maxx, miny, maxy = bbox
    params = {"apiKey": key, "type": "json", "cctvType": 2, "getType": "json",
              "minX": minx, "maxX": maxx, "minY": miny, "maxY": maxy}
    r = requests.get(API, params=params, timeout=20)
    r.raise_for_status()
    data = r.json().get("response", {}).get("data", [])
    return [c for c in data if name_filter in c.get("cctvname", "")] or data


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def grab_frames(mp4: Path, dst: Path, n: int) -> list:
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        t = 0.5 + i * 8.0
        p = dst / f"frame_{i:02d}.png"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t}", "-i", str(mp4),
               "-frames:v", "1", str(p)]
        subprocess.run(cmd, check=False, capture_output=True)
        if p.exists():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--night-only", action="store_true", default=True)
    args = ap.parse_args()

    now = datetime.now()
    if args.night_only and not (now.hour >= 18 or now.hour <= 5):
        print(f"[skip] 낮 시간대({now:%H:%M})에는 수집하지 않습니다. --night-only 해제 시 강제 수집.")
        if not os.environ.get("FORCE_DAY"):
            return

    key = api_key()
    if not key:
        print("ITS API 키가 없습니다(config/secrets.local.json).")
        sys.exit(2)

    day = now.strftime("%Y-%m-%d_%H%M")
    out_dir = OUT_ROOT / day
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"runs": []}
    seen_clip = {r["clip_sha"] for r in manifest.get("runs", [])}
    seen_frame = {f["sha"] for r in manifest.get("runs", []) for f in r.get("frames", [])}

    got = 0
    for rname, *bbox in REGIONS:
        if got >= args.clips:
            break
        try:
            cams = cameras(key, rname, bbox)
        except Exception as e:
            print(f"[warn] {rname} 목록 실패: {e}")
            continue
        for cam in cams:
            if got >= args.clips:
                break
            name = cam.get("cctvname", "unknown")
            url = cam.get("cctvurl", "")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=25)
                r.raise_for_status()
            except Exception as e:
                print(f"[warn] {name} 다운로드 실패: {e}")
                continue
            h = sha(r.content)
            if h in seen_clip:
                continue
            mp4 = out_dir / f"{name.replace('/', '_')}.mp4"
            mp4.write_bytes(r.content)
            frames = grab_frames(mp4, out_dir / name.replace("/", "_"), args.frames)
            entry = {"camera": name, "region": rname, "collected_at": now.isoformat(timespec="seconds"),
                     "clip_sha": h, "clip_bytes": len(r.content), "frames": []}
            for fp in frames:
                fb = fp.read_bytes()
                if sha(fb) in seen_frame:
                    fp.unlink(missing_ok=True)
                    continue
                entry["frames"].append({"file": str(fp), "sha": sha(fb), "bytes": len(fb)})
                seen_frame.add(sha(fb))
            mp4.unlink(missing_ok=True)      # 원본 클립은 용량 절감을 위해 삭제
            if entry["frames"]:
                manifest["runs"].append(entry)
                got += 1
                print(f"[ok] {name}: {len(entry['frames'])} frames")
            time.sleep(0.4)

    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"collected_clips": got, "out": str(out_dir),
                      "total_runs": len(manifest.get("runs", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

