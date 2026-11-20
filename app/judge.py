"""이벤트 단위 판정 — 프레임별 점수를 트랙에 누적(EWMA)하고 히스테리시스로 알림을 확정한다.

핵심 아이디어
- 프레임 하나의 판정은 흔들린다. 차량 트랙(연속 프레임의 같은 차량)에 증거를 누적한다.
- 단순 로그우도 합은 프레임 상관성 때문에 과대평가되므로 EWMA(지수이동평균)로 감쇠 누적한다.
- 진입 임계값(enter)과 해제 임계값(exit)을 다르게 두어 깜빡임을 막는다.
- 판독 불가(작음/흐림/가림)는 UNKNOWN으로 라우팅해 "소등"과 구분한다.
"""
from dataclasses import dataclass, field


@dataclass
class EventConfig:
    enter: float = 0.60          # 알림 확정 임계값
    exit: float = 0.40           # 알림 해제 임계값
    ewma_alpha: float = 0.45     # 최근 프레임 가중치
    min_hits: int = 2            # 최소 관측 프레임 수
    min_width_px: int = 18       # 판정 가능 최소 차량 폭
    max_entropy: float = 0.55    # 상태 분류 엔트로피 상한(정규화)
    iou_match: float = 0.30      # 트랙 연결 IoU


@dataclass
class Track:
    tid: int
    camera: str
    box: tuple
    score: float = 0.0
    hits: int = 0
    state: str = "unknown"
    stale: int = 0
    alerted: bool = False
    history: list = field(default_factory=list)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class EventJudge:
    """프레임 단위 관측 -> 트랙 단위 이벤트 판정"""

    def __init__(self, cfg: EventConfig | None = None):
        self.cfg = cfg or EventConfig()
        self.tracks: dict[int, Track] = {}
        self._next = 1
        self.events: list[dict] = []

    # ---- 트랙 연결 ----
    def _match(self, box, camera):
        best, best_iou = None, 0.0
        for t in self.tracks.values():
            if t.camera != camera:
                continue
            v = iou(box, t.box)
            if v > best_iou:
                best, best_iou = t, v
        if best is not None and best_iou >= self.cfg.iou_match:
            return best
        t = Track(tid=self._next, camera=camera, box=box)
        self._next += 1
        self.tracks[t.tid] = t
        return t

    # ---- 관측 1건 반영 ----
    def observe(self, frame_id: str, box, p_off: float, entropy: float, width_px: int, camera: str = ""):
        # 오래된 트랙 만료(카메라별 12회 관측 동안 갱신 없으면 종료)
        for tid in [k for k, v in self.tracks.items() if v.camera == camera and v.hits > 0 and v.stale >= 12]:
            self.tracks.pop(tid, None)
        t = self._match(box, camera)
        t.box = box
        t.hits += 1

        if width_px < self.cfg.min_width_px or entropy > self.cfg.max_entropy:
            t.state = "unknown"
            t.history.append((frame_id, "unknown", round(p_off, 3)))
            return t

        t.score = (1 - self.cfg.ewma_alpha) * t.score + self.cfg.ewma_alpha * p_off

        if not t.alerted and t.score >= self.cfg.enter and t.hits >= self.cfg.min_hits:
            t.alerted = True
            t.state = "off"
            self.events.append({
                "event": "alert_on", "track_id": t.tid, "frame": frame_id,
                "score": round(t.score, 4), "hits": t.hits,
                "box": [int(v) for v in box],
            })
        elif t.alerted and t.score <= self.cfg.exit:
            t.alerted = False
            t.state = "on"
            self.events.append({
                "event": "alert_off", "track_id": t.tid, "frame": frame_id,
                "score": round(t.score, 4), "hits": t.hits,
                "box": [int(v) for v in box],
            })
        else:
            t.state = "off" if t.alerted else "on"

        t.history.append((frame_id, t.state, round(t.score, 3)))
        return t

    # ---- 요약 ----
    def summary(self):
        confirmed = [t for t in self.tracks.values() if any(e["event"] == "alert_on" and e["track_id"] == t.tid for e in self.events)]
        unknown = [t for t in self.tracks.values() if t.state == "unknown"]
        return {
            "tracks": len(self.tracks),
            "confirmed_events": len([e for e in self.events if e["event"] == "alert_on"]),
            "confirmed_tracks": len(confirmed),
            "unknown_tracks": len(unknown),
            "repeat_alerts_same_track": len([e for e in self.events if e["event"] == "alert_on"]) - len({e["track_id"] for e in self.events if e["event"] == "alert_on"}),
        }

