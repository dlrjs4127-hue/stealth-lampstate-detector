"""A4 5페이지 보고서 HTML 생성 (이미지 base64 임베드, 인쇄용 CSS)"""
import base64
import json
from pathlib import Path

ASSETS = Path(r"D:\Cache\stealth_w4\report_assets")
RUN = Path(r"D:\Cache\its_watch\run_final")
OUT = Path(r"D:\Cache\stealth_w4\report")
OUT.mkdir(parents=True, exist_ok=True)

def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()

# 이미지 선택
hero = ASSETS / "hero_composite.png"
night = ASSETS / "night_frame.png"
panel = ASSETS / "degrade_panel.png"
annot = sorted((RUN / "annotated").glob("*.png"))
annot_img = annot[0] if annot else hero

metrics = json.loads(Path(r"D:\Cache\stealth_w4\y26_1cls\metrics.json").read_text(encoding="utf-8"))
its = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
pair = json.loads(Path(r"D:\Cache\stealth_w4\pairs108\pair_eval.json").read_text(encoding="utf-8"))["summary"]

CSS = """
@page { size: A4; margin: 13mm 14mm 12mm 14mm; }
* { box-sizing: border-box; }
body { font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; color: #17252a; font-size: 9.0pt; line-height: 1.45; margin: 0; width: 178mm; }
h1 { font-size: 15.5pt; margin: 0 0 4px; letter-spacing: -0.3px; }
h2 { font-size: 11.5pt; margin: 10px 0 5px; padding-bottom: 3px; border-bottom: 1.5px solid #2d6a4f; color: #173f35; }
h3 { font-size: 9.8pt; margin: 8px 0 3px; color: #2d6a4f; }
p { margin: 4px 0; }
.small { font-size: 8.2pt; color: #5c696e; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
.titleblock { border-top: 6px solid #173f35; padding-top: 10px; }
.sub { color: #2d6a4f; font-weight: 600; font-size: 10pt; margin-bottom: 8px; }
.meta { font-size: 8.8pt; color: #5c696e; margin-bottom: 10px; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin: 10px 0 6px; }
.kpi { border: 1px solid #d5e3db; border-radius: 4px; padding: 7px; background: #f7faf8; }
.kpi .v { font-size: 14pt; font-weight: 700; color: #173f35; }
.kpi .l { font-size: 7.6pt; color: #5c696e; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 8px; }
th { background: #173f35; color: #fff; font-size: 7.9pt; padding: 4px 5px; text-align: left; font-weight: 600; }
td { border-bottom: 1px solid #d5e3db; padding: 3px 5px; font-size: 8.1pt; vertical-align: top; }
tr:nth-child(even) td { background: #f4f9f6; }
figure { margin: 6px 0 8px; }
figure img { width: 100%; max-height: 54mm; object-fit: contain; display: block; border: 1px solid #d5e3db; border-radius: 3px; }
figcaption { font-size: 7.8pt; color: #5c696e; margin-top: 3px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.box { border: 1px solid #d5e3db; background: #f7faf8; border-radius: 4px; padding: 7px 9px; }
.flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; margin: 8px 0; }
.step { border: 1.2px solid #2d6a4f; border-radius: 4px; padding: 6px 5px; text-align: center; font-size: 7.8pt; font-weight: 600; background: #fff; }
.url { font-family: Consolas, monospace; font-size: 8.2pt; color: #1a4d8f; }
ul { margin: 4px 0 4px 16px; padding: 0; }
li { margin: 2px 0; }
.note { border-left: 3px solid #b23a30; background: #fcf3f1; padding: 6px 9px; font-size: 8.6pt; margin: 6px 0; }
.good { border-left: 3px solid #2d6a4f; background: #eef7f2; padding: 6px 9px; font-size: 8.6pt; margin: 6px 0; }
"""

P1 = f"""
<div class="page">
  <div class="titleblock">
    <div class="small">경희대학교 산업경영공학과 · 독립심화학습2 · 4차 진행보고</div>
    <h1>야간 CCTV 스텔스(램프 소등) 차량 검출</h1>
    <div class="sub">실제 ITS 프레임 기반 ROI 편집 합성과 2단계 판정 시스템</div>
    <div class="meta">2020103950 이건 · 지도교수 김영훈 · 2026. 11. 20. · 저장소 <span class="url">github.com/dlrjs4127-hue/stealth-lampstate-detector</span></div>
  </div>

  <h2>초록</h2>
  <p>야간 고정 CCTV에서 전조등·후미등을 끄고 주행하는 차량(스텔스 차량)을 찾는 문제를 다룬다. 공개된 ‘소등 차량’ 라벨 데이터가 사실상 없으므로, 실제 ITS 프레임의 빈 차로 영역만 생성 모델로 편집해 소등 차량을 삽입하는 데이터 파이프라인을 구축했다. 검출은 YOLO26n 1클래스, 상태 판정은 YOLOv8n-cls로 분리한 2단계 구조를 쓰고, 프레임 단위가 아니라 차량 트랙 단위로 증거를 누적(EWMA·히스테리시스)해 알림을 확정한다. 실제 ITS Open API로 12개 카메라 36프레임을 수집해 프로그램을 실행한 결과, 확정 알림 0건·처리 지연 54.3ms를 기록했다. 또한 램프만 다른 통제쌍 실험에서 현재 분류기가 램프가 아닌 차체 어두움에 의존한다는 사실을 확인했고, 이를 다음 단계의 핵심 과제로 정리했다.</p>

  <div class="kpis">
    <div class="kpi"><div class="v">177장</div><div class="l">ROI 편집 합성본<br/>(ROI 밖 픽셀 변경 0)</div></div>
    <div class="kpi"><div class="v">{metrics['mAP50']:.3f}</div><div class="l">1클래스 검출 mAP50<br/>P {metrics['precision']:.3f} / R {metrics['recall']:.3f}</div></div>
    <div class="kpi"><div class="v">0건</div><div class="l">ITS 36프레임 실행<br/>확정 알림(오탐 0)</div></div>
    <div class="kpi"><div class="v">{its['latency_mean_ms']:.1f}ms</div><div class="l">프레임당 처리 지연<br/>p90 {its['latency_p90_ms']:.1f}ms</div></div>
  </div>

  <h2>1. 연구 배경과 문제 정의</h2>
  <ul>
    <li><b>데이터 부재</b>: 고정 CCTV 시점에서 ‘램프 소등’ 라벨이 붙은 공개 데이터는 없다. 있는 것은 근접 차량 ON/OFF 크롭(공개 데이터)과 라벨 없는 ITS 스트림뿐이다.</li>
    <li><b>신호의 취약성</b>: 야간 CCTV에서 차량 폭은 중앙값 50px(18~110px)이고, 후미등은 2~4px 수준이다. 램프가 꺼지면 차체는 배경과 거의 같은 밝기가 된다.</li>
    <li><b>운영 요구</b>: 단일 프레임 판정은 흔들린다. 실제 적용에는 ① 실시간 처리 ② 카메라별 오탐 억제 ③ 판정 근거의 감사 가능성이 필요하다.</li>
  </ul>
  <h3>1.1 연구 범위와 용어</h3>
  <table>
    <tr><th style="width:20%">용어</th><th>정의</th></tr>
    <tr><td>스텔스 차량</td><td>야간 도로에서 전조등·후미등을 모두 끄고 주행하는 차량. 본 연구는 <b>후면(멀어지는) 뷰</b>를 1차 대상으로 한다.</td></tr>
    <tr><td>ON / OFF / UNKNOWN</td><td>램프 점등 확인 / 소등 확인 / 판독 불가(학습·평가에서 제외).</td></tr>
    <tr><td>트랙</td><td>연속 프레임에서 같은 차량으로 연결된 관측 묶음. 알림은 트랙 단위로 확정한다.</td></tr>
    <tr><td>합성본(합성 데이터)</td><td>실제 프레임의 빈 차로 ROI만 생성 모델로 편집해 만든 이미지. 배경·조도·노이즈는 실제 값을 유지한다.</td></tr>
  </table>
  <p class="small">범위 밖: 번호판 인식, 차량 속도 추정, 다중 카메라 재식별(re-ID).</p>
  <figure><img src="{b64(night)}"/><figcaption>그림 1. 실제 ITS 야간 프레임(고정 CCTV). 차량이 어두운 덩어리로 보이고, 램프 점등 여부가 유일한 단서인 경우가 많다.</figcaption></figure>
</div>
"""

P2 = f"""
<div class="page">
  <h2>2. 데이터 구성</h2>
  <table>
    <tr><th style="width:26%">구분</th><th style="width:14%">규모</th><th>내용·용도</th></tr>
    <tr><td>실제 ITS 야간 프레임</td><td>51장(고유)</td><td>배경·평가. 동일 파일명 중복 제거 후 사용</td></tr>
    <tr><td>ROI 편집 합성본</td><td>177장</td><td>실제 프레임의 빈 차로 ROI만 생성 모델로 편집. 학습 85 / 검증 40 / 나머지 보존</td></tr>
    <tr><td>추가 야간 수집</td><td>36프레임</td><td>ITS Open API 12개 카메라(20:00 수집). 검출기 학습·실행 평가에 사용</td></tr>
    <tr><td>실제 ON/OFF 대응쌍</td><td>628쌍</td><td>공개 데이터(CC BY 4.0). 상태 판정 학습의 핵심 신호</td></tr>
    <tr><td>램프 균형쌍</td><td>20쌍</td><td>동일 장면·동일 차량에서 램프만 합성해 만든 통제쌍(평가용)</td></tr>
  </table>

  <h3>2.1 ROI 편집 합성 파이프라인</h3>
  <p>도로를 통째로 생성하면 카메라 특성(압축·노이즈·조도)이 달라져 전이가 되지 않는다. 그래서 <b>실제 프레임은 그대로 두고 빈 차로 영역(ROI)만</b> 생성 모델로 편집한다.</p>
  <div class="flow">
    <div class="step">① 실제 프레임<br/><span class="small">ITS 야간</span></div>
    <div class="step">② 빈 차로 ROI<br/><span class="small">200×150 자동 선정</span></div>
    <div class="step">③ ROI 편집<br/><span class="small">소등 차량 1대</span></div>
    <div class="step">④ 원위치 복원<br/><span class="small">8px feather</span></div>
    <div class="step">⑤ BBOX 라벨<br/><span class="small">검출기로 재확인</span></div>
  </div>
  <table>
    <tr><th style="width:24%">검증 항목</th><th>결과</th></tr>
    <tr><td>ROI 밖 픽셀 변경</td><td><b>0건</b>(전 장). 마스크 밖은 원본과 완전 동일</td></tr>
    <tr><td>차량 BBOX 확보</td><td>94 / 112 편집본에서 자동 검출 성공(미검출 18장은 제외)</td></tr>
    <tr><td>차량 폭 분포</td><td>18~110px, 중앙값 50px</td></tr>
    <tr><td>조건 축</td><td>거리 3구간(80px↑ / 40~79 / 18~39) × 조도 3구간 × 시점 4종</td></tr>
  </table>
  <h3>2.2 라벨 규칙과 분할 원칙</h3>
  <table>
    <tr><th style="width:16%">라벨</th><th style="width:38%">정의</th><th>처리</th></tr>
    <tr><td>on</td><td>전조등 또는 후미등 점등이 픽셀에서 확인됨</td><td>상태 판정 정상 클래스</td></tr>
    <tr><td>off</td><td>점등 흔적이 없고 차체 형태가 확인됨(그림자와 구분)</td><td>상태 판정 소등 클래스</td></tr>
    <tr><td>unknown</td><td>흐림·가림·원거리로 판독 불가</td><td><b>학습·평가 모두 제외</b>, UNKNOWN으로 라우팅</td></tr>
  </table>
  <ul>
    <li>분할은 <b>카메라 단위</b>: train 6대 / val 3대 / test 3대(카메라 간 완전 분리, 동일 클립·파생 이미지 중복 금지).</li>
    <li>합성본은 train에만 넣고, 검증·테스트는 실제 프레임으로만 구성한다(합성 검증 성능을 실제 성능으로 대체하지 않는다).</li>
  </ul>
  <figure><img src="{b64(hero)}"/><figcaption>그림 2. ROI 편집 합성 예시. 실제 프레임의 빈 차로에 생성 차량을 넣고 BBOX를 기록한다(빨간 박스).</figcaption></figure>
</div>
"""

P3 = f"""
<div class="page">
  <h2>3. 방법</h2>
  <h3>3.1 2단계 구조</h3>
  <table>
    <tr><th style="width:18%">단계</th><th style="width:30%">모델·설정</th><th>역할</th></tr>
    <tr><td>Stage-1 검출</td><td>YOLO26n, 1클래스(vehicle), 960px</td><td>차량 위치 검출. 소등 여부는 판단하지 않는다(2클래스에서 출처 차이를 학습하는 문제 제거)</td></tr>
    <tr><td>Stage-2 상태</td><td>YOLOv8n-cls, 160px</td><td>차량 크롭의 램프 상태(ON/OFF) 판정. 실제 ON/OFF 628쌍 + 합성 124장으로 학습</td></tr>
    <tr><td>이벤트 판정</td><td>EWMA α=0.45, 진입 0.80 / 해제 0.40, 최소 3회</td><td>프레임별 점수를 트랙에 누적. 중복 알림 억제, 깜빡임 방지</td></tr>
    <tr><td>UNKNOWN</td><td>폭 40px 미만·판독불가·불확실</td><td>작은/가려진 객체를 ‘소등’으로 확정하지 않고 보류</td></tr>
  </table>

  <h3>3.2 실행 프로그램</h3>
  <p><span class="url">app/its_watch.py</span>는 두 모드로 동작한다. <b>collect</b>는 ITS Open API에서 클립을 받아 프레임을 추출하고, <b>run</b>은 프레임 폴더에 대해 검출→상태판정→이벤트 판정을 수행한 뒤 알림(JSONL)과 HTML 리포트를 만든다. 검출기·상태 판정기는 슬롯으로 분리해 교체할 수 있고, 실제 로드한 가중치 경로를 로그로 남긴다.</p>
  <ul>
    <li>카메라별 트랙 분리·만료: 서로 다른 카메라의 정지 프레임이 같은 트랙으로 묶이지 않게 처리(요청 12프레임 무갱신 시 종료)</li>
    <li>오버레이 가드: 상단 자막 영역, 흰 글자 과다 영역, 무특징(그림자) 영역 제외 — 36프레임에서 33건 차단</li>
    <li>감사 로그: BBOX·상태·신뢰도·트랙 ID·UNKNOWN 사유·판정 시각 저장</li>
  </ul>
  <figure><img src="{b64(annot_img)}"/><figcaption>그림 3. 프로그램 실행 화면(주석 결과). 초록 상자 ON, 빨강 상자 소등 후보. 36프레임 실행에서 확정 알림은 0건이었다.</figcaption></figure>

  <h3>3.3 판정 규칙(의사코드)</h3>
  <table>
    <tr><th style="width:12%">단계</th><th>동작</th></tr>
    <tr><td>관측</td><td>프레임 t에서 차량 bbox·폭 w·상태확률 p_off·엔트로피 H를 얻는다</td></tr>
    <tr><td>트랙</td><td>같은 카메라의 기존 트랙과 IoU ≥ 0.30이면 연결, 아니면 새 트랙 생성(카메라별 독립)</td></tr>
    <tr><td>보류</td><td>w &lt; 40px 또는 H &gt; 0.55 또는 램프 판독 불가 → UNKNOWN(알림 없음)</td></tr>
    <tr><td>누적</td><td>score ← (1−α)·score + α·p_off, α = 0.45</td></tr>
    <tr><td>확정</td><td>score ≥ 0.80 그리고 관측 ≥ 3회 → 알림 ON, score ≤ 0.40이면 해제(트랙당 최초 1회만 알림)</td></tr>
  </table>
  <p class="small">단순 로그우도 합은 연속 프레임 상관성 때문에 증거를 과대평가하므로 감쇠 누적(EWMA)과 서로 다른 진입·해제 임계값(히스테리시스)을 사용한다.</p>
  <h3>3.4 처리 지연 예산</h3>
  <table>
    <tr><th>입력·전처리</th><th>검출(960px)</th><th>크롭 분류(160px)</th><th>트래킹·판정</th><th>로깅·오버레이</th></tr>
    <tr><td>5ms</td><td>25ms</td><td>16ms</td><td>4ms</td><td>5ms</td></tr>
  </table>
</div>
"""

P4 = f"""
<div class="page">
  <h2>4. 실험 및 결과</h2>
  <table>
    <tr><th style="width:22%">실험</th><th style="width:30%">구성</th><th style="width:24%">결과</th><th>해석</th></tr>
    <tr><td>E1 완전 생성 장면</td><td>도로 전체 생성 59장, 2클래스</td><td>mAP50 0.782</td><td>실제 프레임에서 표지판 오탐 다수 → 배경을 실제로 써야 함</td></tr>
    <tr><td>E2 ITS 합성(초기)</td><td>합성 58 + 실제 51</td><td>mAP50 0.961 / P 0.90</td><td>합성 검증 성능은 높지만 실제 소등 후보 0</td></tr>
    <tr><td>E3 ROI 편집 50장</td><td>실제 ROI 편집 50 + 실제 51</td><td>mAP50 0.976</td><td>장면 정합성 확보(전 쌍 ROI 밖 픽셀 변경 0)</td></tr>
    <tr><td>E4 누수 제거 분할</td><td>프레임 단위 train 26/val 10/test 15</td><td>mAP50 0.798</td><td>독립 평가 기준 확립. 검증셋 합성 포함 한계 명시</td></tr>
    <tr><td>E9 확장</td><td>합성 177장(85/40)</td><td>mAP50 0.781</td><td>합성 수를 늘려도 stealth AP 0.917에서 포화</td></tr>
    <tr><td>E11 1클래스 전환</td><td>YOLO26n 1클래스, 야간 36 추가</td><td>mAP50 {metrics['mAP50']:.3f} / P {metrics['precision']:.3f}</td><td>일반 차량 검출 개선, 실제 실행 오탐 0</td></tr>
    <tr><td>E12 램프 균형쌍</td><td>동일 장면 ON/OFF 20쌍(램프만 차이)</td><td>OFF {pair['off_acc']:.2f} / ON {pair['on_acc']:.2f}</td><td>분류기가 램프가 아닌 차체 어두움에 의존</td></tr>
  </table>

  <div class="note"><b>핵심 발견(E12)</b> — 같은 장면·같은 차량에서 램프만 켠 이미지를 만들었더니, 기존 분류기는 ‘OFF’는 100% 맞히는데 ‘ON’은 30%만 맞혔다(p_off 평균 1.00 vs 0.72). 즉 합성 데이터의 질감/어두움을 램프 단서로 오인하고 있다. 이는 다음 주 데이터·학습 설계의 직접 근거다.</div>

  <h3>4.1 열화 환경 시험</h3>
  <table>
    <tr><th>조건</th><th>차량 검출 유지율</th><th>지연</th></tr>
    <tr><td>clean</td><td>1.00 (기준)</td><td>11.9ms</td></tr>
    <tr><td>모션블러</td><td>0.81</td><td>11.9ms</td></tr>
    <tr><td>카메라 흔들림(회전+이동)</td><td>0.95</td><td>15.1ms</td></tr>
    <tr><td>저화질(JPEG q35)</td><td>0.81</td><td>12.7ms</td></tr>
    <tr><td>야간 고감도 노이즈</td><td>1.14</td><td>16.3ms</td></tr>
  </table>
  <h3>4.2 실제 실행 결과(ITS API, 2026-11-20 20:00)</h3>
  <table>
    <tr><th>프레임</th><th>관측</th><th>트랙</th><th>확정 알림</th><th>UNKNOWN</th><th>가드 차단</th><th>지연 평균/p90</th></tr>
    <tr><td>36 (12카메라)</td><td>165</td><td>138</td><td><b>0</b></td><td>49</td><td>33</td><td>54.3 / 62.6 ms</td></tr>
  </table>
  <h3>4.3 실패 사례와 원인</h3>
  <table>
    <tr><th style="width:26%">사례</th><th>원인</th><th style="width:30%">대응</th></tr>
    <tr><td>자막 그림자 오탐</td><td>어두운 문자 그림자가 차량으로 검출되고, 분류기가 소등으로 판정</td><td>오버레이 가드 + 도로 ROI 마스크(1주차)</td></tr>
    <tr><td>램프 균형쌍 ON 오분류</td><td>분류기가 램프가 아닌 차체 어두움/질감에 의존</td><td>균형쌍 108 + 재학습(1주차)</td></tr>
    <tr><td>카메라 간 트랙 혼동</td><td>트랙 관리가 카메라 단위로 분리되지 않음</td><td>카메라별 독립 ID·만료(수정 완료)</td></tr>
  </table>
  <figure><img src="{b64(panel)}"/><figcaption>그림 4. clean(좌)과 모션블러(우)에서의 검출 비교. 노란 숫자는 검출 수.</figcaption></figure>
</div>
"""

P5 = f"""
<div class="page">
  <h2>5. 논의와 다음 단계</h2>
  <div class="good"><b>기여</b> — ① 실제 CCTV 프레임의 ROI만 편집해 소등 라벨을 만드는 통제된 합성 파이프라인(ROI 밖 픽셀 변경 0 검증) ② 소형 램프 상태 판정에서 특화 CNN이 파운데이션 특징(DINOv2-S linear probe 0.726)보다 우세함을 실측 ③ 불확실성·시간적 증거를 결합한 감사 가능한 이벤트 판정 시스템.</div>
  <p class="small">주의: 위 기여는 실제 소등 차량 검출 성능의 입증이 아니다. 실제 소등 라벨이 없어 소등 재현율은 아직 N/A(n=0)로 둔다.</p>

  <h3>5.1 4주 계획 (GPU 180분 배분)</h3>
  <table>
    <tr><th style="width:18%">주차</th><th>작업</th><th style="width:14%">산출물</th></tr>
    <tr><td>11/21~27</td><td>램프 균형쌍 108쌍(거리3×조도3×시점4×3) 확보, 분류기 재학습으로 램프 단서 강제, 카메라 단위 분할 정비</td><td>균형쌍 108·재학습</td></tr>
    <tr><td>11/28~12/4</td><td>1클래스 검출기 재검증(mAP50 ≥ 0.65), 저데이터·열화 조건 상태 F1 측정</td><td>조건별 성능표</td></tr>
    <tr><td>12/5~11</td><td>트랙 판정·UNKNOWN·캘리브레이션 확정, 연속 영상에서 시간당 오경보 측정</td><td>이벤트 리포트</td></tr>
    <tr><td>12/12~18</td><td>시스템 통합·지연 측정, 최종보고</td><td>최종보고서</td></tr>
  </table>

  <h3>5.2 판정 목표(사전 고정)</h3>
  <table>
    <tr><th>지표</th><th>기준</th><th>대상</th></tr>
    <tr><td>실제 소등 재현율</td><td>≥ 0.70 (라벨 확보 시)</td><td>실제 ITS</td></tr>
    <tr><td>시간당 오경보</td><td>≤ 1건 / 카메라·시간</td><td>연속 영상</td></tr>
    <tr><td>동일 트랙 중복 알림</td><td>0건</td><td>연속 영상</td></tr>
    <tr><td>알림 지연</td><td>p95 ≤ 2초</td><td>5fps 이벤트</td></tr>
    <tr><td>처리 지연</td><td>평균 ≤ 50ms / p90 ≤ 60ms</td><td>RTX 3080</td></tr>
    <tr><td>캘리브레이션 오차</td><td>ECE ≤ 0.08</td><td>독립 상태분류</td></tr>
  </table>

  <h3>5.3 남은 리스크</h3>
  <ul>
    <li><b>실제 소등 라벨 부재</b>: 소등 재현율은 현재 N/A(n=0). 균형쌍은 광도 합성이라 실제 소등과 다르다.</li>
    <li><b>조건 편중</b>: 원본 프레임이 고조도 구간에 몰려 조도 3구간을 채우지 못했다(현재 8셀/36셀). 야간 수집으로 보완한다.</li>
    <li><b>지연 여유</b>: 평균 54ms로 목표(50ms)에 근접, p90 63ms는 초과. 카메라 수를 늘리면 배치 처리·ONNX 변환이 필요하다.</li>
  </ul>
  <h2>6. 결론</h2>
  <p>공개 데이터가 없는 문제에서 ‘실제 프레임을 배경으로 유지하고 필요한 부분만 생성’하는 방식이 검출 학습에 유효함을 확인했고, 1클래스 검출 + 상태 판정 + 이벤트 판정으로 실제 ITS 스트림에 붙여 실행까지 완료했다. 남은 핵심은 램프 단서를 실제로 학습시키는 것(균형쌍·실제 라벨)이며, 통제쌍 실험으로 그 필요성을 수치로 확인했다.</p>

  <h3>저장소·산출물</h3>
  <ul>
    <li>GitHub (public): <span class="url">https://github.com/dlrjs4127-hue/stealth-lampstate-detector</span></li>
    <li>코드: <span class="url">app/</span>(its_watch, judge, models), <span class="url">src/</span>(합성·학습·평가·수집), <span class="url">configs/</span></li>
    <li>문서: <span class="url">docs/PLAN_FINAL.md</span>, <span class="url">EVALUATION_PLAN.md</span>, <span class="url">DATASET_PLAN.md</span>, <span class="url">MODEL_DESIGN.md</span>, <span class="url">EXPERIMENT_LOG.md</span></li>
    <li>결과: <span class="url">results/</span> (1클래스·균형쌍·실행 지표 JSON, manifest)</li>
  </ul>
</div>
"""

html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{P1}{P2}{P3}{P4}{P5}</body></html>"""

(OUT / "report.html").write_text(html, encoding="utf-8")
print("html written", len(html))


