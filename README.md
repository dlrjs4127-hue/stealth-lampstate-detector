# 야간 CCTV 스텔스(램프 소등) 차량 검출 연구

야간 고정 CCTV에서 **전조등·후미등을 끄고 주행하는 차량(스텔스 차량)** 을 찾는 연구입니다.
경희대학교 산업경영공학과 독립심화학습2(2026-2학기) 과제로 진행하며, 데이터 파이프라인·모델·판정 시스템·실험 기록을 함께 관리합니다.

- **보고서(PDF, 5p)**: [docs/REPORT_4차보고서.pdf](docs/REPORT_4차보고서.pdf)
- **최종 계획(확정)**: [docs/PLAN_FINAL.md](docs/PLAN_FINAL.md)
- **실행 프로그램**: [pp/its_watch.py](app/its_watch.py)

## 핵심 결과 요약

| 항목 | 값 | 데이터 |
|---|---|---|
| ROI 편집 합성 | 177장 (ROI 밖 픽셀 변경 0) | SYN-ROI |
| 1클래스 차량 검출 (YOLO26n) | mAP50 **0.746**, P 0.891, R 0.646 | SYN-ROI 학습 / 실제 프레임 학습 포함 |
| 상태 판정 (YOLOv8n-cls) | top-1 1.000 (628 ON/OFF 쌍) | PUBLIC-PAIRED |
| 파운데이션 특징 비교 (DINOv2-S + probe) | 0.726 (JPEG35 0.578) — 특화 CNN 열세 확인 | PUBLIC-PAIRED |
| 램프 균형쌍 통제 실험 | OFF 1.00 / ON **0.30** → 분류기가 램프 아닌 차체 어두움에 의존 | SYN-ROI |
| ITS API 실행 | 36프레임·12카메라, 관측 165, 확정 알림 **0**, 54.3ms/p90 62.6ms | REAL-ITS |
| 열화 시험 | 블러 0.81 / 흔들림 0.95 / JPEG35 0.81 / 노이즈 1.14 유지 | REAL-ITS |

> 지표 표기 규칙: REAL-ITS(실제 프레임) / PUBLIC-PAIRED(공개 ON/OFF 대응쌍) / SYN-ROI(ROI 편집 합성). 실제 소등 라벨이 없으므로 실제 소등 재현율은 **N/A(n=0)** 입니다.

## 문제 정의

- **데이터 부재**: 고정 CCTV 시점에서 '램프 소등' 라벨이 붙은 공개 데이터는 사실상 없습니다. 있는 것은 근접 차량 ON/OFF 크롭과 라벨 없는 ITS 스트림뿐입니다.
- **신호 취약성**: 차량 폭 중앙값 50px(18~110px), 후미등은 2~4px. 램프가 꺼지면 차체가 배경과 거의 같은 밝기가 됩니다.
- **운영 요구**: 실시간 처리(≤50ms/프레임), 카메라별 오탐 억제, 판정 근거의 감사 가능성.

## 방법

### 1) ROI 편집 합성 (데이터)
도로 전체를 생성하면 카메라 특성이 달라 전이가 되지 않습니다. **실제 프레임은 그대로 두고 빈 차로 ROI(200×150)만** 생성 모델로 편집합니다.

`
실제 프레임 → 빈 차로 ROI 자동 선정 → 생성 모델 편집(소등 차량 1대)
            → 원위치 복원(8px feather) → BBOX 자동 검출·기록
`
검증: ROI 밖 픽셀 변경 **0건**, 차량 BBOX 확보 94/112, 차량 폭 18~110px.

### 2) 2단계 판정 (모델)
| 단계 | 모델 | 역할 |
|---|---|---|
| Stage-1 | YOLO26n, 1클래스(vehicle), 960px | 차량 위치 검출. 소등 판단은 하지 않음 |
| Stage-2 | YOLOv8n-cls, 160px | 차량 크롭의 램프 상태(ON/OFF) |
| 이벤트 판정 | EWMA α=0.45, 진입 0.80 / 해제 0.40, 최소 3회 | 트랙 단위 증거 누적·중복 알림 억제 |
| UNKNOWN | 폭 40px 미만·판독불가·불확실 | 소등으로 확정하지 않고 보류 |

### 3) 실행 프로그램 (pp/)
- --mode collect: ITS Open API에서 클립 수집 → ffmpeg 프레임 추출(중복 sha256 제거)
- --mode run: 검출 → 상태판정 → 이벤트 판정 → 알림(JSONL) + HTML 리포트
- 카메라별 트랙 분리·만료, 오버레이(자막) 가드, 감사 로그(BBOX·상태·신뢰도·트랙ID·UNKNOWN 사유)

## 저장소 구조

| 경로 | 내용 |
|---|---|
| pp/ | 실행 프로그램(its_watch.py), 이벤트 판정(judge.py), 모델 슬롯(models.py) |
| src/ | ROI 합성(배치1·2), 균형쌍 생성, 1클래스/2클래스 학습, DINO probe, 열화 시험, 야간 수집, 지표 계산 |
| configs/ | 데이터셋·학습 설정 |
| docs/ | **REPORT_4차보고서.pdf**, PLAN_FINAL, EVALUATION_PLAN, DATASET_PLAN, MODEL_DESIGN, EXPERIMENT_LOG, MAP |
| esults/ | 실험 지표 JSON, 장면 manifest, 실행 결과(JSONL) |

## 재현

`ash
pip install ultralytics opencv-python numpy requests
python src/build_balanced_pairs.py        # 램프 균형쌍 생성
python src/train_1class_vehicle.py        # 1클래스 검출기 학습
python src/train_stage2_and_pipeline.py   # 상태 판정 + 파이프라인 평가
python app/its_watch.py --mode collect --clips 12 --frames 3
python app/its_watch.py --mode run --frames <수집폴더>
`

## 데이터·권리 주의
- 원본 ITS 프레임과 합성 이미지는 저장소에 포함하지 않습니다(권리 확인 전). 코드·지표·manifest만 공개합니다.
- 공개 ON/OFF 크롭은 CC BY 4.0 출처를 따릅니다.

## 다음 단계 (요약)
1. 램프 균형쌍 108쌍 확보 및 상태분류 재학습(램프 단서 강제)
2. 1클래스 검출기 재검증(mAP50 ≥ 0.65) 및 저데이터·열화 조건 평가
3. 이벤트 판정 확정(트랙·UNKNOWN·캘리브레이션), 연속 영상에서 시간당 오경보 측정
4. 최종보고(2026-12-18)
