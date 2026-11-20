# 저장소 구조 (MAP)

| 경로 | 내용 |
|---|---|
| `README.md` | 연구 개요, 방법, 결과 요약, 재현 방법 |
| `configs/` | 데이터셋·학습 하이퍼파라미터 설정 |
| `docs/DATASET_CARD.md` | 데이터 구성·라벨 규칙·권리와 배포 범위 |
| `docs/EVALUATION_PLAN.md` | 평가 단위·지표 정의·판정 기준·실험 매트릭스 |
| `docs/REPORT_4차보고서.pdf` | 제출용 A4 5페이지 보고서(문서형) |
| `docs/WORKLOG_2026-11-20.md` | 2026-11-20 작업 정리(한 일·수치·제출 이력·남은 일) |
| `docs/EXPERIMENT_LOG.md` | 실험별 조건·수치·관찰 기록 |
| `docs/MAP.md` | 이 문서. 저장소 구조 안내 |
| `results/metrics_summary.md` | 실험 5종 결과 요약표 |
| `src/` | 데이터 구축·학습·평가 스크립트 |

## 작업 흐름
1. `src/build_dataset_from_generated_scenes.py` — 생성 장면/합성본에서 2클래스 데이터셋 구성
2. `src/train_detector_its_composite.py` — ITS 합성본 2클래스 학습 + 실제 프레임 평가
3. `src/train_detector_lighting_variants.py` — 조도·노출·색온도 변형 7종 + 실제 OFF 혼합 학습
4. `src/train_detector_photometric_off.py` — 실제 프레임 차량의 램프만 광도 변환해 소등 표본 생성 후 학습
5. `src/eval_on_its_frames.py` — 실제 프레임 추론·주석 이미지 생성
6. `src/evaluate_predictions.py` — 라벨/예측 CSV로 계획서 지표 계산(재현율·정밀도·F1·FP/frame·Wilson 95%)


