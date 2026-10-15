# GUISyn GUI 클릭 좌표 학습 실험

MolmoPoint-GUISyn 모바일 화면과 자연어 명령을 입력받아 클릭 좌표를 생성하도록 **Qwen2.5-VL-3B-Instruct**를 NF4 4-bit + LoRA로 추가학습한 실험입니다. Molmo 모델 자체를 학습한 프로젝트는 아닙니다.

## 교수님 검토용 안내

- [실제 모델 예측 이미지 48개](output/v4_step768/): PNG 상단에 명령어·예측 좌표·적중 여부가 있고, 빨간 십자는 예측 지점, 초록 사각형은 원본 정답 박스입니다.
- [예측 좌표 및 명령어 CSV](output/v4_step768/predictions.csv)
- [라벨 수정 전후 이미지](output/v5_label_review/): 빨강은 기존 박스, 초록은 교정 영역, 파랑은 교정 중심입니다.
- [실험별 설정·검증 기록](experiments/)
- [상세 실험 기록과 실행 환경](training/README.md): 과거 시점의 진행 상태도 포함되어 있습니다.

`output/v4_step768/index.html`은 내려받아서 브라우저로 열면 갤러리로 볼 수 있습니다. GitHub에서는 PNG 파일을 직접 열어 확인할 수 있습니다.

## 결과와 해석

| 버전 | 검증 | 별도 테스트 |
|---|---|---|
| v3 | 최고 적중 수 23/48 (47.92%) | 16/47 (34.04%) |
| v4 | 선택 모델 step 768, 24/48 (50.00%) | 학습 직후 자동 평가가 GPU 메모리 부족으로 실패하여 완료 결과 없음 |
| v5 | 시작 모델 24/48 (50.00%), 1,024 step 추가학습 실행 | 완료 후 평가 예정 |

검증은 12개 화면·48개 명령이며 최고 체크포인트 선택에 사용합니다. 별도 테스트는 12개 화면·47개 명령입니다. v3/v4/v5의 학습 화면과 분리되어 있습니다. 초기 버전까지 포함한 미노출 여부는 별도 확인이 필요합니다. 이 결과는 작은 로컬 분할에 대한 수치이며 공식 벤치마크 점수가 아닙니다.

v4의 선택 모델에서 오답 24개 중 12개가 정답 박스 밖 10픽셀 이내였습니다. 일부 라벨은 실제 버튼보다 글자·커서 영역에 가깝게 잡혀 있습니다. 다만 근접 예측이 곧 올바른 대상 인식을 증명하지는 않습니다.

## v5 변경 사항

- v4 최고 체크포인트에서 1,024 step 추가학습, 학습률 1e-5.
- 직접 화면으로 확인한 학습 라벨 6개만 교정. 전체 데이터 라벨을 수정한 것은 아닙니다.
- 학습 데이터에서 박스 밖 10픽셀 이내 오답을 선별하여 일반 샘플과 50:50 재학습.
- 64 step마다 엄격한 박스 적중률과 박스까지 거리 5/10픽셀 지표 기록.
- 검증·테스트 라벨은 유지. 적중 수 최대, 동률 시 중심 오차 최소 모델 선택.
- 학습 종료 후 GPU 컨텍스트를 반환하고 별도 테스트 실행, 좌표 표시 결과 저장.

## 파일 구성

- `training/train_coordinates.py`: 학습, 라벨 교정 적용, 오답 선별, 주기적 검증
- `training/evaluate_coordinates.py`: 정답 응답을 주지 않는 실제 좌표 생성 평가
- `training/visualize_predictions.py`: 명령어·좌표·정답 박스 이미지와 CSV/JSON/HTML 생성
- `data/guisyn-v5-label-overrides.json`: 검수한 라벨 수정 내역
- `experiments/`: 업로드 시점의 설정 및 지표 스냅샷이며 실시간 갱신되지 않음

## 실행 및 재현 범위

Python 3.11과 CUDA GPU를 사용했습니다. CUDA에 맞는 PyTorch를 설치한 뒤 `pip install -r requirements.txt`로 나머지 의존성을 설치합니다. 의존성 목록은 기존 환경을 참고해 작성했으며 새 환경에서의 설치 재현은 아직 검증하지 않았습니다.

**원본 데이터, Hugging Face 캐시, 모델/LoRA 가중치, optimizer 체크포인트는 크기 때문에 포함하지 않았습니다.** 기존 학습의 연속 실행에는 데이터 manifest와 이전 체크포인트를 별도로 준비해야 합니다. 일부 기록에는 실행 당시 절대 경로가 남아 있어 다른 환경에서는 경로 조정이 필요합니다. 이 저장소만으로 v5 학습을 즉시 재개할 수는 없습니다.

```bash
python training/launch.py --trainer train_coordinates.py \
  --output runs/qwen2.5-vl-3b-guisyn-v5 \
  --previous runs/qwen2.5-vl-3b-guisyn-v4 \
  --init-adapter runs/qwen2.5-vl-3b-guisyn-v4/best \
  --additional-sources data/guisyn-additional/source_annotations.json \
  --steps 1024 --max-pixels 802816 --learning-rate 1e-5 \
  --coordinate-loss-weight 1 --bbox-loss-weight 0.5 \
  --select-by bbox --hard-replay --near-miss-pixels 10 \
  --label-overrides data/guisyn-v5-label-overrides.json \
  --filter-ambiguous --validate-every 64
```

## 출처

- 데이터와 출력 이미지의 원본: [allenai/MolmoPoint-GUISyn](https://huggingface.co/datasets/allenai/MolmoPoint-GUISyn)
- 기반 모델: [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

원본 데이터와 모델의 이용 조건은 각각의 제공처를 따릅니다.
