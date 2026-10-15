# MolmoPoint-GUISyn 데이터로 GUI 클릭 좌표 학습하기

이 문서는 로컬 GPU 문제 해결, 1차·2차 학습 완료 결과와 새 화면 추가 학습(v3)을 정리한 작업 기록입니다.
**최신 적중률 중심 추가 학습(v4)은 13절을 참고하세요. 12절은 v3, 아래 1절의 진행률은 과거 v2 기록입니다.**
**MolmoPoint-GUISyn 데이터를 이용해 Qwen2.5-VL-3B를 미세 조정하는 작업**이며, MolmoPoint 모델 자체를 학습하는 것은 아닙니다.
모든 시간은 한국 시간(KST) 기준입니다.

## 1. 학습 중 상태 기록

**최신 완료 결과와 1차·2차 학습 방법·정확도 비교는 문서 마지막 11절에 있습니다.**

**2026-10-14 22:49 확인 기록입니다. 아래 진행률은 실시간으로 갱신되지 않습니다.**

| 항목 | 확인 결과 |
|---|---|
| 현재 작업 | 좌표 오차를 줄이기 위한 2차 학습(v2) |
| 모델 | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 학습 방식 | CUDA 전용, NF4 4비트 양자화 + LoRA |
| 프로세스 | PID `32895` — 이번 실행의 PID이며 다음 실행에서는 달라짐 |
| 진행 | 540 / 1,024 step, 약 52.7% |
| 최근 전체 손실 | 1.25349 |
| 응답 토큰 교차 엔트로피 | 1.13764 |
| 좌표 보조 손실 | 0.03862 — 전체 손실에는 3배 가중치로 반영 |
| 최근 처리 속도 | step당 약 5.9초 |
| 완료 예상 | 검증·최종 평가 포함 약 50~60분 후, 23:40~23:50경 추정 |
| 최신 저장 체크포인트 | `checkpoint-512` |
| 현재 선택된 최적 체크포인트 | `best`, 256 step — 검증 평균 좌표 오차가 가장 낮음 |
| 최종 완료 여부 | 아직 완료되지 않음. `result.json`과 `final`은 완료 후 생성 |
| 출력 경로 | `runs/qwen2.5-vl-3b-guisyn-v2` |

GPU는 22:40 호스트 측정에서 온도 **87°C**, 사용률 **96%**, 전체 메모리 **4,604 / 6,144 MiB**였습니다.
학습 프로세스의 메모리는 4,584 MiB이며, 22:49까지 PyTorch 최대 할당량은 4,230.5 MiB입니다.
전체 GPU 점유량에는 CUDA 실행 환경과 메모리 할당기의 예약 영역 등이 포함되므로 PyTorch 할당량과 다릅니다.
22:13 진단에서는 `SW Thermal Slowdown: Active`가 확인되어 온도에 따른 클록 조절이 적용되고 있었습니다.
확인 시점까지 학습은 계속 진행됐고, 새 Xid/GSP 초기화 오류는 발견되지 않았습니다.

## 2. 실행 환경과 데이터

| 항목 | 설정 |
|---|---|
| 프로젝트 | `/home/codebiney/project/robot_sim/molmo2` |
| 학습 명령 실행 위치 | `/home/codebiney/project/robot_sim/molmo2/molmo` |
| Python | `/home/codebiney/project/robot_sim/.venv/bin/python` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB |
| NVIDIA 드라이버 | `595.99.02-open` |
| 커널 | `7.0.0-38-generic` |
| PyTorch | `2.14.1+cu130` |
| Transformers | `4.57.6` |
| 프로젝트 내부 패키지 | `vendor`의 PEFT 0.17.1, bitsandbytes 0.50.2, num2words 0.5.14, docopt 0.6.2 |
| 기본 모델 캐시 | `data/huggingface` |
| 데이터와 원본 주석 | `data/guisyn` |

- 데이터: [allenai/MolmoPoint-GUISyn](https://huggingface.co/datasets/allenai/MolmoPoint-GUISyn)
- 현재 모델: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- 초기 준비 모델: [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)

GUISyn의 `mobile` 구성, `train` 분할에서 준비한 128개 화면을 사용합니다.
정답은 화면 너비·높이를 각각 기준으로 0~1000 정수 좌표로 정규화한 `(x, y)`입니다.
예를 들어 `(500, 500)`은 화면 중앙입니다. 픽셀 좌표는 `x * 너비 / 1000`, `y * 높이 / 1000`으로 환산합니다.

**GPU 진단과 학습은 승인된 호스트 실행에서 해야 합니다.** 실행 샌드박스에는 GPU 장치가 숨겨져 있습니다.
학습 코드에는 자동 CPU 대체나 CPU로 모델 가중치를 옮기는 동작이 없습니다.

## 3. 재학습(v2)을 설계한 이유

1차 학습은 CUDA 연산, 역전파, LoRA 가중치 갱신과 저장이 정상 동작함을 확인했습니다.
하지만 실제 좌표 생성 평가에서는 목표 버튼·아이콘 적중률이 2/47(4.26%)에 그쳤습니다.
손실이 낮아지는 것만으로 정확한 클릭 위치를 찾는다고 판단할 수 없었습니다.

이에 따라 다음을 바꿨습니다.

1. 화면당 최대 4개 주석을 쓰던 방식에서 저장된 학습용 화면의 전체 유효 주석을 활용하도록 확대했습니다.
2. 이미지 면적 상한을 시각 토큰 256개에서 768개로 높여 작은 글자와 컨트롤을 더 선명하게 입력합니다.
3. LoRA 학습 대상을 언어 모델의 query/value 투영에서 attention·MLP의 7개 투영 전체로 확대했습니다.
4. 좌표 숫자 토큰을 더 중시하고, 예측 숫자 확률에서 계산한 좌표와 정답의 거리를 줄이는 보조 손실을 추가했습니다.
5. 학습 손실이 아니라 별도 검증 화면에서 직접 생성한 좌표 오차로 최적 모델을 선택합니다.

## 4. v2 데이터 분리와 학습 설정

### 데이터 분리

| 용도 | 화면 수 | 데이터 |
|---|---:|---|
| 학습 | 104 | 2,574개 클릭 대상, 12,870개 지시문 표현 |
| 학습 중 검증 | 12 | 48개 지시문 |
| 최종 평가 | 12 | 기존 1차 평가와 동일한 47개 지시문 |

같은 화면이 학습·검증·최종 평가에 겹치지 않게 분리합니다.
원래 최종 평가에 쓰던 12개 화면은 계속 제외하고, 나머지 116개 중 12개를 검증용으로 별도 분리했습니다.
학습 대상은 매 순회마다 섞고, 같은 대상의 지시문 표현은 방문할 때마다 순환합니다.
12,870개는 준비된 표현의 수이며, 현재 1,024 step에서는 총 4,096번의 개별 샘플 학습을 수행합니다.
모든 표현을 한 번씩 학습한다는 뜻은 아닙니다.

분리 내역과 실제 사용 데이터는 `screenshot_split.json`, `train_targets.json`, `validation.json`, `heldout.json`에 저장됩니다.

### 모델·최적화 설정

| 항목 | 설정 |
|---|---|
| 기본 모델 | Qwen2.5-VL-3B-Instruct, 캐시에서 로드 |
| 시작 가중치 | 기본 모델에서 새 LoRA 생성 — 1차 LoRA를 이어 학습하지 않음 |
| 양자화 | NF4 4비트, 이중 양자화 |
| 연산 정밀도 | BF16 autocast, 작은 정규화·bias 파라미터는 FP32 |
| 동결 부분 | 기본 모델, 이미지 인코더, 임베딩 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA 대상 | 언어 모델의 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| 학습 파라미터 수 | 29,933,568개 |
| 배치 / 누적 | 배치 1, gradient accumulation 4 |
| optimizer step | 1,024 |
| 초기 학습률 | 1e-4 |
| 학습률 변화 | 64 step warmup 후 cosine 감소, 마지막에는 초기값의 10% |
| 이미지 면적 상한 | 602,112 pixels = `768 * 28 * 28`, 시각 토큰 최대 768개 |
| 메모리 절약 | gradient checkpointing, 응답 토큰에 대해서만 vocabulary logits 계산 |
| 검증·저장 | 첫 step 저장, 이후 128 step마다 검증과 체크포인트 저장 |
| 난수 seed | 6198 |

시각 토큰 상한은 이미지 **면적** 제한입니다. 화면의 가로·세로 길이가 각각 768픽셀이라는 뜻은 아닙니다.

### 좌표 오차를 반영한 손실

전체 손실은 다음과 같습니다.

```text
전체 손실 = 숫자 토큰 가중 교차 엔트로피 + 3 × 좌표 보조 손실
```

교차 엔트로피는 정답 응답 토큰에만 계산하며, 숫자 토큰의 가중치는 2, 나머지는 1입니다.
좌표 보조 손실은 각 자리의 0~9 숫자 확률에서 기대 숫자를 구하고, 자릿값을 적용한 x/y 좌표를
0~1로 정규화한 뒤 정답과 SmoothL1(beta=0.05)로 비교합니다.
이는 좌표 오차 감소를 목표로 추가한 실험적 손실입니다. 학습 중 기대 좌표가 좋아져도 실제 생성 정확도는 별도 검증해야 합니다.

### 최적 모델 선택과 완료 처리

128 step마다 정답 응답을 넣지 않고 좌표를 직접 생성하여 검증합니다.
`best`는 검증 평균 정규화 중심점 오차가 가장 낮은 체크포인트입니다.
잘못된 형식이나 범위 밖 좌표는 최대 거리 `sqrt(2) * 1000`인 실패로 처리합니다.

학습 완료 후에는 다음 순서로 처리합니다.

1. 마지막 학습 가중치를 `last`에 저장합니다.
2. 검증에서 선택한 `best` 가중치와 해당 optimizer 상태를 `final`에 저장합니다.
3. `result.json`에 완료 step, 선택 step과 검증 결과를 기록합니다.
4. `evaluate_coordinates.py`로 기존 최종 평가 47개 문항을 자동 평가합니다.

`final`은 마지막 step의 가중치와 다를 수 있습니다.
`result.json`은 학습·최적 모델 저장 완료를 뜻하며, 최종 좌표 평가 완료는
`coordinate_eval/summary.json`과 `train.log`의 `coordinate evaluation complete` 메시지로 확인합니다.

## 5. 현재까지의 평가 결과

### 1차 모델: 최종 평가 47개 문항

1차 모델은 rank-8 query/value LoRA, 256개 시각 토큰, 128 step으로 학습했습니다.
116개 학습 화면의 463개 지시문을 사용했고, 학습은 21:26경 정상 완료했습니다.
학습 step 소요 시간은 347.8초였으며, 최종 학습 loss는 1.06933,
16개 문항의 teacher-forced 검증 loss는 1.07161이었습니다.

실제 좌표 생성 결과는 다음과 같습니다.

| 지표 | 기본 모델(LoRA 비활성화) | 1차 학습 모델 |
|---|---:|---:|
| 목표 영역 안에 들어간 좌표 | 1/47, 2.13% | 2/47, 4.26% |
| 유효한 좌표 형식 | 45/47, 95.74% | 47/47, 100% |
| 중심점 픽셀 오차 중앙값 | 255.86px | 64.90px |
| 중심점 픽셀 오차 평균 | 287.21px | 191.33px |
| 정규화 중심점 거리 50 이하 | 4.26% | 23.40% |

픽셀 오차는 유효한 좌표 출력만을 대상으로 합니다.
유효하지 않은 출력은 적중률과 거리 임계값 적중률 계산에서는 실패로 포함합니다.

결과는 `runs/qwen2.5-vl-3b-guisyn/coordinate_eval`의 `summary.json`, `report.md`, `predictions.jsonl`에 있습니다.

### v2: 학습 중 검증 48개 문항

**아래는 학습 중 모델 선택용 검증 결과이며, 위의 최종 평가 47개 문항과 다른 데이터입니다.**
두 표의 적중률을 직접 비교해 최종 개선 폭으로 해석하면 안 됩니다.

| step | 좌표 형식 성공 | 목표 영역 적중 | 픽셀 오차 중앙값 | 모델 선택용 평균 정규화 오차 |
|---:|---:|---:|---:|---:|
| 0 | 22/48 | 0/48, 0% | 202.45px | 938.59 |
| 128 | 48/48 | 5/48, 10.42% | 53.96px | 175.33 |
| 256 | 48/48 | 10/48, 20.83% | 45.29px | **136.15** |
| 384 | 48/48 | 17/48, 35.42% | 33.94px | 138.00 |
| 512 | 48/48 | 15/48, 31.25% | 30.86px | 151.79 |

픽셀 오차 중앙값은 유효한 출력만 사용합니다. 0 step은 유효한 출력이 22개뿐이므로 이후 수치와 비교할 때 주의해야 합니다.
선택용 평균 오차는 형식 오류도 최대 거리 실패로 포함합니다.
현재 `best`가 256 step인 이유는 적중률이나 중앙값이 아니라 **평균 정규화 오차**로 선택하기 때문입니다.
최종 개선 여부는 학습 완료 후 원래 47개 문항을 다시 평가하여 확인합니다.
이 평가는 128개 준비 화면에서 나눈 작은 로컬 평가이며 공식 GUISyn/ScreenSpot 벤치마크 점수가 아닙니다.

## 6. 실행·진행 확인 방법

아래 명령은 프로젝트의 `molmo` 디렉터리에서 실행합니다.
GPU 관련 명령은 호스트 실행이 필요합니다.

```bash
cd /home/codebiney/project/robot_sim/molmo2/molmo
```

### 현재 재학습을 시작한 명령 — 기록용

```bash
/home/codebiney/project/robot_sim/.venv/bin/python training/launch.py \
  --trainer train_coordinates.py \
  --output runs/qwen2.5-vl-3b-guisyn-v2 \
  --steps 1024
```

`launch.py`는 백그라운드 프로세스를 만들고 PID와 로그를 저장합니다.
현재 실행 중인 작업에 대해 이 명령을 다시 실행할 필요는 없습니다.
새 학습은 새로운 출력 디렉터리를 사용해야 하며, 같은 출력 경로로 실행한다고 기존 학습이 재개되지 않습니다.

### 실시간 진행 확인

```bash
tail -f runs/qwen2.5-vl-3b-guisyn-v2/train.log
tail -n 5 runs/qwen2.5-vl-3b-guisyn-v2/metrics.jsonl
tail -n 5 runs/qwen2.5-vl-3b-guisyn-v2/validation_metrics.jsonl
nvidia-smi
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
```

### 완료 후 결과 확인

```bash
cat runs/qwen2.5-vl-3b-guisyn-v2/result.json
cat runs/qwen2.5-vl-3b-guisyn-v2/coordinate_eval/summary.json
```

최종 좌표 평가를 다시 실행할 때는 `final`이 저장된 후 아래 명령을 사용합니다.
기존 평가 결과 파일은 다시 기록됩니다.

```bash
/home/codebiney/project/robot_sim/.venv/bin/python training/evaluate_coordinates.py \
  --run runs/qwen2.5-vl-3b-guisyn-v2
```

## 7. 중단할 때 저장되는 범위와 재개 제한

**현재 실행 중인 v2 코드는 중단 신호를 받는 순간 최신 가중치를 자동 저장하지 않습니다.**
`SIGINT`, `SIGTERM`, 예외 종료에 대한 저장 처리나 `finally` 저장 처리가 없습니다.
저장이 끝난 체크포인트는 디스크에 남지만, 마지막 저장 이후의 학습분은 잃습니다.
예를 들어 540 step에서 중단하면 최신 주기 저장인 512 step까지 남고 513~540 step은 저장되지 않습니다.

각 체크포인트에는 다음이 들어 있습니다.

- `adapter_model.safetensors`: 학습한 LoRA 가중치
- `adapter_config.json`: LoRA 설정
- 전처리기·토크나이저·대화 템플릿 파일
- `training_state.pt`: 저장 step과 optimizer 상태

기본 모델 전체는 체크포인트마다 복제하지 않고 `data/huggingface` 캐시를 이용합니다.
`checkpoint-N`은 해당 step의 저장본이고, `best`는 검증 평균 오차로 선택한 저장본입니다.
저장 중 강제 종료하면 파일이 완성되지 않을 수 있으므로 `saved ...` 로그와 저장 파일을 확인해야 합니다.

**현재 스크립트에는 체크포인트에서 학습을 이어가는 `--resume` 기능이 없습니다.**
또한 난수 상태와 샘플 순회 상태는 저장하지 않습니다.
저장된 LoRA를 추론·평가하는 것은 가능하지만, 중단 지점부터 동일한 학습을 재개하려면 코드 보완이 필요합니다.
사용자가 중단을 요청하기 전까지 학습을 유지합니다. 앞선 중단 관련 요청에서는 상태만 확인했고 학습을 멈추지 않았습니다.

## 8. 저장 파일과 코드 역할

| 경로 | 용도 |
|---|---|
| `training/verify_gpu.py` | 실제 CUDA 행렬 연산 확인 |
| `training/repair_runtime_pm.sh` | GPU runtime D3 비활성화 설정과 initramfs 갱신 |
| `training/launch.py` | 선택한 학습 스크립트를 백그라운드 실행, PID·로그 저장 |
| `training/train_small.py` | 초기 소규모 학습. 기본값은 SmolVLM2이며 Qwen 1차 학습에도 사용 |
| `training/train_coordinates.py` | 현재 좌표 오차 중심 재학습(v2) |
| `training/evaluate_coordinates.py` | 좌표 직접 생성, 원본 목표 영역 적중·중심점 거리 평가 |
| `data/guisyn/examples.json` | 초기 화면당 최대 4개 주석의 샘플 목록 |
| `data/guisyn/source_annotations.json` | 준비된 128개 화면의 원본 전체 주석 |
| `runs/smolvlm2-guisyn` | 초기 SmolVLM 실행 실패 기록 |
| `runs/qwen2.5-vl-3b-guisyn` | 1차 완료 모델과 최종 좌표 평가 |
| `runs/qwen2.5-vl-3b-guisyn-v2` | 현재 재학습 결과와 로그 |

v2 결과 경로의 주요 파일은 다음과 같습니다.

```text
train.pid                       이번 실행 PID
train.log                       학습·검증·저장 로그
run_config.json                 실제 학습 설정
metrics.jsonl                   step별 손실·gradient norm·GPU 최대 할당량
validation_metrics.jsonl        검증별 좌표 지표
validation-N.json               N step의 검증 지표와 개별 예측
screenshot_split.json           학습·검증·최종 평가 화면 구분
train_targets.json              학습 대상과 지시문 표현
validation.json                 학습 중 검증용 48개 문항
heldout.json                    최종 평가용 47개 문항
checkpoint-N/                   N step의 가중치·optimizer·전처리기
best/                           현재까지 검증 평균 좌표 오차가 가장 낮은 저장본
last/                           전체 학습 완료 시 마지막 가중치
final/                          완료 시 best에서 선택한 최종 가중치
result.json                     학습 완료·선택 step·검증 결과
coordinate_eval.log             완료 후 좌표 평가 로그
coordinate_eval/summary.json    최종 좌표 평가 집계
coordinate_eval/report.md       최종 좌표 평가 보고서
coordinate_eval/predictions.jsonl  문항별 좌표·정답 영역·오차
```

## 9. GPU 문제 해결 기록

### 초기 문제

2026-10-14 GPU 절전 복귀 시 GSP 초기화에 실패하여 Xid 119(GSP RPC timeout), Xid 154(reset 필요)가 발생했습니다.
`nvidia-smi -q`에 `GPU requires reset`이 표시됐고 PyTorch 2.6.0+cu124와 2.14.1+cu130 모두 CUDA 초기화에 실패했습니다.
20:10 재부팅 후에도 20:12~20:13에 GFW boot failure와 같은 Xid가 재현됐습니다.
당시 드라이버의 `DynamicPowerManagement: 2`와 약 104초의 runtime suspend 기록이 확인되어 절전 복귀 문제가 의심됐습니다.

### 적용한 조치

20:17 관리자 인증을 통해 아래 조치를 적용했습니다.

- `/etc/modprobe.d/nvidia-runtimepm.conf`: `options nvidia NVreg_DynamicPowerManagement=0x00`
- `/etc/udev/rules.d/99-nvidia-no-runtime-pm.rules`: NVIDIA 화면 장치의 `power/control=on` 규칙
- 커널 `7.0.0-38-generic`의 initramfs 갱신
- 원래 설정 백업: `/var/backups/nvidia-runtimepm.TGqtRzK1`

`repair_runtime_pm.sh`는 위 조치를 수행하며 데스크톱 종료나 자동 재부팅을 하지 않습니다.
관리자 인증이 필요하고, runtime D3를 비활성화하면 유휴 상태의 전력 사용이 늘 수 있습니다.
이 조치는 이미 적용됐습니다.

### 완전 종료·재전원 후 확인

20:21:49 부팅 후 호스트에서 다음을 확인했습니다.

- `nvidia-smi` 정상 응답, RTX 3060 Laptop GPU와 정상 온도·전력 표시
- 적용된 모듈 설정: `DynamicPowerManagement: 0`
- `/proc/driver/nvidia/gpus/0000:01:00.0/power`: `Runtime D3: Disabled`
- `verify_gpu.py`: CUDA 사용 가능, 64×64 행렬 연산 결과 `64.0`
- 새 부팅 로그에 Xid 119/154, GSP 초기화 timeout 없음

다만 GPU PCI `power/control`은 udev 규칙의 `on`과 달리 `auto`였습니다.
확인 당시 runtime 상태는 `active`, suspend 누적 시간은 0이었으며 드라이버의 D3 비활성화는 적용됐습니다.
PCI의 `on` 설정이 부팅 후에도 유지되는지는 확인되지 않았습니다.
따라서 절전 관련 모든 설정이 의도대로 적용됐다고 단정하지 않고 이 차이를 기록합니다.

호스트에서의 확인 명령은 다음과 같습니다.

```bash
nvidia-smi
cat /proc/driver/nvidia/params | rg DynamicPowerManagement
cat /proc/driver/nvidia/gpus/0000:01:00.0/power
/home/codebiney/project/robot_sim/.venv/bin/python training/verify_gpu.py
```

CUDA 확인에 실패하면 CPU 학습으로 대체하지 않습니다.
Xid/GSP 문제가 다시 발생하면 그 부팅의 커널 로그를 확인해야 합니다.

## 10. 학습 준비 중 해결한 문제와 확인 범위

- SmolVLM 재시도는 CUDA를 통과했지만 `num2words` 누락으로 전처리기 로드에 실패했습니다.
  필요한 패키지는 `vendor`에 설치했고 기존 PyTorch·Transformers를 교체하지 않았습니다.
- Qwen 원본 가중치는 약 7.5GB이며, Xet 병렬 전송으로 내려받아 현재 캐시에 있습니다.
  20:31경 시작한 전송은 21:18~21:19경 완료됐습니다. 현재 재학습은 다시 다운로드하지 않습니다.
- 처음 Qwen을 준비할 때 PEFT의 일반 준비 함수가 큰 BF16 임베딩·이미지 가중치를 FP32로 확장해 GPU 메모리 부족이 발생했습니다.
  기본 가중치를 직접 동결하고 작은 1차원 파라미터만 FP32로 유지하도록 수정해 해결했습니다.
- LoRA 첫 step 갱신 확인은 B 행렬을 사용합니다. B가 처음에 0이므로 A 행렬은 첫 step에서 그대로일 수 있습니다.
- 1차 학습 중 일시적인 메모리 할당 경고는 할당기 재시도 후 회복되어 128 step과 최종 저장이 완료됐습니다.
  로그는 누적 기록이므로 이전 실패와 최신 실행 상태를 구분해야 합니다.

모델은 이미지 부분을 포함해 3,754,622,976개의 고유 파라미터를 갖고 입력·출력 임베딩을 공유합니다.
모두 4비트라면 순수 가중치는 약 1,790 MiB입니다.
현재처럼 이미지 인코더·임베딩을 BF16으로 남기는 구성의 기본 가중치와 양자화 정보는 약 3,235 MiB로 추정됩니다.
LoRA, gradient, optimizer, 중간 계산값과 CUDA 실행 환경은 별도이므로 실제 학습 점유량은 더 큽니다.

CUDA NF4 순전파·역전파와 LoRA optimizer 갱신을 실제 GPU에서 확인했습니다.
평가·새 손실 코드에서는 좌표 형식, 목표 영역 경계, 비정사각형 화면의 픽셀 환산,
잘못된 출력의 분모 처리, 정답 좌표의 보조 손실, 가까운/먼 예측의 벌점 순서,
유한한 gradient, 응답 전용 교차 엔트로피 정렬, 화면 분리의 비중복을 검증했습니다.

## 11. 1차·2차 학습 방법과 최종 정확도 비교 — 완료 기록

**2차 학습과 최종 좌표 평가가 모두 완료됐습니다.**

- 1차 학습: 2026-10-14 21:26경 완료, 128/128 step.
- 2차 학습·최종 가중치 저장: **2026-10-14 23:41:23 KST**, 1,024/1,024 step.
- 2차 최종 좌표 평가: **2026-10-14 23:44:02 KST**, 47/47개 문항 평가 완료.
- 2차 최종 모델: 검증 평균 정규화 오차로 선택한 **1024 step**의 LoRA.
- 두 실행 모두 모든 학습 step의 loss와 gradient norm이 유한함을 확인했습니다.
- 2차 `best`와 `final`의 LoRA 파일 SHA-256이 일치하여 선택된 가중치가 최종 저장됐음을 확인했습니다.

### 11.1 실제 사용한 데이터 범위

**1차와 2차 모두 `allenai/MolmoPoint-GUISyn`의 `mobile`, `train` 데이터 한 종류만 사용했습니다.**
원본 행 0~127의 128개 화면으로 실험했으며, 데이터셋 전체나 세 종류 데이터의 혼합 학습을 수행한 결과가 아닙니다.
학습·검증·최종 평가라는 세 분할은 서로 다른 데이터셋 세 개를 뜻하지 않습니다.

최종 평가에는 두 실행에서 **동일한 12개 화면·47개 지시문**을 사용했습니다.
1차·2차 `heldout.json`의 내용이 동일함을 대조했고, 예측 파일에서 적중 수와 픽셀 오차를 다시 계산해 집계와 일치함을 확인했습니다.

### 11.2 학습 방법 비교

두 실행 모두 화면 이미지와 지시문을 입력받아 0~1000 정수 `(x, y)` 좌표를 출력하도록 가르치는 지도학습입니다.
기본 모델·이미지 인코더·임베딩은 동결하고, 추가된 LoRA만 학습합니다.
언어 모델의 주요 가중치는 NF4 4비트, 이미지 인코더와 임베딩은 BF16이며 자동 CPU 대체는 없습니다.

| 항목 | 1차 학습 | 2차 재학습(v2) |
|---|---|---|
| 기본 모델 | Qwen2.5-VL-3B-Instruct | 동일한 기본 모델에서 새 LoRA 생성 |
| 학습 데이터 | 116개 화면의 선별된 463개 지시문 | 104개 화면의 2,574개 대상·12,870개 표현 준비 |
| 별도 모델 선택용 검증 | 없음 | 12개 화면·48개 지시문 |
| 최종 평가 | 12개 화면·47개 지시문 | 1차와 동일한 47개 지시문 |
| 배치 / 누적 | 1 / 4 | 1 / 4 |
| optimizer step | 128 | 1,024 |
| 개별 샘플 학습 횟수 | 512회 | 4,096회 — 준비된 모든 표현을 한 번씩 사용한 것은 아님 |
| LoRA rank / alpha | 8 / 16 | 16 / 32 |
| LoRA 대상 | 언어 attention의 `q_proj`, `v_proj` | 언어 attention·MLP의 7개 투영 전체 |
| 학습 파라미터 | 1,843,200개 | 29,933,568개 |
| 시각 토큰 상한 | 256, 면적 200,704 pixels | 768, 면적 602,112 pixels |
| 손실 | 정답 응답 토큰 교차 엔트로피 | 숫자 토큰 2배 가중 교차 엔트로피 + 3 × 좌표 SmoothL1 보조 손실 |
| 학습률 | 고정 2e-4 | 1e-4, 64 step warmup 후 cosine 감소 |
| 최종 모델 선택 | 마지막 128 step | 별도 검증 평균 좌표 오차가 최소인 step; 이번에는 1,024 step |
| PyTorch 최대 할당량 | 3,938.0 MiB | 4,230.5 MiB |

2차의 좌표 보조 손실은 정답 응답을 보면서 다음 토큰을 예측하는 학습 과정에서,
숫자 확률과 자릿값으로 계산한 기대 x/y 좌표를 정답과 비교하는 실험적 방법입니다.
실제 평가에서는 정답 좌표를 입력하지 않고 이미지·지시문만으로 좌표를 생성합니다.
2차에서는 큰 이미지·프롬프트 전체의 vocabulary logits 대신 응답 예측에 필요한 logits만 계산해 GPU 메모리를 줄였습니다.

1차는 128 step까지 약 347.8초가 걸렸고, 2차는 학습 중 검증·저장까지 포함해
6746.7초(약 112.4분)가 걸렸습니다.
두 시간의 측정 범위는 다릅니다. 모델 로드와 완료 후 최종 좌표 평가는 포함하지 않습니다.
2차 최종 좌표 평가 자체는 101.66초가 걸렸습니다.

1차 마지막 학습 loss는 1.06933, 2차 마지막 전체 loss는 0.99239입니다.
손실 정의와 사용 샘플이 다르므로 이 두 값의 차이를 좌표 정확도 개선 폭으로 해석하지 않습니다.

### 11.3 적중률·완전 일치율·좌표 오차의 정의

- **목표 영역 적중률**: 예측한 좌표가 원본 주석의 정답 버튼·아이콘 사각형 안에 들어가는 비율입니다. 경계도 포함합니다.
  정답 중심 좌표와 완전히 같을 필요는 없습니다. 정답 중심이 `(500, 500)`이어도 `(510, 505)`가 같은 버튼 안이면 적중입니다.
- **좌표 완전 일치율**: 예측한 정규화 정수 x/y가 정답 좌표의 두 값과 모두 일치하는 비율입니다.
- **중심점 픽셀 오차**: 예측 좌표를 원래 화면의 픽셀 좌표로 환산하고, 반올림 전 원본 정답 중심과의 유클리드 거리를 계산합니다.
- **정규화 중심점 거리**: x축은 화면 너비, y축은 화면 높이를 각각 0~1000으로 환산한 거리입니다. 픽셀 거리나 화면 대각선 백분율과 다릅니다.
- 좌표 형식이 틀리거나 0~1000 범위를 벗어나면 적중률과 거리 임계값 지표에서는 실패입니다.
  픽셀 오차의 평균·중앙값은 유효한 출력만 사용하며, 두 학습 모델은 이번 최종 평가에서 모두 47/47개가 유효했습니다.

### 11.4 동일한 47개 문항의 최종 평가 결과

| 지표 | 1차 최종 모델 | 2차 최종 모델 |
|---|---:|---:|
| 목표 영역 적중 | 2/47, **4.26%** | 15/47, **31.91%** |
| 좌표 완전 일치 | 0/47, 0.00% | 1/47, 2.13% |
| 유효한 좌표 형식 | 47/47, 100.00% | 47/47, 100.00% |
| 중심점 픽셀 오차 중앙값 | 64.90px | **23.54px** |
| 중심점 픽셀 오차 평균 | 191.33px | **88.90px** |
| 중심점 정규화 오차 평균 | 219.26 | 104.67 |
| 중심점 정규화 오차 중앙값 | 131.36 | 32.99 |
| 정규화 중심점 거리 10 이하 | 2.13% | 19.15% |
| 정규화 중심점 거리 25 이하 | 10.64% | 38.30% |
| 정규화 중심점 거리 50 이하 | 23.40% | 57.45% |
| 정규화 중심점 거리 100 이하 | 42.55% | 76.60% |

이 로컬 평가에서 2차 모델의 목표 영역 적중률은 **27.66%p 증가**했고,
픽셀 오차 중앙값은 **63.7% 감소**, 평균은 **53.5% 감소**했습니다.
2차의 **31.91%는 목표 영역 적중률**이며, 정답 좌표 완전 일치율은 **2.13%**입니다.
47개 중 32개는 여전히 목표 영역을 벗어났습니다.

평가 문항·프롬프트·greedy 생성·채점 기준은 같지만, 입력 해상도·데이터 사용·LoRA 범위·손실·학습 횟수는 변경됐습니다.
따라서 이 결과는 전체 재학습 설계 변경의 결과이며 특정 변경 하나의 효과를 분리해서 증명하지 않습니다.
12개 화면의 작은 로컬 평가이므로 공식 벤치마크 성능으로 일반화하지 않습니다.

참고로 2차와 같은 768개 시각 토큰 설정에서 LoRA를 끈 기본 모델은
0/47 영역 적중, 28/47 유효 좌표였습니다.
이는 같은 좌표 출력 형식·프롬프트의 평가 결과이며 기본 모델의 모든 용도에서의 성능을 뜻하지 않습니다.

### 11.5 모델 선택용 검증 결과 — 최종 평가와 별도

2차의 별도 48개 검증 문항에서 최종 선택된 1024 step 모델은 다음 결과를 냈습니다.

- 목표 영역 적중: **21/48, 43.75%**
- 좌표 형식 성공: 48/48
- 중심점 픽셀 오차 중앙값: 19.38px
- 중심점 픽셀 오차 평균: 78.83px
- 모델 선택 기준인 평균 정규화 중심점 오차: 115.21

이 검증의 43.75%와 최종 평가의 31.91%는 문항이 다르므로 구분해야 합니다.
최종 평가 47개는 체크포인트 선택에 사용하지 않았습니다.

### 11.6 결과와 모델 저장 위치

| 실행 | 설정·완료 기록 | 좌표 평가 | 최종 LoRA |
|---|---|---|---|
| 1차 | [설정](../runs/qwen2.5-vl-3b-guisyn/run_config.json), [완료](../runs/qwen2.5-vl-3b-guisyn/result.json) | [집계](../runs/qwen2.5-vl-3b-guisyn/coordinate_eval/summary.json), [보고서](../runs/qwen2.5-vl-3b-guisyn/coordinate_eval/report.md) | [가중치](../runs/qwen2.5-vl-3b-guisyn/final/adapter_model.safetensors) |
| 2차 | [설정](../runs/qwen2.5-vl-3b-guisyn-v2/run_config.json), [완료](../runs/qwen2.5-vl-3b-guisyn-v2/result.json) | [집계](../runs/qwen2.5-vl-3b-guisyn-v2/coordinate_eval/summary.json), [보고서](../runs/qwen2.5-vl-3b-guisyn-v2/coordinate_eval/report.md) | [가중치](../runs/qwen2.5-vl-3b-guisyn-v2/final/adapter_model.safetensors) |

각 실행의 `coordinate_eval/predictions.jsonl`에 모든 문항의 지시문·예측 좌표·정답 영역·오차가 있습니다.
학습 구현은 [1차 train_small.py](train_small.py), [2차 train_coordinates.py](train_coordinates.py),
좌표 평가는 [evaluate_coordinates.py](evaluate_coordinates.py)에 있습니다.

2차 최종 LoRA SHA-256: `576b607ecba5e0359d4015cb4a4c69de5f006f66edf6f96e3e74afd0639bf80b`.

## 12. 새 화면 추가 학습(v3) — 2026-10-15 준비

v2 `final` LoRA에서 **추가 2,048 optimizer step**을 수행하도록 구성했습니다.
이는 optimizer 상태까지 복원하는 중단 재개가 아니라, 기존 LoRA 가중치에서
새 optimizer와 학습률 스케줄로 시작하는 추가 미세 조정입니다.

- 데이터: 동일한 GUISyn `mobile/train`에서 새 화면 256개 추가, 기존 128개 보존.
- 새 화면은 `data/guisyn-additional`에 별도 저장하고 source ID와 RGB 픽셀 해시로 중복 제외.
- 기존 학습 104개 + 새 화면 256개 = 학습 화면 360개.
- v2 검증 12개 화면·48문항과 최종 평가 12개 화면·47문항을 그대로 유지.
- 학습용 대상 전체를 섞어서 순회하며 지시문을 번갈아 사용. 이번 실행에는 오답 우선 샘플링을 적용하지 않음.
- 시작 학습률 `2e-5`, warmup 64 step 후 cosine 감소. NF4, rank-16 LoRA, 768 시각 토큰, gradient accumulation 4 유지.
- 추가 2,048 step에서 개별 샘플 8,192회를 학습. 128 step마다 검증·저장.
- step 0에 v2 초기 가중치를 검증하고 `best`로 저장하므로, 검증 평균 오차가 악화되면 초기 모델도 최종 선택 가능.
- 완료 후 기존 47문항으로 자동 최종 평가. 학습 중단 시 자동 저장 및 optimizer/RNG 상태 복원 기능은 아직 없음.

`molmo` 디렉터리에서 실행:

```bash
/home/codebiney/project/robot_sim/.venv/bin/python training/prepare_additional.py --count 256
/home/codebiney/project/robot_sim/.venv/bin/python training/launch.py \
  --trainer train_coordinates.py \
  --output runs/qwen2.5-vl-3b-guisyn-v3 \
  --previous runs/qwen2.5-vl-3b-guisyn-v2 \
  --init-adapter runs/qwen2.5-vl-3b-guisyn-v2/final \
  --additional-sources data/guisyn-additional/source_annotations.json \
  --steps 2048 --learning-rate 2e-5
```

다운로드에는 외부 네트워크 접근, 학습에는 GPU에 접근 가능한 호스트 실행이 필요합니다.
실제 실행 상태는 `runs/qwen2.5-vl-3b-guisyn-v3/train.log`, `metrics.jsonl`, `train.pid`에서 확인합니다.

### 12.1 실행 시작 확인

2026-10-15 10:36 KST에 PID `11559`로 v3 학습 프로세스를 시작했습니다.
호스트 `nvidia-smi`에서 해당 PID의 GPU 사용을 확인했고, 로그에서 v2 `final`을
학습 가능한 LoRA로 불러온 것을 확인했습니다. 이 시각에는 초기 검증 단계입니다.
실제 준비 결과는 학습 화면 360개, 클릭 대상 8,306개, 지시문 표현 41,530개입니다.
총 384개 화면 중 검증 12개·최종 평가 12개는 기존 v2와 동일합니다.

검증: 기존 데이터만 사용할 때 v2의 학습 대상·검증·분할 재현 일치,
중복 source ID 거부, 새 화면 256개 전체의 학습 분할 포함,
검증·최종 평가 문항 불변, 좌표 범위 검사를 통과했습니다.

이후 초기 검증에서 v2 최종 검증값(21/48 적중, 평균 정규화 오차 115.2055)을
동일하게 재현했습니다. 1 step의 실제 LoRA 가중치 갱신 및 `checkpoint-1` 저장을
확인했고, 2 step까지 loss와 gradient norm이 유한했습니다.
2 step 확인 시 PyTorch 최대 GPU 할당량은 4,031.5 MiB였습니다.
학습 완료 기록이 아니라 정상 시작 확인 기록이며, 프로세스는 백그라운드에서 계속 실행됩니다.

### 12.2 v3 완료 결과

추가 2,048 step과 최종 평가 완료. 검증 평균 좌표 오차로 선택된 step은 1,536입니다.
학습·검증 시간 11,627.8초(약 3시간 14분). 동일한 최종 47문항에서 영역 적중
16/47(34.04%), 중심점 픽셀 오차 중앙값 19.01px, 평균 91.00px입니다.
v2의 15/47(31.91%), 중앙값 23.54px, 평균 88.90px와 비교하면 개선은 제한적입니다.

## 13. 적중률 중심 추가 학습(v4) — 추가 1,024 step

사용자의 요청에 따라 v3 `final`을 시작 가중치로 사용하는 새 실험입니다.
최종 평가 문항은 오답 수집·학습·체크포인트 선택에 사용하지 않습니다.

### 설계

- 추가 step: **1,024**, 배치 1 × gradient accumulation 4 = 샘플 학습 4,096회.
- 학습 360개 화면, 검증 12개·48문항, 최종 평가 12개·47문항으로 v3 분할 유지.
- 같은 화면에서 정규화한 동일 지시문이 서로 다른 좌표를 가리키면 학습 표현에서 제외.
  96개 모호한 지시문 키, 표현 231개, 유효 표현이 없어진 대상 14개를 제외했습니다.
  정제 후 대상 8,292개, 표현 41,299개입니다. 검증·평가 문항은 변경하지 않습니다.
- 입력 상한: 768 → **1,024 시각 토큰**(802,816 pixels). 전체 화면과 좌표계를 유지합니다.
- 학습 전 화면마다 1개씩 총 360개 대상을 정답 없이 생성합니다.
  무작위 순서의 화면 절반은 크기가 작은 하위 25% 대상에서, 나머지는 전체 대상에서 무작위 선택합니다.
  영역 밖 예측과 잘못된 좌표 형식을 오답으로 수집합니다. 이는 모든 8,292개 대상의 전수 평가가 아닙니다.
- 각 optimizer step의 샘플 4개 중 2개는 수집한 오답, 2개는 전체 정제 학습 대상에서 순회합니다.
  각 풀을 섞어 순회하고 지시문 표현도 순환합니다. 오답 풀이 비면 일반 샘플만 사용합니다.
- 손실: 숫자 토큰 가중 CE + **1 × 좌표 SmoothL1 + 0.5 × 영역 크기 보정 손실**.
  영역 손실은 정답 응답의 숫자 확률에서 얻은 기대 좌표가 목표 영역 안쪽 80%를 벗어난 거리를
  영역 폭·높이로 나눈 후 `log1p` 평균으로 계산합니다. 폭·높이 분모의 하한은 해당 이미지 축의 1%입니다.
  작은 목표의 상대 오차를 강조하고 중심부 여유를 유도하는 실험적 대리 손실이며,
  실제 자유 생성의 적중률 개선을 보장하는 직접적인 적중률 손실은 아닙니다.
- v3 LoRA를 로드하고 optimizer/scheduler는 새로 시작. 학습률 `2e-5`, warmup 64, cosine 감소.
- **128 step마다 검증 적중 수가 가장 많은 모델 선택**, 동률일 때 평균 정규화 중심점 오차가 작은 모델 선택.
  step 0의 초기 모델도 포함합니다. 마지막 step과 `final`은 다를 수 있습니다.
- 최종 자동 평가는 기존과 동일한 47문항에 대해 실행합니다.

### 실행 명령

`molmo` 디렉터리에서 GPU 접근 가능한 호스트 실행:

```bash
/home/codebiney/project/robot_sim/.venv/bin/python training/launch.py \
  --trainer train_coordinates.py \
  --output runs/qwen2.5-vl-3b-guisyn-v4 \
  --previous runs/qwen2.5-vl-3b-guisyn-v3 \
  --init-adapter runs/qwen2.5-vl-3b-guisyn-v3/final \
  --additional-sources data/guisyn-additional/source_annotations.json \
  --steps 1024 --learning-rate 2e-5 --max-pixels 802816 \
  --coordinate-loss-weight 1 --bbox-loss-weight 0.5 \
  --select-by bbox --hard-replay --filter-ambiguous
```

`hard_mining.jsonl`에는 학습 대상 예측, `hard_mining_summary.json`에는 오답 인덱스가 남습니다.
`data_cleaning.json`은 정제 집계, `metrics.jsonl`의 `sampling_counts`는 실제 풀별 누적 학습 횟수입니다.
완료 후 `sampling_summary.json`에서 사용한 고유 대상 수도 확인할 수 있습니다.

CPU 검증(`training/test_hit_training.py`): 비정사각형 화면의 영역 변환, 영역 밖 거리의
벌점·gradient 방향·유한성, 다양한 좌표 자릿수의 합성 손실 역전파, 모호한 지시문 제거,
오답/일반 50% 혼합과 빈 오답 풀 처리, 적중률 우선 모델 선택,
실제 데이터의 검증·최종 평가 분할 불변 및 오답 수집의 학습 분할 한정을 통과했습니다.

### 13.1 실행 초기 확인

2026-10-15 15:06경(KST) PID `51349`로 시작했습니다.
1,024 시각 토큰에서의 step 0 검증은 20/48 적중(41.67%), 평균 정규화 오차 101.4377입니다.
동일한 시작 가중치의 v3 기존 768 시각 토큰 검증은 23/48(47.92%)이므로,
해상도 증가만으로 성능이 좋아진 것은 아닙니다. 추가 학습 후의 결과를 별도로 확인해야 합니다.
초기 모델과 검증 결과를 저장했고, 학습용 화면만 사용하여 오답 수집을 진행 중입니다.

### 13.2 오답 수집 완료 및 실제 학습 시작 확인

학습 화면 360개의 대상 360개를 자유 생성으로 검사하여 오답 223개를 수집했습니다.
이 오답 수는 학습용 표본에서 측정한 값이며 검증/최종 평가 성능이 아닙니다.
1 step의 실제 LoRA 갱신과 `checkpoint-1` 저장을 확인했고, 2/1,024 step까지
loss와 gradient norm이 유한했습니다. 샘플 누적 횟수는 오답 4회·일반 4회로 50% 혼합을 확인했습니다.
2 step 기준 최대 PyTorch 할당량은 4,368.2 MiB입니다. v4는 백그라운드에서 계속 학습 중이며
성능 개선 여부는 이후 검증과 최종 평가로 판단합니다.
