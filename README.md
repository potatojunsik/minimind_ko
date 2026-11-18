<div align="center">

# minimind_ko

[![License](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900)
![GPU](https://img.shields.io/badge/GPU-RTX%205050%20Laptop%208GB-76B900)

</div>

<div align="center">
  <h3>"노트북 GPU 한 장으로, 한국어 LLM을 처음부터"</h3>
</div>

<div align="center">

Based on [MiniMind](https://github.com/jingyaogong/minimind)

</div>

<div align="center">

한국어 | [English](./README_en.md)

</div>

* [jingyaogong/minimind](https://github.com/jingyaogong/minimind)(64M 초소형 LLM)를 바탕으로 **한국어 전용 초소형 LLM**을 직접 학습해 보는 개인 공부 기록입니다. 원본 저자가 코드·데이터·문서를 무료로 공개해 준 덕분에 가능했습니다 ([출처 및 참고](#-출처-및-참고)).
* 노트북 GPU 한 장(RTX 5050 Laptop, VRAM 8GB)에서 모든 학습을 돌립니다.
* 한국어 위키백과로 **토크나이저부터 직접** 만들고(vocab 6,400 / 16,000), 그 토크나이저로 모델을 처음부터 사전학습합니다.
* 원본 MiniMind 코드는 **수정하지 않고**, 원본 함수(`train_tokenizer()`, `train_epoch()`)를 `scripts/`에서 불러 재사용합니다.
* 단계마다 명령어, 결과, 배운 점, 겪은 문제를 [docs/](docs)에 기록했습니다. LLM 입문자를 위한 [용어집](docs/00_개요.md#용어집)도 있습니다.
* 문서의 숫자(토큰 수, loss, 파라미터 수 등)는 실제 로그와 스크립트 출력에서 가져왔습니다.

> 참고: 공부·체험용 프로젝트입니다. 만들어지는 모델은 실사용 수준이 아닙니다. 원본 MiniMind와 같은 Apache License 2.0을 따릅니다.

---

# 📌 프로젝트 소개

LLM을 API로 쓰는 것과 직접 학습시켜 보는 것은 전혀 다른 경험입니다. 수십억 파라미터 모델은 개인 PC로 학습할 수 없지만, MiniMind처럼 64M 정도의 작은 모델이라면 노트북 GPU로도 토크나이저 → 사전학습 → 대화 학습의 전 과정을 직접 돌려 볼 수 있습니다.

원본 MiniMind는 중국어·영어 위주로 만들어져 있어서, 한국어 문장을 넣으면 바이트 단위로 잘게 쪼개지고 한국어로는 거의 대화하지 못합니다 ([docs/01](docs/01_환경구성.md)). "한국어 전용으로 처음부터 다시 만들면 어떻게 될까?"를 직접 확인해 보는 것이 이 프로젝트의 출발점입니다.

---

#### 🎉 이 저장소에 포함된 것

- 단계별 공부 기록 [docs/](docs) (00 개요·용어집 ~ 05 비교)
- 한국어 위키백과를 학습용 데이터로 바꾸는 스크립트 (`scripts/prepare_kowiki.py`)
- 원본 함수를 재사용한 한국어 토크나이저 학습 / 한국어 사전학습 스크립트 (`scripts/train_tokenizer_ko.py`, `scripts/train_pretrain_ko.py`)
- 토크나이저 비교, 검증 세트 BPC 평가, 이어 쓰기 비교, loss 그래프 스크립트
- 직접 학습한 한국어 토크나이저 2종 (`tokenizers/`)
- 학습 로그 (`logs/`), loss 그래프 (`images/`), 실행 환경 고정 목록 (`requirements-lock.txt`)

#### 🎉 만든 결과물

| 결과물 | 설명 | 크기 | git |
|---|---|---|---|
| [tokenizers/ko_6400](tokenizers/ko_6400) | 한국어 위키백과로 학습한 Byte-level BPE 토크나이저 | vocab 6,400 | 포함 |
| [tokenizers/ko_16000](tokenizers/ko_16000) | 〃 | vocab 16,000 | 포함 |
| `out_ko/pretrain_ko16000_768.pth` | 한국어 사전학습 모델 (vocab 16,000) | 71.28M | 미포함 (재학습으로 생성) |
| `out_ko/pretrain_ko6400_768.pth` | 한국어 사전학습 모델 (vocab 6,400) | 63.91M | 미포함 (재학습으로 생성) |

---

#### 📝 진행 기록

| 단계 | 내용 | 상태 | 기록 |
|---|---|---|---|
| 1 | 환경 구성, 원본 모델 실행 | ✅ 완료 | [docs/01](docs/01_환경구성.md) |
| 1-2 | 작은 데이터로 사전학습 체험 | ✅ 완료 | [docs/02](docs/02_사전학습_체험.md) |
| 2 | 한국어 토크나이저 만들기 | ✅ 완료 | [docs/03](docs/03_한국어_토크나이저.md) |
| 3 | 한국어 사전학습(Pretrain), vocab 16,000 | ✅ 완료 | [docs/04](docs/04_한국어_사전학습.md) |
| 3-1 | vocab 6,400 vs 16,000 비교 | ✅ 완료 | [docs/05](docs/05_vocab_비교.md) |
| 4 | 한국어 대화 학습(SFT) | ⏳ 예정 | |
| 5 | (선택) LoRA, DPO 체험 | ⏳ 예정 | |

<details open>
<summary> <b>2026-11-18 (3단계: 한국어 사전학습)</b> </summary>

- 한국어 위키 조각을 학습 10만 / 검증 2천으로 분할 (겹치는 줄 0개)
- 검증 세트의 토큰 길이 분포를 재서 `max_seq_len` 512로 결정
- `scripts/train_pretrain_ko.py` 작성 (원본 `train_epoch()` 재사용, 토크나이저 경로·vocab 크기만 교체)
- vocab 16,000 사전학습 완료 (71.28M, 6,250 step, 약 52분, loss 8.50 → 4.45 / 마지막 구간 평균 약 4.3)
- 학습 도중 GPU 전력 상한이 21W로 묶여 느려지는 문제 발견, 제조사 성능 모드로 해결
- vocab 6,400을 같은 조건으로 학습(63.91M, 약 35분)해서 검증 세트 BPC로 비교: **이 규모에서는 6,400이 더 좋았다** (3.2290 vs 3.2997)

</details>

<details>
<summary> <b>2026-11-18 (1~2단계: 환경 구성, 토크나이저)</b> </summary>

- uv로 Python 3.11 환경 구성, RTX 5050(Compute 12.0)용 PyTorch cu128 설치
- 원본 `minimind-3` 실행: 한국어는 바이트 단위로 쪼개지고 대화가 되지 않음을 확인
- 원본 학습 코드로 중국어 2만 줄 사전학습 체험 (약 6분, loss 8.13 → 5.13)
- 한국어 위키백과 전처리 (1,185,371 조각), 토크나이저 vocab 6,400 / 16,000 학습

</details>

---

# 📌 빠른 시작

<details>
<summary>내 PC 사양 (참고)</summary>

* OS: Windows 11 Pro
* CPU: Intel Core Ultra 7 255H
* RAM: 32 GB
* GPU: NVIDIA RTX 5050 Laptop (VRAM 8GB, Compute Capability 12.0)
* Python: 3.11 (uv 가상환경. 시스템 Python 3.14와 별개)
* PyTorch 2.11.0 + CUDA 12.8
* 원본 MiniMind 커밋: [`233d00f`](https://github.com/jingyaogong/minimind/commit/233d00fa068ec2b8c0d4f07b703934d6391a5a73) (2026-10-27)

팁: 노트북에서 학습이 갑자기 느려지면 `nvidia-smi -q -d POWER`로 GPU 전력 상한을 확인하세요. 이 PC는 Windows 전원 모드와 별개로 제조사 성능 모드가 GPU 전력을 21W로 묶고 있었습니다 ([docs/04](docs/04_한국어_사전학습.md#겪은-문제)).

</details>

명령은 Windows 명령 프롬프트(cmd) 기준입니다. `D:\minimind_ko`는 작업 폴더 예시이니 원하는 위치로 바꿔도 됩니다. Git Bash에서는 경로 구분자를 `/`로 바꾸세요 ([docs/01](docs/01_환경구성.md#설치-순서)).

## 0단계

```bat
REM 저장소와 원본 MiniMind 받기 (REPO_URL은 이 저장소 주소로 바꾼다)
git clone REPO_URL D:\minimind_ko
cd /d D:\minimind_ko
git clone https://github.com/jingyaogong/minimind.git
git -C minimind checkout 233d00fa068ec2b8c0d4f07b703934d6391a5a73

REM Python 3.11 가상환경 (uv 필요: https://github.com/astral-sh/uv)
uv venv --python 3.11 .venv

REM 권장: 기록과 똑같은 버전으로 설치
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128

REM GPU 인식 확인
.venv\Scripts\python.exe scripts\check_gpu.py
```

주요 버전: torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 4.57.6, tokenizers 0.22.2, datasets 3.6.0, numpy 1.26.4, pyarrow 20.0.0, peft 0.21.3

<details>
<summary>기록 당시 실제로 설치한 순서 (lock 파일 없이)</summary>

```bat
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r minimind\requirements.txt peft
uv pip install --python .venv\Scripts\python.exe "pyarrow==20.0.0"
```

* RTX 50 시리즈는 CUDA 12.8 이상 빌드의 PyTorch가 필요합니다.
* `pyarrow==20.0.0` 고정 이유: requirements.txt의 `numpy==1.26.4`와 최신 pyarrow가 충돌합니다 ([docs/01](docs/01_환경구성.md#겪은-문제)).
* loss 그래프(`scripts/plot_loss.py`)를 그리려면 matplotlib도 필요합니다 (lock 파일에는 포함).

</details>

<details>
<summary>(선택) 원본 모델 실행과 작은 사전학습 체험 — docs/01, 02</summary>

```bat
cd /d D:\minimind_ko\minimind
..\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('jingyaogong/minimind-3', local_dir='minimind-3')"
..\.venv\Scripts\python.exe -X utf8 ..\scripts\test_chat.py ./minimind-3
```

작은 사전학습 체험 명령은 [docs/02](docs/02_사전학습_체험.md)를 따르세요. 한국어 단계에는 필요 없습니다.

</details>

## Ⅰ 🔤 토크나이저

### 1' 한국어 위키백과 준비

```bat
cd /d D:\minimind_ko
mkdir data
.venv\Scripts\python.exe -X utf8 scripts\prepare_kowiki.py data\kowiki.jsonl
```

### 2' (선택) 토크나이저 학습과 비교

학습된 토크나이저는 `tokenizers/`에 이미 들어 있으므로 다시 만들지 않아도 됩니다.

```bat
.venv\Scripts\python.exe -X utf8 scripts\train_tokenizer_ko.py data\kowiki.jsonl 6400
.venv\Scripts\python.exe -X utf8 scripts\train_tokenizer_ko.py data\kowiki.jsonl 16000
.venv\Scripts\python.exe -X utf8 scripts\compare_tokenizers.py minimind/model tokenizers/ko_6400 tokenizers/ko_16000
```

## Ⅱ 🛠️ 사전학습

### 1' 학습 / 검증 데이터 나누기

```bat
.venv\Scripts\python.exe scripts\make_subset.py data/kowiki.jsonl data/kowiki_train_100k.jsonl 100000 11 0
.venv\Scripts\python.exe scripts\make_subset.py data/kowiki.jsonl data/kowiki_heldout_2k.jsonl 2000 11 5
```

### 2' 사전학습

결과는 `out_ko/pretrain_ko<vocab>_768.pth`에 저장됩니다. 설정값은 모두 스크립트 기본값입니다 ([docs/04](docs/04_한국어_사전학습.md#4-학습-설정)).

```bat
.venv\Scripts\python.exe -X utf8 scripts\train_pretrain_ko.py --tokenizer_dir tokenizers/ko_16000 --data_path data/kowiki_train_100k.jsonl
.venv\Scripts\python.exe -X utf8 scripts\train_pretrain_ko.py --tokenizer_dir tokenizers/ko_6400 --data_path data/kowiki_train_100k.jsonl
```

## Ⅲ 🧪 평가 / 테스트

### 1' 검증 세트 BPC (글자당 비트)

```bat
.venv\Scripts\python.exe -X utf8 scripts\eval_bpc.py --tokenizer_dir tokenizers/ko_16000 --weight out_ko/pretrain_ko16000_768.pth
.venv\Scripts\python.exe -X utf8 scripts\eval_bpc.py --tokenizer_dir tokenizers/ko_6400 --weight out_ko/pretrain_ko6400_768.pth
```

### 2' 이어 쓰기 비교

```bat
.venv\Scripts\python.exe -X utf8 scripts\compare_generation.py --model tokenizers/ko_6400 out_ko/pretrain_ko6400_768.pth --model tokenizers/ko_16000 out_ko/pretrain_ko16000_768.pth
```

### 3' loss 그래프

```bat
.venv\Scripts\python.exe -X utf8 scripts\plot_loss.py images\loss_pretrain_small_20k.png "사전학습 체험: 중국어 2만 줄, 원본 토크나이저" "원본 토크나이저=logs\01_pretrain_small_20k.log"
.venv\Scripts\python.exe -X utf8 scripts\plot_loss.py images\loss_pretrain_ko_compare.png "한국어 사전학습 loss (vocab이 달라 토큰당 loss는 직접 비교 불가, 비교는 BPC로)" "vocab 6,400=logs\03_pretrain_ko6400_100k.log" "vocab 16,000=logs\02_pretrain_ko16000_100k.log"
```

<details>
<summary>스크립트 전체 목록</summary>

모든 스크립트는 파일 맨 위 docstring에 사용법이 있습니다. 실행 위치가 스크립트마다 다르니 주의하세요.

| 파일 | 실행 위치 | 설명 |
|---|---|---|
| [check_gpu.py](scripts/check_gpu.py) | `minimind_ko` | PyTorch의 GPU 인식 여부, VRAM, bf16 행렬곱 속도(TFLOPS) 확인 (학습 중이 아닐 때 실행) |
| [test_chat.py](scripts/test_chat.py) | `minimind` | transformers 형식 모델의 토크나이저 효율과 대화 응답 확인 |
| [test_pretrain.py](scripts/test_pretrain.py) | `minimind` | 원본 토크나이저(vocab 6,400)로 사전학습한 `.pth`의 이어 쓰기 확인 (한국어 토크나이저 모델에는 사용 불가) |
| [make_subset.py](scripts/make_subset.py) | 어디서나 | 큰 jsonl에서 일부 줄만 뽑기. stride·offset으로 서로 겹치지 않는 학습/검증 세트 생성 |
| [prepare_kowiki.py](scripts/prepare_kowiki.py) | `minimind_ko` | 한국어 위키백과를 받아 최대 800자 조각의 학습용 jsonl로 변환 |
| [train_tokenizer_ko.py](scripts/train_tokenizer_ko.py) | `minimind_ko` | 한국어 토크나이저 학습 (원본 `train_tokenizer()` 재사용) |
| [compare_tokenizers.py](scripts/compare_tokenizers.py) | `minimind_ko` | 여러 토크나이저의 압축률과 쪼개는 모양 비교 |
| [train_pretrain_ko.py](scripts/train_pretrain_ko.py) | `minimind_ko` | 한국어 사전학습 (원본 `train_epoch()` 재사용, 토크나이저 경로·vocab 크기만 교체) |
| [eval_bpc.py](scripts/eval_bpc.py) | `minimind_ko` | held-out 데이터로 글자당 비트(BPC) 계산. vocab이 다른 모델끼리 비교하기 위함 |
| [paired_compare.py](scripts/paired_compare.py) | `minimind_ko` | `eval_bpc.py --per_doc_out` 결과 두 개로 문서별 우세와 BPC 차이의 부트스트랩 95% 구간 계산 |
| [compare_generation.py](scripts/compare_generation.py) | `minimind_ko` | 여러 사전학습 모델에 같은 한국어 프롬프트를 주고 이어 쓰기 결과를 나란히 비교 |
| [plot_loss.py](scripts/plot_loss.py) | `minimind_ko` | 학습 로그에서 step별 loss를 읽어 PNG 그래프로 저장 (batch loss + 이동평균 추세선) |

</details>

<details>
<summary>git에 포함되지 않는 것과 다시 만드는 방법</summary>

용량이 크거나 다시 만들 수 있는 것은 저장소에 올리지 않습니다 ([.gitignore](.gitignore)).

| 항목 | 다시 만드는 방법 |
|---|---|
| `minimind/` (원본 저장소) | `git clone https://github.com/jingyaogong/minimind.git` 후 `git -C minimind checkout 233d00fa068ec2b8c0d4f07b703934d6391a5a73` |
| `.venv/` (가상환경) | 0단계 |
| `minimind/minimind-3/` (원본 공개 모델) | `minimind` 폴더에서 `snapshot_download('jingyaogong/minimind-3', local_dir='minimind-3')` |
| `minimind/dataset/pretrain_t2t_mini.jsonl` (원본 사전학습 데이터, 1.2GB) | `minimind` 폴더에서 `hf_hub_download('jingyaogong/minimind_dataset','pretrain_t2t_mini.jsonl',repo_type='dataset',local_dir='dataset')` ([docs/02](docs/02_사전학습_체험.md)) |
| `data/kowiki.jsonl`, `data/raw/` (한국어 위키백과) | `scripts/prepare_kowiki.py` (Ⅰ-1') |
| `data/kowiki_train_100k.jsonl`, `data/kowiki_heldout_2k.jsonl` | `scripts/make_subset.py` (Ⅱ-1') |
| `data/tokenizer_sample.jsonl` | `scripts/train_tokenizer_ko.py` 첫 실행 시 자동 생성 |
| `out*/`, `checkpoints/`, `*.pth` (학습 가중치) | 각 학습 명령을 다시 실행 |

</details>

<details>
<summary>폴더 구조</summary>

```
minimind_ko/
├── README.md              한국어 README (이 파일)
├── README_en.md           영어 README
├── LICENSE                Apache License 2.0 (원본 MiniMind와 같은 라이선스)
├── requirements-lock.txt  실행 환경 고정 목록 (uv pip freeze)
├── .gitignore
├── .gitattributes         줄바꿈 통일, 바이너리 파일 지정
├── docs/                  단계별 기록
├── scripts/               직접 작성한 스크립트
├── logs/                  학습 로그
├── images/                loss 그래프
├── tokenizers/            직접 학습한 한국어 토크나이저 (ko_6400, ko_16000)
├── data/                  한국어 데이터          (git 제외)
├── out_ko/                한국어 사전학습 가중치  (git 제외)
├── minimind/              원본 저장소            (git 제외)
└── .venv/                 가상환경               (git 제외)
```

</details>

---

# 📌 데이터

## Ⅰ 한국어 위키백과

| 항목 | 값 |
|---|---|
| 출처 | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.ko` (parquet 3개, 약 780MB) |
| 라이선스 | CC BY-SA 4.0 (HF 데이터셋 카드 표기: CC BY-SA 3.0 / GFDL) |
| 문서 수 | 647,897 |
| 전처리 | 문단 단위로 묶어 최대 800자 조각으로 나눔, 50자 미만 조각은 버림 |
| 조각 수 | 1,185,371 |
| 총 글자 수 | 593,722,771 (약 5.9억 자) |
| 형식 | 한 줄에 `{"text": "..."}` (jsonl, 1.35GB) |

## Ⅱ 용도별 분할

| 파일 | 줄 수 | 뽑는 방법 | 용도 |
|---|---|---|---|
| `data/tokenizer_sample.jsonl` | 200,000 | 5줄마다 1줄 (20만 줄을 채우면 멈춰서 실제로는 0~999,995번째 줄, 앞쪽 약 84%) | 토크나이저 학습 |
| `data/kowiki_train_100k.jsonl` | 100,000 | 11줄마다 1줄, 0번째부터 | 사전학습 |
| `data/kowiki_heldout_2k.jsonl` | 2,000 | 11줄마다 1줄, 5번째부터 | 검증(held-out). 학습 세트와 겹치는 줄 0개 |

<details>
<summary>분할의 한계 (공부용이라 그대로 진행)</summary>

* 검증 세트는 원본 파일 앞부분(5~21,994번째 줄)에서만 뽑힙니다.
* 같은 위키 문서의 다른 조각이 학습/검증 세트에 나뉘어 들어갈 수 있습니다.
* 토크나이저 학습 샘플에 검증 세트 2,000줄 중 400줄(20%)이 들어 있습니다.

자세한 내용은 [docs/04](docs/04_한국어_사전학습.md#2-데이터-분할-학습-10만--검증-2천)를 참고하세요.

</details>

---

# 📌 모델

모델 구조는 원본 `minimind-3`(Transformer Decoder-Only, RMSNorm, SwiGLU, RoPE, GQA)와 같고, **vocab 크기만** 토크나이저에 맞춰 바꿨습니다. 아래 값은 `minimind/model/model_minimind.py`의 `MiniMindConfig` 기본값과 학습 로그로 확인했습니다. 용어(임베딩, lm_head, GQA, RoPE, RMSNorm, SwiGLU 등)는 [용어집](docs/00_개요.md#용어집)을 참고하세요.

| 모델 | params | vocab | d_model | n_layers | q_heads | kv_heads | FFN 차원 | 학습 max_seq_len |
|---|---|---|---|---|---|---|---|---|
| 원본 구조 (docs/02 체험) | 63.91M | 6,400 | 768 | 8 | 8 | 4 | 2,432 | 340 |
| ko_16000 | 71.28M | 16,000 | 768 | 8 | 8 | 4 | 2,432 | 512 |
| ko_6400 | 63.91M | 6,400 | 768 | 8 | 8 | 4 | 2,432 | 512 |

* 입력 임베딩과 출력층이 가중치를 공유(`tie_word_embeddings=True`)해서, vocab이 늘어난 만큼(9,600 × 768 ≈ 7.4M)만 파라미터가 늘어납니다.

---

# 📌 실험

## Ⅰ 학습 비용

RTX 5050 Laptop (VRAM 8GB) 기준입니다.

| 단계 | 데이터 | 설정 | 소요 시간 | VRAM |
|---|---|---|---|---|
| 작은 사전학습 체험 (docs/02) | 중국어 2만 줄 | batch 16, seq 340, 1,250 step | 약 6분 | 약 4GB |
| 토크나이저 학습 (docs/03) | 위키 20만 조각 | vocab 6,400 / 16,000 | 각 약 50초 (CPU) | - |
| 한국어 사전학습 ko_16000 (docs/04) | 위키 10만 조각 | batch 16 × accum 8, seq 512, 6,250 step | 약 52분 ※ | 약 6.8GB |
| 한국어 사전학습 ko_6400 (docs/05) | 위키 10만 조각 | 같은 설정 | 약 35분 | 약 5.3GB |

* VRAM은 학습 중 nvidia-smi로 본 GPU 전체 사용량입니다 (6,400: 약 5,310MB, 16,000: 약 6,770MB).
* ※ 16,000 학습 시간은 도중에 GPU 전력 제한(21W)으로 느려진 구간이 섞인 참고치라 6,400과 직접 비교하면 안 됩니다 ([docs/04](docs/04_한국어_사전학습.md#겪은-문제)). 두 학습이 모두 성능 모드였던 같은 구간(2,850~6,200 step)의 step당 시간은 6,400 0.335초, 16,000 0.362초입니다.

## Ⅱ 결과

### 1' 토크나이저 압축률

괄호 안은 글자 수 ÷ 토큰 수 (높을수록 같은 문장을 더 적은 토큰으로 표현).

| 문장 | 글자 | 원본 (6,400) | 한국어 6,400 | 한국어 16,000 |
|---|---|---|---|---|
| 한국어(일상) | 14 | 30토큰 (0.47) | 11토큰 (1.27) | 9토큰 (1.56) |
| 한국어(설명) | 113 | 231토큰 (0.49) | 57토큰 (1.98) | **45토큰 (2.51)** |
| 한국어(대화) | 58 | 117토큰 (0.50) | 32토큰 (1.81) | 30토큰 (1.93) |
| 중국어 | 27 | **17토큰 (1.59)** | 57토큰 (0.47) | 42토큰 (0.64) |
| 영어 | 63 | **16토큰 (3.94)** | 32토큰 (1.97) | 27토큰 (2.33) |

```text
"오늘 날씨가 정말 좋네요."
원본          : 바이트 조각 30개 (한 조각만으로는 글자가 안 됨)
한국어 6,400  : 오 | 늘 | 날 | 씨 | 가 | 정 | 말 | 좋 | 네 | 요 | .
한국어 16,000 : 오늘 | 날 | 씨가 | 정 | 말 | 좋 | 네 | 요 | .
```

### 2' 사전학습 loss

작은 사전학습 체험 (중국어 2만 줄, vocab 6,400). 옅은 선은 로그에 찍힌 batch loss, 진한 선은 이동평균 추세입니다.

![loss_pretrain_small_20k](images/loss_pretrain_small_20k.png)

한국어 사전학습 ko_16000: 첫 기록 8.50 → 마지막 4.45 (마지막 250 step 평균 4.32). 무작위 추측 수준은 ln(16000) ≈ 9.68입니다 ([docs/04](docs/04_한국어_사전학습.md#6-결과)).

한국어 사전학습 (vocab 6,400 vs 16,000):

![loss_pretrain_ko_compare](images/loss_pretrain_ko_compare.png)

* vocab이 다르면 토큰 하나에 담긴 글자 수가 달라서, **토큰당 loss를 그대로 비교하면 안 됩니다.** 그래프에서 6,400의 loss가 낮은 것은 토큰이 더 작기 때문이기도 합니다. 공정한 비교는 아래 글자당 비트(BPC)로 합니다.

### 3' vocab 6,400 vs 16,000 비교 (BPC)

학습에 쓰지 않은 검증 조각 2,000개(1,257,882자)로 평가했습니다. BPC는 한 글자를 맞히는 데 드는 평균 비트 수로, **낮을수록 좋습니다** ([docs/05](docs/05_vocab_비교.md)).

| 지표 | vocab 6,400 | vocab 16,000 |
|---|---|---|
| **BPC (텍스트+eos)** | **3.2290** | 3.2997 |
| 토큰당 loss (참고, 직접 비교 불가) | 3.8849 | 4.5627 |
| 토큰당 평균 글자 수 | 1.741 | 2.001 |
| 문서별 비교에서 더 낮은 쪽 | 1,666 / 2,000개 (83%) | 나머지 |
| 파라미터 | 63.91M | 71.28M |

* **예상과 반대였습니다.** 토크나이저 압축률만 보고 16,000을 주력으로 골랐지만, 이 실험 규모(위키 10만 조각, 1 epoch, 6,250 step)에서는 6,400의 BPC가 0.071(약 2.1%) 낮았습니다. 검증 문서를 다시 뽑는 부트스트랩 95% 구간은 0.068~0.074입니다.
* 추정 이유 (검증하지 않음): 데이터가 같으면 vocab이 클수록 토큰 하나가 학습 중 나오는 횟수가 줄어듭니다(6,400 약 4,560번 vs 16,000 약 1,600번). 또 512칸 패딩이라 계산량은 같은데, 6,400은 같은 step에서 실제 토큰을 약 14% 더 많이 학습합니다.
* 즉 **작은 데이터에서는 작은 vocab이 유리할 수 있다**는 결과이고, 16,000이 원래 나쁘다는 뜻은 아닙니다. 학습 시드는 1개뿐이고, 데이터와 학습량을 늘리면 결과가 바뀔 수 있습니다.

### 4' 이어 쓰기 예시

사전학습만 한 모델은 대화를 모르므로, 문장 앞부분을 주고 뒤를 이어 쓰게 합니다.

<details>
<summary>작은 사전학습 체험 (중국어 데이터, docs/02)</summary>

```text
[입력] 人工智能是
[출력] 人工智能是人工智能的。…自然语言处理、广泛、智能、安全、文化等。人工智能技术在人工智能在…

[입력] 오늘 날씨가
[출력] 오늘 날씨가，她在天空中，我是我去在我们心”…
```

문법은 엉망이지만 실제 중국어 단어와 관련 단어 연결을 배웠고, 한국어 입력은 바로 중국어로 넘어갑니다 (한국어를 배우지 않았기 때문).

</details>

<details>
<summary>한국어 사전학습 모델 (vocab 6,400 / 16,000, docs/05)</summary>

```text
[입력] 대한민국의 수도는
[6,400 ] 동해안에 있다. 역사 1985년 7월 3일 : 상하이 시, 남쪽으로는 동해안에 의해 동해안에 편입되었다. …
[16,000] 1960년 1월 30일, 중국인민군(FI)에서 제1차 세계 대전 때 소련군에서 제2차 세계 대전에 대항한 영국군은 …

[입력] 조선 시대에는
[6,400 ] 남해에 조선후기에 이르렀다. 조선 초기에 조선의 종실과 함께 조선의 제후들이 …
[16,000] 조선 시대에 의해서, 한나라의 제9대 황제(1661년 ~ 1779년)는 조선시대에 세워진 고원이다. 본관은 고흥(高興), …
```

두 모델 모두 위키 문체(연도, 한자 병기, "각주", "같이 보기" — 뒤의 둘은 [docs/05](docs/05_vocab_비교.md) 표의 예시)는 흉내 내지만 사실과 의미는 맞지 않습니다. 이 정도 샘플로는 눈으로 우열을 가리기 어려워서, 우열은 BPC로 판단했습니다.

</details>

---

# 📌 문서

| 문서 | 내용 |
|---|---|
| [00_개요.md](docs/00_개요.md) | 전체 흐름, 문서 읽는 순서, 주요 결과 요약, 용어집 |
| [01_환경구성.md](docs/01_환경구성.md) | 설치, GPU 확인, 원본 모델(minimind-3) 실행 |
| [02_사전학습_체험.md](docs/02_사전학습_체험.md) | 원본 학습 코드로 작은 중국어 데이터 사전학습 |
| [03_한국어_토크나이저.md](docs/03_한국어_토크나이저.md) | 한국어 위키백과 준비, 토크나이저(vocab 6,400 / 16,000) 학습과 비교 |
| [04_한국어_사전학습.md](docs/04_한국어_사전학습.md) | 한국어 사전학습 스크립트, 데이터 분할, max_seq_len 결정, 학습 결과 |
| [05_vocab_비교.md](docs/05_vocab_비교.md) | vocab 6,400과 16,000 모델을 검증 세트 BPC와 이어 쓰기로 비교 (이 규모에서는 6,400이 더 좋았음) |

---

# 📌 출처 및 참고

이 저장소는 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)를 바탕으로 한 개인 학습 기록입니다. 원본에서 가져오거나 원본에 기대고 있는 것은 다음과 같습니다.

| 항목 | 원본에서 온 것 |
|---|---|
| 학습 코드 | `scripts/train_pretrain_ko.py`는 원본 `trainer/train_pretrain.py`의 `train_epoch()`를, `scripts/train_tokenizer_ko.py`는 원본 `trainer/train_tokenizer.py`의 `train_tokenizer()`를 import해서 그대로 재사용합니다. 원본 코드는 이 저장소에 포함하지 않으며, 커밋 [`233d00f`](https://github.com/jingyaogong/minimind/commit/233d00fa068ec2b8c0d4f07b703934d6391a5a73)로 고정해 따로 받습니다. |
| 토크나이저 설정 | `tokenizers/*/tokenizer_config.json`의 특수 토큰과 chat_template은 원본 `train_tokenizer()`가 생성한 것입니다. 어휘(vocab)만 한국어 위키백과로 새로 학습했습니다. |
| 모델 구조 | 원본 `model/model_minimind.py`를 그대로 씁니다 (vocab 크기만 인자로 지정). |
| 체험용 데이터와 모델 | 중국어 사전학습 데이터 `pretrain_t2t_mini.jsonl`과 공개 모델 `minimind-3`는 원본 저자가 Hugging Face에 공개한 것입니다. |
| README 구성 | 원본 README의 형식을 참고했습니다. |

**변경 사항**: 원본 파일은 수정하지 않았습니다. 원본과 달라진 동작은 바깥 래퍼 스크립트에서 토크나이저 경로와 vocab 크기를 지정할 수 있게 한 것뿐입니다 ([docs/04](docs/04_한국어_사전학습.md#1-학습-스크립트-만들기-scriptstrain_pretrain_kopy)).

그 밖에 고마운 곳:

* [한국어 위키백과](https://ko.wikipedia.org) 기여자들: 한국어 학습 데이터.
* [Hugging Face](https://huggingface.co): `wikimedia/wikipedia` 데이터셋 배포, `transformers` / `tokenizers` / `datasets` 라이브러리.

---

# 🎓 인용

이 저장소를 참고했다면 원본 MiniMind를 인용해 주세요.

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  url = {https://github.com/jingyaogong/minimind},
  note = {GitHub repository, accessed 2026}
}
```

---

# ⚖️ 라이선스

* 이 저장소의 스크립트와 문서는 원본 MiniMind와 같은 Apache License 2.0을 따릅니다. 전문은 [LICENSE](LICENSE)에 있습니다.
* 한국어 학습 데이터(한국어 위키백과)는 CC BY-SA 4.0입니다 (HF 데이터셋 카드 표기: CC BY-SA 3.0 / GFDL). 데이터 자체는 저장소에 포함하지 않으며 `scripts/prepare_kowiki.py`로 직접 받아 변환합니다.
* `tokenizers/`의 어휘 파일은 한국어 위키백과에서 학습한 결과물로 CC BY-SA 4.0을 따릅니다 (출처: 한국어 위키백과). 스크립트와 문서는 위와 같이 Apache License 2.0입니다.
* 원본 공개 모델과 데이터셋(minimind-3, minimind_dataset)은 원 저작자의 Hugging Face 저장소에서 받습니다.
