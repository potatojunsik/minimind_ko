# 데이터 양별 vocab 비교 실험 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (사용자 규칙상 subagent-driven-development는 쓰지 않는다). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 문서 단위로 분리한 새 검증 세트에서 5만/10만/20만/40만 조각 × vocab 6,400/16,000 (8회) 학습·평가를 끊겨도 이어지게 돌리고, 설계 7절 규칙으로 SFT 토크나이저를 판정한다.

**Architecture:** ① parquet에서 doc_id를 붙여 조각을 다시 만들고(기존 kowiki.jsonl과 줄 단위 일치 확인) 검증/중첩 학습 세트를 만든다. ② Claude 세션과 독립된 창에서 도는 실행기가 상태 파일을 보며 학습→평가를 차례로 하고 끝난 건 건너뛴다. ③ 판정 스크립트가 문서 단위 쌍 부트스트랩으로 Δ(BPB)와 구간을 구해 1/2/3 판정을 낸다. ④ 결과를 docs/06, README, 그래프에 정리한다.

**Tech Stack:** Python 3.11 (`D:\minimind_ko\.venv`), torch 2.11 cu128, pyarrow 20.0.0, 표준 `unittest`(pytest 없음), matplotlib.

**Spec:** `docs/superpowers/specs/2026-11-18-data-scale-bpc-design.md` (실행자는 이 문서도 읽는다)

## Global Constraints

- 모든 명령은 `D:\minimind_ko`에서 `.venv\Scripts\python.exe -X utf8 ...`으로 실행 (cp949 오류 방지).
- 학습 설정은 docs/05와 동일: hidden 768, layer 8, max_seq_len 512, batch 16, accum 8, lr 5e-4, seed 42, bf16, 1 epoch. `train_pretrain_ko.py` 기본값이 이미 이 값이므로 `--tokenizer_dir --data_path --out_dir`만 넘긴다.
- 학습 크기 `SIZES = [50000, 100000, 200000, 400000]`, vocab `[6400, 16000]`, 순서는 크기 오름차순, 크기마다 6400 → 16000.
- 토크나이저(`tokenizers/ko_6400`, `tokenizers/ko_16000`)와 기존 데이터·가중치(`data/kowiki*.jsonl`, `out_ko/`)는 수정·삭제하지 않는다.
- BPB = Σ손실(nats)/ln2 ÷ Σ UTF-8 바이트, BPC = 같은 Σ손실/ln2 ÷ Σ글자. 문서별 비율의 평균 금지. 부트스트랩도 매회 합산 후 나눈다.
- Δ = BPB(6400) − BPB(16000). 부트스트랩 단위 = 위키 원문 문서(doc_id), 10,000회, seed 0, 95% 백분위 구간.
- 판정 2번이 나오면 100만 조각 실험을 시작하지 않고 사용자에게 묻는다.
- 공개 문서에 노트북 제조사/모델명을 쓰지 않는다. 새 산출 데이터·가중치는 기존 .gitignore 규칙(`data/`, `out*/`, `*.jsonl`, `*.pth`)에 걸리는 위치에 둔다.
- git commit/push는 사용자가 지시할 때만. 각 태스크의 "커밋" 단계는 생략한다. 커밋할 때는 메시지 끝에 `Co-Authored-By: GPT 6.2 Astra`(이메일 없음)와 Claude 공동 작성자 줄을 함께 넣는다.
- 실행 전 제조사 성능 모드 확인 (`scripts/check_gpu.py`로 약 23 TFLOPS 이상).

## Review Focus

1. **중간 저장본 오판**: 원본 train_epoch가 1,000 step마다 같은 이름으로 가중치를 덮어쓴다 → 가중치 파일이 있어도 "완료"가 아니다. 종료 코드 0 + 로그의 `완료:` 줄이 둘 다 있어야 완료 (Task 3 테스트).
2. **실행기 재시작 시 이중 실행/누락**: 상태가 `training`으로 남은 실행(강제 종료됨)은 처음부터 다시, `done`은 건너뜀 (Task 3 테스트). (2026-11-19 실행 중 변경: 체크포인트가 있으면 처음부터가 아니라 이어서 학습 — [docs/06 실험 조건](../../06_데이터양_비교.md#실험-조건))
3. **텍스트 중복 조각**: 위키에는 다른 문서에 똑같은 조각(틀 문구 등)이 있을 수 있다 → 그런 조각을 가진 문서는 검증 후보에서 뺀다 (Task 1 테스트).
4. **비율의 평균 실수**: 문서별 BPB 평균과 합산 BPB는 다르다 (Task 2 테스트로 고정).
5. **점추정은 양수인데 구간이 0을 포함**: 반드시 판정 3(6,400) (Task 2 테스트).

---

### Task 1: 문서 id 데이터와 검증/학습 분할

**Files:**
- Create: `scripts/prepare_scale_data.py`
- Create: `tests/test_scale_split.py`
- 산출: `data/scale/heldout.jsonl`, `data/scale/train_{2000,50000,100000,200000,400000}.jsonl`, `data/scale/split_report.json`

**Interfaces:**
- Produces:
  - `select_heldout(chunks: list[tuple[int, str]], candidate_start: int, target_chunks: int, seed: int) -> set[int]` — `chunks[i] = (doc_idx, text)`, i는 kowiki.jsonl 줄 번호. 후보 = 모든 조각의 줄 번호 ≥ candidate_start 이고, 어떤 조각 text도 다른 문서에 나오지 않는 문서. 후보 문서를 `random.Random(seed)`로 섞어 누적 조각 수가 target_chunks 이상이 될 때까지 문서 단위로 담는다.
  - `make_nested_train(chunks, heldout_docs: set[int], sizes: list[int], seed: int) -> dict[int, list[int]]` — 검증 문서 조각을 뺀 줄 번호 풀을 한 번 섞고 앞에서 size개씩 자른 줄 번호 목록.
  - 상수: `TOKENIZER_SAMPLE_END = 1_000_000`, `HELDOUT_TARGET = 2000`, `SPLIT_SEED = 20261118`, `SIZES = [2000, 50000, 100000, 200000, 400000]` (2000은 실행기 동작 확인용).
  - `data/scale/heldout.jsonl` 각 줄 `{"text", "doc_id"}` (doc_id = parquet `id` 문자열), 학습 파일 각 줄 `{"text"}`, 학습 doc_id는 `data/scale/train_400000_docids.txt`(줄 순서 일치, 작은 세트는 앞부분).

- [ ] **Step 1: 테스트 작성** — `tests/test_scale_split.py` (`unittest`)
  - `test_heldout_only_after_start`: 문서 0(줄 0~2), 문서 1(줄 3~5, start=3 걸침 아님), 문서 2(줄 2~4 처럼 경계 걸침) 구성 → 결과에 경계를 걸친 문서와 start 이전 문서 없음.
  - `test_heldout_excludes_duplicate_text`: 문서 1의 조각 text가 문서 0에도 있으면 문서 1 제외.
  - `test_heldout_whole_documents`: 결과 문서들의 조각 전부가 담기고, 누적 조각 수 ≥ target.
  - `test_nested_and_disjoint`: `make_nested_train` 결과에서 `set(r[2]) ⊂ set(r[4])` 이고 r[4]의 앞 2개가 r[2]와 같음, 어떤 줄도 검증 문서에 속하지 않음, 같은 seed면 같은 결과.
- [ ] **Step 2: 실행해 실패 확인** — `.venv\Scripts\python.exe -X utf8 -m unittest tests.test_scale_split -v` → ImportError로 FAIL.
- [ ] **Step 3: 구현** — 두 함수 + `main()`:
  - `scripts/prepare_kowiki.py`의 `split_paragraphs`, `PARQUET_FILES`를 import해 같은 순서로 parquet(`data/raw/20231101.ko`, columns `id`,`text`)를 읽는다(로컬 파일 직접 `pq.read_table`, 다운로드 호출 없음).
  - 생성한 조각 text를 `data/kowiki.jsonl`과 줄 단위로 전수 비교, 줄 수·내용 하나라도 다르면 중단.
  - `split_report.json`에 기록: 검증 문서 수/조각 수/글자 수/바이트 수, 그리고 겹침 확인 4종 —
    (a) 검증 조각 text 중 `data/tokenizer_sample.jsonl`에 있는 수, (b) 토크나이저 샘플 text를 doc으로 역매핑했을 때 검증 문서에 속하는 수,
    (c) 각 학습 세트에서 검증 doc에 속하는 줄 수, (d) 참고: 검증 조각 text 중 기존 `kowiki_train_100k.jsonl`에 있는 수.
    (e) (사용자 추가 요청) 생성한 학습 파일 4개(5만~40만)를 **디스크에서 다시 읽어** 검증 조각 text와 같은 줄 수 — 파일별로 기록.
    보고서에는 파일별로 `doc_id_overlap`(c)과 `text_overlap`(e)을 따로 적는다.
    (a)(b)(c)(e) 중 하나라도 0이 아니면 예외로 중단.
- [ ] **Step 4: 테스트 통과 확인** — 같은 명령 → 4개 OK.
- [ ] **Step 5: 실제 데이터 생성** — `.venv\Scripts\python.exe -X utf8 scripts\prepare_scale_data.py` → 끝에 `split_report.json` 출력, (a)(b)(c)=0, 학습 파일 줄 수가 각 SIZES와 일치(`wc -l`).

### Task 2: BPB 평가와 판정 스크립트

**Files:**
- Modify: `scripts/eval_bpc.py` (per_doc 행에 `"bytes": len(text.encode("utf-8"))` 추가, 요약 출력에 `BPB (텍스트+eos)`, `평가 텍스트 sha256: <hex>`(평가한 text들을 `\n`으로 이어 UTF-8로 해시) 두 줄 추가. 기존 출력 줄과 기본값은 유지)
- Create: `scripts/decide_vocab.py`
- Create: `tests/test_decide_vocab.py`

**Interfaces:**
- Consumes: Task 1의 `data/scale/heldout.jsonl` (줄 순서 = per_doc 행 순서).
- Produces:
  - `group_by_doc(rows: list[dict], doc_ids: list[str]) -> dict[str, tuple[float, int, int]]` — doc_id → (nll 합, bytes 합, chars 합).
  - `rate(groups, ids: list[str], unit: str) -> float` — unit `"bytes"`/`"chars"`, Σnll/ln2/Σunit.
  - `bootstrap_ids(doc_ids: list[str], rounds: int, seed: int) -> list[list[str]]` — 매 회 문서를 복원추출한 목록(네 모델에 같은 목록 사용).
  - `ci(values: list[float]) -> tuple[float, float]` — 2.5%/97.5% 백분위 (정렬 후 `int(0.025*n)`, `int(0.975*n)-1`, paired_compare와 같은 방식).
  - `decide(delta400: float, delta400_lo: float, growth_lo: float) -> int` — 1: `delta400_lo > 0`; 2: `delta400 < 0 and growth_lo > 0`; 그 외 3.
  - CLI: `decide_vocab.py --status out_scale/status.json --heldout data/scale/heldout.jsonl [--rounds 10000]` → 크기별 표(BPB·BPC 각 vocab, Δ, Δ 95% 구간, 6,400 우세 비율을 **조각 단위와 문서 단위 둘 다**), 20만·40만 각각의 Δ와 95% 구간, `Δ(40만)−Δ(20만)` 점추정과 95% 구간, 판정 번호와 문구를 출력하고 `out_scale/decision.json`에 저장. 상태 파일에 끝난 크기만 있으면 있는 것만 표로 내고 판정은 "미완료".
  - 상수 `ROUNDS = 10000`, `SEED = 0`.

- [ ] **Step 1: 테스트 작성** — `tests/test_decide_vocab.py`
  - `test_rate_is_ratio_of_sums`: 문서 A(nll=ln2·10, bytes 10), B(nll=ln2·1, bytes 100) → `rate == 11/110` (문서별 평균 0.505가 아님).
  - `test_group_by_doc_sums_chunks`: 같은 doc_id 두 행이 합쳐짐.
  - `test_decide_cases`: `decide(0.02, 0.005, -1) == 1`; `decide(-0.03, -0.04, 0.01) == 2`; `decide(-0.03, -0.04, -0.001) == 3`; `decide(0.01, -0.002, 0.5) == 3` (점추정 양수지만 구간에 0 포함); `decide(0.0, -0.01, 0.5) == 3`.
  - `test_bootstrap_same_ids_reused`: 같은 seed → 같은 목록, 각 회 길이 = 문서 수.
- [ ] **Step 2: 실패 확인** — `.venv\Scripts\python.exe -X utf8 -m unittest tests.test_decide_vocab -v` → FAIL.
- [ ] **Step 3: 구현** — 위 시그니처대로. eval_bpc 수정은 위 두 줄만.
- [ ] **Step 4: 통과 확인** — 같은 명령 OK, 그리고 기존 모델로 회귀 확인: `eval_bpc.py --tokenizer_dir tokenizers/ko_6400 --weight out_ko/pretrain_ko6400_768.pth --limit 50` 의 BPC 줄이 수정 전 같은 명령 값과 동일(수정 전에 한 번 실행해 값 기록).

### Task 3: 이어 돌릴 수 있는 실행기

**Files:**
- Modify: `scripts/train_pretrain_ko.py` — `--ckpt_dir` 인자 추가(기본값 `""` = 기존 동작 그대로 `minimind/checkpoints`). 값이 있으면 `base.lm_checkpoint`를 감싸 `save_dir`를 그 경로로 강제(원본 파일은 수정하지 않음). 기존 `minimind/checkpoints`의 파일은 건드리지 않는다.
- Create: `scripts/run_scale.py`
- Create: `tests/test_run_scale.py`

**(사용자 추가 요청)**
- 학습 호출에 `--ckpt_dir {out_root}/ko{V}_n{N}/checkpoints`를 넘긴다 (시험·본 실행 모두).
- 중복 실행 방지: 실행기는 시작 시 `{out_root}/runner.lock`(실행기 PID)과 상태 파일의 `child_pid`(학습/평가 프로세스)를 확인해, 둘 중 하나라도 **살아 있는 python 프로세스**면 아무것도 하지 않고 종료. 생존 확인은 Windows API(ctypes `OpenProcess` + `GetExitCodeProcess` == 259, 이미지 경로에 `python` 포함). `os.kill(pid, 0)`은 Windows에서 프로세스를 종료시키므로 쓰지 않는다. 정상 종료 시 lock 삭제.
- 평가 확인: 평가 호출에 그 실행의 `--tokenizer_dir`, `--weight`(`{run}/pretrain_ko{V}_768.pth`), `--limit {eval_limit}`(0이면 생략)이 들어가는지 명령줄을 로그에 그대로 남긴다. 평가 출력의 `문서 N개`와 `평가 텍스트 sha256`을 파싱해, 실행기가 직접 계산한 기대값(검증 파일 앞 N줄 또는 전체)과 같을 때만 `done`. 다르면 `failed` 후 중단.
- 산출: `out_scale/status.json`, `out_scale/ko{V}_n{N}/pretrain_ko{V}_768.pth`, `out_scale/ko{V}_n{N}/per_doc.jsonl`, `logs/06_scale_ko{V}_n{N}.log`, `logs/06_scale_ko{V}_n{N}_eval.log`

**Interfaces:**
- Consumes: Task 1 학습/검증 파일, Task 2 `eval_bpc.py`(`--per_doc_out`), `decide_vocab.py` CLI.
- Produces:
  - `is_train_complete(returncode: int, log_path: str) -> bool` — `returncode == 0` 이고 로그에 `완료:`로 시작하는 줄(Logger 접두어 허용, 포함 검사)이 있을 때만 True.
  - `load_status(path) -> dict` / `save_status(path, status)` — 임시 파일에 쓰고 `os.replace`로 교체. 형식: `{"runs": {"ko6400_n50000": {"state": "pending|training|evaluating|done|failed", "train_start", "train_end", "eval_end", "bpb", "bpc"}}}` (시각은 ISO 문자열).
  - `next_runs(status, sizes, vocabs) -> list[tuple[int, int]]` — `done`이 아닌 것 전부, 정해진 순서대로(`training`/`evaluating`/`failed`로 남은 것도 처음부터 다시).
  - CLI: `run_scale.py [--sizes 50000 100000 200000 400000] [--out_root out_scale] [--eval_limit 0]`. 학습은 `subprocess.run`으로 `train_pretrain_ko.py --tokenizer_dir tokenizers/ko_{V} --data_path data/scale/train_{N}.jsonl --out_dir {out_root}/ko{V}_n{N}`, stdout/stderr를 로그 파일로. 실패(완료 판정 False) 시 상태 `failed` 기록 후 **전체 중단**(다음 실행으로 넘어가지 않음 — 원인 확인 필요). 평가는 `eval_bpc.py --data_path data/scale/heldout.jsonl --per_doc_out ...`, 출력에서 BPB/BPC 줄을 파싱해 상태에 저장. 전부 끝나면 `decide_vocab.py`를 실행해 출력도 `logs/06_scale_decision.log`에 남긴다. 100만 실행 기능은 넣지 않는다.

- [ ] **Step 1: 테스트 작성** — `tests/test_run_scale.py`
  - `test_complete_needs_both`: (0, 완료 줄 있음) True; (0, 없음) False; (1, 있음) False.
  - `test_next_runs_order_and_skip`: ko6400_n50000 done, ko16000_n50000 training → 결과 첫 항목 (50000, 16000), done 항목 없음, 길이 7.
  - `test_save_status_roundtrip`: 임시 폴더에 저장 후 로드 동일, `.tmp` 파일 남지 않음.
  - `test_pid_alive`: 자기 자신 PID → True, 존재하지 않는 PID(예: 끝난 subprocess의 PID) → False.
  - `test_eval_check`: `eval_matches(output_text, expected_n, expected_sha)` — 문서 수나 해시가 하나라도 다르면 False.
- [ ] **Step 2: 실패 확인** — `.venv\Scripts\python.exe -X utf8 -m unittest tests.test_run_scale -v` → FAIL.
- [ ] **Step 3: 구현**.
- [ ] **Step 4: 통과 확인** — 같은 명령 OK.
- [ ] **Step 5: 실제 동작 확인(스모크)** — `.venv\Scripts\python.exe -X utf8 scripts\run_scale.py --sizes 2000 --out_root out_scale_smoke --eval_limit 50` → 두 vocab 모두 `done`, BPB 값 기록됨, `decide_vocab`이 "미완료"로 정상 종료. 확인할 것: 평가 명령줄에 각 실행의 tokenizer/weight/`--limit 50`이 들어 있음, 두 모델의 sha256·문서 수가 같음, 체크포인트가 `out_scale_smoke/ko{V}_n2000/checkpoints/`에만 생기고 `minimind/checkpoints`의 기존 6개 파일 수정 시각이 그대로임. 같은 명령을 다시 실행하면 학습 없이 바로 끝남(건너뛰기 확인). 실행 중에 두 번째 실행기를 띄우면 바로 종료됨(중복 방지 확인). 확인 후 `out_scale_smoke/`, `logs/06_scale_*_n2000*` 삭제.

### Task 4: 본 실행 시작과 재개 정보 기록

**Files:**
- Modify: 메모리 `minimind-ko-study.md` (Claude 메모리 폴더)
- Create: `scripts/run_scale.bat` (새 창에서 실행기를 띄우는 한 줄 배치: `start "scale" cmd /k ".venv\Scripts\python.exe -X utf8 scripts\run_scale.py"`, 배치 파일 위치 기준으로 `cd /d %~dp0..`)

- [ ] **Step 1: 성능 모드 확인** — 사용자에게 제조사 성능 모드 확인을 요청하고 `scripts\check_gpu.py` 결과가 약 23 TFLOPS인지 확인.
- [ ] **Step 2: 메모리 기록** — "데이터 양 실험 진행 중, 상태 파일 `D:\minimind_ko\out_scale\status.json`, 재개 = `scripts\run_scale.bat` 다시 실행, 끝나면 `logs\06_scale_decision.log` 확인, 판정 2면 100만 전에 사용자에게 질문, 설계/계획 문서 경로"를 적는다.
- [ ] **Step 3: 실행** — `cmd //c start "" "D:\minimind_ko\scripts\run_scale.bat"` (Claude 세션과 독립된 창). 1~2분 뒤 `status.json`이 `ko6400_n50000: training`이고 로그에 loss 줄이 찍히는지 확인.
- [ ] **Step 4: 대기** — 약 9시간. 중간 확인은 `status.json`만 읽는다.

### Task 5: 결과 정리 (실행 완료 후)

**Files:**
- Create: `scripts/plot_scale.py` (`out_scale/decision.json` → `images/scale_bpc.png`: x = 학습 조각 수 로그 축, y = held-out BPC(설계 6절), vocab별 선 2개, 색·폰트는 `plot_loss.py`의 `SERIES_COLORS`, `pick_korean_font` 재사용 import)
- Create: `docs/06_데이터양_비교.md` (docs/05 형식: 목적, 분할 방법과 `split_report.json` 겹침 수치, 설정, 결과 표(BPB·BPC·Δ·95% 구간·문서 우세), 표 바로 아래 **"95% 신뢰구간은 검증 문서 표본에 따른 변동을 나타낸다."** 한 줄, 그래프, 판정과 근거(7절 규칙 인용, Δ 부호가 docs/05와 반대임 명시), 한계(시드 1개, 검증 문서가 파일 뒤쪽이라 최근 문서 치우침 가능), 만든 스크립트 표)
- Modify: `README.md` 진행표·실험 표·스크립트 목록, `README_en.md` 같은 부분, `docs/05_vocab_비교.md` 한계 절의 "다음 실험부터는 문서 단위로 나눈다" 뒤에 docs/06 링크
- Modify: 메모리 (판정 결과, 다음 단계)

- [ ] **Step 1: 판정 확인** — `logs/06_scale_decision.log`의 판정 번호. **2번이면 여기서 멈추고, 20만·40만의 Δ와 각 95% 구간, `Δ(40만)−Δ(20만)`의 95% 구간을 보여 주며 사용자에게 100만 실험 여부를 묻는다.**
- [ ] **Step 2: 그래프** — `.venv\Scripts\python.exe -X utf8 scripts\plot_scale.py` → `images/scale_bpc.png` 생성, 이미지를 열어 점 8개 확인. 결과표에는 BPB(판정용)와 BPC(비교용), 조각 단위·문서 단위 우세 비율을 모두 싣는다.
- [ ] **Step 3: 문서 작성** — 위 파일들. 표의 모든 숫자는 `decision.json`/`split_report.json`/로그에서 복사.
- [ ] **Step 4: 검증** — 문서의 숫자를 원본 파일과 대조(verification-before-completion), 전체 테스트 `.venv\Scripts\python.exe -X utf8 -m unittest discover tests -v` OK, 문서에 노트북 제조사·모델명 없음(검색으로 확인).
