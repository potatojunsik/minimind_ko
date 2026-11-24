# 최종 사전학습 구현 계획

> 실행: 작성자가 이 세션에서 직접 진행한다(사용자 지시). 커밋은 사용자가 지시할 때만 한다.

**목표:** 패킹 학습 기능과 실행기·판정 스크립트를 만들어, 실험 1·2 → 최종 사전학습 → 최종 SFT를 사전 기준대로 돌린다.

**구조:** 패킹은 `scripts/packing.py`(순수 함수 + Dataset)로 분리하고 `train_pretrain_ko.py`에 옵션으로 붙인다.
실행기 `run_final.py`는 `run_scale.py`의 상태 관리·자식 프로세스 함수를 import해 재사용하고, 판정은 `decide_final.py`가
`decide_vocab.py`의 문서 묶기·부트스트랩 함수를 import해 계산한다.

**기술:** Python, PyTorch, numpy, HF tokenizers, unittest. **설계:** `docs/superpowers/specs/2026-11-23-final-pretrain-design.md`

## 전체 제약

- 원본 `minimind/` 파일은 고치지 않는다. 06번의 `out_scale/` 결과 파일은 읽기만 한다.
- 판정 숫자는 설계 문서 그대로: 패킹 상한 +0.005, 2 epoch g/G ≥ 0.5 · g 하한 > 0 · 2 epoch 끝 < 1 epoch 끝, 최종 Δf 상한 < 0.
- 부트스트랩: 문서 단위, 10,000회, `random.Random(0)`, `decide_vocab.ci` 백분위.
- 산출물 `out_final/`, 로그 `logs/08_final_*.log`.

## 확인할 실패 경우 (테스트로 고정)

1. 패킹 줄을 이었을 때 원래 토큰 열과 다르다(토큰 누락·순서 바뀜) → 테스트로 막는다.
2. 512보다 긴 조각이 잘린다 → 테스트로 막는다.
3. 토큰 id가 uint16 범위를 넘는다(vocab > 65,535) → 오류로 멈춘다.
4. 판정 경계값(정확히 +0.005, g/G 정확히 0.5) → 경계 테스트.
5. 비교 대상 평가 결과가 다른 검증 세트(sha 불일치)거나 문서 수가 다르다 → 판정 스크립트가 멈춘다.

---

### Task 1: 패킹 (`scripts/packing.py`, 테스트 `tests/test_final_pretrain.py`)

- `pack_token_lists(token_lists, bos_id, eos_id, pad_id, seq_len) -> np.ndarray[uint16] (rows, seq_len)`:
  `[bos]+tokens+[eos]`를 이어 seq_len씩 자르고 마지막 줄만 pad로 채운다.
- `tokenize_and_pack(texts, tokenizer, seq_len, batch_size=10000) -> (rows, stats)`: 배치로 토큰화해 numpy로 바로 옮긴다
  (Python int 리스트를 전부 들고 있지 않는다). stats = 조각 수, 토큰 수(bos·eos 포함), 줄 수, pad 수.
- `PackedPretrainDataset(rows, bos_id, pad_id)`: `__getitem__` → (input_ids long, labels), labels는 pad·bos 위치 -100.
- 테스트: 줄에서 pad를 빼고 이으면 원래 열과 같음 / 600토큰 조각이 잘리지 않고 다음 줄로 이어짐 / 정확히 나눠떨어지면 pad 0 /
  라벨 마스크 위치 / vocab 65,536 이상 id면 ValueError / 실제 ko_16000 토크나이저로 `tokenize_and_pack` 결과가 조각별 토큰화와 같음.

### Task 2: `train_pretrain_ko.py` 옵션

- `--pack 0|1`: 1이면 `PretrainDataset` 대신 `tokenize_and_pack` + `PackedPretrainDataset`. stats를 로그에 남긴다.
- `--keep_epoch_weights 0|1`: 1이면 epoch가 끝날 때마다 `<save_weight>_768.pth`를 `<save_weight>_768_epoch<N>.pth`로 복사.
- 확인: `train_2000.jsonl`로 `--pack 1 --epochs 2 --keep_epoch_weights 1` 짧은 실행이 "완료:" 줄과 epoch 파일 2개를 남김.

### Task 3: 판정 (`scripts/decide_final.py`)

- `paired(groups_a, groups_b, unique_ids, samples) -> {"point", "ci"}`: BPB(a) − BPB(b). `samples`는 `decide_vocab.bootstrap_ids`로
  한 번 만든 부트스트랩 표본(10,000회, seed 0)을 모든 비교에 같이 쓴다.
- `decide_packing(hi) -> bool` (hi < 0.005), `decide_epochs(g, G, g_lo, bpb_final, bpb_epoch1) -> bool`, `decide_accept(hi) -> bool` (hi < 0).
- `load_groups(path, doc_ids, prefix=False)`: per_doc 행 수가 검증 조각 수와 다르면 멈춘다.
- `check_old_runs(keys)`: 06번 비교 대상이 같은 검증 세트(문서 수·sha256) 전체로 평가됐는지 확인한다.
- 하위 명령 `setup`(실험 판정 → `out_final/decision.json`의 pack·epochs), `accept`(최종 판정),
  `sft`(최종 SFT 대화 형식 평가 → `data/chat/chat_eval_final_sft_ko16000.json`, 07번과 답 본문 BPB 비교, 기록용).
- 테스트: 경계값, 같은 입력이면 Δ 0, 행 수 불일치 오류, 06번 저장값(40만 Δ·구간) 재현.

### Task 4: 실행기 (`scripts/run_final.py`)

- 단계: `exp_pack400k` → `exp_ep2_200k`(epoch1 가중치도 평가) → setup 판정 → `final`(pack·epochs는 decision.json) →
  accept 판정 → `sft`(채택일 때만) → 대화 평가(`decide_final.py sft --tag final_sft`).
- `run_scale`의 `load_status/save_status/pid_alive/run_child/is_train_complete/eval_matches/expected_eval/parse_metric`를 재사용.
- 확인: `--smoke`(train_2000, eval_limit 50, out_final_smoke)로 전 단계가 끝까지 돈다.

### Task 5: 학습 풀 전체 파일 (`scripts/prepare_final_data.py`)

- `prepare_scale_data`의 `build_chunks / verify_same_as_kowiki / select_heldout / make_nested_train`을 같은 시드로 불러
  `data/scale/train_1183370.jsonl`을 쓴다. 앞 100만 줄이 `train_1000000.jsonl`과 같고, 검증 문서·텍스트 겹침 0인지 확인한다.
