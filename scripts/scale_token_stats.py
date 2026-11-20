"""
데이터 양 실험(docs/06)의 학습·검증 세트를 vocab별로 학습 때와 똑같이 토큰화해서 토큰 수 표를 출력한다.

학습 때 조각 하나가 모델에 들어가는 모양 (minimind/dataset/lm_dataset.py의 PretrainDataset):
    텍스트 토큰 = tokenizer(text, add_special_tokens=False, max_length=510, truncation=True)
    입력 = [bos] + 텍스트 토큰 + [eos] + [pad] × (512 − 길이)
    채점 위치 = 입력 위치 1~511 중 pad가 아닌 곳 = 자른 뒤 텍스트 토큰 + eos
              (텍스트 안에 pad와 같은 id의 토큰이 있으면 그만큼 빠진다)

출력 항목 (vocab 6,400 / 16,000):
    글자 수, 자른 뒤 텍스트 토큰, 채점 토큰, 잘린 조각(텍스트 토큰 510개 초과), 잘려서 못 본 글자,
    패딩 비율(512칸 중 pad 칸), 글자/토큰·바이트/토큰(자르기 전 전체 텍스트 기준 = 토크나이저 압축률),
    vocab 한 항목당 평균 등장 횟수(자른 뒤 텍스트 토큰 ÷ vocab 크기)
검증 세트는 평가 때 자르지 않는다(scripts/eval_bpc.py의 슬라이딩 윈도우). 그래서 검증 세트는 자르기 전 값을 본다.

작은 학습 세트가 큰 세트의 앞부분이면(5만~40만은 100만의 앞부분) 큰 세트만 토큰화하고 앞부분만 따로 센다.
잘린 조각 전부와 안 잘린 조각 일부는 실제 학습 호출(truncation=True)과 결과가 같은지 확인해서 출력한다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_token_stats.py
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_token_stats.py --out out_scale\\token_stats.json
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_token_stats.py --data data\\kowiki_train_100k.jsonl   (05번 문서의 10만 값 재현)
"""
import argparse
import json
import os
import time

import numpy as np
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZERS = {6400: "tokenizers/ko_6400", 16000: "tokenizers/ko_16000"}
EVAL_DATA = "data/scale/heldout.jsonl"  # 검증 세트: 평가 때 자르지 않는다
DEFAULT_DATA = [
    "data/scale/train_50000.jsonl",
    "data/scale/train_100000.jsonl",
    "data/scale/train_200000.jsonl",
    "data/scale/train_400000.jsonl",
    "data/scale/train_1000000.jsonl",
    EVAL_DATA,
]
MAX_SEQ_LEN = 512  # train_pretrain_ko.py 기본값
MAX_TEXT = MAX_SEQ_LEN - 2  # bos, eos 자리를 뺀 텍스트 토큰 최대 수
BATCH = 2048
# 안 잘린 조각은 이 간격마다 하나씩만 실제 학습 호출과 비교한다 (전부 비교하면 시간이 두 배로 든다)
CHECK_EVERY = 97


def load_texts(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [str(json.loads(line)["text"]) for line in f]


def group_by_prefix(paths):
    """앞부분 관계인 파일끼리 묶는다. 반환: [(토큰화할 파일, 텍스트, {파일: 앞부분 줄 수})]"""
    groups = []
    for path in sorted(paths, key=lambda p: os.path.getsize(os.path.join(ROOT, p)), reverse=True):
        texts = load_texts(path)
        n = len(texts)
        for _, base_texts, members in groups:
            if len(base_texts) >= n and base_texts[:n] == texts:
                members[path] = n
                break
        else:
            groups.append((path, texts, {path: n}))
    return groups


def count_tokens(texts, tok):
    """조각마다 자르기 전 텍스트 토큰 수, 잘려서 못 본 글자 수, 자른 뒤 텍스트 안의 pad id 토큰 수를 센다."""
    backend = tok.backend_tokenizer
    n = len(texts)
    full = np.zeros(n, dtype=np.int64)
    unseen_chars = np.zeros(n, dtype=np.int64)
    pad_in_text = np.zeros(n, dtype=np.int64)
    checked = mismatched = 0
    for s in range(0, n, BATCH):
        batch = texts[s:s + BATCH]
        # 아래 확인용 tok(..., truncation=True) 호출이 자르기 설정을 남기므로, 자르지 않는 토큰화 전에 매번 해제한다
        backend.no_truncation()
        encs = backend.encode_batch(batch, add_special_tokens=False)
        check = []
        for j, enc in enumerate(encs):
            i = s + j
            full[i] = len(enc)
            if full[i] > MAX_TEXT:
                # 남은 마지막 토큰이 끝나는 글자 위치 뒤가 학습에서 못 본 부분이다
                unseen_chars[i] = len(batch[j]) - enc.offsets[MAX_TEXT - 1][1]
                check.append(j)
            elif i % CHECK_EVERY == 0:
                check.append(j)
            # pad id는 특수 토큰 문자열이 그대로 들어 있을 때만 나오므로 그 경우만 센다
            if tok.pad_token in batch[j]:
                pad_in_text[i] = enc.ids[:MAX_TEXT].count(tok.pad_token_id)
        ref = tok([batch[j] for j in check], add_special_tokens=False, max_length=MAX_TEXT, truncation=True).input_ids
        for j, r in zip(check, ref):
            checked += 1
            mismatched += r != encs[j].ids[:MAX_TEXT]
    return {"full": full, "unseen_chars": unseen_chars, "pad_in_text": pad_in_text,
            "checked": checked, "mismatched": mismatched}


def summarize(counts, chars, nbytes, n, vocab_size):
    full = counts["full"][:n]
    kept = np.minimum(full, MAX_TEXT)
    trunc = full > MAX_TEXT
    total_chars = int(chars[:n].sum())
    total_bytes = int(nbytes[:n].sum())
    kept_sum = int(kept.sum())
    unseen = int(counts["unseen_chars"][:n].sum())
    pad = MAX_SEQ_LEN * n - (kept_sum + 2 * n)
    return {
        "pieces": n,
        "chars": total_chars,
        "bytes": total_bytes,
        "text_tokens_full": int(full.sum()),
        "text_tokens_kept": kept_sum,
        "scored_tokens": kept_sum + n - int(counts["pad_in_text"][:n].sum()),
        "truncated_pieces": int(trunc.sum()),
        "truncated_pct": 100 * float(trunc.mean()),
        "unseen_chars": unseen,
        "unseen_chars_pct": 100 * unseen / total_chars,
        "pad_pct": 100 * pad / (MAX_SEQ_LEN * n),
        "chars_per_token": total_chars / int(full.sum()),
        "bytes_per_token": total_bytes / int(full.sum()),
        "uses_per_vocab_entry": kept_sum / vocab_size,
    }


ROWS = [
    ("조각 수", "pieces", "{:,}"),
    ("글자 수", "chars", "{:,}"),
    ("UTF-8 바이트", "bytes", "{:,}"),
    ("자르기 전 텍스트 토큰", "text_tokens_full", "{:,}"),
    ("자른 뒤 텍스트 토큰", "text_tokens_kept", "{:,}"),
    ("채점 토큰 (자른 뒤 텍스트 + eos)", "scored_tokens", "{:,}"),
    ("잘린 조각 (텍스트 토큰 > 510)", "truncated_pieces", "{:,}"),
    ("잘린 조각 비율", "truncated_pct", "{:.2f}%"),
    ("잘려서 못 본 글자", "unseen_chars", "{:,}"),
    ("못 본 글자 비율", "unseen_chars_pct", "{:.2f}%"),
    ("패딩 비율 (512칸 중)", "pad_pct", "{:.2f}%"),
    ("글자/토큰 (자르기 전)", "chars_per_token", "{:.3f}"),
    ("바이트/토큰 (자르기 전)", "bytes_per_token", "{:.3f}"),
    ("vocab 한 항목당 평균 등장 횟수 (자른 뒤)", "uses_per_vocab_entry", "{:,.0f}"),
]
RATIO_KEYS = {"text_tokens_full", "text_tokens_kept", "scored_tokens", "chars_per_token", "bytes_per_token"}


def print_set_table(path, stats):
    a, b = stats[6400], stats[16000]
    print(f"\n### {path}\n")
    print("| 항목 | vocab 6,400 | vocab 16,000 | 16,000 ÷ 6,400 |")
    print("|---|---|---|---|")
    for label, key, fmt in ROWS:
        ratio = f"{b[key] / a[key]:.4f}" if key in RATIO_KEYS else ""
        print(f"| {label} | {fmt.format(a[key])} | {fmt.format(b[key])} | {ratio} |")
    if os.path.normpath(path) == os.path.normpath(EVAL_DATA):
        print("\n검증 세트는 평가 때 자르지 않으므로(scripts/eval_bpc.py) '자르기 전' 행을 본다. "
              "자른 뒤·채점·잘린 조각·패딩 행은 학습 방식으로 넣었을 때의 참고값이다.")


def print_overview(results):
    print("\n### 세트별 요약 (비율은 모두 16,000 ÷ 6,400)\n")
    print("| 세트 | 자른 뒤 텍스트 토큰 6,400 | 16,000 | 자르기 전 텍스트 토큰 비 | 자른 뒤 텍스트 토큰 비 | 채점 토큰 비 "
          "| 잘린 조각 6,400 / 16,000 | 패딩 6,400 / 16,000 |")
    print("|---|---|---|---|---|---|---|---|")
    for path, stats in results.items():
        a, b = stats[6400], stats[16000]
        print(f"| {path} | {a['text_tokens_kept']:,} | {b['text_tokens_kept']:,} "
              f"| {b['text_tokens_full'] / a['text_tokens_full']:.4f} "
              f"| {b['text_tokens_kept'] / a['text_tokens_kept']:.4f} "
              f"| {b['scored_tokens'] / a['scored_tokens']:.4f} "
              f"| {a['truncated_pct']:.2f}% / {b['truncated_pct']:.2f}% | {a['pad_pct']:.2f}% / {b['pad_pct']:.2f}% |")
    print("\n비율 r은 16,000이 (1 − r)만큼 적다는 뜻이고, 같은 차이를 6,400 기준으로 쓰면 6,400이 1 ÷ r배다. "
          "자른 뒤 비율이 자르기 전보다 1에 가까우면 6,400이 512칸에서 더 많이 잘렸다는 뜻이다.")


def main():
    p = argparse.ArgumentParser(description="학습·검증 세트의 vocab별 토큰 수 (학습 때와 같은 자르기·패딩)")
    p.add_argument("--data", nargs="+", default=DEFAULT_DATA, help="jsonl 파일들 (minimind_ko 폴더 기준 경로)")
    p.add_argument("--out", default="", help="결과를 저장할 JSON 경로 (선택)")
    args = p.parse_args()

    t0 = time.time()
    groups = group_by_prefix(args.data)
    lengths = {base: (np.array([len(t) for t in texts], dtype=np.int64),
                      np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64))
               for base, texts, _ in groups}
    results = {path: {} for path in args.data}
    verify = {}
    for vocab, tok_dir in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, tok_dir))
        for base, texts, members in groups:
            counts = count_tokens(texts, tok)
            for path, n in members.items():
                results[path][vocab] = summarize(counts, *lengths[base], n, len(tok))
            verify[f"{base} / vocab {vocab:,}"] = {"checked": counts["checked"], "mismatched": counts["mismatched"]}
            print(f"  vocab {vocab:,}: {base} 토큰화 끝 ({time.time() - t0:.0f}초)", flush=True)

    print("\n학습 호출(truncation=True)과 같은지 확인 (잘린 조각 전부 + 안 잘린 조각 일부):")
    for key, v in verify.items():
        print(f"  {key}: {v['checked']:,}개 확인, 다른 결과 {v['mismatched']:,}개")
    for base, _, members in groups:
        for path, n in members.items():
            if path != base:
                print(f"  {path} = {base}의 앞 {n:,}줄과 같아서 따로 토큰화하지 않음")
    for path, stats in results.items():
        print_set_table(path, stats)
    if len(results) > 1:
        print_overview(results)
    if args.out:
        with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
            json.dump({"max_seq_len": MAX_SEQ_LEN, "verify": verify, "sets": results}, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
