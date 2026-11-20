"""
데이터 양 실험(docs/06)용 검증/학습 세트를 만든다.

1. 로컬 parquet(data/raw/20231101.ko)를 prepare_kowiki.py와 같은 규칙으로 다시 조각내며 원문 문서 id를 붙이고,
   결과가 기존 data/kowiki.jsonl과 줄 단위로 완전히 같은지 확인한다.
2. 검증 세트는 원문 문서 단위로 고른다. 후보는 모든 조각이 TOKENIZER_SAMPLE_END번째 줄 이후에 있는 문서
   (토크나이저 학습 샘플은 그보다 앞쪽 줄에서만 뽑혔다)이고, 다른 문서와 똑같은 조각 글이 하나도 없는 문서다.
3. 학습 풀(검증 문서의 조각을 모두 뺀 나머지)을 한 번 섞고 앞에서부터 잘라 중첩된 학습 세트를 만든다.
4. 겹침을 확인해 data/scale/split_report.json에 남긴다. 겹침이 하나라도 있으면 중단한다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\prepare_scale_data.py
"""
import json
import os
import random
import sys
from collections import defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_kowiki import DOWNLOAD_DIR, PARQUET_FILES, split_paragraphs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KOWIKI_PATH = "data/kowiki.jsonl"
TOKENIZER_SAMPLE_PATH = "data/tokenizer_sample.jsonl"
OLD_TRAIN_PATH = "data/kowiki_train_100k.jsonl"
OUT_DIR = "data/scale"

TOKENIZER_SAMPLE_END = 1_000_000  # 토크나이저 샘플은 0~999,995번째 줄에서 5줄마다 1줄 (docs/05 한계 절)
HELDOUT_TARGET = 2000             # 검증 조각 수 목표 (docs/05와 같은 규모, 문서 단위라 조금 넘을 수 있음)
SPLIT_SEED = 20261118
SIZES = [2000, 50000, 100000, 200000, 400000, 1000000]  # 2000은 실행기 동작 확인용, 100만은 판정 2 후 추가 (앞 40만은 그대로)
CHECKED_SIZES = [50000, 100000, 200000, 400000, 1000000]


def select_heldout(chunks, candidate_start, target_chunks, seed):
    """chunks[i] = (doc_idx, text), i는 줄 번호. 검증으로 쓸 문서 번호 집합을 돌려준다."""
    first_line = {}
    text_docs = defaultdict(set)
    for line, (doc, text) in enumerate(chunks):
        first_line.setdefault(doc, line)
        text_docs[text].add(doc)
    dup_docs = {d for docs in text_docs.values() if len(docs) > 1 for d in docs}
    doc_size = defaultdict(int)
    for doc, _ in chunks:
        doc_size[doc] += 1

    candidates = sorted(d for d, line in first_line.items() if line >= candidate_start and d not in dup_docs)
    random.Random(seed).shuffle(candidates)
    picked, count = set(), 0
    for doc in candidates:
        if count >= target_chunks:
            break
        picked.add(doc)
        count += doc_size[doc]
    return picked


def make_nested_train(chunks, heldout_docs, sizes, seed):
    """검증 문서를 뺀 줄 번호를 한 번 섞고, 크기별로 앞에서부터 자른 목록을 돌려준다 (작은 세트 ⊂ 큰 세트)."""
    pool = [line for line, (doc, _) in enumerate(chunks) if doc not in heldout_docs]
    if max(sizes) > len(pool):
        raise ValueError(f"학습 풀({len(pool)}줄)보다 큰 크기 요청: {max(sizes)}")
    random.Random(seed).shuffle(pool)
    return {size: pool[:size] for size in sizes}


def build_chunks():
    """parquet를 prepare_kowiki.py와 같은 순서·규칙으로 조각낸다. (chunks, doc_ids) 반환."""
    chunks, doc_ids = [], []
    for name in PARQUET_FILES:
        table = pq.read_table(os.path.join(ROOT, DOWNLOAD_DIR, name), columns=["id", "text"])
        for doc_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
            doc_idx = len(doc_ids)
            doc_ids.append(doc_id)
            chunks.extend((doc_idx, c) for c in split_paragraphs(text))
        print(f"{name} | 문서 {len(doc_ids):,} | 조각 {len(chunks):,}", flush=True)
    return chunks, doc_ids


def read_texts(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def verify_same_as_kowiki(chunks):
    old = read_texts(KOWIKI_PATH)
    if len(old) != len(chunks):
        raise RuntimeError(f"줄 수 다름: kowiki.jsonl {len(old):,} vs 새로 만든 조각 {len(chunks):,}")
    for i, (text, (_, new)) in enumerate(zip(old, chunks)):
        if text != new:
            raise RuntimeError(f"{i}번째 줄 내용이 kowiki.jsonl과 다름")
    print(f"kowiki.jsonl과 {len(old):,}줄 모두 일치")


def write_jsonl(path, rows):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    os.makedirs(os.path.join(ROOT, OUT_DIR), exist_ok=True)
    chunks, doc_ids = build_chunks()
    verify_same_as_kowiki(chunks)

    heldout_docs = select_heldout(chunks, TOKENIZER_SAMPLE_END, HELDOUT_TARGET, SPLIT_SEED)
    heldout_lines = [line for line, (doc, _) in enumerate(chunks) if doc in heldout_docs]
    heldout_texts = [chunks[line][1] for line in heldout_lines]
    write_jsonl(f"{OUT_DIR}/heldout.jsonl",
                ({"text": chunks[line][1], "doc_id": doc_ids[chunks[line][0]]} for line in heldout_lines))

    train = make_nested_train(chunks, heldout_docs, SIZES, SPLIT_SEED)
    for size, lines in train.items():
        write_jsonl(f"{OUT_DIR}/train_{size}.jsonl", ({"text": chunks[line][1]} for line in lines))
    with open(os.path.join(ROOT, OUT_DIR, f"train_{max(SIZES)}_docids.txt"), "w", encoding="utf-8") as f:
        for line in train[max(SIZES)]:
            f.write(doc_ids[chunks[line][0]] + "\n")

    # ---- 겹침 확인 ----
    heldout_text_set = set(heldout_texts)
    text_docs = defaultdict(set)
    text_line = {}
    for line, (doc, text) in enumerate(chunks):
        text_docs[text].add(doc)
        text_line.setdefault(text, line)

    tok_texts = read_texts(TOKENIZER_SAMPLE_PATH)
    tok_text_overlap = sum(1 for t in tok_texts if t in heldout_text_set)                       # (a)
    tok_doc_overlap = sum(1 for t in tok_texts if text_docs.get(t, set()) & heldout_docs)       # (b)
    tok_unmatched = sum(1 for t in tok_texts if t not in text_docs)
    tok_max_line = max(text_line[t] for t in tok_texts if t in text_line)

    train_checks = {}
    for size in CHECKED_SIZES:
        doc_overlap = sum(1 for line in train[size] if chunks[line][0] in heldout_docs)          # (c)
        disk_texts = read_texts(f"{OUT_DIR}/train_{size}.jsonl")                                  # (e) 디스크에서 다시 읽음
        text_overlap = sum(1 for t in disk_texts if t in heldout_text_set)
        train_checks[str(size)] = {"lines_on_disk": len(disk_texts), "doc_id_overlap": doc_overlap,
                                   "text_overlap": text_overlap}
    old_train_overlap = sum(1 for t in read_texts(OLD_TRAIN_PATH) if t in heldout_text_set)      # (d) 참고용

    report = {
        "seed": SPLIT_SEED,
        "candidate_start_line": TOKENIZER_SAMPLE_END,
        "total_docs": len(doc_ids),
        "total_chunks": len(chunks),
        "heldout": {
            "docs": len(heldout_docs),
            "chunks": len(heldout_lines),
            "chars": sum(len(t) for t in heldout_texts),
            "bytes": sum(len(t.encode("utf-8")) for t in heldout_texts),
            "first_line": min(heldout_lines),
        },
        "tokenizer_sample": {
            "lines": len(tok_texts),
            "max_line_in_kowiki": tok_max_line,
            "unmatched_lines": tok_unmatched,
            "a_text_overlap_with_heldout": tok_text_overlap,
            "b_doc_overlap_with_heldout": tok_doc_overlap,
        },
        "train": train_checks,
        "d_old_train_100k_text_overlap_reference": old_train_overlap,
    }
    with open(os.path.join(ROOT, OUT_DIR, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    problems = [tok_text_overlap, tok_doc_overlap] + [
        v for c in train_checks.values() for k, v in c.items() if k != "lines_on_disk"]
    if any(problems):
        raise RuntimeError("검증 세트와 겹침이 있음 (split_report.json 확인)")
    for size in CHECKED_SIZES:
        if train_checks[str(size)]["lines_on_disk"] != size:
            raise RuntimeError(f"train_{size}.jsonl 줄 수가 다름")
    print("겹침 확인 통과: (a)(b)(c)(e) 모두 0")


if __name__ == "__main__":
    main()
