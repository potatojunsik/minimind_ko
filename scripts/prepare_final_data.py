"""
최종 사전학습용 학습 풀 전체 파일을 만든다 (설계: docs/superpowers/specs/2026-11-23-final-pretrain-design.md "최종 사전학습").

prepare_scale_data.py와 같은 함수·시드로 위키를 다시 조각내고 검증 문서를 고른 뒤, 학습 풀 전체를 같은 순서로 섞어
data/scale/train_<풀 크기>.jsonl로 쓴다. 기존 파일은 고치지 않는다. 다음을 확인하고 하나라도 어긋나면 멈춘다.
  1. 다시 고른 검증 세트가 data/scale/heldout.jsonl과 줄 단위로 같다
  2. 새 파일의 앞 100만 줄이 data/scale/train_1000000.jsonl과 같다 (같은 섞기 순서)
  3. 새 파일에 검증 문서의 조각이 없고, 검증 조각과 같은 글도 없다

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\prepare_final_data.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_scale_data import (HELDOUT_TARGET, OUT_DIR, SPLIT_SEED, TOKENIZER_SAMPLE_END, build_chunks,  # noqa: E402
                                make_nested_train, read_texts, select_heldout, verify_same_as_kowiki, write_jsonl)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX_SIZE = 1000000


def check_prefix(new_texts, old_texts):
    """new_texts의 앞부분이 old_texts와 줄 단위로 같은지. 다른 첫 줄 번호를 돌려준다(같으면 None)."""
    if len(new_texts) < len(old_texts):
        return len(new_texts)
    for i, (a, b) in enumerate(zip(new_texts, old_texts)):
        if a != b:
            return i
    return None


def main():
    chunks, doc_ids = build_chunks()
    verify_same_as_kowiki(chunks)
    heldout_docs = select_heldout(chunks, TOKENIZER_SAMPLE_END, HELDOUT_TARGET, SPLIT_SEED)
    heldout_texts = [text for doc, text in chunks if doc in heldout_docs]
    if heldout_texts != read_texts(f"{OUT_DIR}/heldout.jsonl"):
        raise RuntimeError("다시 고른 검증 세트가 heldout.jsonl과 다름")
    pool_size = sum(1 for doc, _ in chunks if doc not in heldout_docs)
    lines = make_nested_train(chunks, heldout_docs, [pool_size], SPLIT_SEED)[pool_size]
    texts = [chunks[line][1] for line in lines]

    bad = check_prefix(texts, read_texts(f"{OUT_DIR}/train_{PREFIX_SIZE}.jsonl"))
    if bad is not None:
        raise RuntimeError(f"앞 {PREFIX_SIZE:,}줄이 train_{PREFIX_SIZE}.jsonl과 다름 ({bad}번째 줄)")
    doc_overlap = sum(1 for line in lines if chunks[line][0] in heldout_docs)
    heldout_set = set(heldout_texts)
    text_overlap = sum(1 for t in texts if t in heldout_set)
    if doc_overlap or text_overlap:
        raise RuntimeError(f"검증 세트와 겹침: 문서 {doc_overlap}, 같은 글 {text_overlap}")

    path = f"{OUT_DIR}/train_{pool_size}.jsonl"
    write_jsonl(path, ({"text": t} for t in texts))
    on_disk = read_texts(path)
    if on_disk != texts:
        raise RuntimeError(f"{path}를 다시 읽은 내용이 다름")
    report = {"path": path, "lines": pool_size, "prefix_same_as": f"train_{PREFIX_SIZE}.jsonl",
              "heldout_same": True, "doc_id_overlap": doc_overlap, "text_overlap": text_overlap,
              "chars": sum(len(t) for t in texts)}
    with open(os.path.join(ROOT, OUT_DIR, "final_pool_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
