"""
DPO 선호 쌍 만들기 (설계: docs/superpowers/specs/2026-11-24-lora-dpo-design.md, docs/09).

SFT 후보 9,749쌍의 질문에 최종 SFT 모델로 greedy 생성(docs/08과 같은 generate_one)하고,
생성한 답 안에서 MIN_COUNT번 이상 나온 같은 답을 '상투 답'으로 정한다. 모델 답이 상투 답이면서
정답과 다른(띄어쓰기·문장부호 무시) 쌍만 골라 좋은 답 = 정답, 나쁜 답 = 모델 답으로 짝짓는다.
평가용 2,001쌍과 07번 200문항은 쓰지 않는다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\make_dpo_data.py
결과: data/dpo/sft_pool_answers.jsonl (생성 답), data/dpo/dpo_pairs.jsonl, data/dpo/report.json
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_chat_eval import normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = "tokenizers/ko_16000"
WEIGHT = "out_final/sft/sft_ko16000_768.pth"
SFT_POOL = "data/chat/sft_pool.jsonl"
OUT_DIR = "data/dpo"
MIN_COUNT = 20


def select_pairs(rows, answers, min_count=MIN_COUNT):
    """rows[i] = {"q","a"}, answers[i] = generate_one 결과. (DPO 쌍 목록, {상투 답: 횟수})."""
    counts = Counter(a["text"] for a in answers)
    stock = {t: c for t, c in counts.most_common() if c >= min_count}
    pairs = [{"q": r["q"], "chosen": r["a"], "rejected": a["text"]}
             for r, a in zip(rows, answers)
             if a["text"] in stock and normalize(a["text"]) != normalize(r["a"])]
    return pairs, stock


def write_jsonl(path, rows):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    import torch
    from blind_chat_eval import generate_one
    from eval_bpc import load_model
    with open(os.path.join(ROOT, SFT_POOL), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(TOKENIZER, WEIGHT, device)
    os.makedirs(os.path.join(ROOT, OUT_DIR), exist_ok=True)
    answers, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        answers.append(generate_one(model, tokenizer, r["q"], device))
        if i % 1000 == 0:
            print(f"생성 {i:,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(f"{OUT_DIR}/sft_pool_answers.jsonl", [dict(q=r["q"], **a) for r, a in zip(rows, answers)])
    pairs, stock = select_pairs(rows, answers)
    write_jsonl(f"{OUT_DIR}/dpo_pairs.jsonl", pairs)
    report = {"weight": WEIGHT, "questions": len(rows), "min_count": MIN_COUNT, "stock_answers": stock,
              "stock_answer_items": sum(stock.values()), "pairs": len(pairs),
              "truncated": sum(not a["stopped"] for a in answers), "distinct_answers": len({a["text"] for a in answers})}
    with open(os.path.join(ROOT, OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"상투 답 {len(stock)}종({report['stock_answer_items']:,}문항), DPO 쌍 {len(pairs):,}개 → {OUT_DIR}/dpo_pairs.jsonl")
    for t, c in stock.items():
        print(f"  {c:>5}  {t}")
    print("완료:", f"{OUT_DIR}/report.json")


if __name__ == "__main__":
    main()
