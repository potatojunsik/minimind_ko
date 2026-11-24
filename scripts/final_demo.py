"""
README 대화 데모와 생성 자동 집계 (docs/08). 최종 SFT 모델이 실제로 어떤 답을 내는지 보여 주기 위한 예시이며 판정이 아니다.

두 단계로 쓴다.
  select   : 데모 질문 10개와 생성 설정을 docs/08_demo_questions.json에 고정한다. 답을 보기 전에 실행하고 커밋한다.
             - 평가용 대화 쌍(학습에 쓰지 않음)에서 시드 20261124로 묶음 5개를 뽑아 질문 하나씩(민감 단어가 든 질문은 건너뜀)
             - 직접 쓴 질문 5개: 일상 1, 위키 지식 2, 학습 범위 밖 2. SFT 후보·평가 쌍과 띄어쓰기·문장부호를 무시하고 같으면 멈춘다.
  generate : 질문 파일이 커밋된 상태인지 확인한 뒤, 최종 SFT 모델로 greedy 생성한다.
             - 데모 10문항의 답 → docs/08_demo_results.json (고르거나 고치지 않고 전부, 잘린 답은 표시)
             - 07번 블라인드 비교의 200문항(data/chat/blind_items.json)을 같은 설정으로 생성해 자동 집계하고
               07번 16,000 SFT 답(data/chat/blind_outputs.json)과 같은 기준으로 나란히 남긴다.

생성 설정은 07번 블라인드 비교와 같다(blind_chat_eval.generate_one): 필터 없는 greedy, 새 토큰 상한 96, <|im_end|>에서 멈춤, fp32.
repetition_penalty는 쓰지 않는다. 원본 구현은 질문까지 포함한 모든 입력 토큰에 벌점을 줘서, 질문 부분의 <|im_end|>(eos) 때문에
답을 끝내는 것 자체가 억제되고 반복 루프도 가려지기 때문이다(minimind/model/model_minimind.py generate).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\final_demo.py select
    .venv\\Scripts\\python.exe -X utf8 scripts\\final_demo.py generate
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blind_chat_eval import MAX_NEW_TOKENS, TRUNC_MARK, chat_prompt, generate_one  # noqa: E402
from chat_ko import answer_text  # noqa: E402
from prepare_chat_eval import normalize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = "tokenizers/ko_16000"
WEIGHT = "out_final/sft/sft_ko16000_768.pth"
QUESTIONS = "docs/08_demo_questions.json"
RESULTS = "docs/08_demo_results.json"
EVAL_PAIRS = "data/chat/eval_pairs.jsonl"
SFT_POOL = "data/chat/sft_pool.jsonl"
BLIND_ITEMS = "data/chat/blind_items.json"
BLIND_OUTPUTS_07 = "data/chat/blind_outputs.json"
FINAL_BLIND_OUTPUTS = "data/chat/final_blind_outputs.json"
DEMO_SEED = 20261124
N_SAMPLED = 5
# 민감한 주제(죽음·자해·의료)를 데모에서 빼기 위한 단어. 뽑기 전에 정한 규칙이며, 질문만 보고 건너뛴다.
SENSITIVE = ["죽", "자살", "자해", "병원", "수술", "약 먹", "우울"]
WRITTEN = [
    {"category": "일상 (직접 씀)", "q": "오늘 비가 와서 그런지 기분이 좀 가라앉아",
     "expect": "학습 데이터와 같은 말투라 짧은 공감 답을 예상"},
    {"category": "위키 지식", "q": "대한민국의 수도는 어디야?",
     "expect": "SFT 데이터에 지식 문답이 없어 틀리거나 엉뚱한 답을 예상"},
    {"category": "위키 지식", "q": "세종대왕은 어떤 일을 했어?",
     "expect": "SFT 데이터에 지식 문답이 없어 틀리거나 엉뚱한 답을 예상"},
    {"category": "범위 밖 (지시)", "q": "다음 문장을 영어로 번역해 줘: 나는 학생이다",
     "expect": "번역 지시를 배운 적이 없어 실패를 예상"},
    {"category": "범위 밖 (계산)", "q": "123 더하기 456은 얼마야?",
     "expect": "계산을 배운 적이 없어 실패를 예상"},
]
STAT_PHRASES = ["그런 사람 만나지 마세요.", "그런 사람 만나세요."]
REPEAT_MIN = 3


def load_jsonl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sample_eval_questions(pairs, n, seed, banned=SENSITIVE):
    """평가 쌍에서 묶음을 시드 순서대로 돌며 질문 하나씩 고른다. 민감 단어가 든 질문은 건너뛴다."""
    members = {}
    for p in pairs:
        members.setdefault(p["group"], []).append(p)
    rng = random.Random(seed)
    order = sorted(members)
    rng.shuffle(order)
    picked, skipped = [], []
    for g in order:
        p = rng.choice(members[g])
        if any(word in p["q"] for word in banned):
            skipped.append(p["q"])
            continue
        picked.append({"category": "평가용 대화 (학습에 안 씀)", "q": p["q"], "reference": p["a"], "group": g,
                       "expect": "학습 데이터와 같은 분포라 짧은 공감·일상 답을 예상"})
        if len(picked) == n:
            break
    return picked, skipped


def check_written_unseen(written, pool, pairs):
    """직접 쓴 질문이 SFT 후보·평가 쌍의 질문과 정규화 기준으로 같으면 멈춘다."""
    seen = {normalize(r["q"]) for r in pool} | {normalize(r["q"]) for r in pairs}
    clash = [w["q"] for w in written if normalize(w["q"]) in seen]
    if clash:
        raise SystemExit(f"학습·평가 데이터에 같은 질문이 있음: {clash}")


def generation_stats(outputs):
    """outputs[id] = {"text", "stopped"}. 07번 문서(docs/07 4-2절)와 같은 기준의 자동 집계."""
    texts = [o["text"] for o in outputs.values()]
    counts = Counter(texts)
    return {"items": len(texts),
            "truncated": sum(not o["stopped"] for o in outputs.values()),
            "distinct_answers": len(counts),
            "items_with_answer_repeated_3plus": sum(c for c in counts.values() if c >= REPEAT_MIN),
            "phrases": {p: counts.get(p, 0) for p in STAT_PHRASES},
            "top5": counts.most_common(5)}


def settings():
    return {"weight": WEIGHT, "tokenizer": TOKENIZER, "prompt": chat_prompt("{질문}"),
            "decoding": "greedy (do_sample=False, temperature 1.0, top_k 0, top_p 1.0, repetition_penalty 1.0)",
            "max_new_tokens": MAX_NEW_TOKENS, "stop": "<|im_end|>", "dtype": "fp32",
            "truncated_mark": TRUNC_MARK, "same_as": "07번 블라인드 비교 blind_chat_eval.generate_one"}


def cmd_select(_args):
    pairs, pool = load_jsonl(EVAL_PAIRS), load_jsonl(SFT_POOL)
    sampled, skipped = sample_eval_questions(pairs, N_SAMPLED, DEMO_SEED)
    check_written_unseen(WRITTEN, pool, pairs)
    items = [dict(item, id=i) for i, item in enumerate(sampled + WRITTEN, 1)]
    out = {"note": "답을 생성하기 전에 고정한 질문과 설정. 생성 결과는 고르거나 고치지 않고 전부 싣는다.",
           "seed": DEMO_SEED, "sensitive_words": SENSITIVE, "skipped_sensitive": skipped,
           "settings": settings(), "weight_sha256": sha256_file(WEIGHT), "items": items}
    with open(os.path.join(ROOT, QUESTIONS), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for it in items:
        print(f"{it['id']:>2}. [{it['category']}] {it['q']}")
    print(f"건너뛴 민감 질문 {len(skipped)}개. 저장: {QUESTIONS}")


def committed(path):
    """파일이 git에 커밋돼 있고 바뀐 것이 없는지."""
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", path], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return tracked and not dirty


def cmd_generate(_args):
    import torch
    from eval_bpc import load_model
    if not committed(QUESTIONS):
        raise SystemExit(f"{QUESTIONS}를 먼저 커밋해야 한다(답을 보기 전에 질문을 고정했다는 기록)")
    with open(os.path.join(ROOT, QUESTIONS), encoding="utf-8") as f:
        fixed = json.load(f)
    if fixed["weight_sha256"] != sha256_file(WEIGHT):
        raise SystemExit("가중치 sha256이 질문을 고정할 때와 다르다")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(TOKENIZER, WEIGHT, device)

    demo = []
    for it in fixed["items"]:
        o = generate_one(model, tokenizer, it["q"], device)
        demo.append(dict(it, answer=answer_text(o), **o))
        print(f"나> {it['q']}\n모델> {demo[-1]['answer']}\n")

    with open(os.path.join(ROOT, BLIND_ITEMS), encoding="utf-8") as f:
        items = json.load(f)
    final_out = {str(it["id"]): generate_one(model, tokenizer, it["q"], device) for it in items}
    with open(os.path.join(ROOT, FINAL_BLIND_OUTPUTS), "w", encoding="utf-8") as f:
        json.dump(final_out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, BLIND_OUTPUTS_07), encoding="utf-8") as f:
        old_out = json.load(f)["16000"]

    result = {"settings": fixed["settings"], "weight_sha256": fixed["weight_sha256"],
              "environment": {"torch": torch.__version__,
                              "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu"},
              "demo": demo,
              "blind200": {"final_sft": generation_stats(final_out), "sft_07_ko16000": generation_stats(old_out)}}
    with open(os.path.join(ROOT, RESULTS), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for name, s in result["blind200"].items():
        print(f"{name}: 잘림 {s['truncated']} | 서로 다른 답 {s['distinct_answers']} | "
              f"3번 이상 반복된 답의 문항 {s['items_with_answer_repeated_3plus']} | {s['phrases']}")
    print(f"저장: {RESULTS}, {FINAL_BLIND_OUTPUTS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["select", "generate"])
    args = p.parse_args()
    {"select": cmd_select, "generate": cmd_generate}[args.step](args)


if __name__ == "__main__":
    main()
