"""
대화 SFT 비교의 블라인드 생성 평가 (설계: docs/superpowers/specs/2026-11-20-chat-sft-design.md 4-2절, 5절).

세 단계로 쓴다.
  select   : SFT 전에 평가 묶음 200개를 시드 20261121로 뽑고 묶음마다 질문 하나를 고정한다 → data/chat/blind_items.json
  generate : 두 SFT 모델로 같은 질문에 greedy 생성 → 평가자용 비교표 data/chat/blind_sheet.csv
             (모델 이름 없음, 문항마다 좌우를 시드 20261122로 섞음)와 정답표 data/chat/blind_key.json
             정답표와 data/chat/blind_outputs.json은 블라인드 평가가 끝날 때까지 열지 않는다.
  score    : 평가자가 '선택' 열에 왼쪽/오른쪽/동률을 적은 비교표와 정답표, SFT 후 BPB 비교 결과로 설계 5절을 판정한다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\blind_chat_eval.py select
    .venv\\Scripts\\python.exe -X utf8 scripts\\blind_chat_eval.py generate ^
        --model tokenizers/ko_6400 out_sft/sft_ko6400_768.pth --model tokenizers/ko_16000 out_sft/sft_ko16000_768.pth
    .venv\\Scripts\\python.exe -X utf8 scripts\\blind_chat_eval.py score --bpb data/chat/chat_compare_chat_sft.json
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decide_vocab import ci  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS_SEED = 20261121
SHEET_SEED = 20261122
N_ITEMS = 200
MAX_NEW_TOKENS = 96
TRUNC_MARK = "[상한에서 잘림]"
SMALL, LARGE = "6400", "16000"
# 설계 5절: 6,400 승수가 16,000 승수보다 20건 이상 많아야 한다(200문항의 순선호 10%p)
MIN_WIN_MARGIN = 20
BOOT_ROUNDS = 10000
BOOT_SEED = 0
CHOICES = {"왼쪽": "left", "오른쪽": "right", "동률": "tie", "l": "left", "r": "right", "t": "tie"}
SHEET_COLUMNS = ["번호", "질문", "왼쪽", "오른쪽", "선택"]


def select_items(pairs, n, seed):
    """묶음 n개를 뽑고 묶음마다 질문 하나를 고른다. 같은 묶음의 쌍은 같은 질문이나 답을 공유하므로 하나만 쓴다."""
    members = {}
    for p in pairs:
        members.setdefault(p["group"], []).append(p)
    rng = random.Random(seed)
    groups = rng.sample(sorted(members), n)
    items = []
    for i, g in enumerate(groups, 1):
        p = rng.choice(members[g])
        items.append({"id": i, "group": g, "q": p["q"], "a": p["a"]})
    return items


def make_sheet(items, outputs, seed):
    """평가자용 비교표 행과 정답표. outputs[vocab][id] = {"text", "stopped"}."""
    rng = random.Random(seed)
    rows, key = [], {}
    for it in items:
        left, right = (SMALL, LARGE) if rng.random() < 0.5 else (LARGE, SMALL)

        def shown(name):
            o = outputs[name][it["id"]]
            return o["text"] if o["stopped"] else f"{o['text']} {TRUNC_MARK}"

        rows.append({"id": it["id"], "question": it["q"], "left": shown(left), "right": shown(right), "choice": ""})
        key[str(it["id"])] = {"left": left, "right": right}
    return rows, key


def outcomes(choices, key):
    """문항별 점수 목록(번호 순): 6,400 승 +1 / 동률 0 / 16,000 승 −1. 빠진 문항이나 알 수 없는 선택이 있으면 멈춘다."""
    if set(choices) != set(key):
        missing = sorted(set(key) - set(choices), key=int)
        extra = sorted(set(choices) - set(key))
        raise ValueError(f"선택이 빠진 문항: {missing[:10]}, 정답표에 없는 번호: {extra[:10]}")
    scores = []
    for qid in sorted(key, key=int):
        side = CHOICES.get(choices[qid].strip().lower())
        if side is None:
            raise ValueError(f"{qid}번의 선택 '{choices[qid]}'을 알 수 없다 (왼쪽/오른쪽/동률)")
        scores.append(0 if side == "tie" else (1 if key[qid][side] == SMALL else -1))
    return scores


def decide(bpb_ci_hi, scores, rounds=BOOT_ROUNDS, seed=BOOT_SEED):
    """설계 5절 판정. bpb_ci_hi: SFT 후 답 본문 BPB Δ(6,400 − 16,000)의 95% 구간 상한."""
    n = len(scores)
    rng = random.Random(seed)
    means = [sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds)]
    lo, hi = ci(means)
    wins_small, wins_large = scores.count(1), scores.count(-1)
    cond_bpb = bpb_ci_hi < 0
    cond_blind = wins_small - wins_large >= MIN_WIN_MARGIN and lo > 0
    return {"wins_6400": wins_small, "wins_16000": wins_large, "ties": scores.count(0), "items": n,
            "mean_score": sum(scores) / n, "ci": [lo, hi], "bpb_ci_hi": bpb_ci_hi,
            "cond_bpb": cond_bpb, "cond_blind": cond_blind, "revisit": cond_bpb and cond_blind}


def chat_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def generate_one(model, tokenizer, q, device):
    """질문 하나에 greedy로 답한다. {"text", "stopped"(스스로 <|im_end|>를 냄), "new_tokens"}."""
    import torch
    ids = tokenizer(chat_prompt(q), add_special_tokens=False).input_ids
    x = torch.tensor([ids], device=device)
    # 필터를 모두 끈 greedy(매 단계 확률이 가장 높은 토큰): 원본 generate의 do_sample=False 경로
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=1.0, top_k=0,
                           top_p=1.0, repetition_penalty=1.0, eos_token_id=tokenizer.eos_token_id)
    new = y[0, len(ids):].tolist()
    stopped = tokenizer.eos_token_id in new
    body = new[:new.index(tokenizer.eos_token_id)] if stopped else new
    return {"text": tokenizer.decode(body, skip_special_tokens=True).strip(), "stopped": stopped,
            "new_tokens": len(new)}


def generate_answers(tokenizer_dir, weight, items, device):
    from eval_bpc import load_model
    tokenizer, model = load_model(tokenizer_dir, weight, device)
    return str(len(tokenizer)), {it["id"]: generate_one(model, tokenizer, it["q"], device) for it in items}


def check_bpb_source(bpb):
    """판정에 쓸 BPB 비교 결과가 'SFT 후, 대화 형식, 6,400 vs 16,000'인지 확인한다. 아니면 멈춘다."""
    if bpb.get("format") != "chat":
        raise SystemExit(f"BPB 비교 결과의 형식이 chat이 아니다({bpb.get('format')}). eval_chat_bpb.py --format chat 결과를 준다")
    if bpb.get("vocabs") != [int(SMALL), int(LARGE)]:
        raise SystemExit(f"vocab 6,400과 16,000의 비교가 아니다({bpb.get('vocabs')})")
    weights = bpb.get("weights", {})
    if not weights or not all("sft" in os.path.basename(w) for w in weights.values()):
        raise SystemExit(f"SFT 가중치로 만든 비교 결과가 아니다(가중치: {weights})")


def read_choices(path):
    """비교표 CSV의 번호 → 선택. 같은 번호가 두 번 있으면 멈춘다."""
    choices = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["번호"].strip()
            if qid in choices:
                raise ValueError(f"비교표에 {qid}번이 두 번 있다")
            choices[qid] = row["선택"]
    return choices


def score_from_files(sheet_path, key_path, bpb_path):
    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)
    with open(bpb_path, encoding="utf-8") as f:
        bpb = json.load(f)
    check_bpb_source(bpb)
    # 조건 1은 답 본문(텍스트만) BPB의 Δ 구간 상한으로 판정한다(설계 4-1, 5절)
    d = decide(bpb["delta_text_only"]["ci"][1], outcomes(read_choices(sheet_path), key))
    d["bpb_text_only_delta"] = bpb["delta_text_only"]
    d["bpb_with_eos_delta"] = bpb["delta"]
    d["bpb_weights"] = bpb["weights"]
    return d


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cmd_select(args):
    path = os.path.join(ROOT, args.items)
    if os.path.exists(path) and not args.force:
        raise SystemExit(f"이미 있다: {path} (결과를 본 뒤 다시 뽑지 않도록 덮어쓰지 않는다)")
    with open(os.path.join(ROOT, args.data), encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    items = select_items(pairs, N_ITEMS, ITEMS_SEED)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    print(f"저장: {path} ({len(items)}문항, 묶음 {len({it['group'] for it in items})}개) sha256 {sha256_file(path)}")


def cmd_generate(args):
    if len(args.model) != 2:
        raise SystemExit("--model을 두 번 준다 (vocab 6,400 / 16,000)")
    with open(os.path.join(ROOT, args.items), encoding="utf-8") as f:
        items = json.load(f)
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)  # 생성(수 분)을 마친 뒤에 저장이 실패하지 않게 먼저 만든다
    outputs = {}
    for tokenizer_dir, weight in args.model:
        vocab, outs = generate_answers(tokenizer_dir, weight, items, args.device)
        outputs[vocab] = outs
    if set(outputs) != {SMALL, LARGE}:
        raise SystemExit("vocab 6,400과 16,000 모델을 하나씩 준다")
    rows, key = make_sheet(items, outputs, SHEET_SEED)
    with open(os.path.join(out_dir, "blind_outputs.json"), "w", encoding="utf-8") as f:
        json.dump({v: {str(k): o for k, o in outs.items()} for v, outs in outputs.items()}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "blind_key.json"), "w", encoding="utf-8") as f:
        json.dump(key, f, indent=1)
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않게
    with open(os.path.join(out_dir, "blind_sheet.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        for r in rows:
            w.writerow([r["id"], r["question"], r["left"], r["right"], r["choice"]])
    print(f"비교표: {os.path.join(out_dir, 'blind_sheet.csv')} ({len(rows)}문항). 정답표와 출력 파일은 평가가 끝날 때까지 열지 않는다.")


def cmd_score(args):
    d = score_from_files(os.path.join(ROOT, args.sheet), os.path.join(ROOT, args.key), os.path.join(ROOT, args.bpb))
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    dt, de = d["bpb_text_only_delta"], d["bpb_with_eos_delta"]
    print(f"블라인드 {d['items']}문항: 6,400 승 {d['wins_6400']} / 16,000 승 {d['wins_16000']} / 동률 {d['ties']}")
    print(f"문항 점수 평균 {d['mean_score']:+.3f}, 95% 구간 [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    print(f"SFT 후 답 본문 BPB Δ {dt['delta']:+.4f} [{dt['ci'][0]:+.4f}, {dt['ci'][1]:+.4f}], "
          f"EOS 포함 Δ {de['delta']:+.4f} [{de['ci'][0]:+.4f}, {de['ci'][1]:+.4f}]")
    print(f"조건 1 (답 본문 BPB Δ 구간 상한 < 0): {'충족' if d['cond_bpb'] else '불충족'}")
    print(f"조건 2 (승수 차이 ≥ {MIN_WIN_MARGIN}, 점수 구간 하한 > 0): {'충족' if d['cond_blind'] else '불충족'}")
    if d["revisit"]:
        print("판정: 두 조건 모두 충족 → 바로 교체하지 않고, SFT 시드를 바꿔 같은 비교를 반복할지 사용자에게 묻는다.")
    else:
        only_eos = (not d["cond_bpb"]) and de["ci"][1] < 0
        print("판정: 16,000 유지." + (" (EOS 포함 BPB에서만 6,400이 앞섬)" if only_eos else ""))


def main():
    p = argparse.ArgumentParser(description="대화 SFT 비교의 블라인드 생성 평가")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--data", default="data/chat/eval_pairs.jsonl")
    s.add_argument("--items", default="data/chat/blind_items.json")
    s.add_argument("--force", action="store_true", help="이미 있는 문항 파일을 덮어씀(결과를 보기 전에만)")
    g = sub.add_parser("generate")
    g.add_argument("--model", nargs=2, action="append", metavar=("TOKENIZER_DIR", "WEIGHT"), default=[])
    g.add_argument("--items", default="data/chat/blind_items.json")
    g.add_argument("--out_dir", default="data/chat")
    g.add_argument("--device", default="cuda:0")
    c = sub.add_parser("score")
    c.add_argument("--sheet", default="data/chat/blind_sheet.csv")
    c.add_argument("--key", default="data/chat/blind_key.json")
    c.add_argument("--bpb", required=True, help="SFT 후 대화 형식 비교 결과 (eval_chat_bpb.py --format chat --tag chat_sft)")
    c.add_argument("--out", default="data/chat/blind_result.json")
    args = p.parse_args()
    {"select": cmd_select, "generate": cmd_generate, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
