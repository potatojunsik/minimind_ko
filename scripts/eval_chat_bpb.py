"""
사전학습 모델(또는 SFT 모델)이 한국어 대화체 답변을 얼마나 잘 예측하는지 BPB로 재고, 두 모델을 비교한다.

평가 규칙 (eval_bpc.py와 같은 원칙, 두 토크나이저에 똑같이 적용):
  - 입력 형식은 두 가지다. plain(기본, 아래 설명)과 chat(--format chat, SFT와 같은 대화 형식, build_chat_ids).
  - 입력: [bos] + 질문 토큰 + 줄바꿈 토큰 + 답 토큰 + [eos]. 질문·줄바꿈·답을 따로 토큰화해 이어 붙인다
    (한 번에 토큰화하면 경계의 글자가 한 토큰으로 합쳐져 답 토큰만 골라낼 수 없다).
  - 채점: 답 토큰 전부 + eos 1개. 질문은 문맥으로만 쓰고 채점하지 않는다. eos 포함 값과 텍스트만(eos 제외) 값을 함께 낸다.
    계획에서는 eos 포함을 기본으로 했지만, 결과를 본 뒤 텍스트만 값을 대표로 바꿨다. 대화 답은 짧아서(대부분 50자 미만)
    답 끝 eos가 사전학습(50자 미만 조각은 버림)에서 거의 보지 못한 상황이고, 그 몫이 차이의 대부분을 차지했기 때문이다.
  - BPB = 답 손실 합(nats) / ln 2 / 답의 UTF-8 바이트 합. 쌍별 비율의 평균이 아니라 합의 비율이다.
  - fp32, 512토큰 창(eval_bpc.WINDOW). 대화 쌍은 짧아서 넘지 않아야 정상이고, 넘으면 멈춘다.
  - 두 모델 비교: Δ = BPB(작은 vocab) − BPB(큰 vocab), 양수면 큰 vocab이 앞섬.
    95% 구간은 prepare_chat_eval.py가 만든 묶음(같은 질문이나 같은 답을 공유하는 쌍) 단위 쌍 부트스트랩
    (10,000회, seed 0, decide_vocab.py와 같은 함수). plain 결과는 판정에 쓰지 않는 참고값이고(docs/06), chat 형식의 SFT 후 결과는 docs/07 조건 1의 판정에 쓴다.

사용법 (minimind_ko 폴더에서 실행, 먼저 prepare_chat_eval.py로 data/chat/eval_pairs.jsonl을 만든다):
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_chat_bpb.py ^
        --model tokenizers/ko_6400 out_scale/ko6400_n1000000/pretrain_ko6400_768.pth ^
        --model tokenizers/ko_16000 out_scale/ko16000_n1000000/pretrain_ko16000_768.pth
    (저장된 결과만 다시 비교: --compare data/chat/chat_eval_ko6400.json data/chat/chat_eval_ko16000.json)
    대화 형식(docs/07)은 --format chat과 --tag를 함께 준다. 예: SFT 후 모델
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_chat_bpb.py --format chat --tag chat_sft ^
        --model tokenizers/ko_6400 out_sft/sft_ko6400_768.pth --model tokenizers/ko_16000 out_sft/sft_ko16000_768.pth
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from decide_vocab import bootstrap_ids, ci, group_by_doc, rate  # noqa: E402
from eval_bpc import ROOT, WINDOW, load_model  # noqa: E402
from prepare_chat_eval import eval_sha256  # noqa: E402

SEP = "\n"
BOOT_ROUNDS = 10000
BOOT_SEED = 0
REPLACEMENT_CHAR = "\ufffd"
WIKI_DECISION = "out_scale/decision.json"


def build_ids(tokenizer, q, a, sep=SEP):
    """([bos] + 질문 + 구분자 + 답 + [eos], 답 첫 토큰의 위치)."""
    q_ids = tokenizer(q, add_special_tokens=False).input_ids
    sep_ids = tokenizer(sep, add_special_tokens=False).input_ids
    a_ids = tokenizer(a, add_special_tokens=False).input_ids
    ids = [tokenizer.bos_token_id] + q_ids + sep_ids + a_ids + [tokenizer.eos_token_id]
    return ids, 1 + len(q_ids) + len(sep_ids)


def build_chat_ids(tokenizer, q, a):
    """SFT와 같은 대화 형식(train_sft_ko.chat_text): user 질문 → assistant 답 → <|im_end|>.
    형식 앞부분과 답을 따로 토큰화해 이어 붙인다. 형식 끝의 줄바꿈 뒤라서 한 번에 토큰화한 결과와 같다(테스트로 확인).
    학습 때 <|im_end|> 뒤에 붙는 줄바꿈은 답의 일부가 아니라 평가하지 않는다."""
    prefix = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    p_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    a_ids = tokenizer(a, add_special_tokens=False).input_ids
    return p_ids + a_ids + [tokenizer.eos_token_id], len(p_ids)


BUILDERS = {"plain": build_ids, "chat": build_chat_ids}


@torch.no_grad()
def score_answer(model, ids, answer_start, device):
    """답 토큰과 eos의 NLL(nats) 목록. 위치 t의 토큰은 위치 t-1의 출력으로 예측하므로 nll[answer_start-1:]이 답 몫이다."""
    if len(ids) > WINDOW:
        raise ValueError(f"쌍이 {len(ids)}토큰으로 창({WINDOW})보다 길다")
    x = torch.tensor([ids], device=device)
    logits = model(x).logits[0, :-1].float()
    nll = F.cross_entropy(logits, x[0, 1:], reduction="none").tolist()
    return nll[answer_start - 1:]


def byte_fragment_share(tokenizer, token_ids, cache):
    """토큰 하나만 decode했을 때 U+FFFD가 나오는(글자 하나를 다 담지 못한 바이트 조각) 토큰의 비율."""
    for t in token_ids:
        if t not in cache:
            cache[t] = REPLACEMENT_CHAR in tokenizer.decode([t])
    return sum(cache[t] for t in token_ids) / len(token_ids)


def pair_row(pair, nll):
    """쌍 하나의 결과. nll은 score_answer의 반환값(답 토큰들 + 마지막 eos)."""
    return {"group": pair["group"], "label": pair["label"], "chars": len(pair["a"]),
            "bytes": len(pair["a"].encode("utf-8")), "tokens": len(nll) - 1,
            "nll": sum(nll), "nll_eos": nll[-1]}


def evaluate(tokenizer_dir, weight, pairs, device, fmt="plain"):
    tokenizer, model = load_model(tokenizer_dir, weight, device)
    builder = BUILDERS[fmt]
    rows, q_tokens, a_tokens, max_len = [], [], [], 0
    t0 = time.time()
    for p in pairs:
        ids, start = builder(tokenizer, p["q"], p["a"])
        nll = score_answer(model, ids, start, device)
        max_len = max(max_len, len(ids))
        q_tokens.extend(tokenizer(p["q"], add_special_tokens=False).input_ids)
        a_tokens.extend(ids[start:-1])
        rows.append(pair_row(p, nll))
    cache = {}
    stats = {
        "q_chars_per_token": sum(len(p["q"]) for p in pairs) / len(q_tokens),
        "a_chars_per_token": sum(len(p["a"]) for p in pairs) / len(a_tokens),
        "q_byte_fragment_share": byte_fragment_share(tokenizer, q_tokens, cache),
        "a_byte_fragment_share": byte_fragment_share(tokenizer, a_tokens, cache),
        "max_ids_len": max_len,
    }
    print(f"  {tokenizer_dir}: {len(pairs)}쌍, {time.time() - t0:.0f}s, 가장 긴 입력 {max_len}토큰")
    return {"tokenizer": tokenizer_dir, "vocab": len(tokenizer), "weight": weight, "format": fmt,
            "sha256": eval_sha256(pairs), "pairs": rows, "tokenizer_stats": stats}


def check_same_eval(a, b):
    """두 결과가 같은 평가 쌍을 같은 순서·같은 입력 형식으로 봤는지 확인한다. 다르면 비교하지 않는다."""
    if a.get("format", "plain") != b.get("format", "plain"):
        raise SystemExit(f"입력 형식이 다르다: {a.get('format', 'plain')} vs {b.get('format', 'plain')}")
    if a["sha256"] != b["sha256"]:
        raise SystemExit(f"평가 텍스트 sha256이 다르다: {a['sha256']} vs {b['sha256']}")
    if [p["group"] for p in a["pairs"]] != [p["group"] for p in b["pairs"]]:
        raise SystemExit("평가 쌍의 묶음 순서가 다르다")


def paired_delta(rows_a, rows_b, groups, rounds=BOOT_ROUNDS, seed=BOOT_SEED):
    """Δ = BPB(a) − BPB(b)와 묶음 단위 쌍 부트스트랩 95% 구간. 두 모델에 같은 묶음 표본을 쓴다."""
    ga, gb = group_by_doc(rows_a, groups), group_by_doc(rows_b, groups)
    ids = list(ga)
    point = rate(ga, ids, "bytes") - rate(gb, ids, "bytes")
    values = [rate(ga, s, "bytes") - rate(gb, s, "bytes") for s in bootstrap_ids(ids, rounds, seed)]
    lo, hi = ci(values)
    return {"delta": point, "ci": [lo, hi], "share_le_0": sum(v <= 0 for v in values) / len(values),
            "units": len(ids)}


def bpb(rows):
    return sum(r["nll"] for r in rows) / math.log(2) / sum(r["bytes"] for r in rows)


def text_only(rows):
    return [dict(r, nll=r["nll"] - r["nll_eos"]) for r in rows]


def large_wins(rows_small, rows_large):
    """쌍 하나씩 비교해 큰 vocab의 답변 BPB(손실 ÷ 바이트)가 더 낮은 쌍의 비율."""
    return sum(y["nll"] / y["bytes"] < x["nll"] / x["bytes"] for x, y in zip(rows_small, rows_large)) / len(rows_small)


def compare(res_small, res_large, rounds, seed):
    check_same_eval(res_small, res_large)
    if res_small["vocab"] == res_large["vocab"]:
        raise SystemExit(f"두 결과의 vocab이 같다({res_small['vocab']}). 서로 다른 vocab의 모델 두 개를 비교한다")
    a, b = res_small["pairs"], res_large["pairs"]
    ta, tb = text_only(a), text_only(b)
    groups = [p["group"] for p in a]
    vs, vl = res_small["vocab"], res_large["vocab"]
    out = {"vocabs": [vs, vl], "pairs": len(a), "groups": len(set(groups)), "sha256": res_small["sha256"],
           "bpb": {str(vs): bpb(a), str(vl): bpb(b)},
           "bpb_text_only": {str(vs): bpb(ta), str(vl): bpb(tb)},
           "delta": paired_delta(a, b, groups, rounds, seed),
           "delta_text_only": paired_delta(ta, tb, groups, rounds, seed),
           "large_wins_pair": large_wins(a, b),
           "large_wins_pair_text_only": large_wins(ta, tb),
           "eos_nll_mean": {str(vs): sum(r["nll_eos"] for r in a) / len(a), str(vl): sum(r["nll_eos"] for r in b) / len(b)},
           "answer_chars_mean": sum(r["chars"] for r in a) / len(a),
           "answer_bytes_mean": sum(r["bytes"] for r in a) / len(a),
           "answers_under_50_chars": sum(r["chars"] < 50 for r in a) / len(a),
           "tokens_per_pair": {str(vs): sum(r["tokens"] for r in a) / len(a), str(vl): sum(r["tokens"] for r in b) / len(b)},
           "tokenizer_stats": {str(vs): res_small["tokenizer_stats"], str(vl): res_large["tokenizer_stats"]},
           "by_label_text_only": {}}
    for label in sorted({p["label"] for p in a}):
        idx = [i for i, p in enumerate(a) if p["label"] == label]
        out["by_label_text_only"][label] = dict(pairs=len(idx), **paired_delta(
            [ta[i] for i in idx], [tb[i] for i in idx], [groups[i] for i in idx], rounds, seed))
    out["format"] = res_small.get("format", "plain")
    out["weights"] = {str(vs): res_small.get("weight", ""), str(vl): res_large.get("weight", "")}
    wiki = os.path.join(ROOT, WIKI_DECISION)
    # 위키 BPB는 사전학습 100만 모델의 값이라, 같은 모델을 평문 형식으로 잰 비교(docs/06)에만 참고로 붙인다
    if out["format"] == "plain" and os.path.exists(wiki):
        with open(wiki, encoding="utf-8") as f:
            size = json.load(f).get("sizes", {}).get("1000000", {})
        if size.get(f"bpb_{vs}") is not None and size.get(f"bpb_{vl}") is not None:
            out["wiki_bpb_1m"] = {str(vs): size[f"bpb_{vs}"], str(vl): size[f"bpb_{vl}"]}
    return out


def report(c):
    vs, vl = (str(v) for v in c["vocabs"])
    d, dt = c["delta"], c["delta_text_only"]
    print(f"\n평가 쌍 {c['pairs']:,}개, 묶음 {c['groups']:,}개, sha256 {c['sha256']}")
    print(f"| 항목 | vocab {int(vs):,} | vocab {int(vl):,} |")
    print("|---|---|---|")
    print(f"| 답변 BPB (텍스트만, 대표값) | {c['bpb_text_only'][vs]:.4f} | {c['bpb_text_only'][vl]:.4f} |")
    print(f"| 답변 BPB (eos 포함) | {c['bpb'][vs]:.4f} | {c['bpb'][vl]:.4f} |")
    print(f"| 답 끝 eos 하나의 평균 손실 (nats) | {c['eos_nll_mean'][vs]:.2f} | {c['eos_nll_mean'][vl]:.2f} |")
    if "wiki_bpb_1m" in c:
        print(f"| (참고) 위키 검증 BPB, 100만 | {c['wiki_bpb_1m'][vs]:.4f} | {c['wiki_bpb_1m'][vl]:.4f} |")
    print(f"| 답 하나의 평균 토큰 수 | {c['tokens_per_pair'][vs]:.2f} | {c['tokens_per_pair'][vl]:.2f} |")
    for key, name in [("q_chars_per_token", "질문 글자/토큰"), ("a_chars_per_token", "답 글자/토큰"),
                      ("q_byte_fragment_share", "질문 바이트 조각 토큰 비율"), ("a_byte_fragment_share", "답 바이트 조각 토큰 비율")]:
        fmt = "{:.2%}" if "share" in key else "{:.3f}"
        print(f"| {name} | {fmt.format(c['tokenizer_stats'][vs][key])} | {fmt.format(c['tokenizer_stats'][vl][key])} |")
    print(f"\nΔ = BPB({int(vs):,}) − BPB({int(vl):,}), 양수면 {int(vl):,}이 앞섬 (묶음 단위 부트스트랩 95% 구간)")
    print(f"  텍스트만(대표): {dt['delta']:+.4f} [{dt['ci'][0]:+.4f}, {dt['ci'][1]:+.4f}] (재표본 중 Δ≤0 {dt['share_le_0']:.1%})")
    print(f"  eos 포함:       {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] (재표본 중 Δ≤0 {d['share_le_0']:.1%})")
    print(f"  쌍 단위로 {int(vl):,}의 답변 BPB가 더 낮은 비율: 텍스트만 {c['large_wins_pair_text_only']:.1%}, "
          f"eos 포함 {c['large_wins_pair']:.1%}")
    print(f"  답 평균 {c['answer_chars_mean']:.1f}자 ({c['answer_bytes_mean']:.1f}바이트), "
          f"50자 미만인 답 {c['answers_under_50_chars']:.1%}")
    print("\n라벨별 Δ, 텍스트만 (탐색적, 0 일상 / 1 이별 / 2 사랑):")
    for label, v in c["by_label_text_only"].items():
        print(f"  {label}: {v['pairs']:,}쌍 {v['delta']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")


def main():
    p = argparse.ArgumentParser(description="대화체 답변 BPB 평가와 두 모델 비교")
    p.add_argument("--model", nargs=2, action="append", metavar=("TOKENIZER_DIR", "WEIGHT"), default=[])
    p.add_argument("--compare", nargs=2, metavar=("RESULT_A", "RESULT_B"), help="저장된 결과 두 개만 비교")
    p.add_argument("--data", default="data/chat/eval_pairs.jsonl")
    p.add_argument("--out_dir", default="data/chat")
    p.add_argument("--rounds", type=int, default=BOOT_ROUNDS)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--format", choices=sorted(BUILDERS), default="plain",
                   help="plain: [bos] 질문 줄바꿈 답 [eos] / chat: SFT와 같은 대화 형식")
    p.add_argument("--tag", default="", help="결과 파일 이름에 붙일 이름 (예: chat_pre, chat_sft)")
    args = p.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.compare and args.model:
        raise SystemExit("--model과 --compare는 함께 쓰지 않는다")
    # 대화 형식 결과가 tag 없이 저장되면 평문 결과(docs/06) 파일 이름과 겹쳐 덮어쓴다
    no_tag = "대화 형식 결과는 --tag로 이름을 붙인다(평문 결과 파일 chat_eval_ko*.json, chat_compare.json을 덮어쓰지 않게)"
    if args.model and args.format != "plain" and not args.tag:
        raise SystemExit(no_tag)
    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    if args.compare:
        results = []
        for path in args.compare:
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                results.append(json.load(f))
        if results[0].get("format", "plain") != "plain" and not args.tag:
            raise SystemExit(no_tag)
    else:
        if len(args.model) != 2:
            raise SystemExit("--model을 두 번(토크나이저 폴더, 가중치) 주거나 --compare를 쓴다")
        with open(os.path.join(ROOT, args.data), encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        results = []
        for tokenizer_dir, weight in args.model:
            res = evaluate(tokenizer_dir, weight, pairs, args.device, args.format)
            path = os.path.join(ROOT, args.out_dir, f"chat_eval{suffix}_ko{res['vocab']}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)
            print(f"  저장: {path}")
            results.append(res)
    results.sort(key=lambda r: r["vocab"])
    c = compare(results[0], results[1], args.rounds, args.seed)
    c["tag"] = args.tag
    with open(os.path.join(ROOT, args.out_dir, f"chat_compare{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=1)
    report(c)


if __name__ == "__main__":
    main()
