"""
데이터 양 실험(docs/06)의 결과를 모아 SFT 토크나이저를 판정한다 (설계 문서 7절 규칙, 결과를 보기 전에 확정).

    BPB = 손실 합(nats) / ln2 / UTF-8 바이트 합      (판정용)
    BPC = 같은 손실 합 / ln2 / 글자 수 합             (docs/05와 이어 보기용)
    Δ   = BPB(6,400) − BPB(16,000)                    (양수면 16,000이 앞섬. docs/05의 "16,000 − 6,400"과 부호가 반대)

신뢰구간은 위키 원문 문서 단위로 복원추출한다. 한 번 뽑은 문서 목록을 모든 모델에 똑같이 적용한다(쌍 부트스트랩).
판정 (40만 조각 기준):
  1. Δ(40만)의 95% 구간 하한 > 0                              → 16,000 선택
  2. Δ(40만) < 0 이고 Δ(40만) − Δ(20만)의 95% 구간 하한 > 0    → 100만 조각 실험 여부를 사용자에게 묻는다
  3. 그 밖                                                     → 6,400 선택
100만 최종 선택 (판정 2 뒤 추가, 2026-11-19 09:00 사용자 확정, 100만 결과 확인 전):
  Δ(100만)의 95% 구간 하한 > 0 → 16,000, 하한 ≤ 0 → 6,400 (반올림 전 수치). Δ(100만) − Δ(40만)은 경향 설명용.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\decide_vocab.py --status out_scale/status.json --heldout data/scale/heldout.jsonl
"""
import argparse
import json
import math
import os
import random
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUNDS = 10000
SEED = 0
VOCABS = [6400, 16000]
DECISION_SIZE = 400000
PREVIOUS_SIZE = 200000
EXTRA_SIZE = 1000000  # 판정 2 뒤 사용자 승인으로 추가한 크기. 판정은 40만 기준 그대로, 100만은 같은 방식으로 함께 보여 준다
UNIT_INDEX = {"bytes": 1, "chars": 2}
MESSAGES = {
    1: "판정 1: 16,000 선택 (Δ(40만)의 95% 구간 하한 > 0)",
    2: "판정 2: 100만 조각 실험 여부를 사용자에게 먼저 묻는다 (Δ(40만) < 0, 격차 증가 구간 하한 > 0)",
    3: "판정 3: 6,400 선택",
}


def group_by_doc(rows, doc_ids):
    """조각별 결과 행을 원문 문서별 (nll 합, bytes 합, chars 합)으로 묶는다. 순서는 처음 나온 순서."""
    if len(rows) != len(doc_ids):
        raise ValueError(f"평가 행 수({len(rows)})와 문서 id 수({len(doc_ids)})가 다름")
    groups = OrderedDict()
    for row, doc in zip(rows, doc_ids):
        nll, nbytes, nchars = groups.get(doc, (0.0, 0, 0))
        groups[doc] = (nll + row["nll"], nbytes + row["bytes"], nchars + row["chars"])
    return groups


def rate(groups, ids, unit):
    """뽑힌 문서들의 손실 합 / ln2 / 단위 합. 문서별 비율의 평균이 아니다."""
    k = UNIT_INDEX[unit]
    nll = sum(groups[d][0] for d in ids)
    total = sum(groups[d][k] for d in ids)
    return nll / math.log(2) / total


def bootstrap_ids(doc_ids, rounds, seed):
    rng = random.Random(seed)
    n = len(doc_ids)
    return [[doc_ids[rng.randrange(n)] for _ in range(n)] for _ in range(rounds)]


def ci(values):
    """95% 백분위 구간 (paired_compare.py와 같은 위치 계산)."""
    s = sorted(values)
    n = len(s)
    return s[int(0.025 * n)], s[int(0.975 * n) - 1]


def growth_ci(boot, sizes_out, big, small):
    """같은 부트스트랩 표본에서 Δ(big) − Δ(small)의 점추정과 95% 구간."""
    point = sizes_out[str(big)]["delta_bpb"] - sizes_out[str(small)]["delta_bpb"]
    lo, hi = ci([a - b for a, b in zip(boot[big], boot[small])])
    return {"point": point, "ci": [lo, hi]}


def format_delta(label, d):
    return f"{label} {d['delta_bpb']:+.4f} [{d['delta_bpb_ci'][0]:+.4f}, {d['delta_bpb_ci'][1]:+.4f}]"


def decide_extra(lo, hi):
    """100만 최종 선택 (2026-11-19 09:00 사용자 확정, 결과 확인 전). 반올림 전 하한으로 판정. (vocab, 문구) 반환."""
    if lo > 0:
        return 16000, "최종 선택: 16,000 (Δ(100만)의 95% 구간 하한 > 0)"
    if hi >= 0:
        return 6400, "최종 선택: 6,400 — 16,000의 우위가 확인되지 않아 사전 규칙에 따라 6,400을 선택했다 (구간에 0 포함)"
    return 6400, "최종 선택: 6,400 (Δ(100만)의 95% 구간 하한 ≤ 0, 구간 전체가 0보다 작음)"


def decide(delta400, delta400_lo, growth_lo):
    if delta400_lo > 0:
        return 1
    if delta400 < 0 and growth_lo > 0:
        return 2
    return 3


def win_rate(a_vals, b_vals):
    """a가 b보다 손실이 낮은 비율 (같은 텍스트라 손실 비교 = BPB 비교)."""
    return sum(1 for a, b in zip(a_vals, b_vals) if a < b) / len(a_vals)


def load_jsonl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--status", default="out_scale/status.json")
    p.add_argument("--heldout", default="data/scale/heldout.jsonl")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    args = p.parse_args()

    with open(os.path.join(ROOT, args.status), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    doc_ids = [row["doc_id"] for row in load_jsonl(args.heldout)]

    sizes = sorted({int(k.split("_n")[1]) for k, r in runs.items() if r.get("state") == "done"})
    sizes = [n for n in sizes if all(runs.get(f"ko{v}_n{n}", {}).get("state") == "done" for v in VOCABS)]
    rows, groups = {}, {}
    for n in sizes:
        for v in VOCABS:
            r = load_jsonl(runs[f"ko{v}_n{n}"]["per_doc"])
            ids = doc_ids[:len(r)]  # 시험 실행(--eval_limit)이면 앞부분만 평가됨
            rows[(n, v)], groups[(n, v)] = r, group_by_doc(r, ids)

    out = {"rounds": args.rounds, "seed": SEED, "sizes": {}, "growth": None, "growth_extra": None, "decision": None, "final_vocab": None}
    if not sizes:
        print("완료된 크기가 없음")
    else:
        unique_ids = list(groups[(sizes[0], VOCABS[0])].keys())
        samples = bootstrap_ids(unique_ids, args.rounds, SEED)
        boot = {n: [] for n in sizes}
        for ids in samples:
            for n in sizes:
                boot[n].append(rate(groups[(n, 6400)], ids, "bytes") - rate(groups[(n, 16000)], ids, "bytes"))

        print(f"{'조각 수':>8} | {'BPB 6400':>8} {'BPB 16000':>9} | {'Δ BPB':>8} {'95% 구간':>19} | "
              f"{'BPC 6400':>8} {'BPC 16000':>9} | 6400 우세(조각/문서)")
        for n in sizes:
            g64, g160 = groups[(n, 6400)], groups[(n, 16000)]
            bpb64, bpb160 = rate(g64, unique_ids, "bytes"), rate(g160, unique_ids, "bytes")
            bpc64, bpc160 = rate(g64, unique_ids, "chars"), rate(g160, unique_ids, "chars")
            lo, hi = ci(boot[n])
            chunk_win = win_rate([r["nll"] for r in rows[(n, 6400)]], [r["nll"] for r in rows[(n, 16000)]])
            doc_win = win_rate([g64[d][0] for d in unique_ids], [g160[d][0] for d in unique_ids])
            out["sizes"][str(n)] = {"bpb_6400": bpb64, "bpb_16000": bpb160, "bpc_6400": bpc64, "bpc_16000": bpc160,
                                    "delta_bpb": bpb64 - bpb160, "delta_bpb_ci": [lo, hi],
                                    "delta_bpc": bpc64 - bpc160, "win_6400_chunk": chunk_win, "win_6400_doc": doc_win,
                                    "chunks": len(rows[(n, 6400)]), "docs": len(unique_ids)}
            print(f"{n:>8,} | {bpb64:8.4f} {bpb160:9.4f} | {bpb64 - bpb160:+8.4f} [{lo:+.4f}, {hi:+.4f}] | "
                  f"{bpc64:8.4f} {bpc160:9.4f} | {chunk_win:.1%} / {doc_win:.1%}")
        print("95% 신뢰구간은 검증 문서 표본에 따른 변동을 나타낸다.")

        if DECISION_SIZE in sizes and PREVIOUS_SIZE in sizes:
            out["growth"] = g = growth_ci(boot, out["sizes"], DECISION_SIZE, PREVIOUS_SIZE)
            d400 = out["sizes"][str(DECISION_SIZE)]
            out["decision"] = decide(d400["delta_bpb"], d400["delta_bpb_ci"][0], g["ci"][0])
            print(format_delta("Δ(20만)", out["sizes"][str(PREVIOUS_SIZE)]))
            print(format_delta("Δ(40만)", d400))
            print(f"Δ(40만) − Δ(20만) {g['point']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}]")
            print(MESSAGES[out["decision"]])
            if EXTRA_SIZE in sizes:
                out["growth_extra"] = g = growth_ci(boot, out["sizes"], EXTRA_SIZE, DECISION_SIZE)
                print(format_delta("Δ(100만)", out["sizes"][str(EXTRA_SIZE)]))
                print(f"Δ(100만) − Δ(40만) {g['point']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] (경향 설명용)")
                out["final_vocab"], message = decide_extra(*out["sizes"][str(EXTRA_SIZE)]["delta_bpb_ci"])
                print(message)
        else:
            print("판정: 미완료 (20만·40만 결과가 모두 있어야 판정)")

    out_path = os.path.join(os.path.dirname(os.path.join(ROOT, args.status)), "decision.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
