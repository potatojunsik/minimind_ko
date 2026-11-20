"""
데이터 양 실험(docs/06)에서 이미 저장한 검증 결과를 더 들여다본다. 새 학습·평가 없이 파일만 읽는다.

  0. decide_vocab.py와 같은 방식(문서 단위 쌍 부트스트랩 10,000회, seed 0)으로 Δ와 95% 구간을 다시 계산해
     out_scale/decision.json과 같은지 확인한다. 다르면 멈춘다.
  1. 크기별 검증 토큰당 손실: 토큰 크기가 달라 토큰당 손실(학습 loss도 마찬가지)로는 vocab을 비교할 수 없는 이유
  2. 100만 Δ(6,400 BPB의 약 0.17%)가 어느 정도 크기인지
  3. 결론의 튼튼함: 평가 방식을 바꿔도 100만 Δ의 부호와 구간이 유지되는지
  4. 탐색적 분석: 글 특징별 100만 Δ. 결과를 본 뒤 정한 분석이라 판정 근거가 아니다
     (숫자·길이·한글·라틴 비율은 구간별 표, 처음 탐색에서 함께 본 나머지 4개 특징은 맨 위 − 맨 아래 구간 대비만)
  5. 검증 세트가 학습 데이터(위키 전체 조각의 84%를 무작위로 뽑은 100만 조각)와 얼마나 비슷한지

    Δ   = BPB(6,400) − BPB(16,000)   (양수면 16,000이 앞섬)
    BPB = 손실 합(nats) / ln2 / UTF-8 바이트 합   (조각별 값의 평균이 아니라 합으로 나눈다)
    글자 구성 비율은 공백을 뺀 글자 기준이다 (숫자 = 0-9, 라틴 = A-Z a-z, 한글 = 완성형 음절 가-힣,
    한자 = CJK 통합 한자와 확장 A·호환 한자, 기타 = 그 밖의 글자: 문장부호·가나 등).

입력 (읽기만): data/scale/heldout.jsonl, data/scale/train_1000000.jsonl, data/scale/split_report.json,
              out_scale/decision.json, out_scale/ko<vocab>_n<조각 수>/per_doc.jsonl, logs/06_scale_ko<vocab>_n1000000_eval.log

사용법 (minimind_ko 폴더에서 실행, 1~2분):
    .venv\\Scripts\\python.exe -X utf8 scripts\\analyze_scale_results.py
    (숫자를 JSON으로도 저장: --out scale_analysis.json)
"""
import argparse
import json
import math
import os
import random
import re
import sys
from types import SimpleNamespace

import numpy as np

from decide_vocab import ROOT, ROUNDS, SEED, VOCABS, load_jsonl

SIZES = [50000, 100000, 200000, 400000, 1000000]
BIG, PREV, PREV2 = 1000000, 400000, 200000  # 최종 판정 크기와 그 앞 두 크기
LN2 = math.log(2)
HELDOUT = "data/scale/heldout.jsonl"
TRAIN_BIG = "data/scale/train_1000000.jsonl"
SPLIT_REPORT = "data/scale/split_report.json"
DECISION = "out_scale/decision.json"
PER_DOC = "out_scale/ko{v}_n{n}/per_doc.jsonl"
EVAL_LOG = "logs/06_scale_ko{v}_n{n}_eval.log"
TAIL_95 = 0.025
EXTRA_SEEDS = range(1, 10)  # 공식 seed 0 말고 더 돌려 보는 부트스트랩 seed
LATIN_EDGES = [0.05, 0.15]  # 라틴 비율은 0%인 조각이 1/3이라 사분위 대신 0% / 0~5% / 5~15% / 15% 이상
HANJA_EDGES = [0.01]  # 한자가 있는 조각은 11%뿐이라 0% / 0~1% / 1% 이상
HANGUL_SPLIT = 0.81  # 층별 확인용 한글 비율 경계 (한글 비율의 중앙값 근처)
TOP_DOCS = 10
MATCH_TOL = 1e-9  # decision.json과 비교할 때 허용하는 차이 (더하는 순서가 달라 생기는 부동소수점 오차)

SPACE = re.compile(r"\s")
HANGUL = re.compile("[가-힣]")
DIGIT = re.compile("[0-9]")
LATIN = re.compile("[A-Za-z]")
HANJA = re.compile("[㐀-䶿一-鿿豈-﫿]")


def pct(x):
    return f"{x * 100:.1f}%"


def interval(values, tail=TAIL_95):
    """양쪽 꼬리를 tail씩 뺀 백분위 구간. tail 0.025, 10,000회이면 decide_vocab.ci와 같은 위치(250번째, 9,750번째)."""
    s = np.sort(values)
    k = int(round(tail * len(s)))
    return float(s[k]), float(s[len(s) - k - 1])


def boot_counts(n, rounds=ROUNDS, seed=SEED):
    """decide_vocab.bootstrap_ids와 같은 순서로 문서를 복원추출하고, 회차마다 문서별로 뽑힌 횟수를 센다 (rounds × n)."""
    rng = random.Random(seed)
    return np.stack([np.bincount([rng.randrange(n) for _ in range(n)], minlength=n) for _ in range(rounds)]).astype(float)


def composition(text):
    """(한글 음절, 숫자, 라틴 문자, 한자, 공백 제외 글자) 개수."""
    nonspace = len(text) - len(SPACE.findall(text))
    return len(HANGUL.findall(text)), len(DIGIT.findall(text)), len(LATIN.findall(text)), len(HANJA.findall(text)), nonspace


def load():
    held = load_jsonl(HELDOUT)
    texts = [row["text"] for row in held]
    doc_ids = [row["doc_id"] for row in held]
    order = list(dict.fromkeys(doc_ids))  # decide_vocab.group_by_doc과 같은 '처음 나온 순서'
    position = {d: i for i, d in enumerate(order)}
    h = SimpleNamespace(texts=texts, doc_idx=np.array([position[d] for d in doc_ids]), n_docs=len(order),
                        nchars=np.array([len(t) for t in texts], dtype=float),
                        nbytes=np.array([len(t.encode("utf-8")) for t in texts], dtype=float), tokens={}, nll={})
    for n in SIZES:
        for v in VOCABS:
            rows = load_jsonl(PER_DOC.format(v=v, n=n))
            if len(rows) != len(texts) or any(r["chars"] != c or r["bytes"] != b for r, c, b in zip(rows, h.nchars, h.nbytes)):
                sys.exit(f"{PER_DOC.format(v=v, n=n)}의 조각 순서/글자 수가 {HELDOUT}와 다름")
            tokens = np.array([r["tokens"] for r in rows], dtype=float)
            if v in h.tokens and not np.array_equal(tokens, h.tokens[v]):
                sys.exit(f"vocab {v}의 토큰 수가 크기마다 다름 (같은 토크나이저여야 함)")
            h.tokens[v] = tokens
            h.nll[(n, v)] = np.array([r["nll"] for r in rows])
    comp = np.array([composition(t) for t in texts], dtype=float)
    nonspace = np.maximum(comp[:, 4], 1)
    h.hangul, h.digit, h.latin, h.hanja = (comp[:, k] / nonspace for k in range(4))
    h.other = 1 - comp[:, :4].sum(axis=1) / nonspace
    h.comp_total = comp.sum(axis=0)
    return h


def diff_nats(h, n):
    """조각별 손실 차이 (6,400 − 16,000, nats). 양수면 그 조각에서 16,000이 나음."""
    return h.nll[(n, 6400)] - h.nll[(n, 16000)]


def delta(h, n, mask=None):
    m = np.ones(len(h.texts), dtype=bool) if mask is None else mask
    return float(diff_nats(h, n)[m].sum() / LN2 / h.nbytes[m].sum())


def doc_sums(h, n, mask=None):
    """문서별 (손실 차이 합, 바이트 합). mask가 있으면 그 조각만 더한다."""
    w = np.ones(len(h.texts)) if mask is None else mask.astype(float)
    return (np.bincount(h.doc_idx, weights=diff_nats(h, n) * w, minlength=h.n_docs),
            np.bincount(h.doc_idx, weights=h.nbytes * w, minlength=h.n_docs))


def biggest_doc(h):
    """바이트가 가장 많은 검증 문서의 번호."""
    return int(np.bincount(h.doc_idx, weights=h.nbytes, minlength=h.n_docs).argmax())


def boot_ratio(counts, doc_diff, doc_bytes):
    """부트스트랩 회차마다 뽑힌 문서들의 손실 차이 합 / ln2 / 바이트 합."""
    values = (counts @ doc_diff) / LN2 / (counts @ doc_bytes)
    if not np.all(np.isfinite(values)):
        sys.exit("부트스트랩 회차 중 해당 조각이 하나도 뽑히지 않은 회차가 있음 (구간이 너무 작음)")
    return values


def boot_delta(h, counts, n, mask=None):
    """부트스트랩 회차마다 뽑힌 문서들(그중 mask 조각만)로 다시 계산한 Δ."""
    return boot_ratio(counts, *doc_sums(h, n, mask))


def contrast(h, counts, n, a, b):
    """Δ[a 조각] − Δ[b 조각]의 점추정과, 같은 부트스트랩 표본으로 구한 회차별 값."""
    return delta(h, n, a) - delta(h, n, b), boot_delta(h, counts, n, a) - boot_delta(h, counts, n, b)


def fmt_ci(lo, hi, digits=4):
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}]"


def section(title):
    print()
    print("=" * 100)
    print(title)


# ---------------------------------------------------------------- 0. 판정 재현
def reproduce(h, counts):
    section("0. 판정 재현 (decide_vocab.py 방식: 문서 단위 쌍 부트스트랩 10,000회, seed 0)")
    with open(os.path.join(ROOT, DECISION), encoding="utf-8") as f:
        official = json.load(f)["sizes"]
    out = {}
    for n in SIZES:
        d = delta(h, n)
        lo, hi = interval(boot_delta(h, counts, n))
        o = official[str(n)]
        same = all(abs(a - b) < MATCH_TOL for a, b in [(d, o["delta_bpb"]), (lo, o["delta_bpb_ci"][0]), (hi, o["delta_bpb_ci"][1])])
        print(f"{n:>9,} | Δ {d:+.6f} {fmt_ci(lo, hi, 6)} | decision.json {o['delta_bpb']:+.6f} "
              f"{fmt_ci(*o['delta_bpb_ci'], 6)} | {'일치' if same else '다름'}")
        if not same:
            sys.exit("decision.json과 다름: 입력 파일이나 계산 방식을 확인할 것")
        out[n] = {"delta": d, "ci": [lo, hi]}
    print(f"검증 세트: 조각 {len(h.texts):,}개, 문서 {h.n_docs:,}개, {h.nchars.sum():,.0f}자, {h.nbytes.sum():,.0f}바이트")
    return out


# ---------------------------------------------------------------- 1. 토큰당 손실
def token_loss(h):
    section("1. 검증 토큰당 손실과 BPB (토큰당 손실 = 손실 합 / 텍스트 토큰 수. 손실 합에는 조각마다 eos 1개 몫이 들어 있다)")
    total_bytes = h.nbytes.sum()
    per_token_bytes = {v: total_bytes / h.tokens[v].sum() for v in VOCABS}
    print(f"검증 텍스트 토큰: 6,400 {h.tokens[6400].sum():,.0f}개 / 16,000 {h.tokens[16000].sum():,.0f}개, "
          f"토큰 하나가 담는 바이트: {per_token_bytes[6400]:.3f} / {per_token_bytes[16000]:.3f} "
          f"(16,000이 {per_token_bytes[16000] / per_token_bytes[6400]:.4f}배)")
    print(f"{'조각 수':>9} | {'토큰당 손실 6400':>15} {'16000':>7} {'16000/6400':>10} | {'BPB 6400':>8} {'16000':>7} | {'Δ BPB':>8}")
    out = {"bytes_per_token": per_token_bytes, "sizes": {}}
    for n in SIZES:
        loss = {v: h.nll[(n, v)].sum() / h.tokens[v].sum() for v in VOCABS}
        bpb = {v: h.nll[(n, v)].sum() / LN2 / total_bytes for v in VOCABS}
        out["sizes"][n] = {"loss_per_token": loss, "bpb": bpb}
        print(f"{n:>9,} | {loss[6400]:15.4f} {loss[16000]:7.4f} {loss[16000] / loss[6400]:10.4f} | "
              f"{bpb[6400]:8.4f} {bpb[16000]:7.4f} | {bpb[6400] - bpb[16000]:+8.4f}")
    big = out["sizes"][BIG]
    loss_ratio = big["loss_per_token"][16000] / big["loss_per_token"][6400]
    bytes_ratio = per_token_bytes[16000] / per_token_bytes[6400]
    print(f"100만: 토큰당 손실 비 {loss_ratio:.4f} ÷ 토큰당 바이트 비 {bytes_ratio:.4f} = BPB 비 {loss_ratio / bytes_ratio:.4f} "
          f"(토큰당 손실 = BPB × ln2 × 토큰당 바이트)")
    with_eos = {v: h.nll[(BIG, v)].sum() / (h.tokens[v].sum() + len(h.texts)) for v in VOCABS}
    print(f"(평가 로그의 '토큰당 loss (텍스트+eos)'는 eos도 토큰으로 세서 {with_eos[6400]:.4f} / {with_eos[16000]:.4f})")
    out["ratio_1m"] = {"loss": loss_ratio, "bytes_per_token": bytes_ratio}
    return out


# ---------------------------------------------------------------- 2. 0.17%의 크기
def secant(bpb, a, b):
    """두 크기 a < b의 BPB를 log(조각 수) 눈금에서 직선으로 이은 기울기 (BPB / ln 조각 수)."""
    return (bpb[b] - bpb[a]) / math.log(b / a)


def end_slope(bpb):
    """20만·40만·100만 세 점을 log(조각 수) 눈금에서 지나는 2차 곡선의 100만 지점 기울기.
    데이터가 늘수록 기울기가 완만해지는 것을 반영한, 두 점 직선보다 덜 거친 추정."""
    s_early, s_late = secant(bpb, PREV2, PREV), secant(bpb, PREV, BIG)
    return s_late + (s_late - s_early) / math.log(BIG / PREV2) * math.log(BIG / PREV)


def magnitude(h, losses):
    section("2. 100만 Δ의 크기 감각")
    bpb64 = {n: losses["sizes"][n]["bpb"][6400] for n in SIZES}
    bpb16 = {n: losses["sizes"][n]["bpb"][16000] for n in SIZES}
    d = bpb64[BIG] - bpb16[BIG]
    bits = {v: h.nll[(BIG, v)].sum() / LN2 for v in VOCABS}
    chars = h.nchars.sum()
    gain64 = bpb64[PREV] - bpb64[BIG]
    print(f"Δ {d:+.6f} = 6,400 BPB({bpb64[BIG]:.4f})의 {d / bpb64[BIG]:.2%}")
    print(f"검증 세트 전체 정보량: 6,400 {bits[6400]:,.1f}비트, 16,000 {bits[16000]:,.1f}비트, 차이 {bits[6400] - bits[16000]:,.1f}비트 "
          f"(약 {(bits[6400] - bits[16000]) / 8:,.0f}바이트). 바이트로 바꾸면 {bits[6400] / 8:,.0f} / {bits[16000] / 8:,.0f}바이트")
    print(f"1,000자당: 6,400 {bits[6400] / chars * 1000:,.1f}비트, 16,000 {bits[16000] / chars * 1000:,.1f}비트, "
          f"차이 {(bits[6400] - bits[16000]) / chars * 1000:.2f}비트")
    print(f"6,400의 40만→100만 개선 폭 {gain64:.4f} BPB 대비: {d / gain64:.1%}")
    print(f"10만 Δ {bpb64[100000] - bpb16[100000]:+.4f}의 크기 대비: 1/{abs(bpb64[100000] - bpb16[100000]) / d:.0f}")

    print("(거친 추정) BPB를 Δ만큼 바꾸려면 학습 조각을 몇 % 바꿔야 하나. 기울기 단위는 BPB / ln(조각 수)")
    out = {"delta": d, "relative_to_bpb6400": d / bpb64[BIG], "bits": bits, "bits_per_1000_chars_diff": (bits[6400] - bits[16000]) / chars * 1000,
           "gain6400_400k_to_1m": gain64, "delta_over_gain": d / gain64, "data_change": {}}
    for v, bpb in [(6400, bpb64), (16000, bpb16)]:
        early, late, curve = secant(bpb, PREV2, PREV), secant(bpb, PREV, BIG), end_slope(bpb)
        # 기울기가 음수라 조각 수를 늘리면 BPB가 내려간다. 6,400은 Δ만큼 좋아지는 데 더 볼 양, 16,000은 Δ만큼 나빠지는 덜 볼 양
        change = {name: (math.exp(-d / s) - 1 if v == 6400 else 1 - math.exp(d / s)) for name, s in [("line", late), ("curve", curve)]}
        out["data_change"][v] = {"slope_200k_400k": early, "slope_400k_1m": late, "slope_curve_at_1m": curve, **change}
        word = "더 보면 Δ만큼 좋아짐" if v == 6400 else "덜 보면 Δ만큼 나빠짐"
        print(f"  {v:>6,}: 기울기 20만→40만 {early:+.4f}, 40만→100만 {late:+.4f} (완만해지는 중), 세 점 곡선의 100만 지점 {curve:+.4f}")
        print(f"          학습 조각을 두 점 직선으로 {change['line']:.2%}(100만 기준 약 {change['line'] * BIG:,.0f}조각), "
              f"세 점 곡선으로 {change['curve']:.2%}(약 {change['curve'] * BIG:,.0f}조각) {word}")
    return out


# ---------------------------------------------------------------- 3. 결론의 튼튼함
def read_eval_bpc(v, n):
    """eval_bpc.py 로그의 BPC (텍스트+eos), BPC (텍스트만). 로그에는 소수 넷째 자리까지만 찍혀 있다."""
    out = {}
    with open(os.path.join(ROOT, EVAL_LOG.format(v=v, n=n)), encoding="utf-8") as f:
        for line in f:
            m = re.match(r"BPC \((텍스트\+eos|텍스트만)\)\s*: ([0-9.]+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def robustness(h, counts):
    section("3. 결론의 튼튼함 (100만). 모두 문서 단위 쌍 부트스트랩 10,000회 95% 구간")
    out = {}
    base = boot_delta(h, counts, BIG)
    lo, hi = interval(base)
    out["base"] = {"delta": delta(h, BIG), "ci": [lo, hi], "share_le_0": float(np.mean(base <= 0))}
    print(f"기본 (판정과 같음)            : Δ {delta(h, BIG):+.4f} {fmt_ci(lo, hi)}, 부트스트랩 중 Δ ≤ 0인 회차 {np.mean(base <= 0):.2%}")

    doc_diff, doc_bytes = doc_sums(h, BIG)
    per_doc = doc_diff / LN2 / doc_bytes
    lo, hi = interval(counts @ per_doc / h.n_docs)
    out["equal_doc_weight"] = {"delta": float(per_doc.mean()), "ci": [lo, hi], "median": float(np.median(per_doc)),
                               "doc_win_16000": float(np.mean(per_doc > 0))}
    print(f"문서마다 같은 무게 (문서별 Δ 평균): Δ {per_doc.mean():+.4f} {fmt_ci(lo, hi)}, 문서별 Δ 중앙값 {np.median(per_doc):+.4f}, "
          f"16,000이 나은 문서 {np.mean(per_doc > 0):.1%}")

    biggest = biggest_doc(h)
    keep = np.arange(h.n_docs) != biggest
    d_rest = delta(h, BIG, h.doc_idx != biggest)
    # 남은 문서 수(n−1)로 다시 복원추출한다
    lo, hi = interval(boot_ratio(boot_counts(h.n_docs - 1), doc_diff[keep], doc_bytes[keep]))
    share = doc_bytes[biggest] / doc_bytes.sum()
    out["without_biggest_doc"] = {"chunks": int(np.sum(h.doc_idx == biggest)), "bytes_share": float(share), "delta": d_rest, "ci": [lo, hi]}
    print(f"가장 큰 문서 1개 제외          : Δ {d_rest:+.4f} {fmt_ci(lo, hi)} "
          f"(뺀 문서: 조각 {out['without_biggest_doc']['chunks']}개, 검증 바이트의 {share:.1%})")

    seeds = {SEED: out["base"]["ci"]}
    for s in EXTRA_SEEDS:
        seeds[s] = list(interval(boot_delta(h, boot_counts(h.n_docs, seed=s), BIG)))
    lows, highs = [c[0] for c in seeds.values()], [c[1] for c in seeds.values()]
    out["seeds"] = seeds
    print(f"부트스트랩 seed {min(seeds)}~{max(seeds)} ({len(seeds)}개)   : 하한 {min(lows):+.5f} ~ {max(lows):+.5f}, "
          f"상한 {min(highs):+.5f} ~ {max(highs):+.5f}")

    bits = doc_diff / LN2
    win16, win64 = bits > 0, bits < 0
    print(f"문서별 비트 차이               : 16,000이 나은 문서 {win16.sum()}개의 합 {bits[win16].sum():+,.0f}비트, "
          f"6,400이 나은 문서 {win64.sum()}개의 합 {bits[win64].sum():+,.0f}비트, 둘이 상쇄되고 남은 순합 {bits.sum():+,.0f}비트")
    order = np.argsort(-doc_diff)  # 16,000 쪽으로 기여(비트)가 큰 문서부터
    flip = int(np.argmax(np.cumsum(doc_diff[order]) >= doc_diff.sum())) + 1
    removed, opposite = order[:flip], order[::-1][:flip]  # 반대쪽(6,400 쪽 기여가 큰 문서)도 같은 수만큼
    d_opposite = delta(h, BIG, ~np.isin(h.doc_idx, opposite))
    out["flip"] = {"docs": flip, "doc_share": flip / h.n_docs, "bytes_share": float(doc_bytes[removed].sum() / doc_bytes.sum()),
                   "bits_16000_side": float(bits[win16].sum()), "bits_6400_side": float(bits[win64].sum()),
                   "opposite_bytes_share": float(doc_bytes[opposite].sum() / doc_bytes.sum()), "opposite_delta": d_opposite}
    print(f"부호가 뒤집히려면              : 16,000 쪽 기여가 큰 문서부터 {flip}개(문서의 {flip / h.n_docs:.1%}, "
          f"바이트의 {out['flip']['bytes_share']:.1%})를 빼야 함")
    print(f"반대로 6,400 쪽 기여가 큰 문서 {flip}개(바이트의 {out['flip']['opposite_bytes_share']:.1%})를 빼면: Δ {d_opposite:+.4f}")

    logs = {v: read_eval_bpc(v, BIG) for v in VOCABS}
    ratio = h.nchars.sum() / h.nbytes.sum()
    d_text = logs[6400]["텍스트만"] - logs[16000]["텍스트만"]
    d_all = logs[6400]["텍스트+eos"] - logs[16000]["텍스트+eos"]
    out["text_only"] = {"bpc": {v: logs[v] for v in VOCABS}, "delta_bpc": d_text, "delta_bpb_approx": d_text * ratio,
                        "delta_bpc_with_eos": d_all, "eos_share_approx": 1 - d_text / d_all}
    print(f"텍스트만 (eos 예측 제외, 평가 로그 값): BPC 6,400 {logs[6400]['텍스트만']:.4f} / 16,000 {logs[16000]['텍스트만']:.4f}, "
          f"Δ BPC {d_text:+.4f} (eos 포함 {d_all:+.4f}, eos 몫 약 {1 - d_text / d_all:.0%}), BPB로 바꾸면 약 {d_text * ratio:+.4f} (구간 없음)")
    return out


# ---------------------------------------------------------------- 4. 탐색적: 글 특징별 Δ
def quartile_bins(x, decimals):
    """사분위수(반올림한 값)를 경계로 4구간. 구간 i = 경계[i-1] <= x < 경계[i]."""
    edges = [round(float(q), decimals) for q in np.percentile(x, [25, 50, 75])]
    return np.searchsorted(edges, x, side="right"), edges


def range_names(edges, fmt):
    names = [f"< {fmt(edges[0])}"]
    names += [f"{fmt(a)} ~ {fmt(b)}" for a, b in zip(edges, edges[1:])]
    return names + [f">= {fmt(edges[-1])}"]


def zero_then_bins(x, edges, fmt):
    """0인 조각을 따로 한 구간으로 두고, 나머지를 edges로 나눈다."""
    bins = np.where(x == 0, 0, 1 + np.searchsorted(edges, x, side="right"))
    return bins, ["0%", f"0 초과 ~ {fmt(edges[0])}"] + range_names(edges, fmt)[1:]


def doc_position(h):
    """0 = 조각 1개짜리 문서, 1 = 여러 조각 문서의 첫 조각, 2 = 여러 조각 문서의 이후 조각 (같은 문서 조각은 검증 파일에서 연속)."""
    first = np.r_[True, h.doc_idx[1:] != h.doc_idx[:-1]]
    single = np.bincount(h.doc_idx, minlength=h.n_docs)[h.doc_idx] == 1
    return np.where(single, 0, np.where(first, 1, 2))


def features(h, counts):
    section("4. 탐색적 분석 (결과를 본 뒤 정한 분석이라 판정 근거가 아님). 100만 Δ를 글 특징별 구간으로 나눔")
    digit_bin, digit_edges = quartile_bins(h.digit, 3)
    length_bin, length_edges = quartile_bins(h.nchars, 0)
    hangul_bin, hangul_edges = quartile_bins(h.hangul, 2)
    gain_bin, gain_edges = quartile_bins(h.tokens[6400] / h.tokens[16000], 3)
    other_bin, other_edges = quartile_bins(h.other, 3)
    table = {
        "숫자 비율": (digit_bin, range_names(digit_edges, pct)),
        "길이(글자)": (length_bin, range_names(length_edges, lambda x: f"{x:.0f}자")),
        "한글 비율": (hangul_bin, range_names(hangul_edges, pct)),
        "라틴 비율": zero_then_bins(h.latin, LATIN_EDGES, pct),
    }
    others = {
        "압축 이득(6,400 토큰 수 ÷ 16,000 토큰 수)": (gain_bin, range_names(gain_edges, lambda x: f"{x:.3f}")),
        "한자 비율": zero_then_bins(h.hanja, HANJA_EDGES, pct),
        "기타 글자 비율": (other_bin, range_names(other_edges, pct)),
        "문서 안 위치": (doc_position(h), ["조각 1개짜리 문서", "여러 조각 문서의 첫 조각", "여러 조각 문서의 이후 조각"]),
    }
    total_bits = diff_nats(h, BIG).sum() / LN2
    win16 = h.nll[(BIG, 16000)] < h.nll[(BIG, 6400)]
    out = {"edges": {"digit": digit_edges, "length": length_edges, "hangul": hangul_edges, "latin": LATIN_EDGES}, "features": {}}
    for name, (bins, names) in table.items():
        print()
        print(f"--- {name}")
        print(f"{'구간':<16} {'조각':>5} {'문서':>5} {'바이트 비중':>9} | {'Δ 100만':>8} {'95% 구간':>18} | "
              f"{'16000 우세':>9} | {'Δ 비트 몫':>8} | " + " ".join(f"{'Δ ' + str(n // 10000) + '만':>8}" for n in SIZES[:-1]))
        rows = []
        for k, label in enumerate(names):
            m = bins == k
            lo, hi = interval(boot_delta(h, counts, BIG, m))
            row = {"bin": label, "chunks": int(m.sum()), "docs": len(set(h.doc_idx[m].tolist())),
                   "bytes_share": float(h.nbytes[m].sum() / h.nbytes.sum()), "delta": {n: delta(h, n, m) for n in SIZES},
                   "ci_1m": [lo, hi], "win16000_chunk": float(win16[m].mean()),
                   "share_of_delta_bits": float(diff_nats(h, BIG)[m].sum() / LN2 / total_bits)}
            rows.append(row)
            print(f"{label:<16} {row['chunks']:>5} {row['docs']:>5} {row['bytes_share']:9.1%} | {row['delta'][BIG]:+8.4f} {fmt_ci(lo, hi)} | "
                  f"{row['win16000_chunk']:9.1%} | {row['share_of_delta_bits']:+8.0%} | " + " ".join(f"{row['delta'][n]:+8.4f}" for n in SIZES[:-1]))
        point, values = contrast(h, counts, BIG, bins == len(names) - 1, bins == 0)
        out["features"][name] = {"bins": rows, "contrast_1m": {"point": point, "ci": list(interval(values))}}
        print(f"  대비 Δ[{names[-1]}] − Δ[{names[0]}] = {point:+.4f} {fmt_ci(*interval(values))}")

    print()
    print("--- 처음 탐색에서 함께 본 나머지 특징 (100만, 맨 위 구간 − 맨 아래 구간 대비만)")
    out["other_contrasts"] = {}
    for name, (bins, names) in others.items():
        point, values = contrast(h, counts, BIG, bins == len(names) - 1, bins == 0)
        out["other_contrasts"][name] = {"last": names[-1], "first": names[0], "point": point, "ci": list(interval(values))}
        print(f"  {name}: Δ[{names[-1]}] − Δ[{names[0]}] = {point:+.4f} {fmt_ci(*interval(values))}")

    # 숫자 비율은 처음 탐색에서 본 특징 여러 개 중 결과를 보고 고른 것이라, 특징 수만큼 꼬리를 줄인 넓은 구간(본페로니식)도 보여 준다.
    # 이 보정은 100만의 대비 8개만 센 것이고, 크기별·층별로 더 본 것은 감안하지 않는다
    explored = len(table) + len(others)
    wide_tail = TAIL_95 / explored
    out["contrast_by_size"] = {}
    for name in ["숫자 비율", "라틴 비율"]:
        bins, names = table[name]
        print()
        print(f"--- {name} 대비 Δ[{names[-1]}] − Δ[{names[0]}], 크기별 (같은 검증 문서, 겹치는 학습 세트라 서로 독립된 재현이 아님)")
        by_size = {}
        for n in SIZES:
            point, values = contrast(h, counts, n, bins == len(names) - 1, bins == 0)
            by_size[n] = {"point": point, "ci": list(interval(values)), "ci_wide": list(interval(values, wide_tail))}
            print(f"{n:>9,}: 대비 {point:+.4f} 95% {fmt_ci(*by_size[n]['ci'])}  "
                  f"{1 - 2 * wide_tail:.3%} {fmt_ci(*by_size[n]['ci_wide'])} (100만 특징 {explored}개를 본 것을 감안한 넓은 구간)")
        out["contrast_by_size"][name] = by_size

    high, low = digit_bin == 3, digit_bin == 0
    biggest = biggest_doc(h)
    without = delta(h, BIG, high & (h.doc_idx != biggest)) - delta(h, BIG, low & (h.doc_idx != biggest))
    loo = [delta(h, BIG, high & (h.doc_idx != d)) - delta(h, BIG, low & (h.doc_idx != d)) for d in np.unique(h.doc_idx[high | low])]
    out["digit_contrast_one_doc"] = {"without_biggest": without, "leave_one_doc_out": [min(loo), max(loo)]}
    print(f"--- 숫자 비율 대비 (100만)가 한 문서에 기대는지: 가장 큰 검증 문서를 빼면 {without:+.4f}, "
          f"두 구간의 문서 {len(loo)}개를 하나씩 빼 보면 {min(loo):+.4f} ~ {max(loo):+.4f}")

    print("--- 한글 비율로 층을 나눈 뒤의 숫자 비율 대비 (100만, 숫자 구간 경계는 위와 같음)")
    strata = {}
    for label, s in [(f"한글 < {HANGUL_SPLIT:.0%}", h.hangul < HANGUL_SPLIT), (f"한글 >= {HANGUL_SPLIT:.0%}", h.hangul >= HANGUL_SPLIT)]:
        point, values = contrast(h, counts, BIG, s & high, s & low)
        strata[label] = {"high_chunks": int((s & high).sum()), "high_docs": len(set(h.doc_idx[s & high].tolist())),
                         "low_chunks": int((s & low).sum()), "low_docs": len(set(h.doc_idx[s & low].tolist())),
                         "point": point, "ci": list(interval(values))}
        r = strata[label]
        print(f"  {label:<10}: 숫자 많은 조각 {r['high_chunks']:>4}개(문서 {r['high_docs']:>3}개), "
              f"숫자 적은 조각 {r['low_chunks']:>4}개(문서 {r['low_docs']:>3}개), 대비 {point:+.4f} {fmt_ci(*r['ci'])}")
    out["digit_contrast_by_hangul"] = strata
    means = {k: float(h.hangul[digit_bin == k].mean()) for k in range(4)}
    digit_names = table["숫자 비율"][1]
    print("숫자 비율 구간별 평균 한글 비율: " + ", ".join(f"{digit_names[k]} {means[k]:.0%}" for k in range(4)))
    out["digit_bin_mean_hangul"] = means
    return out


# ---------------------------------------------------------------- 5. 검증 세트 vs 학습 데이터
def representativeness(h, digit_edges):
    section("5. 검증 세트가 학습 데이터와 얼마나 비슷한가")
    with open(os.path.join(ROOT, SPLIT_REPORT), encoding="utf-8") as f:
        report = json.load(f)
    pool = report["total_chunks"] - report["heldout"]["chunks"]
    doc_chunks = np.bincount(h.doc_idx, minlength=h.n_docs)
    doc_bytes = np.bincount(h.doc_idx, weights=h.nbytes, minlength=h.n_docs)
    top = np.sort(doc_bytes)[::-1]

    lengths, digit_ratio, comp_total = [], [], np.zeros(5)
    with open(os.path.join(ROOT, TRAIN_BIG), encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            c = composition(text)
            lengths.append(len(text))
            digit_ratio.append(c[1] / max(c[4], 1))
            comp_total += c
    lengths, digit_ratio = np.array(lengths), np.array(digit_ratio)

    def summary(name, lens, comp, digits):
        q = np.percentile(lens, [25, 50, 75])
        hi_share, lo_share = np.mean(digits >= digit_edges[-1]), np.mean(digits < digit_edges[0])
        print(f"{name:<10}: 조각 {len(lens):>9,}개, 평균 {lens.mean():6.1f}자 (사분위 {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}자), "
              f"공백 제외 글자 중 한글 {comp[0] / comp[4]:.1%} 숫자 {comp[1] / comp[4]:.1%} 라틴 {comp[2] / comp[4]:.1%}, "
              f"숫자 비율 {digit_edges[-1]:.1%} 이상 조각 {hi_share:.1%} / {digit_edges[0]:.1%} 미만 조각 {lo_share:.1%}")
        return {"chunks": len(lens), "mean_chars": float(lens.mean()), "quartiles_chars": q.tolist(),
                "hangul": comp[0] / comp[4], "digit": comp[1] / comp[4], "latin": comp[2] / comp[4],
                "digit_high_share": float(hi_share), "digit_low_share": float(lo_share)}

    out = {"heldout": summary("검증", h.nchars, h.comp_total, h.digit), "train_1m": summary("학습 100만", lengths, comp_total, digit_ratio)}
    out["heldout"].update({"docs": h.n_docs, "chunks_per_doc": len(h.texts) / h.n_docs, "single_chunk_docs": float(np.mean(doc_chunks == 1)),
                           "max_doc_chunks": int(doc_chunks.max()), "top1_bytes_share": float(top[0] / top.sum()),
                           "top10_bytes_share": float(top[:TOP_DOCS].sum() / top.sum())})
    out["wiki"] = {"docs": report["total_docs"], "chunks": report["total_chunks"], "chunks_per_doc": report["total_chunks"] / report["total_docs"],
                   "train_pool_chunks": pool, "train_1m_share_of_pool": BIG / pool}
    print(f"검증 문서 {h.n_docs:,}개: 문서당 조각 {out['heldout']['chunks_per_doc']:.2f}개, 조각 1개짜리 문서 {np.mean(doc_chunks == 1):.1%}, "
          f"가장 큰 문서 조각 {doc_chunks.max()}개(바이트의 {top[0] / top.sum():.1%}), 큰 문서 {TOP_DOCS}개가 바이트의 {top[:TOP_DOCS].sum() / top.sum():.1%}")
    print(f"위키 전체: 문서 {report['total_docs']:,}개, 조각 {report['total_chunks']:,}개, 문서당 조각 {out['wiki']['chunks_per_doc']:.2f}개. "
          f"학습 풀 {pool:,}조각 중 100만 조각 = {BIG / pool:.1%}")
    return out


def main():
    p = argparse.ArgumentParser(description="데이터 양 실험(docs/06) 결과 보충 분석")
    p.add_argument("--out", default="", help="결과 숫자를 저장할 JSON 경로 (선택)")
    args = p.parse_args()

    h = load()
    counts = boot_counts(h.n_docs)
    result = {"reproduce": reproduce(h, counts)}
    result["token_loss"] = token_loss(h)
    result["magnitude"] = magnitude(h, result["token_loss"])
    result["robustness"] = robustness(h, counts)
    result["features"] = features(h, counts)
    result["representativeness"] = representativeness(h, result["features"]["edges"]["digit"])

    if args.out:
        with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
