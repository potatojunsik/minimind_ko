"""
docs/06의 보조 그림 4장을 분석 스크립트가 저장한 JSON으로 그린다. 숫자를 손으로 옮기지 않는다.
  images/scale_ratio.png       16,000 ÷ 6,400 비율 (토큰 수, 학습 step당 시간, 생성 시간, 파라미터, 검증 BPB)
  images/scale_padding.png     100만 학습에서 512칸이 무엇으로 채워졌나 (텍스트 / bos·eos / 패딩)
  images/scale_robustness.png  100만 Δ를 계산 방식별로 (결과를 본 뒤 정한 점검)
  images/scale_digits.png      숫자 비율 구간별 100만 Δ (탐색적 분석)

사용법 (minimind_ko 폴더에서 실행, 앞의 세 명령이 JSON을 만든다):
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_token_stats.py --out out_scale\\token_stats.json
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_costs.py --bench --out out_scale\\costs.json
    .venv\\Scripts\\python.exe -X utf8 scripts\\analyze_scale_results.py --out out_scale\\analysis.json
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_scale_details.py
--bench 없이 만든 costs.json이면 생성 시간 행은 빠진다.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss import GRID, MUTED, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, pick_korean_font  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_1M = "data/scale/train_1000000.jsonl"
VOCABS = ("6400", "16000")
VOCAB_LABEL = {"6400": "vocab 6,400", "16000": "vocab 16,000"}
VOCAB_COLOR = dict(zip(VOCABS, SERIES_COLORS))
MAX_SEQ_LEN = 512
SPECIAL_SLOTS = 2  # 조각마다 bos 1칸 + eos 1칸
PAD_COLOR = "#e1e0d9"
DPI = 150


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def new_figure(height):
    fig, ax = plt.subplots(figsize=(9, height), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def finish(fig, ax, title, note, out):
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=14)
    if note:
        fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06 * (note.count(chr(10)) + 1) if note else 0, 1, 1))
    fig.savefig(os.path.join(ROOT, out), facecolor=SURFACE)
    plt.close(fig)
    print(f"저장: {out}")


def percent_word(ratio, smaller, larger):
    pct = abs(1 - ratio) * 100
    # 1%가 안 되는 차이(검증 BPB 약 0.17%)는 소수 첫째 자리로 반올림하면 문서 값과 달라 보여서 둘째 자리까지 쓴다
    return f"{pct:.{2 if pct < 1 else 1}f}% {smaller if ratio < 1 else larger}"


def plot_ratio(tokens, costs, decision, out):
    full = {v: tokens["sets"][TRAIN_1M][v]["text_tokens_full"] for v in VOCABS}
    params = {v: costs["params"][v]["total"] for v in VOCABS}
    speed = costs["train_speed"]
    bpb = {v: decision["sizes"]["1000000"][f"bpb_{v}"] for v in VOCABS}
    rows = [
        ("같은 글의 토큰 수\n(100만 학습 세트, 자르기 전)", full["16000"] / full["6400"], None, ("적음", "많음")),
        ("학습 step당 시간\n(5만~40만 4쌍 평균, 막대는 범위)", speed["ratio_mean"],
         (speed["ratio_min"], speed["ratio_max"]), ("빠름", "느림")),
    ]
    if "bench" in costs:
        rows.append(("같은 글자 수 생성 시간\n(batch 1, 짝 비교 중앙값)", costs["bench"]["same_chars_time_ratio"], None,
                     ("짧음", "김")))
    rows += [
        ("파라미터 수", params["16000"] / params["6400"], None, ("적음", "많음")),
        ("검증 BPB (100만)", bpb["16000"] / bpb["6400"], None, ("낮음", "높음")),
    ]
    fig, ax = new_figure(0.75 * len(rows) + 1.4)
    ys = list(range(len(rows)))[::-1]
    color = VOCAB_COLOR["16000"]
    for y, (label, ratio, span, words) in zip(ys, rows):
        if span:
            ax.plot(span, (y, y), color=color, linewidth=2, solid_capstyle="round")
        ax.plot(ratio, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(ratio + (0.012 if ratio >= 1 else -0.012), y + 0.28, f"{ratio:.3f} ({percent_word(ratio, *words)})",
                color=TEXT_PRIMARY, fontsize=9.5, ha="left" if ratio >= 1 else "right", va="bottom")
    ax.axvline(1.0, color=TEXT_SECONDARY, linewidth=1)
    ax.text(1.004, len(rows) - 0.35, "1.0 = 같음", color=TEXT_SECONDARY, fontsize=8.5, ha="left", va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=TEXT_PRIMARY, fontsize=9.5)
    ax.set_xlim(0.78, 1.22)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("vocab 16,000 ÷ vocab 6,400", color=TEXT_SECONDARY)
    finish(fig, ax, "vocab 16,000은 6,400과 비교해 얼마나 다른가",
           "판정은 검증 BPB로만 했다. 나머지는 참고 자료(docs/06 \"실제로 얼마나 다른가\").", out)


def plot_padding(tokens, out):
    stats = tokens["sets"][TRAIN_1M]
    fig, ax = new_figure(2.6)
    ax.xaxis.grid(False)
    ys = {"6400": 1, "16000": 0}
    for v in VOCABS:
        s = stats[v]
        text = s["text_tokens_kept"] / s["pieces"]
        pad = MAX_SEQ_LEN * s["pad_pct"] / 100
        # bos·eos 2칸은 막대로 그리면 보이지 않을 만큼 좁아 텍스트 칸에 합쳐 적는다
        parts = [(text + SPECIAL_SLOTS, VOCAB_COLOR[v], f"텍스트 {text:.0f}칸 + bos·eos {SPECIAL_SLOTS}칸"),
                 (pad, PAD_COLOR, f"패딩 {pad:.0f}칸 ({s['pad_pct']:.1f}%)")]
        left = 0
        for width, color, label in parts:
            ax.barh(ys[v], width, left=left, height=0.55, color=color, edgecolor=SURFACE, linewidth=2)
            ink = "white" if color != PAD_COLOR else TEXT_PRIMARY
            ax.text(left + width / 2, ys[v], label, color=ink, fontsize=9.5, ha="center", va="center")
            left += width
    ax.set_yticks(list(ys.values()))
    ax.set_yticklabels([VOCAB_LABEL[v] for v in ys], color=TEXT_PRIMARY)
    ax.set_xlim(0, MAX_SEQ_LEN)
    ax.set_xticks([0, 128, 256, 384, 512])
    ax.set_xlabel("조각 하나가 들어가는 512칸의 평균 구성 (100만 학습 세트)", color=TEXT_SECONDARY)
    finish(fig, ax, "512칸 중 실제 글이 들어간 칸과 패딩",
           "패딩 칸도 계산은 하지만 loss에서는 빠진다. bos는 채점하지 않고 eos는 채점한다.", out)


def forest(ax, rows, color):
    ys = list(range(len(rows)))[::-1]
    for y, (label, delta, ci) in zip(ys, rows):
        if ci:
            ax.plot(ci, (y, y), color=color, linewidth=2, solid_capstyle="round")
            text = f"{delta:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        else:
            text = f"{delta:+.4f}  (구간 없음)"
        ax.plot(delta, y, "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(0.0105, y, text, color=TEXT_PRIMARY, fontsize=9, ha="left", va="center")
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=TEXT_PRIMARY, fontsize=9.5)
    ax.set_xlim(-0.006, 0.019)
    ax.set_xticks([-0.005, 0, 0.005, 0.010])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Δ BPB = BPB(6,400) - BPB(16,000)   (0보다 크면 16,000이 앞섬)", color=TEXT_SECONDARY)


def plot_robustness(analysis, out):
    r = analysis["robustness"]
    rows = [
        ("판정 방식 (손실 합 ÷ 바이트 합)", r["base"]["delta"], r["base"]["ci"]),
        ("문서마다 같은 무게", r["equal_doc_weight"]["delta"], r["equal_doc_weight"]["ci"]),
        (f"가장 큰 문서 빼기 ({r['without_biggest_doc']['chunks']}조각)",
         r["without_biggest_doc"]["delta"], r["without_biggest_doc"]["ci"]),
        ("텍스트만 (eos 빼기, 평가 로그 값)", r["text_only"]["delta_bpb_approx"], None),
    ]
    fig, ax = new_figure(3.2)
    forest(ax, rows, VOCAB_COLOR["16000"])
    finish(fig, ax, "100만 Δ를 계산 방식을 바꿔 다시 계산 (결과를 본 뒤 정한 점검)",
           f"문서를 골라 빼면 크게 움직인다: 16,000 쪽 기여가 큰 문서 {r['flip']['docs']}개(검증 바이트의 "
           f"{r['flip']['bytes_share'] * 100:.0f}%)를 빼면 부호가 뒤집히고, 6,400 쪽 {r['flip']['docs']}개를 빼면 "
           f"{r['flip']['opposite_delta']:+.4f}.\n구간은 문서 단위 부트스트랩 10,000회 95% 구간.", out)


def plot_digits(analysis, out):
    feature = analysis["features"]["features"]["숫자 비율"]
    base = analysis["robustness"]["base"]
    rows = [(f"숫자 비율 {b['bin']}\n(조각 {b['chunks']:,}개, 바이트 {b['bytes_share'] * 100:.0f}%)",
             b["delta"]["1000000"], b["ci_1m"]) for b in feature["bins"]]
    rows.append(("전체", base["delta"], base["ci"]))
    fig, ax = new_figure(3.9)
    forest(ax, rows, VOCAB_COLOR["16000"])
    finish(fig, ax, "숫자 비율 구간별 100만 Δ (탐색적 분석, 판정 근거 아님)",
           "숫자 비율 = 공백을 뺀 글자 중 0~9의 비율. 구간은 조각 수가 같도록 4등분. 문서 단위 부트스트랩 10,000회 95% 구간.", out)


def main():
    p = argparse.ArgumentParser(description="docs/06 보조 그림")
    p.add_argument("--tokens", default="out_scale/token_stats.json")
    p.add_argument("--costs", default="out_scale/costs.json")
    p.add_argument("--analysis", default="out_scale/analysis.json")
    p.add_argument("--decision", default="out_scale/decision.json")
    p.add_argument("--out_dir", default="images")
    args = p.parse_args()

    font = pick_korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False
    tokens, costs, analysis, decision = (load(args.tokens), load(args.costs), load(args.analysis),
                                         load(args.decision))
    plot_ratio(tokens, costs, decision, os.path.join(args.out_dir, "scale_ratio.png"))
    plot_padding(tokens, os.path.join(args.out_dir, "scale_padding.png"))
    plot_robustness(analysis, os.path.join(args.out_dir, "scale_robustness.png"))
    plot_digits(analysis, os.path.join(args.out_dir, "scale_digits.png"))


if __name__ == "__main__":
    main()
