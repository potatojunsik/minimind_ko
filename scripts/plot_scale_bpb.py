"""
데이터 양 실험(docs/06)의 조각 수별 BPB를 vocab 두 개에 대해 한 그래프로 그린다.
입력은 decide_vocab.py가 만든 out_scale/decision.json (같은 검증 세트 data/scale/heldout.jsonl에서 잰 값만 들어 있다).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_scale_bpb.py out_scale/decision.json images\\scale_bpb.png ko
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_scale_bpb.py out_scale/decision.json images\\scale_bpb_en.png en
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402

from plot_loss import GRID, MUTED, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, pick_korean_font  # noqa: E402

VOCABS = [6400, 16000]
# 두 선이 큰 규모에서 거의 겹치므로 끝 라벨 대신 범례를 쓰고, 색만으로 구분하지 않도록 마커 모양도 다르게 한다
MARKERS = ["o", "s"]
LABELS = {
    "ko": {
        "title": "학습 데이터 양에 따른 검증 BPB (같은 검증 세트)",
        "xlabel": "학습 조각 수 (로그 눈금, 각 규모에서 처음부터 학습)",
        "ylabel": "BPB (UTF-8 바이트당 비트, ↓ 낮을수록 좋음)",
        "delta_ylabel": "Δ = BPB(6,400) - BPB(16,000)\n(0보다 위면 16,000 유리)",
        "series": "vocab {:,}",
        "size": lambda n: f"{n // 10000}만",
    },
    "en": {
        "title": "Held-out BPB by training data size (same held-out set)",
        "xlabel": "Training chunks (log scale, each size trained from scratch)",
        "ylabel": "BPB (bits per UTF-8 byte, ↓ lower is better)",
        "delta_ylabel": "Δ = BPB(6,400) - BPB(16,000)\n(above 0 favors 16,000)",
        "series": "vocab {:,}",
        "size": lambda n: f"{n // 1000000}M" if n >= 1000000 else f"{n // 1000}k",
    },
}


def main(decision_path, out_path, lang):
    text = LABELS[lang]
    with open(decision_path, "r", encoding="utf-8") as f:
        sizes_info = json.load(f)["sizes"]
    sizes = sorted(int(n) for n in sizes_info)
    if not sizes:
        raise ValueError(f"조각 수별 결과가 없음: {decision_path}")

    font = pick_korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    # 위: 두 vocab의 BPB 추세. 아래: 판정 근거인 Δ와 95% 구간(위 축척으로는 100만의 작은 역전이 보이지 않아서 따로 그린다)
    fig, (ax, ax_d) = plt.subplots(2, 1, figsize=(8, 6.4), dpi=150, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.15]})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, ax_d):
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(GRID)

    for color, marker, vocab in zip(SERIES_COLORS, MARKERS, VOCABS):
        values = [sizes_info[str(n)][f"bpb_{vocab}"] for n in sizes]
        ax.plot(sizes, values, color=color, linewidth=2, marker=marker, markersize=5,
                label=text["series"].format(vocab))

    deltas = [sizes_info[str(n)]["delta_bpb"] for n in sizes]
    cis = [sizes_info[str(n)]["delta_bpb_ci"] for n in sizes]
    errors = [[d - lo for d, (lo, _) in zip(deltas, cis)], [hi - d for d, (_, hi) in zip(deltas, cis)]]
    ax_d.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax_d.errorbar(sizes, deltas, yerr=errors, fmt="o", color=TEXT_PRIMARY, ecolor=TEXT_SECONDARY,
                  elinewidth=2, capsize=0, markersize=5)
    last, (lo, hi) = deltas[-1], cis[-1]
    ax_d.annotate(f"{last:+.4f} [{lo:+.4f}, {hi:+.4f}]", (sizes[-1], hi), xytext=(0, 8),
                  textcoords="offset points", ha="right", fontsize=8.5, color=TEXT_PRIMARY)

    ax.set_xscale("log")
    ax_d.set_xticks(sizes)
    ax_d.set_xticklabels([text["size"](n) for n in sizes])
    ax_d.minorticks_off()
    ax.set_title(text["title"], loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax_d.set_xlabel(text["xlabel"], color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(text["ylabel"], color=TEXT_SECONDARY, fontsize=9)
    ax_d.set_ylabel(text["delta_ylabel"], color=TEXT_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in LABELS:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
