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
        "series": "vocab {:,}",
        "size": lambda n: f"{n // 10000}만",
    },
    "en": {
        "title": "Held-out BPB by training data size (same held-out set)",
        "xlabel": "Training chunks (log scale, each size trained from scratch)",
        "ylabel": "BPB (bits per UTF-8 byte, ↓ lower is better)",
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

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for color, marker, vocab in zip(SERIES_COLORS, MARKERS, VOCABS):
        values = [sizes_info[str(n)][f"bpb_{vocab}"] for n in sizes]
        ax.plot(sizes, values, color=color, linewidth=2, marker=marker, markersize=5,
                label=text["series"].format(vocab))

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([text["size"](n) for n in sizes])
    ax.minorticks_off()
    ax.set_title(text["title"], loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.set_xlabel(text["xlabel"], color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(text["ylabel"], color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[3] not in LABELS:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
