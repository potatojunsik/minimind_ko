"""
학습 로그(train_pretrain*.py 출력)에서 step별 loss를 읽어 그래프 이미지(PNG)로 저장한다.
로그 한 줄 형식: Epoch:[1/1](50/6250), loss: 8.4956, ...

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_loss.py <출력.png> "<제목>" <이름=로그경로> [<이름=로그경로>]   (로그는 최대 2개)
예:
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_loss.py images\\loss_pretrain_small_20k.png "사전학습 체험: 중국어 2만 줄, 원본 토크나이저" "원본 토크나이저=logs\\01_pretrain_small_20k.log"
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

LOG_PATTERN = re.compile(r"\((\d+)/(\d+)\), loss: ([\d.]+)")
# 파랑·주황: 색각 이상(적록/청황)이 있어도 서로 구분되고, 밝은 배경과 대비 3:1 이상인 조합
SERIES_COLORS = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# 로그 간격마다 찍힌 batch loss는 출렁임이 커서 추세선을 따로 그린다. 창 반폭 = 전체 점 수 / 이 값
SMOOTH_WINDOW_DIVISOR = 30
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]


def pick_korean_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            return name
    return None


def read_log(path):
    steps, losses = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(3)))
    return steps, losses


def smooth(values):
    """중앙 이동평균. 지수이동평균은 기록 점이 적으면 추세선이 실제 값보다 뒤처져 보여서 쓰지 않는다."""
    half = max(len(values) // SMOOTH_WINDOW_DIVISOR, 1)
    out = []
    for i in range(len(values)):
        window = values[max(i - half, 0): i + half + 1]
        out.append(sum(window) / len(window))
    return out


def main(out_path, title, series):
    if len(series) > len(SERIES_COLORS):
        raise ValueError(f"한 그래프에 최대 {len(SERIES_COLORS)}개 로그까지 그릴 수 있음 (입력 {len(series)}개)")
    font = pick_korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (name, path) in enumerate(series):
        steps, losses = read_log(path)
        if not steps:
            raise ValueError(f"loss 줄을 찾지 못함: {path}")
        color = SERIES_COLORS[i]
        ax.plot(steps, losses, color=color, linewidth=1, alpha=0.25)
        trend = smooth(losses)
        ax.plot(steps, trend, color=color, linewidth=2, label=name)
        # 선 끝에 이름과 마지막 추세값을 직접 표시 (색만으로 구분하지 않도록)
        ax.annotate(f"{name}  {trend[-1]:.2f}", xy=(steps[-1], trend[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)

    ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.set_xlabel("step", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("loss (토큰당)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if len(series) >= 2:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper right")
    ax.margins(x=0.02)
    fig.text(0.01, 0.01, "옅은 선: 로그 간격마다 기록된 batch loss / 진한 선: 이동평균 추세", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 0.88, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    pairs = [arg.split("=", 1) for arg in sys.argv[3:]]
    main(sys.argv[1], sys.argv[2], pairs)
