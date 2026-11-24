"""
README "모델 구조" 절의 그림을 그린다. 원본 MiniMind 그림을 이 저장소의 최종 모델(vocab 16,000) 기준으로 다시 그린 것이다.
그림 속 숫자(텐서 크기, 파라미터 수, 예시 문장의 토큰)는 손으로 적지 않고 원본 모델 코드와 토크나이저에서 읽는다.
가중치 파일은 필요 없다 (구조만 만들어서 센다).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_architecture.py images\\architecture.png ko
    .venv\\Scripts\\python.exe -X utf8 scripts\\plot_architecture.py images\\architecture_en.png en
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

from plot_loss import GRID as GRID_LINE, MUTED, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, pick_korean_font  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKENIZER_DIR = os.path.join(ROOT, "tokenizers", "ko_16000")
ORIGINAL_TOKENIZER_DIR = os.path.join(ROOT, "minimind", "model")
HIDDEN_SIZE = 768
NUM_LAYERS = 8
LORA_RANK = 16  # scripts/train_lora_ko.py 기본값 (docs/09)
PRETRAIN_LEN = 512
SFT_LEN = 128
EXAMPLE = "오늘 날씨가 정말 좋네요."

# 구성 요소 종류별 색 (밝은 바탕, 진한 테두리)
COLORS = {
    "io": ("#fde2c8", "#c8742c"),
    "tokenizer": ("#fff1b8", "#b8962a"),
    "linear": ("#d6e6f7", "#3f78b5"),
    "norm": ("#e4e3df", "#6f6e69"),
    "attn": ("#dcefd9", "#4b8a43"),
    "ffn": ("#f1e3f5", "#8a55a0"),
    "softmax": ("#f8d7c4", "#c25f2d"),
    "head": ("#4a4a48", "#2b2b2a"),
    "block": ("#e2d6f0", "#7a5aa6"),
}
PANEL_BG = "#f3f2ee"
LORA_COLOR = "#e0621b"

LABELS = {
    "ko": {
        "title": "minimind_ko 최종 모델 구조 (vocab {vocab:,}, {total} 파라미터)",
        "panel_a": "(a) 전체 흐름",
        "panel_b": "(b) Transformer 블록 (× {layers})",
        "panel_c": "(c) GQA 어텐션 (블록마다)",
        "panel_d": "(d) SwiGLU FFN (블록마다)",
        "input": "입력: “{text}”",
        "tokenizer": "한국어 BPE 토크나이저 (ko_16000)\n한국어 위키백과로 학습, 어휘 {vocab:,}개",
        "tokens": "{n}토큰, _는 앞 공백 (원본 MiniMind 토크나이저는 {orig}토큰)",
        "embed": "입력 임베딩  {vocab:,} × {d}\n{p}",
        "blocks": "Transformer 블록 × {layers}\n블록당 {per}, 합계 {sum}",
        "final_norm": "RMSNorm",
        "lm_head": "출력층  {d} → {vocab:,}\n임베딩과 같은 행렬 (추가 파라미터 0)",
        "softmax": "Softmax → 다음 토큰 확률 ({vocab:,}개 중)",
        "next": "다음 토큰",
        "tied": "같은 행렬",
        "shape": "[B, T, {d}]  T ≤ {pre} (SFT {sft})",
        "attn": "GQA 어텐션\n{p}",
        "ffn": "SwiGLU FFN\n{p}",
        "residual": "잔차 연결",
        "keep_shape": "블록 입출력 모두 [B, T, {d}]",
        "c_in": "RMSNorm 출력 [B, T, {d}]",
        "q_heads": "Q  {n}헤드 × {hd}",
        "kv_heads": "{name}  {n}헤드 × {hd}",
        "qk_norm": "RMSNorm({hd})",
        "rope": "RoPE θ={theta}",
        "attention": "softmax(QK$^\\mathrm{{T}}$ / √{hd} + 인과 마스크) · V",
        "concat": "헤드 {n}개 이어 붙이기 → [B, T, {d}]",
        "share": "Q 헤드 {g}개가 K·V 헤드 1개를 같이 씀",
        "lora": "LoRA rank {r} (docs/09)",
        "d_in": "RMSNorm 출력 [B, T, {d}]",
        "params_title": "파라미터 수 (vocab {vocab:,})",
        "params": [
            ("임베딩 = 출력층 (공유)", "embed"),
            ("블록 {layers}개", "blocks"),
            ("   어텐션", "attn"),
            ("   FFN", "ffn"),
            ("   RMSNorm·QK-Norm", "norms"),
            ("최종 RMSNorm", "final_norm"),
            ("합계", "total"),
        ],
        "vocab_note": "vocab 6,400이면 임베딩만 {small_embed}로 줄어\n합계 {small_total}",
        "footer": "Dropout 확률 0 · MoE 안 씀 · YaRN 안 씀 · 학습 길이 사전학습 {pre}, 대화 SFT {sft}토큰 · "
                  "모델 코드는 원본 MiniMind model/model_minimind.py",
    },
    "en": {
        "title": "minimind_ko final model architecture (vocab {vocab:,}, {total} parameters)",
        "panel_a": "(a) Overall flow",
        "panel_b": "(b) Transformer block (× {layers})",
        "panel_c": "(c) GQA attention (per block)",
        "panel_d": "(d) SwiGLU FFN (per block)",
        "input": "Input: “{text}”",
        "tokenizer": "Korean BPE tokenizer (ko_16000)\ntrained on Korean Wikipedia, {vocab:,} entries",
        "tokens": "{n} tokens, _ = leading space (original MiniMind tokenizer: {orig})",
        "embed": "Input embedding  {vocab:,} × {d}\n{p}",
        "blocks": "Transformer block × {layers}\n{per} each, {sum} total",
        "final_norm": "RMSNorm",
        "lm_head": "Output layer  {d} → {vocab:,}\nsame matrix as embedding (0 extra)",
        "softmax": "Softmax → probabilities of {vocab:,} next tokens",
        "next": "next token",
        "tied": "same matrix",
        "shape": "[B, T, {d}]  T ≤ {pre} (SFT {sft})",
        "attn": "GQA attention\n{p}",
        "ffn": "SwiGLU FFN\n{p}",
        "residual": "residual",
        "keep_shape": "block input and output: [B, T, {d}]",
        "c_in": "RMSNorm output [B, T, {d}]",
        "q_heads": "Q  {n} heads × {hd}",
        "kv_heads": "{name}  {n} heads × {hd}",
        "qk_norm": "RMSNorm({hd})",
        "rope": "RoPE θ={theta}",
        "attention": "softmax(QK$^\\mathrm{{T}}$ / √{hd} + causal mask) · V",
        "concat": "concat {n} heads → [B, T, {d}]",
        "share": "{g} Q heads share 1 K/V head",
        "lora": "LoRA rank {r} (docs/09)",
        "d_in": "RMSNorm output [B, T, {d}]",
        "params_title": "Parameters (vocab {vocab:,})",
        "params": [
            ("Embedding = output (tied)", "embed"),
            ("{layers} blocks", "blocks"),
            ("   attention", "attn"),
            ("   FFN", "ffn"),
            ("   RMSNorm, QK-Norm", "norms"),
            ("Final RMSNorm", "final_norm"),
            ("Total", "total"),
        ],
        "vocab_note": "With vocab 6,400 only the embedding shrinks,\nto {small_embed}; total {small_total}",
        "footer": "Dropout p = 0 · no MoE · no YaRN · training length {pre} (pretraining), {sft} (chat SFT) tokens · "
                  "model code: original MiniMind model/model_minimind.py",
    },
}


def fmt_params(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def fmt_theta(theta):
    """1000000.0 → 1e6 (matplotlib 기본 표기 1e+06 대신 설정 파일과 같은 모양)."""
    exponent = len(str(int(theta))) - 1
    if theta != 10 ** exponent:
        return f"{theta:g}"
    return f"1e{exponent}"


def count_params(vocab_size):
    """원본 모델 코드로 구조만 만들어 구성 요소별 파라미터를 센다. lm_head는 임베딩과 같은 텐서라 한 번만 센다."""
    config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS, vocab_size=vocab_size)
    model = MiniMindForCausalLM(config)
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("임베딩과 출력층이 가중치를 공유하지 않음: 그림의 '같은 행렬' 설명이 틀려진다")
    layer = model.model.layers[0]
    attn = layer.self_attn
    counts = {
        "embed": model.model.embed_tokens.weight.numel(),
        "attn": sum(m.weight.numel() for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj)),
        "ffn": sum(p.numel() for p in layer.mlp.parameters()),
        "norms": sum(p.numel() for p in (layer.input_layernorm.weight, layer.post_attention_layernorm.weight,
                                         attn.q_norm.weight, attn.k_norm.weight)),
        "final_norm": model.model.norm.weight.numel(),
        "total": sum(p.numel() for p in model.parameters()),
    }
    counts["block"] = sum(p.numel() for p in layer.parameters())
    counts["blocks"] = counts["block"] * NUM_LAYERS
    if counts["block"] != counts["attn"] + counts["ffn"] + counts["norms"]:
        raise ValueError("블록 파라미터가 어텐션 + FFN + 정규화 합과 다름: 모델 코드가 바뀌었는지 확인")
    if counts["total"] != counts["embed"] + counts["blocks"] + counts["final_norm"]:
        raise ValueError("전체 파라미터가 구성 요소 합과 다름: 모델 코드가 바뀌었는지 확인")
    return config, counts


class Canvas:
    def __init__(self, ax):
        self.ax = ax

    def box(self, x, y, w, h, text, kind, size=11, weight="normal", text_color=None, lora=False):
        face, edge = COLORS[kind]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                         facecolor=face, edgecolor=edge, linewidth=1.3, zorder=3))
        if lora:
            self.ax.add_patch(FancyBboxPatch((x - 0.7, y - 0.7), w + 1.4, h + 1.4,
                                             boxstyle="round,pad=0,rounding_size=1.2", facecolor="none",
                                             edgecolor=LORA_COLOR, linewidth=1.6, linestyle=(0, (4, 2)), zorder=2))
        color = text_color or ("white" if kind == "head" else TEXT_PRIMARY)
        self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
                     fontweight=weight, zorder=4, linespacing=1.35)

    def arrow(self, x1, y1, x2, y2, color=TEXT_PRIMARY, style="-|>", dashed=False, rad=0.0, lw=1.4):
        self.ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=13, color=color,
                                          linewidth=lw, linestyle=(0, (4, 3)) if dashed else "-",
                                          connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=2))

    def line(self, xs, ys, color=TEXT_PRIMARY, dashed=False, lw=1.4):
        self.ax.plot(xs, ys, color=color, linewidth=lw, linestyle=(0, (4, 3)) if dashed else "-", zorder=2,
                     solid_capstyle="butt")

    def circle_op(self, x, y, symbol, r=1.7):
        self.ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=TEXT_PRIMARY, linewidth=1.4, zorder=3))
        self.ax.text(x, y, symbol, ha="center", va="center", fontsize=13, zorder=4, color=TEXT_PRIMARY)

    def panel(self, x, y, w, h, title):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2",
                                         facecolor=PANEL_BG, edgecolor="#d9d7cf", linewidth=1.0, zorder=0))
        self.ax.text(x + 1.8, y + h - 2.6, title, ha="left", va="center", fontsize=13.5, fontweight="bold",
                     color=TEXT_PRIMARY, zorder=4)

    def text(self, x, y, s, size=10, color=TEXT_SECONDARY, ha="center", va="center", **kw):
        self.ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, zorder=4, **kw)


def draw_overall(c, t, cfg, counts, tokens, orig_count):
    vocab, d = cfg.vocab_size, cfg.hidden_size
    c.panel(1, 1, 52, 104, t["panel_a"])
    cx, w = 29, 40
    x0 = cx - w / 2

    c.box(x0, 4, w, 5.5, t["input"].format(text=EXAMPLE), "io", size=11.5)
    c.arrow(cx, 9.5, cx, 12.5)
    c.box(x0, 12.5, w, 8, t["tokenizer"].format(vocab=vocab), "tokenizer", size=10.5)
    c.arrow(cx, 20.5, cx, 23.5)

    # 실제 토큰 조각. 한글 글꼴에 공백 기호(␣, ▁)가 없어서 앞 공백은 _로 보여 준다
    n = len(tokens)
    gap = 0.5
    chip_w = (w - gap * (n - 1)) / n
    for i, piece in enumerate(tokens):
        c.box(x0 + i * (chip_w + gap), 23.5, chip_w, 4.5, piece.replace(" ", "_"), "io", size=9.5)
    c.text(cx, 30, t["tokens"].format(n=n, orig=orig_count), size=9.5)
    c.arrow(cx, 31.8, cx, 34)

    c.box(x0, 34, w, 7.5, t["embed"].format(vocab=vocab, d=d, p=fmt_params(counts["embed"])), "linear", size=10.5)
    c.arrow(cx, 41.5, cx, 45)
    c.text(cx + 1.5, 43.2, t["shape"].format(d=d, pre=PRETRAIN_LEN, sft=SFT_LEN), size=8.5, ha="left")

    c.box(x0, 46, w, 11, t["blocks"].format(layers=NUM_LAYERS, per=fmt_params(counts["block"]),
                                           sum=fmt_params(counts["blocks"])), "block", size=11, weight="bold")
    c.ax.add_patch(FancyBboxPatch((x0 - 1, 45), w + 2, 13, boxstyle="round,pad=0,rounding_size=1.2",
                                  facecolor="none", edgecolor=COLORS["block"][1], linewidth=1.3,
                                  linestyle=(0, (2, 2)), zorder=2))
    c.arrow(cx, 58, cx, 62)
    c.box(x0, 62, w, 5, t["final_norm"], "norm", size=10.5)
    c.arrow(cx, 67, cx, 70.5)
    c.box(x0, 70.5, w, 8, t["lm_head"].format(d=d, vocab=vocab), "linear", size=10.5)
    c.arrow(cx, 78.5, cx, 82)
    c.box(x0, 82, w, 5, t["softmax"].format(vocab=vocab), "softmax", size=10.5)
    c.arrow(cx, 87, cx, 90.5)
    c.box(cx - 8, 90.5, 16, 5, t["next"], "io", size=10.5)

    # 입력 임베딩 ↔ 출력층 가중치 공유
    c.line([x0, x0 - 4.5], [37.75, 37.75], color=COLORS["linear"][1], dashed=True)
    c.line([x0 - 4.5, x0 - 4.5], [37.75, 74.5], color=COLORS["linear"][1], dashed=True)
    c.arrow(x0 - 4.5, 74.5, x0, 74.5, color=COLORS["linear"][1], dashed=True)
    c.text(x0 - 5.6, 56, t["tied"], size=9, color=COLORS["linear"][1], rotation=90)
    return (x0 + w + 1, 45, 58)


def draw_block(c, t, counts, zoom_from):
    c.panel(56, 1, 36, 104, t["panel_b"].format(layers=NUM_LAYERS))
    # (a)의 블록 상자에서 (b)로 펼쳐지는 띠
    zx, zy0, zy1 = zoom_from
    c.ax.add_patch(Polygon([(zx, zy0), (zx, zy1), (56, 100), (56, 6)], closed=True,
                           facecolor=COLORS["block"][0], edgecolor="none", alpha=0.45, zorder=1))
    cx, w = 72, 22
    x0 = cx - w / 2
    rx = x0 + w + 4  # 잔차 연결 선의 x

    c.arrow(cx, 8, cx, 17)
    c.box(x0, 17, w, 5, "RMSNorm", "norm", size=10.5)
    c.arrow(cx, 22, cx, 26)
    c.box(x0, 26, w, 9, t["attn"].format(p=fmt_params(counts["attn"])), "attn", size=11, weight="bold")
    c.arrow(cx, 35, cx, 40.3)
    c.circle_op(cx, 42, "+")
    c.line([cx, rx, rx], [12, 12, 42])
    c.arrow(rx, 42, cx + 1.7, 42)
    c.text(rx + 0.8, 27, t["residual"], size=8.5, rotation=90, ha="left")

    c.arrow(cx, 43.7, cx, 54)
    c.box(x0, 54, w, 5, "RMSNorm", "norm", size=10.5)
    c.arrow(cx, 59, cx, 63)
    c.box(x0, 63, w, 9, t["ffn"].format(p=fmt_params(counts["ffn"])), "ffn", size=11, weight="bold")
    c.arrow(cx, 72, cx, 77.3)
    c.circle_op(cx, 79, "+")
    c.line([cx, rx, rx], [49, 49, 79])
    c.arrow(rx, 79, cx + 1.7, 79)
    c.text(rx + 0.8, 64, t["residual"], size=8.5, rotation=90, ha="left")
    c.arrow(cx, 80.7, cx, 88)
    c.text(cx, 92, t["keep_shape"].format(d=HIDDEN_SIZE), size=9.5)


def draw_attention(c, t, cfg):
    d, hd = cfg.hidden_size, cfg.head_dim
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    c.panel(95, 1, 62, 104, t["panel_c"])
    cols = [(98, "Q"), (117.5, "K"), (137, "V")]
    bw = 17
    out_dims = {"Q": nq * hd, "K": nkv * hd, "V": nkv * hd}
    proj = {"Q": "q_proj", "K": "k_proj", "V": "v_proj"}

    c.text(126, 5, t["c_in"].format(d=d), size=9.5)
    c.line([106.5, 145.5], [7.5, 7.5])
    for x, name in cols:
        mid = x + bw / 2
        c.line([mid, mid], [7.5, 10])
        c.arrow(mid, 9.9, mid, 10.5)
        c.box(x, 10.5, bw, 7, f"{proj[name]}\n{d} → {out_dims[name]}", "linear", size=10, lora=(name == "Q"))
        c.arrow(mid, 17.5, mid, 21)
        label = t["q_heads"].format(n=nq, hd=hd) if name == "Q" else t["kv_heads"].format(name=name, n=nkv, hd=hd)
        c.box(x, 21, bw, 5, label, "head", size=9.5)
        if name == "V":
            c.line([mid, mid], [26, 48])
            continue
        c.arrow(mid, 26, mid, 29)
        c.box(x + 1.5, 29, bw - 3, 4.2, t["qk_norm"].format(hd=hd), "norm", size=9)
        c.arrow(mid, 33.2, mid, 36)
        c.box(x + 1.5, 36, bw - 3, 4.2, t["rope"].format(theta=fmt_theta(cfg.rope_theta)), "norm", size=9)
        c.line([mid, mid], [40.2, 48])
    c.text(116.5, 34.6, "QK-Norm", size=8.5, color=MUTED)

    c.arrow(126, 47.9, 126, 48)
    c.box(99, 48, 54, 6, t["attention"].format(hd=hd), "softmax", size=10.5)
    c.arrow(126, 54, 126, 58)
    c.box(99, 58, 54, 5, t["concat"].format(n=nq, d=d), "head", size=9.5)
    c.arrow(126, 63, 126, 67)
    c.box(115, 67, 22, 7, f"o_proj\n{d} → {d}", "linear", size=10, lora=True)
    c.arrow(126, 74, 126, 78)
    c.text(126, 79.6, f"[B, T, {d}]", size=9.5)
    # 위쪽 헤드 짝 그림은 흐름의 다음 단계가 아니라 보조 설명이라 선으로 나눈다
    c.line([99, 153], [81.6, 81.6], color=GRID_LINE, lw=1.0)

    # LoRA 범례
    c.ax.add_patch(FancyBboxPatch((99.5, 75.5), 4, 3, boxstyle="round,pad=0,rounding_size=0.5", facecolor="none",
                                  edgecolor=LORA_COLOR, linewidth=1.6, linestyle=(0, (4, 2)), zorder=3))
    c.text(104.5, 77, t["lora"].format(r=LORA_RANK), size=8.5, color=LORA_COLOR, ha="left")

    # Q 헤드와 K·V 헤드 짝
    group = nq // nkv
    qw, gap_q = 5, 0.9
    q_total = nq * qw + (nq - 1) * gap_q
    qx0 = 126 - q_total / 2
    kv_w = 2 * qw + gap_q
    for i in range(nq):
        qx = qx0 + i * (qw + gap_q)
        c.box(qx, 86, qw, 3.6, f"Q{i + 1}", "head", size=8)
        kv = i // group
        kx = qx0 + kv * group * (qw + gap_q) + (group * qw + (group - 1) * gap_q - kv_w) / 2
        c.line([qx + qw / 2, kx + kv_w / 2], [86, 89 - 6.4 + 3.6], color=MUTED, lw=1.0)
    for j in range(nkv):
        kx = qx0 + j * group * (qw + gap_q) + (group * qw + (group - 1) * gap_q - kv_w) / 2
        c.box(kx, 82.6, kv_w, 3.6, f"KV{j + 1}", "norm", size=8)
    c.text(126, 94, t["share"].format(g=group), size=9.5)


def draw_ffn(c, t, cfg, counts, small_counts):
    d, inter = cfg.hidden_size, cfg.intermediate_size
    c.panel(160, 1, 39, 104, t["panel_d"])
    gx, ux, bw = 162, 180.5, 16.5
    gmid, umid, cx = gx + bw / 2, ux + bw / 2, 179.5

    c.text(cx, 5, t["d_in"].format(d=d), size=9.5)
    c.line([gmid, umid], [7.5, 7.5])
    for mid in (gmid, umid):
        c.line([mid, mid], [7.5, 10])
        c.arrow(mid, 9.9, mid, 10.5)
    c.box(gx, 10.5, bw, 7, f"gate_proj\n{d} → {inter:,}", "linear", size=10)
    c.box(ux, 10.5, bw, 7, f"up_proj\n{d} → {inter:,}", "linear", size=10)
    c.arrow(gmid, 17.5, gmid, 21)
    c.box(gx + 3, 21, bw - 6, 4.5, "SiLU", "tokenizer", size=10, weight="bold")
    c.line([gmid, gmid, cx - 1.7], [25.5, 31, 31])
    c.line([umid, umid, cx + 1.7], [17.5, 31, 31])
    c.circle_op(cx, 31, "⊙")
    c.arrow(cx, 32.7, cx, 36)
    c.box(cx - 9, 36, 18, 7, f"down_proj\n{inter:,} → {d}", "linear", size=10)
    c.arrow(cx, 43, cx, 46.5)
    c.text(cx, 48.2, f"[B, T, {d}]", size=9.5)

    # 파라미터 표 (y축이 위로 커지므로 위에서부터 아래로 내려가며 적는다)
    c.text(162, 94, t["params_title"].format(vocab=cfg.vocab_size), size=11, color=TEXT_PRIMARY, ha="left",
           fontweight="bold")
    y = 89
    for label, key in t["params"]:
        bold = key == "total"
        if bold:
            c.line([162, 197], [y + 2.2, y + 2.2], color=MUTED, lw=0.8)
        value = fmt_params(counts[key])
        if key in ("attn", "ffn", "norms"):
            value = f"{fmt_params(counts[key])} × {NUM_LAYERS}"
        c.text(162, y, label.format(layers=NUM_LAYERS), size=9.5, color=TEXT_PRIMARY, ha="left",
               fontweight="bold" if bold else "normal")
        c.text(197, y, value, size=9.5, color=TEXT_PRIMARY, ha="right", fontweight="bold" if bold else "normal")
        y -= 4.4
    c.text(162, y + 2, t["vocab_note"].format(small_embed=fmt_params(small_counts["embed"]),
                                              small_total=fmt_params(small_counts["total"])),
           size=8.5, ha="left", va="top", linespacing=1.4)


def main(out_path, lang):
    t = LABELS[lang]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    original_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_TOKENIZER_DIR)
    ids = tokenizer(EXAMPLE, add_special_tokens=False)["input_ids"]
    tokens = [tokenizer.decode([i]) for i in ids]
    if "".join(tokens) != EXAMPLE:
        raise ValueError(f"토큰을 이어 붙여도 원문이 안 됨: {tokens}")
    orig_count = len(original_tokenizer(EXAMPLE, add_special_tokens=False)["input_ids"])

    cfg, counts = count_params(len(tokenizer))
    _, small_counts = count_params(6400)

    font = pick_korean_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(20, 11.6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 200)
    ax.set_ylim(-5, 111)
    ax.set_aspect("equal")
    ax.axis("off")
    c = Canvas(ax)

    zoom = draw_overall(c, t, cfg, counts, tokens, orig_count)
    draw_block(c, t, counts, zoom)
    draw_attention(c, t, cfg)
    draw_ffn(c, t, cfg, counts, small_counts)

    c.text(100, 108.5, t["title"].format(vocab=cfg.vocab_size, total=fmt_params(counts["total"])), size=17,
           color=TEXT_PRIMARY, fontweight="bold")
    c.text(100, -2.5, t["footer"].format(pre=PRETRAIN_LEN, sft=SFT_LEN), size=9.5, color=MUTED)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"저장: {out_path} | 토큰 {len(tokens)}개(원본 {orig_count}) | 전체 {counts['total']:,} | "
          f"블록 {counts['block']:,} | 임베딩 {counts['embed']:,}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in LABELS:
        sys.exit("사용법: plot_architecture.py <출력 png> <ko|en>")
    main(sys.argv[1], sys.argv[2])
