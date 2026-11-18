"""
사전학습 모델을 held-out 데이터로 평가해 '글자당 비트(BPC, bits per character)'를 구한다.

vocab 크기가 다른 토크나이저끼리는 토큰당 loss를 비교할 수 없다.
(vocab 16,000의 토큰 하나는 6,400의 토큰 하나보다 더 많은 글자를 담는다)
그래서 같은 텍스트 전체를 맞히는 데 든 정보량(NLL 합)을 '글자 수'로 나눠 단위를 맞춘다.

    BPC = (예측한 토큰들의 NLL 합, nats) / ln 2 / (총 글자 수)

평가 규칙 (두 토크나이저에 똑같이 적용):
  - 입력은 학습과 같은 모양: [bos] + 텍스트 토큰 + [eos]
  - 예측 대상은 bos 다음의 모든 토큰 = 텍스트 토큰 전부 + eos 1개. bos 자체는 항상 주어지므로 제외.
  - 기본 BPC는 eos 예측까지 포함한다 (문서 끝을 아는 것도 정보다). 텍스트 토큰만 쓴 BPC도 따로 출력한다.
  - 글자 수는 원문 텍스트의 len() (유니코드 글자 수). bos/eos는 글자로 세지 않는다.
  - 텍스트는 자르지 않는다. 학습 길이(max_seq_len=512)보다 긴 조각은 슬라이딩 윈도우로 나눠 평가한다
    (아래 WINDOW/STRIDE 설명 참고).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_bpc.py ^
        --tokenizer_dir tokenizers/ko_16000 --weight out_ko/pretrain_ko16000_768.pth
    (빠른 동작 확인: --limit 50 --device cpu)
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

DEFAULT_DATA = "data/kowiki_heldout_2k.jsonl"
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 8
# 모델은 학습 때 512 위치까지만 봤다. RoPE는 이론상 더 긴 위치도 계산되지만 학습하지 않은 위치라 성능이 떨어지고,
# 토큰이 더 많은 vocab 6,400 쪽이 그 손해를 더 많이 본다(512 초과 조각 84개 vs 17개). 그래서 학습 길이로 창을 제한한다.
WINDOW = 512
# 긴 조각은 창을 STRIDE만큼 밀면서 새로 들어온 토큰만 채점한다. 각 토큰이 최소 (WINDOW - STRIDE)개의 앞 문맥을 갖는다.
STRIDE = 256


def parse_args():
    p = argparse.ArgumentParser(description="held-out 데이터로 글자당 비트(BPC) 평가")
    p.add_argument("--tokenizer_dir", required=True, help="토크나이저 폴더 (예: tokenizers/ko_16000)")
    p.add_argument("--weight", required=True, help="사전학습 가중치 .pth (fp16 state_dict)")
    p.add_argument("--data_path", default=DEFAULT_DATA, help="평가 데이터 jsonl ({\"text\": ...})")
    p.add_argument("--limit", type=int, default=0, help="앞에서부터 N줄만 평가 (0이면 전체)")
    p.add_argument("--window", type=int, default=WINDOW, help="한 번에 보는 최대 토큰 수 (0이면 제한 없음)")
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 두 모델을 '같은 문서끼리' 짝지어 비교(몇 개 문서에서 이겼는지 등)하려면 문서별 값이 필요하다
    p.add_argument("--per_doc_out", default="", help="문서별 글자 수/NLL을 jsonl로 저장할 경로 (선택)")
    return p.parse_args()


def load_model(tokenizer_dir, weight_path, device):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, tokenizer_dir))
    config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS, vocab_size=len(tokenizer))
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(os.path.join(ROOT, weight_path), map_location="cpu"), strict=True)
    # 저장은 fp16이지만 평가는 fp32로 한다. 작은 확률의 log 값이 반올림 오차 없이 합쳐지도록 하기 위함
    return tokenizer, model.float().eval().to(device)


@torch.no_grad()
def score_tokens(model, ids, bos_id, window, stride, device):
    """ids = [bos] + 텍스트 + [eos]. 위치 1..끝 각 토큰의 NLL(nats) 리스트를 돌려준다."""
    targets_total = len(ids) - 1
    nll = [None] * targets_total
    if window <= 0 or len(ids) <= window:
        spans = [(0, len(ids))]
    else:
        spans = []
        start = 0
        while True:
            end = min(start + window, len(ids))
            spans.append((start, end))
            if end == len(ids):
                break
            start += stride

    for start, end in spans:
        if start == 0:
            chunk = ids[start:end]
        else:
            # 학습 데이터는 항상 bos로 시작했으므로, 중간부터 자른 창도 앞에 bos를 붙여 같은 모양으로 맞춘다.
            # 창 길이를 window 이하로 유지하려고 bos 몫만큼 한 토큰 뒤에서 시작한다.
            chunk = [bos_id] + ids[start + 1:end]
        x = torch.tensor([chunk], device=device)
        logits = model(x).logits[0, :-1].float()
        token_nll = F.cross_entropy(logits, x[0, 1:], reduction="none").tolist()
        # chunk의 j번째 예측은 원래 ids의 위치 (start + j + 1) 토큰이므로 nll 인덱스는 start + j
        for j, v in enumerate(token_nll):
            pos = start + j
            if start != 0 and j == 0:
                # bos 바로 다음 토큰은 앞 문맥 없이 맞힌 값이라 쓰지 않는다 (이전 창에서 이미 채점됨)
                continue
            # 겹치는 구간은 먼저 채점한 창의 값을 쓴다 (그때도 이미 충분한 앞 문맥이 있었다)
            if nll[pos] is None:
                nll[pos] = v
    assert all(v is not None for v in nll), "채점되지 않은 토큰이 있다 (stride/window 확인)"
    return nll


def main():
    args = parse_args()
    if args.window > 0 and not (0 < args.stride < args.window - 1):
        sys.exit("stride는 0보다 크고 window-1보다 작아야 한다")
    tokenizer, model = load_model(args.tokenizer_dir, args.weight, args.device)
    with open(os.path.join(ROOT, args.data_path), encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if args.limit > 0:
        texts = texts[:args.limit]

    total_chars = 0
    total_text_tokens = 0
    nll_text = 0.0
    nll_eos = 0.0
    long_docs = 0
    per_doc = []
    t0 = time.time()
    for i, text in enumerate(texts, 1):
        ids = [tokenizer.bos_token_id] + tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
        if args.window > 0 and len(ids) > args.window:
            long_docs += 1
        nll = score_tokens(model, ids, tokenizer.bos_token_id, args.window, args.stride, args.device)
        nll_text += sum(nll[:-1])
        nll_eos += nll[-1]
        total_text_tokens += len(nll) - 1
        total_chars += len(text)
        per_doc.append({"chars": len(text), "tokens": len(nll) - 1, "nll": sum(nll)})
        if i % 200 == 0:
            print(f"  {i}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)

    if args.per_doc_out:
        with open(os.path.join(ROOT, args.per_doc_out), "w", encoding="utf-8") as f:
            for row in per_doc:
                f.write(json.dumps(row) + "\n")

    ln2 = math.log(2)
    n_docs = len(texts)
    print(f"\ntokenizer: {args.tokenizer_dir} (vocab {len(tokenizer)}) | weight: {args.weight}")
    print(f"문서 {n_docs}개 | 글자 {total_chars:,} | 텍스트 토큰 {total_text_tokens:,} | "
          f"window {args.window} 초과 문서 {long_docs}개 | {time.time() - t0:.0f}s")
    print(f"BPC (텍스트+eos)       : {(nll_text + nll_eos) / ln2 / total_chars:.4f}")
    print(f"BPC (텍스트만)         : {nll_text / ln2 / total_chars:.4f}")
    print(f"토큰당 loss (텍스트+eos): {(nll_text + nll_eos) / (total_text_tokens + n_docs):.4f}")
    print(f"토큰당 평균 글자 수     : {total_chars / total_text_tokens:.3f}")


if __name__ == "__main__":
    main()
