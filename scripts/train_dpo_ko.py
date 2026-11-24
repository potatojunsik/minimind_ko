"""
최종 SFT 모델에 DPO로 상투 답을 줄이도록 가르친다 (설계: docs/superpowers/specs/2026-11-24-lora-dpo-design.md, docs/09).
원본 minimind/trainer/train_dpo.py의 학습 루프(train_epoch, dpo_loss)를 그대로 쓰고, 데이터만 바꾼다.
원본 DPODataset은 원본 대화 템플릿(무작위 생각 태그 처리 포함)을 쓰므로, SFT와 같은 고정 대화 형식으로 만드는
ChatDPODataset을 쓴다(채점 위치 = 답 본문 + <|im_end|> + 줄바꿈, ChatSFTDataset 라벨과 같음).
참조 모델은 시작 모델을 얼려서 쓴다(원본 방식).

입력: data/dpo/dpo_pairs.jsonl (make_dpo_data.py)
사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\train_dpo_ko.py
결과: out_dpo/dpo_ko16000_768.pth
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: F401,E402  # 원본과 동일: Windows pyarrow/torch DLL 충돌 회피 (원본 issue #771)
import torch  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import trainer.train_dpo as base  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.trainer_utils import Logger, SkipBatchSampler, setup_seed  # noqa: E402
from train_sft_ko import ChatSFTDataset, chat_text, epoch_indices  # noqa: E402


class ChatDPODataset(Dataset):
    """원본 DPODataset과 같은 키를 돌려준다. mask[i] = y[i]가 답 본문 + <|im_end|> + 줄바꿈이면 1."""

    def __init__(self, rows, tokenizer, max_length):
        self.chosen = ChatSFTDataset([{"q": r["q"], "a": r["chosen"]} for r in rows], tokenizer, max_length)
        self.rejected = ChatSFTDataset([{"q": r["q"], "a": r["rejected"]} for r in rows], tokenizer, max_length)

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, index):
        out = {}
        for kind, ds in (("chosen", self.chosen), ("rejected", self.rejected)):
            input_ids, labels = ds[index]
            out[f"x_{kind}"] = input_ids[:-1]
            out[f"y_{kind}"] = input_ids[1:]
            out[f"mask_{kind}"] = (labels[1:] != -100).long()
        return out


def load_model(tokenizer, args):
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                            vocab_size=len(tokenizer))
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(os.path.join(ROOT, args.init_weight), map_location="cpu"), strict=True)
    return config, model.to(args.device)


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind 한국어 DPO (상투 답 줄이기)")
    p.add_argument("--tokenizer_dir", default="tokenizers/ko_16000")
    p.add_argument("--init_weight", default="out_final/sft/sft_ko16000_768.pth")
    p.add_argument("--data_path", default="data/dpo/dpo_pairs.jsonl")
    p.add_argument("--out_dir", default="out_dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=100000, help="epoch 끝에는 원본 루프가 항상 저장한다")
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, args.tokenizer_dir))
    lm_config, model = load_model(tokenizer, args)
    _, ref_model = load_model(tokenizer, args)
    ref_model.eval()
    ref_model.requires_grad_(False)
    with open(os.path.join(ROOT, args.data_path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise SystemExit(f"{args.data_path}에 DPO 쌍이 없다(상투 답이 없음). 학습하지 않는다")
    # ChatSFTDataset은 너무 긴 쌍을 꺼낼 때에야 멈추므로, 학습 도중이 아니라 시작 전에 모두 확인한다
    longest = max(len(tokenizer(chat_text(r["q"], r[k]), add_special_tokens=False).input_ids)
                  for r in rows for k in ("chosen", "rejected"))
    if longest > args.max_seq_len:
        raise SystemExit(f"가장 긴 쌍이 {longest}토큰으로 max_seq_len({args.max_seq_len})보다 길다")
    ds = ChatDPODataset(rows, tokenizer, args.max_seq_len)
    Logger(f"init: {args.init_weight} | DPO 쌍 {len(rows):,}개 | beta {args.beta} | lr {args.learning_rate}")

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 원본 train_epoch()가 참조하는 모듈 전역값을 채운다
    args.save_dir = out_dir
    args.save_weight = f"dpo_ko{len(tokenizer)}"
    base.args = args
    base.model = model
    base.optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    base.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    base.autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)
    # 수십 분 안에 끝나는 학습이라 이어서 하기용 체크포인트(약 0.9GB)는 저장하지 않는다
    base.lm_checkpoint = lambda *a, **kw: None

    for epoch in range(args.epochs):
        indices = epoch_indices(len(ds), args.seed, epoch)
        loader = DataLoader(ds, batch_sampler=SkipBatchSampler(indices, args.batch_size, 0), num_workers=0)
        base.train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, None, beta=args.beta)
    Logger(f"완료: {os.path.join(out_dir, f'{args.save_weight}_{args.hidden_size}.pth')}")


if __name__ == "__main__":
    main()
