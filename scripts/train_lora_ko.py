"""
최종 SFT 모델에 LoRA로 자기소개를 가르친다 (설계: docs/superpowers/specs/2026-11-24-lora-dpo-design.md, docs/09).
원본 minimind/trainer/train_lora.py의 학습 루프(train_epoch)와 model/model_lora.py(apply_lora, save_lora)를 그대로 쓰고,
데이터는 train_sft_ko.py와 같은 고정 대화 형식(ChatSFTDataset, 답 본문 + <|im_end|> + 줄바꿈만 채점)으로 넣는다.
원본 LoRA는 입력과 출력 크기가 같은 Linear(이 모델은 층마다 q_proj, o_proj)에만 붙고, 나머지 가중치는 얼린다.

입력: docs/09_lora_identity_ko.json의 "train" (질문·답 48쌍)
사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\train_lora_ko.py
결과: out_lora/lora_identity_ko_768.pth (LoRA 가중치만)
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
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import trainer.train_lora as base  # noqa: E402
from model.model_lora import apply_lora  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.trainer_utils import Logger, SkipBatchSampler, setup_seed  # noqa: E402
from train_sft_ko import ChatSFTDataset, check_first_batch, epoch_indices  # noqa: E402

DATA = "docs/09_lora_identity_ko.json"


def count_lora_params(model):
    """(LoRA 파라미터 수, 전체 파라미터 수)."""
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if ".lora." in n)
    return lora, total


def freeze_except_lora(model):
    """LoRA 파라미터만 학습하고 나머지는 얼린다. 학습할 파라미터 목록을 돌려준다."""
    params = []
    for name, p in model.named_parameters():
        p.requires_grad = ".lora." in name
        if p.requires_grad:
            params.append(p)
    return params


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind 한국어 LoRA (자기소개)")
    p.add_argument("--tokenizer_dir", default="tokenizers/ko_16000")
    p.add_argument("--init_weight", default="out_final/sft/sft_ko16000_768.pth")
    p.add_argument("--out_dir", default="out_lora")
    p.add_argument("--lora_name", default="lora_identity_ko")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=3)
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
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               vocab_size=len(tokenizer))
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(os.path.join(ROOT, args.init_weight), map_location="cpu"), strict=True)
    model = model.to(args.device)
    apply_lora(model, rank=args.lora_rank)
    lora_params = freeze_except_lora(model)
    lora, total = count_lora_params(model)
    Logger(f"init: {args.init_weight} | LoRA rank {args.lora_rank} | LoRA 파라미터 {lora:,} / 전체 {total:,} "
           f"({lora / total * 100:.2f}%)")

    with open(os.path.join(ROOT, DATA), encoding="utf-8") as f:
        rows = json.load(f)["train"]
    ds = ChatSFTDataset(rows, tokenizer, args.max_seq_len)
    Logger(f"학습 쌍 {len(rows)}개")

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 원본 train_epoch()가 참조하는 모듈 전역값을 채운다
    args.save_dir = out_dir
    base.args = args
    base.lm_config = lm_config
    base.model = model
    base.optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    base.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    base.autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)
    # 몇 분이면 끝나는 학습이라 이어서 하기용 체크포인트(전체 가중치+옵티마이저, 약 0.9GB)는 저장하지 않는다
    base.lm_checkpoint = lambda *a, **kw: None

    for epoch in range(args.epochs):
        indices = epoch_indices(len(ds), args.seed, epoch)
        if epoch == 0:
            check_first_batch(ds, tokenizer, indices, args.batch_size)
        loader = DataLoader(ds, batch_sampler=SkipBatchSampler(indices, args.batch_size, 0), num_workers=0)
        base.train_epoch(epoch, loader, len(loader), lora_params, 0, None)
    Logger(f"완료: {os.path.join(out_dir, f'{args.lora_name}_{args.hidden_size}.pth')}")


if __name__ == "__main__":
    main()
