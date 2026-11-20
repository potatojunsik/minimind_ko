"""
한국어 사전학습 모델을 대화 데이터로 SFT한다 (설계: docs/superpowers/specs/2026-11-20-chat-sft-design.md 3절).

원본 minimind/trainer/train_full_sft.py의 학습 루프(train_epoch)를 그대로 재사용하고, 다음만 바꾼다.
  1. 토크나이저 경로와 모델 vocab 크기 (원본: ../model 고정, vocab 6,400 고정)
  2. 데이터셋: 원본 SFTDataset은 20% 확률로 중국어·영어 시스템 프롬프트를 넣고 빈 <think> 태그를 80% 확률로 지운다.
     두 vocab의 조건을 같게 하려고 무작위 처리 없이 형식 하나로 고정한다(ChatSFTDataset).
     채점 위치(답 본문 + <|im_end|> + 줄바꿈)는 원본 generate_labels를 그대로 쓴다.
  3. 시작 가중치를 파일 경로로 받고, 체크포인트 폴더를 실행마다 나눈다. 원본 파일은 고치지 않는다.

입력 jsonl: 줄마다 {"q": 질문, "a": 답} (prepare_chat_eval.py의 sft_pool.jsonl)

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\train_sft_ko.py ^
        --tokenizer_dir tokenizers/ko_16000 --init_weight out_scale/ko16000_n1000000/pretrain_ko16000_768.pth
결과: out_sft/sft_ko<vocab>_768.pth
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINIMIND_DIR = os.path.join(ROOT, "minimind")
sys.path.append(MINIMIND_DIR)

import datasets  # noqa: F401,E402  # 원본과 동일: Windows pyarrow/torch DLL 충돌 회피 (원본 issue #771)
import torch  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import trainer.train_full_sft as base  # noqa: E402
from dataset.lm_dataset import SFTDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.trainer_utils import Logger, SkipBatchSampler, setup_seed  # noqa: E402


def chat_text(q, a):
    """고정 대화 형식 (시스템 프롬프트·빈 생각 태그 없음)."""
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"


class ChatSFTDataset(SFTDataset):
    def __init__(self, rows, tokenizer, max_length):
        # 부모 __init__은 jsonl을 datasets로 읽는다(무작위 전처리와 짝). 여기서는 읽은 행을 받고,
        # 채점 위치를 찾는 표시(bos_id, eos_id)만 부모와 똑같이 만든다
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = rows
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __getitem__(self, index):
        row = self.samples[index]
        input_ids = self.tokenizer(chat_text(row["q"], row["a"]), add_special_tokens=False).input_ids
        if len(input_ids) > self.max_length:
            # 원본은 조용히 자르지만, 자르면 답 끝 <|im_end|>가 빠져 두 vocab의 조건이 달라지므로 멈춘다
            raise ValueError(f"{index}번 쌍이 {len(input_ids)}토큰으로 max_seq_len({self.max_length})보다 길다")
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def epoch_indices(n, seed, epoch):
    """원본 main과 같은 데이터 순서: epoch마다 시드를 다시 정하고 섞는다(두 vocab의 쌍 수가 같아 순서도 같다)."""
    setup_seed(seed + epoch)
    return torch.randperm(n).tolist()


def check_first_batch(ds, tokenizer, indices, batch_size):
    """첫 배치의 채점 위치가 답 본문 + <|im_end|> + 줄바꿈뿐인지 확인한다(질문·형식 토큰·패딩은 빠져야 한다)."""
    for i in indices[:batch_size]:
        input_ids, labels = ds[i]
        scored = [int(t) for t, lab in zip(input_ids, labels) if int(lab) != -100]
        want = tokenizer(ds.samples[i]["a"] + f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        if scored != want:
            raise SystemExit(f"{i}번 쌍의 채점 위치가 답 본문 + 끝 표시와 다르다: {tokenizer.decode(scored)!r}")
    example = ds.samples[indices[0]]
    Logger(f"첫 배치 채점 위치 확인: {batch_size}쌍 모두 답 본문 + {tokenizer.eos_token} + 줄바꿈만 채점. "
           f"예: {example['q']!r} → {example['a']!r}")


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind 한국어 대화 SFT")
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--init_weight", required=True, help="시작 가중치 .pth (사전학습 결과)")
    p.add_argument("--data_path", default="data/chat/sft_pool.jsonl")
    p.add_argument("--out_dir", default="out_sft")
    p.add_argument("--ckpt_dir", default="", help="체크포인트 폴더 (비우면 out_sft/checkpoints_ko<vocab>)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--accumulation_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50)
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
    vocab_size = len(tokenizer)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               vocab_size=vocab_size)
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(torch.load(os.path.join(ROOT, args.init_weight), map_location="cpu"), strict=True)
    model = model.to(args.device)
    Logger(f"tokenizer: {args.tokenizer_dir} | vocab_size: {vocab_size} | init: {args.init_weight}")

    with open(os.path.join(ROOT, args.data_path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    ds = ChatSFTDataset(rows, tokenizer, args.max_seq_len)
    longest = max(len(tokenizer(chat_text(r["q"], r["a"]), add_special_tokens=False).input_ids) for r in rows)
    if longest > args.max_seq_len:
        raise SystemExit(f"가장 긴 쌍이 {longest}토큰으로 max_seq_len({args.max_seq_len})보다 길다")
    Logger(f"학습 쌍 {len(rows):,}개, 가장 긴 쌍 {longest}토큰 (max_seq_len {args.max_seq_len})")

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(ROOT, args.ckpt_dir or f"{args.out_dir}/checkpoints_ko{vocab_size}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 원본 train_epoch()가 참조하는 모듈 전역값을 채운다
    args.save_dir = out_dir
    args.save_weight = f"sft_ko{vocab_size}"
    base.args = args
    base.lm_config = lm_config
    base.model = model
    base.optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    base.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    base.autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)
    # 원본 train_epoch()는 체크포인트를 '../checkpoints'(현재 폴더 기준)에 저장하므로 실행마다 다른 폴더로 바꾼다
    original_lm_checkpoint = base.lm_checkpoint
    base.lm_checkpoint = lambda *a, **kw: original_lm_checkpoint(*a, **{**kw, "save_dir": ckpt_dir})

    for epoch in range(args.epochs):
        indices = epoch_indices(len(ds), args.seed, epoch)
        if epoch == 0:
            check_first_batch(ds, tokenizer, indices, args.batch_size)
        loader = DataLoader(ds, batch_sampler=SkipBatchSampler(indices, args.batch_size, 0),
                            num_workers=args.num_workers, pin_memory=True)
        base.train_epoch(epoch, loader, len(loader), 0, None)
    Logger(f"완료: {os.path.join(out_dir, f'{args.save_weight}_{args.hidden_size}.pth')}")


if __name__ == "__main__":
    main()
