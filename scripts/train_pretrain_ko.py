"""
한국어 토크나이저로 MiniMind를 처음부터 사전학습한다.

원본 minimind/trainer/train_pretrain.py의 학습 루프(train_epoch)를 그대로 재사용하고,
원본에서 고정돼 있던 두 가지만 바꾼다.
  1. 토크나이저 경로 (원본: ../model 고정)
  2. 모델 vocab 크기 (원본: MiniMindConfig 기본값 6400 고정) → 토크나이저 크기에 맞춤

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\train_pretrain_ko.py ^
        --tokenizer_dir tokenizers/ko_16000 --data_path data/kowiki_train_100k.jsonl
결과: out_ko/pretrain_ko<vocab>_768.pth
"""
import argparse
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

import trainer.train_pretrain as base  # noqa: E402
from dataset.lm_dataset import PretrainDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402
from trainer.trainer_utils import Logger, SkipBatchSampler, get_model_params, setup_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="MiniMind 한국어 사전학습")
    p.add_argument("--tokenizer_dir", required=True, help="토크나이저 폴더 (예: tokenizers/ko_16000)")
    p.add_argument("--data_path", required=True, help="학습 데이터 jsonl ({\"text\": ...})")
    p.add_argument("--out_dir", default="out_ko", help="가중치 저장 폴더")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    # 위키 800자 조각 기준 ko_6400은 p95 501토큰, ko_16000은 p95 445토큰이라 두 쪽 모두 거의 안 잘리도록 512
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    # 상대 경로는 minimind_ko 폴더 기준으로 해석한다
    tokenizer_dir = os.path.join(ROOT, args.tokenizer_dir)
    data_path = os.path.join(ROOT, args.data_path)
    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    vocab_size = len(tokenizer)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               vocab_size=vocab_size)
    model = MiniMindForCausalLM(lm_config).to(args.device)
    get_model_params(model, lm_config)
    Logger(f"tokenizer: {args.tokenizer_dir} | vocab_size: {vocab_size} | "
           f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if "cpu" in args.device else torch.cuda.amp.autocast(dtype=dtype)
    train_ds = PretrainDataset(data_path, tokenizer, max_length=args.max_seq_len)

    # 원본 train_epoch()가 참조하는 모듈 전역값을 채운다
    args.save_dir = out_dir
    args.save_weight = f"pretrain_ko{vocab_size}"
    base.args = args
    base.lm_config = lm_config
    base.model = model
    base.optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    base.scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    base.autocast_ctx = autocast_ctx

    # 원본 train_epoch()는 체크포인트를 '../checkpoints'(현재 폴더 기준)에 저장하므로
    # minimind/trainer에서 실행하는 것과 같은 위치(minimind/checkpoints)가 되도록 이동한다
    os.chdir(os.path.join(MINIMIND_DIR, "trainer"))

    for epoch in range(args.epochs):
        setup_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args.batch_size, 0),
                            num_workers=args.num_workers, pin_memory=True)
        base.train_epoch(epoch, loader, len(loader), 0, None)
    Logger(f"완료: {os.path.join(out_dir, f'{args.save_weight}_{args.hidden_size}.pth')}")


if __name__ == "__main__":
    main()
