"""
여러 사전학습 모델에 같은 한국어 프롬프트를 주고 이어 쓰기 결과를 나란히 비교한다.
사전학습만 한 모델은 대화를 모르므로 질문 대신 문장 앞부분을 주고 뒤를 이어 쓰게 한다.

공정하게 비교하기 위해 모든 모델에 같은 시드와 샘플링 설정을 쓴다.
단, vocab이 다르면 토큰 하나의 길이가 달라서 '같은 토큰 수'는 '같은 글자 수'가 아니다.
그래서 넉넉히 생성한 뒤 출력을 MAX_CHARS 글자에서 잘라 길이를 맞춘다.

사용법 (minimind_ko 폴더에서 실행, --model 토크나이저경로 가중치경로 를 원하는 만큼 반복):
    .venv\\Scripts\\python.exe -X utf8 scripts\\compare_generation.py ^
        --model tokenizers/ko_6400 out_ko/pretrain_ko6400_768.pth ^
        --model tokenizers/ko_16000 out_ko/pretrain_ko16000_768.pth
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 8
PROMPTS = [
    "대한민국의 수도는",
    "인공지능은",
    "조선 시대에는",
    "오늘 날씨가",
]
# vocab 6,400은 토큰당 약 1.7글자라 120토큰이면 MAX_CHARS를 넘길 만큼 충분하다
MAX_NEW_TOKENS = 120
MAX_CHARS = 100
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50
# 같은 문장만 반복하는 현상을 조금 줄이되, 모델 차이를 가리지 않도록 약하게만 건다
REPETITION_PENALTY = 1.1


def parse_args():
    p = argparse.ArgumentParser(description="사전학습 모델 이어 쓰기 비교")
    p.add_argument("--model", nargs=2, action="append", required=True, metavar=("TOKENIZER_DIR", "WEIGHT"),
                   help="토크나이저 폴더와 가중치 .pth 쌍 (여러 번 지정)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=2, help="프롬프트마다 뽑을 샘플 수 (운에 덜 휘둘리도록)")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(tokenizer_dir, weight_path, device):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, tokenizer_dir))
    config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS, vocab_size=len(tokenizer))
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(os.path.join(ROOT, weight_path), map_location="cpu"), strict=True)
    return tokenizer, model.float().eval().to(device)


def generate(tokenizer, model, prompt, seed, device):
    x = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    # 모델마다, 프롬프트마다 시드를 다시 고정해야 앞선 생성이 뒤 결과에 영향을 주지 않는다
    torch.manual_seed(seed)
    out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=TEMPERATURE,
                         top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
                         eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.replace("\n", " ")[:MAX_CHARS]


def main():
    args = parse_args()
    models = [(tok_dir, *load_model(tok_dir, weight, args.device)) for tok_dir, weight in args.model]
    print(f"설정: seed {args.seed}, temperature {TEMPERATURE}, top_p {TOP_P}, top_k {TOP_K}, "
          f"repetition_penalty {REPETITION_PENALTY}, 출력 최대 {MAX_CHARS}자")
    for prompt in PROMPTS:
        print(f"\n### {prompt}")
        for tok_dir, tokenizer, model in models:
            for k in range(args.num_samples):
                text = generate(tokenizer, model, prompt, args.seed + k, args.device)
                print(f"- [{tok_dir} #{k + 1}] {prompt}{text}")


if __name__ == "__main__":
    main()
