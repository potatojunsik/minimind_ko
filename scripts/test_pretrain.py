"""
사전학습(pretrain)으로 만든 .pth 가중치를 불러와 '문장 이어 쓰기'를 확인한다.
사전학습만 한 모델은 대화 형식을 모르므로 질문이 아니라 문장 앞부분을 주고 뒤를 이어 쓰게 한다.

사용법 (minimind 폴더에서 실행):
    ..\\.venv\\Scripts\\python.exe -X utf8 ..\\scripts\\test_pretrain.py out_test/pretrain_768.pth
"""
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.append(os.getcwd())  # minimind 폴더의 model 패키지를 불러오기 위함
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

TOKENIZER_DIR = "model"  # 원본 저장소에 포함된 기본 토크나이저
PROMPTS = [
    "秋天的早晨，",
    "人工智能是",
    "我最喜欢的食物是",
    "오늘 날씨가",
]
MAX_NEW_TOKENS = 60


def main(weight_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = MiniMindForCausalLM(MiniMindConfig())
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    # fp16은 GPU에서만 빠르다. CPU에서는 fp16 연산이 느리거나 지원되지 않으므로 fp32로 둔다
    if device == "cuda":
        model = model.half()
    model = model.eval().to(device)

    torch.manual_seed(0)
    for p in PROMPTS:
        x = tok(tok.bos_token + p, return_tensors="pt", return_token_type_ids=False).to(device)
        out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.8, top_p=0.9,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).replace("\n", " ")
        print(f"\n[입력] {p}\n[출력] {p}{text}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
