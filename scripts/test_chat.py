"""
transformers 형식 MiniMind 모델의 토크나이저 효율과 대화 응답을 빠르게 확인한다.

사용법 (minimind 폴더에서 실행):
    ..\\.venv\\Scripts\\python.exe -X utf8 ..\\scripts\\test_chat.py ./minimind-3

Windows 콘솔(cp949)에서 한글 출력 오류를 막기 위해 -X utf8 옵션을 붙인다.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENIZE_SAMPLES = [
    "오늘 날씨가 정말 좋네요.",
    "今天天气真好。",
    "The weather is really nice today.",
]
CHAT_QUESTIONS = [
    "안녕하세요, 자기소개 해주세요.",
    "你好，请介绍一下你自己。",
    "What is the capital of France?",
]
MAX_NEW_TOKENS = 80


def main(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"vocab size: {tok.vocab_size} | params: {params_m:.1f}M | device: {device}")

    # 글자 수 대비 토큰 수가 많을수록 그 언어를 비효율적으로(잘게) 쪼갠다는 뜻
    print("\n[토크나이저 효율]")
    for s in TOKENIZE_SAMPLES:
        ids = tok.encode(s)
        print(f"  토큰 {len(ids):2d} / 글자 {len(s):2d} : {s}")

    print("\n[대화 테스트]")
    torch.manual_seed(0)
    for q in CHAT_QUESTIONS:
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        # MiniMind 모델은 token_type_ids를 받지 않으므로 생성하지 않는다
        x = tok(text, return_tensors="pt", return_token_type_ids=False).to(device)
        out = model.generate(**x, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9)
        answer = tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\nQ: {q}\nA: {answer}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
