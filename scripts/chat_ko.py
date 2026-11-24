"""
한국어 SFT 모델과 콘솔에서 대화한다. 질문을 입력하면 모델이 답한다(빈 줄이나 q를 입력하면 끝).

생성 설정은 docs/08 데모·07번 블라인드 비교와 같다(blind_chat_eval.generate_one): 필터 없는 greedy, 새 토큰 상한 96,
<|im_end|>에서 멈춤. 같은 질문에는 항상 같은 답이 나온다. 상한에서 잘린 답에는 "[상한에서 잘림]"을 붙인다.
대화 기록은 이어 붙이지 않는다(SFT를 질문 하나·답 하나로만 했기 때문에 매번 새 질문으로 묻는다).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\chat_ko.py
    고정 질문을 차례로 넣기(README 화면): .venv\\Scripts\\python.exe -X utf8 scripts\\chat_ko.py --questions docs/08_demo_questions.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blind_chat_eval import TRUNC_MARK, generate_one  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def answer_text(o):
    """generate_one 결과를 화면·문서에 싣는 글로. 스스로 끝내지 못하고 상한에서 잘린 답에는 표시를 붙인다."""
    return o["text"] + ("" if o["stopped"] else f" {TRUNC_MARK}")


def answer(model, tokenizer, q, device):
    return answer_text(generate_one(model, tokenizer, q, device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_dir", default="tokenizers/ko_16000")
    p.add_argument("--weight", default="out_final/sft/sft_ko16000_768.pth")
    p.add_argument("--questions", default="", help="final_demo.py select가 만든 질문 파일. 주면 그 질문을 차례로 넣는다")
    args = p.parse_args()

    import torch
    from eval_bpc import load_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.tokenizer_dir, args.weight, device)
    print(f"MiniMind 한국어 대화 ({args.weight}, greedy). 끝내려면 빈 줄이나 q를 입력하세요.\n")

    if args.questions:
        with open(os.path.join(ROOT, args.questions), encoding="utf-8") as f:
            for it in json.load(f)["items"]:
                print(f"나> {it['q']}")
                print(f"모델> {answer(model, tokenizer, it['q'], device)}\n")
        return
    while True:
        try:
            q = input("나> ").strip()
        except EOFError:
            break
        if not q or q.lower() == "q":
            break
        print(f"모델> {answer(model, tokenizer, q, device)}\n")


if __name__ == "__main__":
    main()
