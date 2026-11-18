"""
여러 토크나이저의 압축률(글자 수 / 토큰 수)과 쪼개는 방식을 비교한다.
압축률이 높을수록 같은 문장을 더 적은 토큰으로 표현한다 = 학습/추론 효율이 좋다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\compare_tokenizers.py minimind/model tokenizers/ko_6400 tokenizers/ko_16000
"""
import sys

from transformers import AutoTokenizer

SAMPLES = {
    "한국어(일상)": "오늘 날씨가 정말 좋네요.",
    "한국어(설명)": "인공지능은 인간의 학습 능력, 추론 능력, 지각 능력을 인공적으로 구현하려는 컴퓨터 과학의 세부 분야 중 하나이다. "
                 "대규모 언어 모델은 방대한 텍스트 데이터로 학습하여 사람처럼 문장을 이해하고 생성한다.",
    "한국어(대화)": "안녕하세요, 저는 서울에 살고 있는 회사원입니다. 주말에는 가족과 함께 공원에서 산책하는 것을 좋아해요.",
    "중국어": "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
    "영어": "Large language models are trained on vast amounts of text data.",
}
DETAIL_SAMPLE = "오늘 날씨가 정말 좋네요."


def main(tokenizer_dirs):
    toks = {d: AutoTokenizer.from_pretrained(d) for d in tokenizer_dirs}

    header = "| 문장 | 글자 | " + " | ".join(f"{d} (vocab {len(t)})" for d, t in toks.items()) + " |"
    print(header)
    print("|" + "---|" * (2 + len(toks)))
    for name, text in SAMPLES.items():
        cells = []
        for t in toks.values():
            n = len(t.encode(text))
            cells.append(f"{n}토큰 ({len(text) / n:.2f})")
        print(f"| {name} | {len(text)} | " + " | ".join(cells) + " |")

    print(f"\n[쪼개진 모양] {DETAIL_SAMPLE}")
    for d, t in toks.items():
        ids = t.encode(DETAIL_SAMPLE)
        pieces = [t.decode([i]) for i in ids]
        print(f"  {d}: {' | '.join(pieces)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
