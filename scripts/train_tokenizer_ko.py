"""
한국어 데이터로 MiniMind 형식 토크나이저(Byte-level BPE)를 학습한다.

원본 minimind/trainer/train_tokenizer.py의 train_tokenizer()를 그대로 불러 쓰므로
특수 토큰, chat_template 등은 원본과 동일하고 어휘(vocab)만 한국어 기준으로 바뀐다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\train_tokenizer_ko.py data\\kowiki.jsonl 6400
결과: tokenizers\\ko_<vocab_size>\\ (tokenizer.json, tokenizer_config.json 등)
"""
import os
import sys
import time

sys.path.append(os.path.abspath("minimind"))  # 원본 저장소의 trainer 패키지를 불러오기 위함
from trainer.train_tokenizer import train_tokenizer  # noqa: E402

SAMPLE_LINES = 200_000                     # 토크나이저 학습에 쓸 최대 줄 수 (전체를 쓰면 RAM/시간이 과도함)
SAMPLE_PATH = "data/tokenizer_sample.jsonl"
OUTPUT_ROOT = "tokenizers"


def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def make_sample(src, dst, sample_lines):
    """문서가 특정 주제에 몰리지 않도록 일정 간격(step)으로 뽑는다.

    step은 내림(total // sample_lines)이고 sample_lines를 채우면 멈추므로,
    파일 끝부분 일부는 뽑히지 않는다 (kowiki 기준 step 5, 0~999,995번째 줄에서만 뽑힘).
    이미 공개한 토크나이저를 똑같이 재현할 수 있도록 이 동작은 바꾸지 않았다.
    """
    total = count_lines(src)
    step = max(total // sample_lines, 1)
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i % step == 0 and written < sample_lines:
                fout.write(line)
                written += 1
    print(f"샘플 {written:,}줄 / 전체 {total:,}줄 (간격 {step})")


def main(data_path, vocab_size):
    if not os.path.exists(SAMPLE_PATH):
        make_sample(data_path, SAMPLE_PATH, SAMPLE_LINES)
    out_dir = os.path.join(OUTPUT_ROOT, f"ko_{vocab_size}")
    start = time.time()
    train_tokenizer(SAMPLE_PATH, out_dir, vocab_size)
    print(f"vocab {vocab_size} 학습 완료: {out_dir} ({time.time() - start:.0f}초)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
