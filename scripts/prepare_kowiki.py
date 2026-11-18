"""
한국어 위키백과(HuggingFace wikimedia/wikipedia, 20231101.ko)를 받아
MiniMind 학습 형식({"text": ...} jsonl)으로 변환한다.

긴 문서는 문단 단위로 묶어 MAX_CHARS 이하의 조각으로 나눈다.
(사전학습의 max_seq_len 안에 한 조각이 대부분 들어가도록 하기 위함)

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\prepare_kowiki.py data\\kowiki.jsonl
"""
import json
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "wikimedia/wikipedia"
PARQUET_FILES = [f"20231101.ko/train-0000{i}-of-00003.parquet" for i in range(3)]
DOWNLOAD_DIR = "data/raw"
MAX_CHARS = 800   # 한 조각의 최대 글자 수 (한국어 토크나이저 기준 대략 300~400 토큰 예상)
MIN_CHARS = 50    # 이보다 짧은 조각은 버린다 (목차, 빈 섹션 등)


def split_paragraphs(text):
    """문단을 MAX_CHARS 이하로 묶어 조각 목록을 만든다."""
    chunks, buf = [], ""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if buf and len(buf) + len(para) + 1 > MAX_CHARS:
            chunks.append(buf)
            buf = ""
        # 문단 하나가 MAX_CHARS보다 길면 그대로 잘라서 넣는다
        while len(para) > MAX_CHARS:
            chunks.append(para[:MAX_CHARS])
            para = para[MAX_CHARS:]
        buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHARS]


def main(out_path):
    articles, chunks_written, chars_written = 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for name in PARQUET_FILES:
            local = hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=DOWNLOAD_DIR)
            table = pq.read_table(local, columns=["text"])
            for text in table.column("text").to_pylist():
                articles += 1
                for chunk in split_paragraphs(text):
                    fout.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                    chunks_written += 1
                    chars_written += len(chunk)
            print(f"{name} 완료 | 누적 문서 {articles:,} | 조각 {chunks_written:,} | 글자 {chars_written:,}")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
