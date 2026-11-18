"""
큰 jsonl 데이터셋에서 일부 줄만 뽑아 작은 데이터셋을 만든다.
stride/offset을 주면 i % stride == offset 인 줄만 뽑는다 (offset을 다르게 주면 서로 겹치지 않는 세트를 만들 수 있다).
지정한 줄 수를 채우면 멈추므로, 뽑히는 범위는 파일 앞쪽에 치우칠 수 있다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe scripts\\make_subset.py <원본.jsonl> <출력.jsonl> <줄 수> [stride=1] [offset=0]
예:
    .venv\\Scripts\\python.exe scripts\\make_subset.py minimind/dataset/pretrain_t2t_mini.jsonl minimind/dataset/pretrain_small_20k.jsonl 20000
    .venv\\Scripts\\python.exe scripts\\make_subset.py data/kowiki.jsonl data/kowiki_train_100k.jsonl 100000 11 0
    .venv\\Scripts\\python.exe scripts\\make_subset.py data/kowiki.jsonl data/kowiki_heldout_2k.jsonl 2000 11 5
"""
import sys


def main(src, dst, line_count, stride=1, offset=0):
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError(f"stride는 1 이상, offset은 0 이상 stride 미만이어야 함 (stride={stride}, offset={offset})")
    written = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if written >= line_count:
                break
            if i % stride == offset:
                fout.write(line)
                written += 1
    print(f"{written}줄 저장: {dst} (stride {stride}, offset {offset})")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6):
        print(__doc__)
        sys.exit(1)
    extra = [int(v) for v in sys.argv[4:]]
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), *extra)
