"""
eval_bpc.py --per_doc_out으로 저장한 두 모델의 문서별 결과를 짝지어 비교한다.
  - 문서별 우세: 같은 문서에서 NLL이 더 낮은 쪽을 센다 (글자 수가 같으므로 NLL 비교 = BPC 비교)
  - 부트스트랩: 문서를 복원추출로 다시 뽑아 BPC 차이(B - A)의 95% 구간을 구한다

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_bpc.py --tokenizer_dir tokenizers/ko_6400 --weight out_ko/pretrain_ko6400_768.pth --per_doc_out doc_6400.jsonl
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_bpc.py --tokenizer_dir tokenizers/ko_16000 --weight out_ko/pretrain_ko16000_768.pth --per_doc_out doc_16000.jsonl
    .venv\\Scripts\\python.exe -X utf8 scripts\\paired_compare.py doc_6400.jsonl doc_16000.jsonl
"""
import json
import math
import random
import sys

BOOTSTRAP_ROUNDS = 2000
SEED = 0


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def bpc(rows, idx):
    return sum(rows[i]["nll"] for i in idx) / math.log(2) / sum(rows[i]["chars"] for i in idx)


def main(path_a, path_b):
    a, b = load(path_a), load(path_b)
    if len(a) != len(b) or any(x["chars"] != y["chars"] for x, y in zip(a, b)):
        raise ValueError("두 파일의 문서 순서/글자 수가 다름 (같은 검증 데이터로 평가했는지 확인)")
    n = len(a)
    all_idx = range(n)
    a_wins = sum(1 for x, y in zip(a, b) if x["nll"] < y["nll"])
    ties = sum(1 for x, y in zip(a, b) if x["nll"] == y["nll"])
    print(f"BPC  A: {bpc(a, all_idx):.4f}  B: {bpc(b, all_idx):.4f}  차이(B-A): {bpc(b, all_idx) - bpc(a, all_idx):.4f}")
    print(f"문서별 A 우세: {a_wins}/{n} ({a_wins / n:.1%}), 동점 {ties}")

    rng = random.Random(SEED)
    diffs = []
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(bpc(b, idx) - bpc(a, idx))
    diffs.sort()
    lo, hi = diffs[int(0.025 * BOOTSTRAP_ROUNDS)], diffs[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
    print(f"부트스트랩 {BOOTSTRAP_ROUNDS}회 (seed {SEED}) BPC 차이 95% 구간: {lo:.4f} ~ {hi:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
