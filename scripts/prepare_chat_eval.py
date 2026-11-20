"""
공개 한국어 대화 데이터(songys/Chatbot_data, MIT)를 정리해 평가용 쌍과 나머지(SFT 후보)로 나눈다.

같은 답이 여러 질문에 반복되므로(최대 22번) 쌍 단위로 무작위로 나누면 같은 답이 평가용과 SFT 후보에 함께 들어간다.
그래서 같은 질문이나 같은 답을 공유하는 쌍을 한 묶음으로 묶고(공유가 이어지면 모두 한 묶음), 묶음째로 평가용을 뽑는다.
띄어쓰기·문장부호만 다른 질문이나 답('아닐거예요.' / '아닐 거예요!')도 같은 글로 보고 묶는다.
부트스트랩도 같은 묶음 단위로 한다(eval_chat_bpb.py).

입력:  data/chat/ChatbotData.csv  (https://github.com/songys/Chatbot_data 커밋 4cf20d1 고정, 열: Q, A, label)
출력:  data/chat/eval_pairs.jsonl, data/chat/sft_pool.jsonl  (줄마다 {"q", "a", "label", "group"})
       data/chat/split_report.json  (원본 sha256, 행 수, 제거한 중복, 묶음 수, 양쪽이 공유하는 질문·답 수, 평가 sha256)

사용법 (minimind_ko 폴더에서 실행):
    curl --create-dirs -L -o data\\chat\\ChatbotData.csv https://raw.githubusercontent.com/songys/Chatbot_data/4cf20d13fc46f5037fd1c531cd566e2dd9f72974/ChatbotData.csv
    .venv\\Scripts\\python.exe -X utf8 scripts\\prepare_chat_eval.py
"""
import argparse
import csv
import hashlib
import json
import os
import random
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_SEED = 20261120
TARGET_EVAL_PAIRS = 2000
# 이 기록에서 쓴 원본(커밋 4cf20d1의 ChatbotData.csv). 다르면 분할과 결과가 문서와 달라진다
EXPECTED_CSV_SHA256 = "287eb129695b577321c80ad397bb3c2279164d4ca577874d129fd3db5b30afe2"
# 유니코드 문장부호(P로 시작하는 범주) 말고도 구어체에서 문장부호처럼 쓰이는 기호
EXTRA_PUNCT = "~"


def normalize(text):
    """띄어쓰기·문장부호만 다른 글을 같게 보기 위한 키. NFKC 정규화 뒤 공백과 문장부호를 지운다."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text
                   if not ch.isspace() and not unicodedata.category(ch).startswith("P") and ch not in EXTRA_PUNCT)


def read_csv(path):
    # utf-8-sig: 파일 맨 앞에 BOM이 있으면 첫 열 이름에 붙지 않게 지운다(받은 파일에는 없지만 다시 저장한 파일 대비)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{"q": r["Q"], "a": r["A"], "label": r["label"].strip()} for r in csv.DictReader(f)]


def load_rows(path):
    """CSV를 읽어 {"q", "a", "label"} 목록으로. 라벨 앞뒤 공백을 지우고, 질문과 답이 같은 쌍은 처음 것만 남긴다."""
    rows, seen = [], set()
    for row in read_csv(path):
        key = (row["q"], row["a"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def label_conflicts(path):
    """질문과 답이 같은데 라벨이 다른 중복 행 수 (load_rows는 처음 행의 라벨을 남긴다)."""
    first, n = {}, 0
    for row in read_csv(path):
        key = (row["q"], row["a"])
        if key in first and first[key] != row["label"]:
            n += 1
        first.setdefault(key, row["label"])
    return n


def group_ids(rows):
    """같은 질문이나 같은 답을 공유하는 쌍을 한 묶음으로 (union-find). 띄어쓰기·문장부호만 다른 경우도 공유로 본다.
    묶음 번호는 묶음에서 처음 나온 행 번호."""
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first = {}
    for i, r in enumerate(rows):
        keys = [("q", r["q"]), ("a", r["a"])]
        for kind in ("q", "a"):
            norm = normalize(r[kind])
            # 문장부호만 있는 글은 정규화하면 빈 문자열이 되므로 키로 쓰지 않는다(서로 다른 쌍이 모두 한 묶음이 되지 않게)
            if norm:
                keys.append((kind + "_norm", norm))
        for key in keys:
            if key in first:
                a, b = find(i), find(first[key])
                parent[max(a, b)] = min(a, b)
            else:
                first[key] = i
    return [find(i) for i in range(len(rows))]


def split(rows, groups, target, seed):
    """묶음을 섞어 평가용 쌍이 target개 이상이 될 때까지 묶음째로 담는다. (평가 행 번호, 나머지 행 번호)."""
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    order = sorted(members)
    random.Random(seed).shuffle(order)
    eval_idx = []
    taken = set()
    for g in order:
        if len(eval_idx) >= target:
            break
        eval_idx.extend(members[g])
        taken.add(g)
    rest_idx = [i for i in range(len(rows)) if groups[i] not in taken]
    return sorted(eval_idx), rest_idx


def eval_sha256(rows):
    """평가한 텍스트가 같은지 확인하는 해시. 쌍마다 '질문\\t답'을 줄바꿈으로 이은 UTF-8."""
    return hashlib.sha256("\n".join(f"{r['q']}\t{r['a']}" for r in rows).encode("utf-8")).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def shared(eval_rows, rest_rows, kind, key=lambda s: s):
    return {key(r[kind]) for r in eval_rows} & {key(r[kind]) for r in rest_rows}


def main():
    p = argparse.ArgumentParser(description="대화 데이터 정리와 묶음 단위 평가용 분할")
    p.add_argument("--csv", default="data/chat/ChatbotData.csv")
    p.add_argument("--out_dir", default="data/chat")
    p.add_argument("--target", type=int, default=TARGET_EVAL_PAIRS)
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = p.parse_args()

    csv_path = os.path.join(ROOT, args.csv)
    with open(csv_path, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()
    if csv_sha != EXPECTED_CSV_SHA256:
        print(f"경고: 원본 CSV sha256이 이 기록과 다르다({csv_sha}). 분할과 결과가 docs/06과 달라질 수 있다")
    raw_rows = len(read_csv(csv_path))
    rows = load_rows(csv_path)
    groups = group_ids(rows)
    eval_idx, rest_idx = split(rows, groups, args.target, args.seed)
    eval_rows = [dict(rows[i], group=groups[i]) for i in eval_idx]
    rest_rows = [dict(rows[i], group=groups[i]) for i in rest_idx]

    checks = {
        "shared_questions": len(shared(eval_rows, rest_rows, "q")),
        "shared_answers": len(shared(eval_rows, rest_rows, "a")),
        "shared_questions_normalized": len(shared(eval_rows, rest_rows, "q", normalize) - {""}),
        "shared_answers_normalized": len(shared(eval_rows, rest_rows, "a", normalize) - {""}),
    }
    if any(checks.values()):
        raise SystemExit(f"평가용과 나머지가 글을 공유한다: {checks}")

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "eval_pairs.jsonl"), eval_rows)
    write_jsonl(os.path.join(out_dir, "sft_pool.jsonl"), rest_rows)
    report = {
        "source_csv_sha256": csv_sha,
        "source_rows": raw_rows,
        "duplicate_pairs_removed": raw_rows - len(rows),
        "duplicate_label_conflicts": label_conflicts(csv_path),
        "pairs": len(rows),
        "groups": len(set(groups)),
        "eval_pairs": len(eval_rows),
        "eval_groups": len({r["group"] for r in eval_rows}),
        "sft_pool_pairs": len(rest_rows),
        **checks,
        "eval_labels": {k: sum(1 for r in eval_rows if r["label"] == k) for k in sorted({r["label"] for r in rows})},
        "eval_sha256": eval_sha256(eval_rows),
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
