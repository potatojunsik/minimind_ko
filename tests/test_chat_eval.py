"""대화체 평가(prepare_chat_eval.py, eval_chat_bpb.py) 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_chat_eval -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import math  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from eval_chat_bpb import build_ids, check_same_eval, compare, pair_row, paired_delta, score_answer  # noqa: E402
from prepare_chat_eval import group_ids, load_rows, normalize, split  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_csv(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="")
    f.write(text)
    f.close()
    return f.name


class LoadRowsTest(unittest.TestCase):
    def test_clean_rows(self):
        # BOM, 라벨 뒤 공백, 완전히 같은 쌍(두 번째는 버림), 따옴표 안 쉼표
        path = write_csv("\ufeffQ,A,label\n안녕,반가워요.,0\n안녕,반가워요.,0\n뭐해,\"그냥, 쉬어요.\",2   \n")
        try:
            rows = load_rows(path)
        finally:
            os.unlink(path)
        self.assertEqual(rows, [{"q": "안녕", "a": "반가워요.", "label": "0"},
                                {"q": "뭐해", "a": "그냥, 쉬어요.", "label": "2"}])


class GroupTest(unittest.TestCase):
    def test_groups_merge_on_shared_q_or_a(self):
        rows = [{"q": "A", "a": "x", "label": "0"},
                {"q": "B", "a": "x", "label": "0"},   # 0과 답 공유
                {"q": "B", "a": "y", "label": "0"},   # 1과 질문 공유 → 0·1·2가 한 묶음
                {"q": "C", "a": "z", "label": "0"}]
        g = group_ids(rows)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[1], g[2])
        self.assertNotEqual(g[0], g[3])

    def test_groups_merge_on_near_duplicates(self):
        # 띄어쓰기·문장부호만 다른 답과 질문도 같은 묶음 (평가용과 SFT 후보에 사실상 같은 글이 갈라지지 않게)
        rows = [{"q": "잠이 안 와", "a": "아닐거예요.", "label": "0"},
                {"q": "잠이 안와", "a": "따뜻한 우유 드세요", "label": "0"},
                {"q": "시험 망했어", "a": "아닐 거예요!", "label": "0"},
                {"q": "배고파", "a": "밥 드세요.", "label": "0"}]
        g = group_ids(rows)
        self.assertEqual(g[0], g[1])  # 질문이 띄어쓰기만 다름
        self.assertEqual(g[0], g[2])  # 답이 띄어쓰기·문장부호만 다름
        self.assertNotEqual(g[0], g[3])

    def test_normalize(self):
        self.assertEqual(normalize("아닐 거예요!"), normalize("아닐거예요."))
        self.assertEqual(normalize("그럴만 해요~"), "그럴만해요")
        self.assertEqual(normalize("...!"), "")


class SplitTest(unittest.TestCase):
    def make_rows(self):
        rows = []
        for i in range(60):
            rows.append({"q": f"q{i}", "a": f"a{i // 3}", "label": "0"})  # 답을 3쌍씩 공유
        return rows

    def test_no_shared_text_across_split(self):
        rows = self.make_rows()
        ev, rest = split(rows, group_ids(rows), target=10, seed=1)
        self.assertTrue(set(rows[i]["q"] for i in ev).isdisjoint(rows[i]["q"] for i in rest))
        self.assertTrue(set(rows[i]["a"] for i in ev).isdisjoint(rows[i]["a"] for i in rest))
        self.assertEqual(sorted(ev + rest), list(range(len(rows))))

    def test_split_keeps_whole_groups(self):
        rows = self.make_rows()
        groups = group_ids(rows)
        ev, rest = split(rows, groups, target=10, seed=1)
        self.assertGreaterEqual(len(ev), 10)
        self.assertEqual(len(ev) % 3, 0)
        self.assertTrue({groups[i] for i in ev}.isdisjoint(groups[i] for i in rest))

    def test_split_is_deterministic(self):
        rows = self.make_rows()
        groups = group_ids(rows)
        self.assertEqual(split(rows, groups, target=10, seed=7), split(rows, groups, target=10, seed=7))
        self.assertNotEqual(split(rows, groups, target=10, seed=7)[0], split(rows, groups, target=10, seed=8)[0])


class ZeroLogitModel:
    """모든 로짓이 0인 가짜 모델. 어떤 토큰이든 NLL = ln(vocab)이 된다."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, x):
        return SimpleNamespace(logits=torch.zeros(1, x.shape[1], self.vocab))


class PositionModel:
    """위치 t에서 정답 토큰(x[t+1])의 로짓만 t로 두는 가짜 모델. 위치마다 NLL이 달라진다."""

    def __init__(self, vocab):
        self.vocab = vocab

    def nll_at(self, t):
        return math.log(math.exp(t) + self.vocab - 1) - t

    def __call__(self, x):
        logits = torch.zeros(1, x.shape[1], self.vocab)
        for t in range(x.shape[1] - 1):
            logits[0, t, x[0, t + 1]] = float(t)
        return SimpleNamespace(logits=logits)


class AnswerScoringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizers/ko_6400"))

    def test_build_ids_boundary(self):
        q, a = "오늘 뭐 먹지?", "맛있는 거 드세요."
        ids, start = build_ids(self.tok, q, a)
        self.assertEqual(ids[0], self.tok.bos_token_id)
        self.assertEqual(ids[-1], self.tok.eos_token_id)
        self.assertEqual(ids[start:-1], self.tok(a, add_special_tokens=False).input_ids)
        q_len = len(self.tok(q, add_special_tokens=False).input_ids)
        self.assertEqual(ids[1:1 + q_len], self.tok(q, add_special_tokens=False).input_ids)
        # 질문과 답 사이는 줄바꿈 토큰이다(위키 학습 조각에서 문단을 잇는 것과 같은 모양)
        self.assertEqual(ids[1 + q_len:start], self.tok("\n", add_special_tokens=False).input_ids)

    def test_answer_positions_only(self):
        model = ZeroLogitModel(len(self.tok))
        a = "맛있는 거 드세요."
        n_answer = len(self.tok(a, add_special_tokens=False).input_ids)
        for q in ("배고파", "오늘 점심은 무엇을 먹으면 좋을지 정말 모르겠어요"):
            ids, start = build_ids(self.tok, q, a)
            nll = score_answer(model, ids, start, "cpu")
            self.assertEqual(len(nll), n_answer + 1)  # 답 토큰 + eos
            self.assertAlmostEqual(sum(nll), (n_answer + 1) * math.log(len(self.tok)), places=4)

    def test_answer_positions_exact(self):
        # 위치마다 손실이 다른 가짜 모델로, 채점한 개수가 아니라 '어느 위치'를 채점했는지 확인한다
        model = PositionModel(len(self.tok))
        ids, start = build_ids(self.tok, "오늘 점심 뭐 먹지", "맛있는 거 드세요.")
        expected = [model.nll_at(t) for t in range(start - 1, len(ids) - 1)]
        nll = score_answer(model, ids, start, "cpu")
        self.assertEqual(len(nll), len(expected))
        for got, want in zip(nll, expected):
            self.assertAlmostEqual(got, want, places=4)
        # 마지막 값은 eos를 예측한 위치(len(ids) - 2)의 손실이다
        self.assertAlmostEqual(nll[-1], model.nll_at(len(ids) - 2), places=4)

    def test_pair_row(self):
        p = {"q": "배고파", "a": "밥 드세요.", "label": "0", "group": 3}
        row = pair_row(p, [1.0, 2.0, 3.0, 0.5])
        self.assertEqual(row, {"group": 3, "label": "0", "chars": 6, "bytes": len("밥 드세요.".encode("utf-8")),
                               "tokens": 3, "nll": 6.5, "nll_eos": 0.5})

    def test_too_long_pair_raises(self):
        ids, start = build_ids(self.tok, "가나다라 " * 400, "네.")
        with self.assertRaises(ValueError):
            score_answer(ZeroLogitModel(len(self.tok)), ids, start, "cpu")


class CompareTest(unittest.TestCase):
    def rows(self, nlls):
        return [{"nll": n, "nll_eos": 0.0, "bytes": 10, "chars": 5} for n in nlls]

    def test_bootstrap_by_group(self):
        a, b = self.rows([7.0, 3.0, 9.0, 1.0]), self.rows([5.0, 4.0, 6.0, 2.0])
        # 모든 쌍이 한 묶음이면 다시 뽑을 단위가 하나라 구간이 점추정 하나로 줄어든다(쌍 단위로 뽑으면 넓어진다)
        one = paired_delta(a, b, [0, 0, 0, 0], rounds=200, seed=0)
        self.assertAlmostEqual(one["ci"][0], one["delta"])
        self.assertAlmostEqual(one["ci"][1], one["delta"])
        each = paired_delta(a, b, [0, 1, 2, 3], rounds=200, seed=0)
        self.assertAlmostEqual(each["delta"], one["delta"])
        self.assertLess(each["ci"][0], each["ci"][1])
        self.assertAlmostEqual(one["delta"], (20.0 - 17.0) / math.log(2) / 40)

    def test_compare_aggregates(self):
        def result(vocab, pairs):
            rows = [{"group": g, "label": "0", "chars": 2, "bytes": b, "tokens": 1, "nll": n, "nll_eos": e}
                    for g, b, n, e in pairs]
            stats = {"q_chars_per_token": 1.0, "a_chars_per_token": 1.0, "q_byte_fragment_share": 0.0,
                     "a_byte_fragment_share": 0.0, "max_ids_len": 5}
            return {"vocab": vocab, "sha256": "s", "pairs": rows, "tokenizer_stats": stats}
        # 쌍 0: 큰 vocab이 텍스트만으로는 이기고(1.0 < 2.0) eos 포함으로는 진다(1.0+3.0 > 2.0+0.5)
        small = result(6400, [(0, 10, 2.5, 0.5), (1, 20, 4.0, 1.0)])
        large = result(16000, [(0, 10, 4.0, 3.0), (1, 20, 3.0, 1.0)])
        c = compare(small, large, rounds=50, seed=0)
        ln2 = math.log(2)
        self.assertAlmostEqual(c["bpb"]["6400"], 6.5 / ln2 / 30)
        self.assertAlmostEqual(c["bpb"]["16000"], 7.0 / ln2 / 30)
        self.assertAlmostEqual(c["bpb_text_only"]["6400"], 5.0 / ln2 / 30)
        self.assertAlmostEqual(c["bpb_text_only"]["16000"], 3.0 / ln2 / 30)
        self.assertAlmostEqual(c["delta"]["delta"], (6.5 - 7.0) / ln2 / 30)
        self.assertAlmostEqual(c["delta_text_only"]["delta"], (5.0 - 3.0) / ln2 / 30)
        self.assertAlmostEqual(c["large_wins_pair"], 0.5)            # eos 포함: 쌍 1만 큰 vocab 승
        self.assertAlmostEqual(c["large_wins_pair_text_only"], 1.0)  # 텍스트만: 둘 다 큰 vocab 승
        self.assertAlmostEqual(c["eos_nll_mean"]["6400"], 0.75)
        self.assertAlmostEqual(c["eos_nll_mean"]["16000"], 2.0)

    def test_refuses_same_vocab(self):
        res = {"vocab": 6400, "sha256": "s", "pairs": [{"group": 0}]}
        with self.assertRaises(SystemExit):
            compare(res, dict(res), rounds=10, seed=0)

    def test_refuses_mismatched_eval(self):
        base = {"sha256": "x", "pairs": [{"group": 0}]}
        check_same_eval(base, dict(base))
        with self.assertRaises(SystemExit):
            check_same_eval(base, {"sha256": "y", "pairs": [{"group": 0}]})
        with self.assertRaises(SystemExit):
            check_same_eval(base, {"sha256": "x", "pairs": [{"group": 1}]})


if __name__ == "__main__":
    unittest.main()
