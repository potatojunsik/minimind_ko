"""대화 SFT 비교(train_sft_ko.py, eval_chat_bpb.py --format chat, blind_chat_eval.py) 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_chat_sft -v
"""
import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from blind_chat_eval import decide, generate_one, make_sheet, outcomes, score_from_files, select_items  # noqa: E402
from eval_chat_bpb import build_chat_ids, check_same_eval  # noqa: E402
from train_sft_ko import ChatSFTDataset, chat_text, check_first_batch, epoch_indices  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCABS = (6400, 16000)
SAMPLE = [{"q": "배고파", "a": "밥 드세요."}, {"q": "오늘 너무 힘들었어 ㅠㅠ", "a": "고생 많았어요. 푹 쉬세요!"},
          {"q": "PPL 심하네", "a": "눈살이 찌푸려지죠."}]


def tokenizers():
    return {v: AutoTokenizer.from_pretrained(os.path.join(ROOT, f"tokenizers/ko_{v}")) for v in VOCABS}


class ChatFormatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toks = tokenizers()

    def test_chat_text(self):
        self.assertEqual(chat_text("배고파", "밥 드세요."),
                         "<|im_start|>user\n배고파<|im_end|>\n<|im_start|>assistant\n밥 드세요.<|im_end|>\n")

    def test_labels_answer_only(self):
        # 채점(label != -100) 위치는 답 본문 + <|im_end|> + 줄바꿈뿐이다. 질문·형식 토큰·패딩은 빠진다
        for v, tok in self.toks.items():
            ds = ChatSFTDataset(SAMPLE, tok, max_length=64)
            for i, pair in enumerate(SAMPLE):
                input_ids, labels = ds[i]
                scored = [int(t) for t, lab in zip(input_ids, labels) if int(lab) != -100]
                want = tok(pair["a"] + "<|im_end|>\n", add_special_tokens=False).input_ids
                self.assertEqual(scored, want, f"vocab {v}, 쌍 {i}")
                self.assertEqual(int(labels[-1]), -100)  # 패딩
                self.assertEqual(len(input_ids), 64)

    def test_too_long_raises(self):
        tok = self.toks[6400]
        ds = ChatSFTDataset([{"q": "가나다라 " * 40, "a": "네."}], tok, max_length=32)
        with self.assertRaises(ValueError):
            ds[0]

    def test_chat_ids_match_joint(self):
        # 평가 입력(따로 토큰화해 이음)이 학습 때처럼 한 번에 토큰화한 결과와 같아야 한다(끝 줄바꿈은 평가에서 뺌)
        for v, tok in self.toks.items():
            for pair in SAMPLE:
                ids, start = build_chat_ids(tok, pair["q"], pair["a"])
                joint = tok(chat_text(pair["q"], pair["a"]), add_special_tokens=False).input_ids
                self.assertEqual(ids, joint[:len(ids)], f"vocab {v}")
                self.assertEqual(ids[-1], tok.eos_token_id)
                self.assertEqual(ids[start:-1], tok(pair["a"], add_special_tokens=False).input_ids)


class FullEvalInputTest(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data/chat/eval_pairs.jsonl")), "평가 데이터 없음")
    def test_all_eval_pairs_match_joint(self):
        # 설계 4-1: 평가 입력이 형식 전체를 한 번에 토큰화한 결과와 같은지 평가용 쌍 전부에서 확인한다
        import json
        with open(os.path.join(ROOT, "data/chat/eval_pairs.jsonl"), encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
        for v, tok in tokenizers().items():
            bad = 0
            for p in pairs:
                ids, _ = build_chat_ids(tok, p["q"], p["a"])
                bad += ids != tok(chat_text(p["q"], p["a"]), add_special_tokens=False).input_ids[:len(ids)]
            self.assertEqual(bad, 0, f"vocab {v}")


class BlindTest(unittest.TestCase):
    def pairs(self):
        rows = []
        for g in range(30):
            for k in range(3 if g % 2 else 1):
                rows.append({"q": f"q{g}-{k}", "a": f"a{g}-{k}", "group": g * 10, "label": "0"})
        return rows

    def test_select_items(self):
        items = select_items(self.pairs(), n=10, seed=5)
        self.assertEqual(len(items), 10)
        self.assertEqual(len({it["group"] for it in items}), 10)  # 묶음마다 하나
        self.assertEqual(items, select_items(self.pairs(), n=10, seed=5))
        self.assertNotEqual(items, select_items(self.pairs(), n=10, seed=6))
        self.assertEqual([it["id"] for it in items], list(range(1, 11)))
        # 쌍이 여러 개인 묶음에서는 질문도 시드로 고른다(항상 첫 쌍이 아니다)
        firsts = {}
        for p in self.pairs():
            firsts.setdefault(p["group"], p["q"])
        picked = [it for s in range(20) for it in select_items(self.pairs(), n=10, seed=s) if it["group"] % 20 == 10]
        self.assertTrue(any(it["q"] != firsts[it["group"]] for it in picked))

    def test_blind_roundtrip(self):
        items = select_items(self.pairs(), n=12, seed=1)
        outs = {"6400": {it["id"]: {"text": f"작은{it['id']}", "stopped": True} for it in items},
                "16000": {it["id"]: {"text": f"큰{it['id']}", "stopped": it["id"] % 3 != 0} for it in items}}
        rows, key = make_sheet(items, outs, seed=2)
        self.assertEqual(len({r["left"].startswith("작은") for r in rows}), 2)  # 좌우가 섞인다
        for r in rows:
            side = key[str(r["id"])]
            self.assertTrue(r["left"].startswith("작은" if side["left"] == "6400" else "큰"))
            if not outs["16000"][r["id"]]["stopped"]:
                self.assertIn("[상한에서 잘림]", r["left"] if side["left"] == "16000" else r["right"])
        # 사용자가 매번 6,400 쪽을 고르면 모두 +1
        choices = {str(r["id"]): ("왼쪽" if key[str(r["id"])]["left"] == "6400" else "오른쪽") for r in rows}
        self.assertEqual(outcomes(choices, key), [1] * len(rows))
        choices = {k: "동률" for k in choices}
        self.assertEqual(outcomes(choices, key), [0] * len(rows))
        with self.assertRaises(ValueError):
            outcomes({**choices, str(rows[0]["id"]): "몰라"}, key)
        with self.assertRaises(ValueError):
            outcomes({k: v for k, v in list(choices.items())[1:]}, key)

    def test_decision_rule(self):
        # 6,400 승 60 / 16,000 승 40 / 동률 100 → 차이 20, 평균 0.1 (구간 하한 > 0일 만큼 충분)
        scores = [1] * 60 + [-1] * 40 + [0] * 100
        d = decide(bpb_ci_hi=-0.001, scores=scores)
        self.assertTrue(d["cond_bpb"] and d["cond_blind"] and d["revisit"])
        self.assertEqual((d["wins_6400"], d["wins_16000"], d["ties"]), (60, 40, 100))
        # 차이 19건이면 조건 2 불충족
        d = decide(bpb_ci_hi=-0.001, scores=[1] * 59 + [-1] * 40 + [0] * 101)
        self.assertFalse(d["cond_blind"])
        self.assertFalse(d["revisit"])
        # 구간 상한이 0이면(0보다 작지 않음) 조건 1 불충족
        d = decide(bpb_ci_hi=0.0, scores=scores)
        self.assertFalse(d["cond_bpb"])
        self.assertFalse(d["revisit"])
        # 차이가 정확히 20건이고 구간 하한 > 0이면 조건 2 충족 (20건 경계 포함)
        d = decide(bpb_ci_hi=-0.001, scores=[1] * 21 + [-1] * 1)
        self.assertEqual(d["wins_6400"] - d["wins_16000"], 20)
        self.assertGreater(d["ci"][0], 0)
        self.assertTrue(d["cond_blind"])
        # 차이는 20건인데 문항 점수 구간 하한이 0 이하면 불충족 (차이 조건만 맞고 구간 조건이 실패하는 경로)
        d = decide(bpb_ci_hi=-0.001, scores=[1] * 110 + [-1] * 90)
        self.assertEqual(d["wins_6400"] - d["wins_16000"], 20)
        self.assertLessEqual(d["ci"][0], 0)
        self.assertFalse(d["cond_blind"])
        self.assertFalse(d["revisit"])


class ScoreFilesTest(unittest.TestCase):
    """판정 CLI 경로(score_from_files): 입력 파일 확인과 조건 1에 쓰는 지표."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.key = {"1": {"left": "6400", "right": "16000"}, "2": {"left": "16000", "right": "6400"}}
        self.bpb = {"format": "chat", "vocabs": [6400, 16000],
                    "weights": {"6400": "out_sft/sft_ko6400_768.pth", "16000": "out_sft/sft_ko16000_768.pth"},
                    # 텍스트만 Δ 구간 상한은 0보다 크고 EOS 포함만 0보다 작다 → 조건 1은 텍스트만으로 판정해야 불충족
                    "delta_text_only": {"delta": 0.001, "ci": [-0.001, 0.003]},
                    "delta": {"delta": -0.005, "ci": [-0.008, -0.002]}}

    def write(self, name, obj=None, rows=None):
        path = os.path.join(self.dir, name)
        if rows is not None:
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f)
                w.writerow(["번호", "질문", "왼쪽", "오른쪽", "선택"])
                w.writerows(rows)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f)
        return path

    def paths(self, bpb=None, rows=None):
        rows = rows or [["1", "q1", "a", "b", "왼쪽"], ["2", "q2", "c", "d", "동률"]]
        return (self.write("sheet.csv", rows=rows), self.write("key.json", self.key),
                self.write("bpb.json", bpb or self.bpb))

    def test_uses_text_only_delta(self):
        d = score_from_files(*self.paths())
        self.assertFalse(d["cond_bpb"])
        self.assertEqual((d["wins_6400"], d["wins_16000"], d["ties"]), (1, 0, 1))

    def test_rejects_wrong_bpb_source(self):
        for change in ({"format": "plain"}, {"vocabs": [6400, 6400]},
                       {"weights": {"6400": "out_scale/ko6400_n1000000/pretrain_ko6400_768.pth",
                                    "16000": "out_sft/sft_ko16000_768.pth"}}):
            with self.assertRaises(SystemExit):
                score_from_files(*self.paths(bpb={**self.bpb, **change}))

    def test_rejects_duplicate_rows(self):
        rows = [["1", "q1", "a", "b", "왼쪽"], ["1", "q1", "a", "b", "오른쪽"], ["2", "q2", "c", "d", "동률"]]
        with self.assertRaises(ValueError):
            score_from_files(*self.paths(rows=rows))


class GenerateOneTest(unittest.TestCase):
    """greedy 생성 한 문항: 원본 generate에 greedy 인자를 넘기고, 스스로 끝냈는지와 답 본문을 바르게 가른다."""

    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizers()[16000]

    def fake_model(self, new_tokens):
        test = self

        class Fake:
            def generate(self, x, **kw):
                test.assertFalse(kw["do_sample"])
                test.assertEqual((kw["top_k"], kw["top_p"], kw["temperature"], kw["repetition_penalty"]), (0, 1.0, 1.0, 1.0))
                test.assertEqual(kw["eos_token_id"], test.tok.eos_token_id)
                return torch.cat([x, torch.tensor([new_tokens])], dim=1)
        return Fake()

    def test_stopped_and_body(self):
        body = self.tok("밥 드세요.", add_special_tokens=False).input_ids
        out = generate_one(self.fake_model(body + [self.tok.eos_token_id]), self.tok, "배고파", "cpu")
        self.assertEqual(out, {"text": "밥 드세요.", "stopped": True, "new_tokens": len(body) + 1})

    def test_truncated(self):
        body = self.tok("저도 저도 저도", add_special_tokens=False).input_ids
        out = generate_one(self.fake_model(body), self.tok, "배고파", "cpu")
        self.assertEqual(out, {"text": "저도 저도 저도", "stopped": False, "new_tokens": len(body)})


class TrainOrderTest(unittest.TestCase):
    def test_epoch_indices_match_original(self):
        # 원본 train_full_sft main: epoch마다 setup_seed(seed + epoch) 뒤 torch.randperm
        from trainer.trainer_utils import setup_seed
        for epoch in range(3):
            setup_seed(42 + epoch)
            want = torch.randperm(100).tolist()
            self.assertEqual(epoch_indices(100, 42, epoch), want)
        self.assertNotEqual(epoch_indices(100, 42, 0), epoch_indices(100, 42, 1))

    @unittest.skipUnless(os.path.exists(os.path.join(ROOT, "data/chat/sft_pool.jsonl")), "SFT 데이터 없음")
    def test_all_train_pairs_labels(self):
        # 학습 쌍 전부에서 채점 위치가 답 본문 + <|im_end|> + 줄바꿈뿐인지 (학습 때는 첫 배치만 확인했다)
        with open(os.path.join(ROOT, "data/chat/sft_pool.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for v, tok in tokenizers().items():
            ds = ChatSFTDataset(rows, tok, max_length=128)
            check_first_batch(ds, tok, list(range(len(rows))), len(rows))


class CompareFormatTest(unittest.TestCase):
    def test_refuses_format_mismatch(self):
        base = {"sha256": "x", "pairs": [{"group": 0}], "format": "chat"}
        with self.assertRaises(SystemExit):
            check_same_eval(base, {**base, "format": "plain"})


if __name__ == "__main__":
    unittest.main()
