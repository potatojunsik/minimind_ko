"""README 대화 데모(scripts/final_demo.py)의 질문 고르기·집계 테스트. 모델은 쓰지 않는다.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_final_demo -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import final_demo as d  # noqa: E402


class SampleTest(unittest.TestCase):
    PAIRS = [{"q": f"질문{i}", "a": "답", "group": i // 2} for i in range(20)] + \
            [{"q": "배고파 죽겠다", "a": "답", "group": 100}]

    def test_one_question_per_group_and_reproducible(self):
        a, _ = d.sample_eval_questions(self.PAIRS, 5, 7)
        b, _ = d.sample_eval_questions(self.PAIRS, 5, 7)
        self.assertEqual(a, b)
        self.assertEqual(len({x["group"] for x in a}), 5)

    def test_sensitive_question_is_skipped(self):
        picked, skipped = d.sample_eval_questions(self.PAIRS, 11, 7)
        self.assertNotIn("배고파 죽겠다", [x["q"] for x in picked])
        self.assertEqual(skipped, ["배고파 죽겠다"])


class UnseenTest(unittest.TestCase):
    def test_same_question_ignoring_spaces_and_punct_stops(self):
        with self.assertRaises(SystemExit):
            d.check_written_unseen([{"q": "12시 땡"}], [{"q": "12시 땡!"}], [])

    def test_new_question_passes(self):
        d.check_written_unseen([{"q": "완전히 새 질문"}], [{"q": "12시 땡!"}], [{"q": "다른 질문"}])


class StatsTest(unittest.TestCase):
    def test_counts(self):
        outs = {"1": {"text": "그런 사람 만나지 마세요.", "stopped": True},
                "2": {"text": "그런 사람 만나지 마세요.", "stopped": True},
                "3": {"text": "그런 사람 만나지 마세요.", "stopped": True},
                "4": {"text": "저도 저도", "stopped": False}}
        s = d.generation_stats(outs)
        self.assertEqual((s["truncated"], s["distinct_answers"], s["items_with_answer_repeated_3plus"]), (1, 2, 3))
        self.assertEqual(s["phrases"]["그런 사람 만나지 마세요."], 3)

    def test_reproduces_07_document_numbers(self):
        """docs/07 4-2절: 16,000 잘림 4, 서로 다른 답 132, 3번 이상 반복 문항 71, '그런 사람 만나지 마세요.' 27."""
        with open(os.path.join(ROOT, d.BLIND_OUTPUTS_07), encoding="utf-8") as f:
            s = d.generation_stats(json.load(f)["16000"])
        self.assertEqual((s["truncated"], s["distinct_answers"], s["items_with_answer_repeated_3plus"]), (4, 132, 71))
        self.assertEqual(s["phrases"]["그런 사람 만나지 마세요."], 27)


if __name__ == "__main__":
    unittest.main()
