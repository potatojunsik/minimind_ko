"""decide_vocab.py의 BPB 집계, 부트스트랩, 판정 규칙 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_decide_vocab -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from decide_vocab import bootstrap_ids, ci, decide, decide_extra, group_by_doc, growth_ci, rate  # noqa: E402

LN2 = math.log(2)


class RateTest(unittest.TestCase):
    def test_rate_is_ratio_of_sums(self):
        groups = {"A": (LN2 * 10, 10, 5), "B": (LN2 * 1, 100, 50)}
        self.assertAlmostEqual(rate(groups, ["A", "B"], "bytes"), 11 / 110)
        self.assertNotAlmostEqual(rate(groups, ["A", "B"], "bytes"), (1.0 + 0.01) / 2)
        self.assertAlmostEqual(rate(groups, ["A", "B"], "chars"), 11 / 55)

    def test_rate_counts_repeated_ids(self):
        groups = {"A": (LN2 * 10, 10, 5), "B": (LN2 * 1, 100, 50)}
        self.assertAlmostEqual(rate(groups, ["A", "A", "B"], "bytes"), 21 / 120)

    def test_group_by_doc_sums_chunks(self):
        rows = [{"nll": 1.0, "bytes": 3, "chars": 1}, {"nll": 2.0, "bytes": 6, "chars": 2},
                {"nll": 4.0, "bytes": 9, "chars": 3}]
        groups = group_by_doc(rows, ["x", "x", "y"])
        self.assertEqual(groups, {"x": (3.0, 9, 3), "y": (4.0, 9, 3)})

    def test_group_by_doc_length_mismatch(self):
        with self.assertRaises(ValueError):
            group_by_doc([{"nll": 1.0, "bytes": 3, "chars": 1}], ["x", "y"])


class DecideTest(unittest.TestCase):
    def test_decide_cases(self):
        self.assertEqual(decide(0.02, 0.005, -1), 1)
        self.assertEqual(decide(-0.03, -0.04, 0.01), 2)
        self.assertEqual(decide(-0.03, -0.04, -0.001), 3)
        self.assertEqual(decide(0.01, -0.002, 0.5), 3)  # 점추정은 양수지만 구간에 0 포함
        self.assertEqual(decide(0.0, -0.01, 0.5), 3)


class BootstrapTest(unittest.TestCase):
    def test_bootstrap_same_ids_reused(self):
        ids = ["a", "b", "c", "d"]
        first = bootstrap_ids(ids, rounds=5, seed=0)
        self.assertEqual(first, bootstrap_ids(ids, rounds=5, seed=0))
        self.assertEqual(len(first), 5)
        self.assertTrue(all(len(r) == 4 and set(r) <= set(ids) for r in first))

    def test_ci_percentiles(self):
        lo, hi = ci([float(i) for i in range(100, 0, -1)])
        self.assertEqual((lo, hi), (3.0, 97.0))

    def test_decide_extra(self):
        self.assertEqual(decide_extra(0.0001, 0.003)[0], 16000)
        self.assertEqual(decide_extra(0.0, 0.003)[0], 6400)
        self.assertEqual(decide_extra(1e-12, 0.003)[0], 16000)  # 반올림하면 +0.0000이어도 0보다 크면 16,000
        self.assertIn("16,000의 우위가 확인되지 않아 사전 규칙에 따라 6,400을 선택했다", decide_extra(-0.001, 0.002)[1])
        self.assertEqual(decide_extra(-0.004, -0.001)[0], 6400)
        self.assertNotIn("우위가 확인되지 않아", decide_extra(-0.004, -0.001)[1])

    def test_growth_ci_pairs_same_round(self):
        boot = {400000: [float(i) for i in range(100)], 1000000: [float(i) + 0.5 for i in range(100)]}
        sizes_out = {"400000": {"delta_bpb": -0.003}, "1000000": {"delta_bpb": 0.001}}
        g = growth_ci(boot, sizes_out, 1000000, 400000)
        self.assertAlmostEqual(g["point"], 0.004)
        self.assertEqual(g["ci"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
