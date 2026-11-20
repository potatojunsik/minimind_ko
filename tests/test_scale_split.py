"""prepare_scale_data.py의 검증/학습 분할 함수 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_scale_split -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from prepare_scale_data import make_nested_train, select_heldout  # noqa: E402

# 줄 번호 = 리스트 인덱스. 문서 0: 줄 0~2, 문서 1: 줄 3~4, 문서 2: 줄 5~6, 문서 3: 줄 7~8
CHUNKS = [
    (0, "가0"), (0, "가1"), (0, "가2"),
    (1, "나0"), (1, "나1"),
    (2, "다0"), (2, "다1"),
    (3, "라0"), (3, "라1"),
]


class SelectHeldoutTest(unittest.TestCase):
    def test_heldout_only_after_start(self):
        # 문서 1은 줄 3이 start(4)보다 앞이라 경계를 걸친다 → 제외
        result = select_heldout(CHUNKS, candidate_start=4, target_chunks=100, seed=1)
        self.assertEqual(result, {2, 3})

    def test_heldout_excludes_duplicate_text(self):
        chunks = list(CHUNKS)
        chunks[7] = (3, "가1")  # 문서 3의 조각이 문서 0의 조각과 같은 글
        result = select_heldout(chunks, candidate_start=4, target_chunks=100, seed=1)
        self.assertEqual(result, {2})

    def test_heldout_whole_documents(self):
        result = select_heldout(CHUNKS, candidate_start=4, target_chunks=1, seed=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(result <= {2, 3})
        picked_lines = [i for i, (d, _) in enumerate(CHUNKS) if d in result]
        self.assertEqual(len(picked_lines), 2)  # 문서의 조각 2개가 함께 들어간다


class NestedTrainTest(unittest.TestCase):
    def test_nested_and_disjoint(self):
        heldout = {2, 3}
        r = make_nested_train(CHUNKS, heldout, sizes=[2, 4], seed=7)
        self.assertEqual(set(r), {2, 4})
        self.assertEqual(len(r[2]), 2)
        self.assertEqual(len(r[4]), 4)
        self.assertEqual(r[4][:2], r[2])
        self.assertTrue(set(r[2]) <= set(r[4]))
        for line in r[4]:
            self.assertNotIn(CHUNKS[line][0], heldout)
        self.assertEqual(r, make_nested_train(CHUNKS, heldout, sizes=[2, 4], seed=7))

    def test_size_larger_than_pool_raises(self):
        with self.assertRaises(ValueError):
            make_nested_train(CHUNKS, {2, 3}, sizes=[6], seed=7)


if __name__ == "__main__":
    unittest.main()
