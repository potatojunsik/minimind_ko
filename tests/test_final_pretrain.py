"""최종 사전학습(설계: docs/superpowers/specs/2026-11-23-final-pretrain-design.md)의 패킹·판정 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_final_pretrain -v
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from packing import IGNORE_INDEX, PackedPretrainDataset, pack_token_lists, tokenize_and_pack  # noqa: E402

BOS, EOS, PAD = 1, 2, 0


def unpad(rows, pad_tail):
    flat = rows.reshape(-1).tolist()
    return flat[:len(flat) - pad_tail] if pad_tail else flat


class PackTest(unittest.TestCase):
    def test_rows_rejoin_to_original_stream(self):
        lists = [[5, 6, 7], [8], [9, 10, 11, 12, 13]]
        rows = pack_token_lists(lists, BOS, EOS, PAD, 4)
        want = [1, 5, 6, 7, 2, 1, 8, 2, 1, 9, 10, 11, 12, 13, 2]
        self.assertEqual(rows.shape, (4, 4))
        self.assertEqual(unpad(rows, 1), want)
        self.assertEqual(rows[-1].tolist(), [12, 13, 2, PAD])

    def test_long_chunk_is_not_truncated(self):
        long = list(range(3, 603))  # 600토큰
        rows = pack_token_lists([long, [7]], BOS, EOS, PAD, 512)
        flat = rows.reshape(-1).tolist()
        self.assertEqual(flat[:602], [BOS] + long + [EOS])
        self.assertEqual(flat[602:605], [BOS, 7, EOS])
        self.assertEqual(rows.shape, (2, 512))
        self.assertEqual(flat[605:], [PAD] * (1024 - 605))

    def test_exact_multiple_has_no_pad(self):
        rows = pack_token_lists([[5, 6], [7, 8]], BOS, EOS, PAD, 4)
        self.assertEqual(rows.tolist(), [[1, 5, 6, 2], [1, 7, 8, 2]])

    def test_token_id_out_of_uint16_range_raises(self):
        with self.assertRaises(ValueError):
            pack_token_lists([[65536]], BOS, EOS, PAD, 4)

    def test_labels_mask_pad_and_bos_only(self):
        rows = pack_token_lists([[5, 6, 7], [8], [9]], BOS, EOS, PAD, 4)  # [1 5 6 7] [2 1 8 2] [1 9 2 0]
        ds = PackedPretrainDataset(rows, BOS, PAD)
        self.assertEqual(len(ds), 3)
        x, y = ds[1]
        self.assertEqual(x.tolist(), [2, 1, 8, 2])
        self.assertEqual(y.tolist(), [2, IGNORE_INDEX, 8, 2])
        x, y = ds[2]
        self.assertEqual(y.tolist(), [IGNORE_INDEX, 9, 2, IGNORE_INDEX])
        self.assertEqual(str(x.dtype), "torch.int64")


class TokenizeAndPackTest(unittest.TestCase):
    def test_matches_per_chunk_tokenization_with_real_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizers", "ko_16000"))
        with open(os.path.join(ROOT, "data", "scale", "train_2000.jsonl"), encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for line in f][:300]
        rows, stats = tokenize_and_pack(texts, tok, 512, batch_size=64)
        want = []
        for t in texts:
            want += [tok.bos_token_id] + tok(t, add_special_tokens=False).input_ids + [tok.eos_token_id]
        self.assertEqual(stats["tokens"], len(want))
        self.assertEqual(stats["chunks"], 300)
        self.assertEqual(stats["rows"], math.ceil(len(want) / 512))
        self.assertEqual(stats["pad"], stats["rows"] * 512 - len(want))
        self.assertEqual(unpad(rows, stats["pad"]), want)


class DecideRulesTest(unittest.TestCase):
    def test_packing_margin_is_strict(self):
        from decide_final import decide_packing
        self.assertTrue(decide_packing(0.0049999))
        self.assertFalse(decide_packing(0.005))
        self.assertTrue(decide_packing(-0.01))

    def test_epochs_needs_all_three(self):
        from decide_final import decide_epochs
        self.assertTrue(decide_epochs(0.08, 0.12, 0.001, 1.16, 1.20))
        self.assertTrue(decide_epochs(0.25, 0.5, 0.001, 1.16, 1.20))  # g/G 정확히 0.5는 채택
        self.assertFalse(decide_epochs(0.0599, 0.12, 0.001, 1.16, 1.20))
        self.assertFalse(decide_epochs(0.08, 0.12, 0.0, 1.16, 1.20))
        self.assertFalse(decide_epochs(0.08, 0.12, 0.001, 1.20, 1.20))
        self.assertFalse(decide_epochs(0.08, 0.0, 0.001, 1.16, 1.20))

    def test_accept_is_strict(self):
        from decide_final import decide_accept
        self.assertTrue(decide_accept(-1e-9))
        self.assertFalse(decide_accept(0.0))


class DecideDataTest(unittest.TestCase):
    """06번 실제 결과로 계산이 decide_vocab.py와 같은지, 행 수 검사가 동작하는지 확인한다."""

    def setUp(self):
        from decide_vocab import bootstrap_ids, load_jsonl
        self.doc_ids = [r["doc_id"] for r in load_jsonl("data/scale/heldout.jsonl")]
        self.unique = list(dict.fromkeys(self.doc_ids))
        self.samples = bootstrap_ids(self.unique, 2000, 0)

    def test_same_as_decide_vocab_400k_delta(self):
        from decide_final import load_groups, paired
        from decide_vocab import bootstrap_ids
        a = load_groups("out_scale/ko6400_n400000/per_doc.jsonl", self.doc_ids)
        b = load_groups("out_scale/ko16000_n400000/per_doc.jsonl", self.doc_ids)
        samples = bootstrap_ids(self.unique, 10000, 0)
        got = paired(a, b, self.unique, samples)
        with open(os.path.join(ROOT, "out_scale", "decision.json"), encoding="utf-8") as f:
            want = json.load(f)["sizes"]["400000"]
        self.assertAlmostEqual(got["point"], want["delta_bpb"], places=12)
        self.assertAlmostEqual(got["ci"][0], want["delta_bpb_ci"][0], places=12)
        self.assertAlmostEqual(got["ci"][1], want["delta_bpb_ci"][1], places=12)

    def test_same_model_gives_zero(self):
        from decide_final import load_groups, paired
        a = load_groups("out_scale/ko16000_n400000/per_doc.jsonl", self.doc_ids)
        got = paired(a, a, self.unique, self.samples)
        self.assertEqual(got, {"point": 0.0, "ci": [0.0, 0.0]})

    def test_row_count_mismatch_stops(self):
        from decide_final import load_groups
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "per_doc.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"chars": 1, "bytes": 1, "tokens": 1, "nll": 1.0}) + chr(10))
            with self.assertRaises(SystemExit):
                load_groups(path, self.doc_ids)

    def test_check_prefix(self):
        from prepare_final_data import check_prefix
        self.assertIsNone(check_prefix(["a", "b", "c"], ["a", "b"]))
        self.assertEqual(check_prefix(["a", "x", "c"], ["a", "b"]), 1)
        self.assertEqual(check_prefix(["a"], ["a", "b"]), 1)

    def test_old_runs_are_full_heldout(self):
        from decide_final import check_old_runs
        check_old_runs(["400k", "200k", "1m"])


class RunnerSafetyTest(unittest.TestCase):
    """코드 점검(2026-11-24)에서 나온 실행기·지킴이 문제의 회귀 테스트."""

    def test_read_lock_pid_handles_missing_empty_broken(self):
        from run_final import read_lock_pid
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "runner.lock")
            self.assertIsNone(read_lock_pid(path))
            for text, want in [("", None), ("abc", None), ("1234", 1234)]:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.assertEqual(read_lock_pid(path), want)

    def test_unknown_battery_values(self):
        from battery_guard import read_power
        self.assertEqual(read_power(255, 1, 255), (None, True, False))
        self.assertEqual(read_power(40, 1, 8 | 1), (40, True, True))
        self.assertEqual(read_power(40, 0, 1), (40, False, False))

    def test_evaluating_state_skips_training(self):
        """평가 중에 끊긴 단계는 다시 학습하지 않고 평가만 한다(끝난 가중치를 덮어쓰지 않음)."""
        import argparse
        import run_final
        with tempfile.TemporaryDirectory() as d:
            out_root = os.path.relpath(d, ROOT) if os.path.splitdrive(d)[0] == os.path.splitdrive(ROOT)[0] else d
            with open(os.path.join(d, "status.json"), "w", encoding="utf-8") as f:
                json.dump({"runs": {"final": {"state": "evaluating"}}}, f)
            runner = run_final.Runner(argparse.Namespace(out_root=out_root, smoke=True))
            calls = []
            runner.evaluate = lambda key, run_dir, evals: calls.append(key) or True
            self.assertTrue(runner.pretrain("final", "없는/데이터.jsonl", [], []))
            self.assertEqual(calls, ["final"])

    def test_full_pool_comes_from_report(self):
        from run_final import full_pool
        self.assertEqual(full_pool(), "data/scale/train_1183370.jsonl")


if __name__ == "__main__":
    unittest.main()
