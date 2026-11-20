"""run_scale.py(데이터 양 실험 실행기)의 완료 판정, 재개 순서, 상태 저장, 중복 실행 방지, 평가 확인 테스트.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_run_scale -v
"""
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from run_scale import (build_train_cmd, eval_matches, is_train_complete, load_status, next_runs,  # noqa: E402
                       pid_alive, save_status)

SHA = "a" * 64


class CompleteTest(unittest.TestCase):
    def _log(self, text):
        f = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".log", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.remove, f.name)
        return f.name

    def test_complete_needs_both(self):
        done = self._log("Epoch:[1/1](10/10), loss: 5.0\n완료: D:\\out\\pretrain_ko6400_768.pth\n")
        not_done = self._log("Epoch:[1/1](5/10), loss: 5.0\n")
        self.assertTrue(is_train_complete(0, done))
        self.assertFalse(is_train_complete(0, not_done))
        self.assertFalse(is_train_complete(1, done))


class NextRunsTest(unittest.TestCase):
    def test_next_runs_order_and_skip(self):
        status = {"runs": {"ko6400_n50000": {"state": "done"}, "ko16000_n50000": {"state": "training"}}}
        runs = next_runs(status, [50000, 100000, 200000, 400000], [6400, 16000])
        self.assertEqual(runs[0], (50000, 16000))
        self.assertNotIn((50000, 6400), runs)
        self.assertEqual(len(runs), 7)
        self.assertEqual(runs[1:3], [(100000, 6400), (100000, 16000)])


class StatusTest(unittest.TestCase):
    def test_save_status_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "status.json")
            status = {"runs": {"ko6400_n2000": {"state": "done", "bpb": 1.5}}}
            save_status(path, status)
            self.assertEqual(load_status(path), status)
            self.assertEqual(os.listdir(d), ["status.json"])

    def test_load_missing_status(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_status(os.path.join(d, "none.json")), {"runs": {}})


class PidTest(unittest.TestCase):
    def test_pid_alive(self):
        self.assertTrue(pid_alive(os.getpid()))
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertFalse(pid_alive(p.pid))
        self.assertFalse(pid_alive(0))


class EvalCheckTest(unittest.TestCase):
    def test_eval_check(self):
        out = f"문서 50개 | 글자 33,538 | ...\nBPC (텍스트+eos)       : 3.2839\n평가 텍스트 sha256: {SHA}\n"
        self.assertTrue(eval_matches(out, 50, SHA))
        self.assertFalse(eval_matches(out, 49, SHA))
        self.assertFalse(eval_matches(out, 50, "b" * 64))
        self.assertFalse(eval_matches("출력 없음", 50, SHA))


class TrainCmdTest(unittest.TestCase):
    def test_resume_flag_only_when_resuming(self):
        fresh = build_train_cmd(1000000, 16000, "out_scale/ko16000_n1000000", False)
        resumed = build_train_cmd(1000000, 16000, "out_scale/ko16000_n1000000", True)
        self.assertNotIn("--resume", fresh)
        self.assertEqual(resumed[:len(fresh)], fresh)
        self.assertEqual(resumed[len(fresh):], ["--resume", "1"])
        self.assertIn("out_scale/ko16000_n1000000/checkpoints", fresh)
        self.assertEqual(fresh[fresh.index("--num_workers") + 1], "0")


if __name__ == "__main__":
    unittest.main()
