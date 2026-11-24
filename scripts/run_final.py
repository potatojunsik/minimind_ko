"""
최종 사전학습 실행기 (설계: docs/superpowers/specs/2026-11-23-final-pretrain-design.md).

순서: 실험 1(40만 패킹) → 실험 2(20만 × 2 epoch) → setup 판정 → 최종 사전학습 → accept 판정
      → (새 모델 채택 시) 최종 SFT → SFT 기록용 평가.
run_scale.py와 같은 방식으로 끊겨도 이어간다: 상태는 <out_root>/status.json, 완료는 "종료 코드 0 + 완료: 줄",
평가는 문서 수·sha256 확인, 끊긴 사전학습은 체크포인트에서 --resume 1로 이어서 한다. 판정은 decision.json에 한 번만 남긴다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\run_final.py
    시험 실행: .venv\\Scripts\\python.exe -X utf8 scripts\\run_final.py --smoke
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_scale import (HELDOUT, eval_matches, expected_eval, is_train_complete, load_status, now,  # noqa: E402
                       parse_metric, pid_alive, resume_checkpoint, run_child, save_status)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCAB = 16000
TOKENIZER = f"tokenizers/ko_{VOCAB}"
WEIGHT = f"pretrain_ko{VOCAB}_768.pth"
POOL_REPORT = "data/scale/final_pool_report.json"  # prepare_final_data.py가 학습 풀 전체 파일 경로를 적어 둔다
SMOKE_DATA = "data/scale/train_2000.jsonl"
SMOKE_EVAL_LIMIT = 50


def experiments(smoke):
    """(이름, 학습 데이터, 추가 인자, [(평가할 가중치, 문서별 결과 파일)])"""
    data400 = SMOKE_DATA if smoke else "data/scale/train_400000.jsonl"
    data200 = SMOKE_DATA if smoke else "data/scale/train_200000.jsonl"
    return [
        ("exp_pack400k", data400, ["--pack", "1"], [(WEIGHT, "per_doc.jsonl")]),
        ("exp_ep2_200k", data200, ["--epochs", "2", "--keep_epoch_weights", "1"],
         [(WEIGHT, "per_doc.jsonl"), (WEIGHT.replace(".pth", "_epoch1.pth"), "per_doc_epoch1.jsonl")]),
    ]


def final_run(setup, smoke):
    extra = ["--pack", "1" if setup["pack"] else "0", "--epochs", str(setup["epochs"])]
    evals = [(WEIGHT, "per_doc.jsonl")]
    if setup["epochs"] > 1:
        extra += ["--keep_epoch_weights", "1"]
        # 1 epoch 끝 가중치는 참고용(두 번째 epoch 동안 검증 성능 변화 설명)으로만 평가한다. 채택 판정에는 쓰지 않는다.
        evals.append((WEIGHT.replace(".pth", "_epoch1.pth"), "per_doc_epoch1.jsonl"))
    return ("final", SMOKE_DATA if smoke else full_pool(), extra, evals)


def full_pool():
    with open(os.path.join(ROOT, POOL_REPORT), encoding="utf-8") as f:
        return json.load(f)["path"]


def read_lock_pid(path):
    """실행기 락 파일의 PID. 파일이 없거나 비었거나 깨졌으면 None."""
    try:
        with open(path, encoding="utf-8") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pretrain_cmd(data, extra, run_dir, resume):
    # 데이터 로더 작업 프로세스 0개: run_scale.py와 같은 이유(Windows 가상 메모리 커밋 한도)
    cmd = [sys.executable, "-u", "-X", "utf8", "scripts/train_pretrain_ko.py", "--tokenizer_dir", TOKENIZER,
           "--data_path", data, "--out_dir", run_dir, "--ckpt_dir", f"{run_dir}/checkpoints", "--num_workers", "0"] + extra
    return cmd + ["--resume", "1"] if resume else cmd


class Runner:
    def __init__(self, args):
        self.args = args
        self.out_root = args.out_root
        self.status_path = os.path.join(ROOT, args.out_root, "status.json")
        self.status = load_status(self.status_path)
        self.log_dir = os.path.join(ROOT, "logs") if not args.smoke else os.path.join(ROOT, args.out_root, "logs")
        self.eval_limit = SMOKE_EVAL_LIMIT if args.smoke else 0
        self.expected = expected_eval(self.eval_limit)

    def log_path(self, name):
        return os.path.join(self.log_dir, f"08_final_{name}.log")

    def done(self, key):
        return self.status["runs"].get(key, {}).get("state") == "done"

    def fail(self, key, message):
        self.status["runs"][key].update({"state": "failed", "error": message})
        save_status(self.status_path, self.status)
        print(f"[{now()}] {key} 실패: {message}\n원인을 확인한 뒤 다시 실행하세요. 다음 단계로 넘어가지 않습니다.", flush=True)
        return False

    def pretrain(self, key, data, extra, evals):
        if self.done(key):
            return True
        run_dir = f"{self.out_root}/{key}"
        if self.status["runs"].get(key, {}).get("state") == "evaluating":
            # "evaluating"은 학습이 끝까지 성공한 뒤에만 기록된다. 평가 중에 끊겼으면 학습은 건너뛰고 평가만 다시 한다
            # (다시 학습하면 끝난 가중치를 덮어쓴다)
            print(f"[{now()}] {key}: 학습은 끝났고 평가 중에 끊김 → 평가만 다시 함", flush=True)
            return self.evaluate(key, run_dir, evals)
        if not os.path.exists(os.path.join(ROOT, data)):
            print(f"[{now()}] {key}: 학습 데이터가 없음 ({data})", flush=True)
            return False
        previous = self.status["runs"].get(key, {})
        resume = previous.get("state") in ("training", "failed") and resume_checkpoint(run_dir, VOCAB) is not None
        if resume:
            entry = {**previous, "state": "training", "child_pid": None, "resumes": previous.get("resumes", []) + [now()]}
            entry.pop("error", None)
        else:
            entry = {"state": "training", "train_start": now(), "data": data, "extra": extra}
        self.status["runs"][key] = entry
        save_status(self.status_path, self.status)
        print(f"[{now()}] {key} 학습 시작{' (이어서)' if resume else ''}: {data} {' '.join(extra)}", flush=True)

        train_log = self.log_path(key)
        code = run_child(pretrain_cmd(data, extra, run_dir, resume), train_log, self.status, self.status_path, key,
                         append=resume)
        entry["train_end"] = now()
        if not is_train_complete(code, train_log):
            return self.fail(key, f"학습 미완료 (종료 코드 {code}, 로그 {train_log})")
        return self.evaluate(key, run_dir, evals)

    def evaluate(self, key, run_dir, evals):
        entry = self.status["runs"][key]
        entry["state"] = "evaluating"
        save_status(self.status_path, self.status)
        entry["evals"] = {}
        for weight, per_doc in evals:
            eval_log = self.log_path(f"{key}_eval_{per_doc.replace('.jsonl', '')}")
            cmd = [sys.executable, "-X", "utf8", "scripts/eval_bpc.py", "--tokenizer_dir", TOKENIZER,
                   "--weight", f"{run_dir}/{weight}", "--data_path", HELDOUT, "--per_doc_out", f"{run_dir}/{per_doc}"]
            if self.eval_limit:
                cmd += ["--limit", str(self.eval_limit)]
            code = run_child(cmd, eval_log, self.status, self.status_path, key)
            with open(eval_log, encoding="utf-8", errors="replace") as f:
                output = f.read()
            if code != 0 or not eval_matches(output, *self.expected):
                return self.fail(key, f"평가 실패 또는 평가 텍스트 불일치 (종료 코드 {code}, 로그 {eval_log})")
            entry["evals"][per_doc] = {"weight": f"{run_dir}/{weight}", "bpb": parse_metric(output, "BPB (텍스트+eos)")}
        entry.update({"state": "done", "eval_end": now(), "eval_docs": self.expected[0], "eval_sha256": self.expected[1]})
        save_status(self.status_path, self.status)
        print(f"[{now()}] {key} 완료 | " + ", ".join(f"{k} BPB {v['bpb']:.4f}" for k, v in entry["evals"].items()),
              flush=True)
        return True

    def decide(self, step, extra=()):
        """판정은 한 번만 낸다(이미 decision.json에 있으면 다시 계산하지 않는다)."""
        decision = self.decision()
        if step in decision:
            return decision[step]
        cmd = [sys.executable, "-X", "utf8", "scripts/decide_final.py", step, "--out_root", self.out_root] + list(extra)
        if self.args.smoke and step != "sft":
            cmd += ["--limit", str(self.eval_limit)]
        log = self.log_path(f"decide_{step}")
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
        with open(log, encoding="utf-8") as f:
            print(f.read(), flush=True)
        decision = self.decision()
        if code != 0 or step not in decision:
            print(f"[{now()}] 판정 {step} 실패 (로그 {log})", flush=True)
            return None
        return decision[step]

    def decision(self):
        path = os.path.join(ROOT, self.out_root, "decision.json")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def sft(self):
        key = "sft"
        if self.done(key):
            return True
        run_dir = f"{self.out_root}/{key}"
        self.status["runs"][key] = {"state": "training", "train_start": now()}
        save_status(self.status_path, self.status)
        cmd = [sys.executable, "-u", "-X", "utf8", "scripts/train_sft_ko.py", "--tokenizer_dir", TOKENIZER,
               "--init_weight", f"{self.out_root}/final/{WEIGHT}", "--out_dir", run_dir, "--ckpt_dir", f"{run_dir}/checkpoints"]
        if self.args.smoke:
            cmd += ["--epochs", "1"]
        print(f"[{now()}] 최종 SFT 시작", flush=True)
        log = self.log_path(key)
        code = run_child(cmd, log, self.status, self.status_path, key)
        if not is_train_complete(code, log):
            return self.fail(key, f"SFT 미완료 (종료 코드 {code}, 로그 {log})")
        self.status["runs"][key].update({"state": "done", "train_end": now(), "weight": f"{run_dir}/sft_ko{VOCAB}_768.pth"})
        save_status(self.status_path, self.status)
        return True

    def run(self):
        for key, data, extra, evals in experiments(self.args.smoke):
            if not self.pretrain(key, data, extra, evals):
                return
        setup = self.decide("setup")
        if setup is None:
            return
        if not self.pretrain(*final_run(setup, self.args.smoke)):
            return
        accept = self.decide("accept")
        if accept is None:
            return
        if not accept["use_new"] and not self.args.smoke:
            print(f"[{now()}] 기존 16,000·100만 모델을 유지한다. 07번 SFT 모델이 같은 설정의 결과라 최종 SFT는 하지 않는다.")
            return
        if not self.sft():
            return
        tag = "smoke_final_sft" if self.args.smoke else "final_sft"
        if self.decide("sft", ["--weight", self.status["runs"]["sft"]["weight"], "--tag", tag]) is None:
            return
        print(f"[{now()}] 전체 완료. 판정 기록: {os.path.join(self.out_root, 'decision.json')}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="")
    p.add_argument("--smoke", action="store_true", help="시험 실행: 학습 2,000조각, 검증 앞 50조각, out_final_smoke")
    args = p.parse_args()
    args.out_root = args.out_root or ("out_final_smoke" if args.smoke else "out_final")

    out_root = os.path.join(ROOT, args.out_root)
    os.makedirs(out_root, exist_ok=True)
    lock_path = os.path.join(out_root, "runner.lock")
    other = read_lock_pid(lock_path)
    if other is not None and other != os.getpid() and pid_alive(other):
        print(f"실행기가 이미 돌고 있음 (PID {other}). 새로 시작하지 않음.")
        return
    runner = Runner(args)
    os.makedirs(runner.log_dir, exist_ok=True)
    alive = [(k, r["child_pid"]) for k, r in runner.status["runs"].items() if pid_alive(r.get("child_pid"))]
    if alive:
        print(f"이전 학습/평가 프로세스가 아직 살아 있음 {alive}. 새로 시작하지 않음.")
        return
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        runner.run()
    finally:
        os.remove(lock_path)


if __name__ == "__main__":
    main()
