"""
데이터 양 실험(docs/06) 실행기: 크기별 × vocab별 사전학습 → 검증 평가를 차례로 돌리고, 끝나면 판정까지 낸다.

끊겨도 이어갈 수 있게 만든다.
  - 진행 상태는 <out_root>/status.json에만 둔다. 다시 실행하면 done인 것은 건너뛴다.
    끊긴 학습은 실행 폴더의 재개용 체크포인트(1,000 step마다 저장)가 있으면 거기서 이어서 하고(로그는 이어 씀), 없으면 처음부터 한다.
  - 학습 완료는 '종료 코드 0 + 로그의 "완료:" 줄'일 때만 인정한다
    (원본 train_epoch이 학습 도중에도 같은 이름으로 가중치를 저장하므로, 가중치 파일이 있다고 끝난 게 아니다).
  - 평가는 문서 수와 평가 텍스트 sha256이 기대값과 같을 때만 done으로 기록한다.
  - 실행기(<out_root>/runner.lock)나 학습/평가 프로세스(status.json의 child_pid)가 살아 있으면 새로 시작하지 않는다.
  - 재개용 체크포인트는 실행 폴더 안(<out_root>/ko<V>_n<N>/checkpoints)에만 저장한다.

사용법 (minimind_ko 폴더에서 실행, 세션과 무관하게 돌리려면 scripts\\run_scale.bat):
    .venv\\Scripts\\python.exe -X utf8 scripts\\run_scale.py
    시험 실행: .venv\\Scripts\\python.exe -X utf8 scripts\\run_scale.py --sizes 2000 --out_root out_scale_smoke --eval_limit 50
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [50000, 100000, 200000, 400000, 1000000]  # 100만은 판정 2 후 사용자 승인으로 추가
VOCABS = [6400, 16000]
HELDOUT = "data/scale/heldout.jsonl"
MAIN_OUT_ROOT = "out_scale"
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def now():
    return datetime.now().isoformat(timespec="seconds")


def run_key(size, vocab):
    return f"ko{vocab}_n{size}"


def is_train_complete(returncode, log_path):
    if returncode != 0 or not os.path.exists(log_path):
        return False
    with open(log_path, encoding="utf-8", errors="replace") as f:
        return any("완료:" in line for line in f)


def load_status(path):
    if not os.path.exists(path):
        return {"runs": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_status(path, status):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def next_runs(status, sizes, vocabs):
    return [(n, v) for n in sizes for v in vocabs
            if status["runs"].get(run_key(n, v), {}).get("state") != "done"]


def pid_alive(pid):
    """살아 있는 python 프로세스인지. Windows에서 os.kill(pid, 0)은 프로세스를 종료시키므로 쓰지 않는다."""
    if not pid:
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != STILL_ACTIVE:
            return False
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return False
        return "python" in os.path.basename(buf.value).lower()
    finally:
        kernel32.CloseHandle(handle)


def eval_matches(output, expected_n, expected_sha):
    n = re.search(r"문서 (\d+)개", output)
    sha = re.search(r"평가 텍스트 sha256: ([0-9a-f]{64})", output)
    return bool(n and sha) and int(n.group(1)) == expected_n and sha.group(1) == expected_sha


def parse_metric(output, label):
    m = re.search(rf"{re.escape(label)}\s*: ([\d.]+)", output)
    if not m:
        raise RuntimeError(f"평가 출력에서 '{label}'를 찾지 못함")
    return float(m.group(1))


def expected_eval(eval_limit):
    with open(os.path.join(ROOT, HELDOUT), encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f]
    if eval_limit > 0:
        texts = texts[:eval_limit]
    return len(texts), hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def resume_checkpoint(run_dir, vocab):
    """이어서 학습할 체크포인트 경로. 없으면 None."""
    path = os.path.join(ROOT, run_dir, "checkpoints", f"pretrain_ko{vocab}_768_resume.pth")
    return path if os.path.exists(path) else None


def build_train_cmd(size, vocab, run_dir, resume):
    # 데이터 로더 작업 프로세스는 0개(본 프로세스에서 읽음). 작업 프로세스마다 torch를 따로 불러와 Windows 가상 메모리(커밋)를
    # 약 2.2GB씩 더 쓰는데, 2026-11-19 ko16000_n1000000이 커밋 한도를 넘어 CUDA out of memory로 멈춰서 기본값 4에서 줄였다.
    # 데이터 순서(본 프로세스의 randperm)와 배치 내용은 작업 프로세스 수와 무관해서 학습 결과는 같다.
    cmd = [sys.executable, "-u", "-X", "utf8", "scripts/train_pretrain_ko.py",
           "--tokenizer_dir", f"tokenizers/ko_{vocab}", "--data_path", f"data/scale/train_{size}.jsonl",
           "--out_dir", run_dir, "--ckpt_dir", f"{run_dir}/checkpoints", "--num_workers", "0"]
    return cmd + ["--resume", "1"] if resume else cmd


def run_child(cmd, log_path, status, status_path, key, append=False):
    """자식 프로세스를 띄우고 pid를 상태 파일에 남긴 뒤 끝날 때까지 기다린다. 종료 코드를 돌려준다."""
    with open(log_path, "a" if append else "w", encoding="utf-8") as log:
        log.write("명령: " + subprocess.list2cmdline(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status["runs"][key]["child_pid"] = proc.pid
        save_status(status_path, status)
        code = proc.wait()
    status["runs"][key]["child_pid"] = None
    return code


def train_and_eval(size, vocab, args, status, status_path, expected):
    key = run_key(size, vocab)
    run_dir = f"{args.out_root}/{key}"
    previous = status["runs"].get(key, {})
    resume = previous.get("state") in ("training", "failed") and resume_checkpoint(run_dir, vocab) is not None
    if resume:
        # 끊기기 전 시작 시각은 남기고, 이어서 시작한 시각을 따로 쌓는다
        entry = status["runs"][key] = {**previous, "state": "training", "child_pid": None,
                                       "resumes": previous.get("resumes", []) + [now()]}
        entry.pop("error", None)
    else:
        entry = status["runs"][key] = {"state": "training", "train_start": now()}
    save_status(status_path, status)

    train_log = os.path.join(ROOT, "logs", f"06_scale_{key}.log")
    code = run_child(build_train_cmd(size, vocab, run_dir, resume), train_log, status, status_path, key, append=resume)
    entry["train_end"] = now()
    if not is_train_complete(code, train_log):
        entry["state"] = "failed"
        entry["error"] = f"학습 미완료 (종료 코드 {code}, 로그 {train_log})"
        save_status(status_path, status)
        return False

    entry["state"] = "evaluating"
    save_status(status_path, status)
    eval_log = os.path.join(ROOT, "logs", f"06_scale_{key}_eval.log")
    eval_cmd = [sys.executable, "-X", "utf8", "scripts/eval_bpc.py",
                "--tokenizer_dir", f"tokenizers/ko_{vocab}", "--weight", f"{run_dir}/pretrain_ko{vocab}_768.pth",
                "--data_path", HELDOUT, "--per_doc_out", f"{run_dir}/per_doc.jsonl"]
    if args.eval_limit > 0:
        eval_cmd += ["--limit", str(args.eval_limit)]
    code = run_child(eval_cmd, eval_log, status, status_path, key)
    with open(eval_log, encoding="utf-8", errors="replace") as f:
        output = f.read()
    entry["eval_end"] = now()
    entry["eval_cmd"] = subprocess.list2cmdline(eval_cmd)
    if code != 0 or not eval_matches(output, *expected):
        entry["state"] = "failed"
        entry["error"] = f"평가 실패 또는 평가 텍스트 불일치 (종료 코드 {code}, 로그 {eval_log})"
        save_status(status_path, status)
        return False

    entry.update({"state": "done", "per_doc": f"{run_dir}/per_doc.jsonl", "eval_docs": expected[0],
                  "eval_sha256": expected[1], "bpb": parse_metric(output, "BPB (텍스트+eos)"),
                  "bpc": parse_metric(output, "BPC (텍스트+eos)")})
    save_status(status_path, status)
    print(f"[{now()}] {key} 완료 | BPB {entry['bpb']:.4f} | BPC {entry['bpc']:.4f}", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    p.add_argument("--out_root", default=MAIN_OUT_ROOT)
    p.add_argument("--eval_limit", type=int, default=0, help="검증 앞 N조각만 평가 (시험 실행용, 0이면 전체)")
    args = p.parse_args()

    out_root = os.path.join(ROOT, args.out_root)
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    status_path = os.path.join(out_root, "status.json")
    lock_path = os.path.join(out_root, "runner.lock")

    status = load_status(status_path)
    if os.path.exists(lock_path):
        with open(lock_path, encoding="utf-8") as f:
            other = int(f.read().strip() or 0)
        if other != os.getpid() and pid_alive(other):
            print(f"실행기가 이미 돌고 있음 (PID {other}). 새로 시작하지 않음.")
            return
    alive_children = [(k, r["child_pid"]) for k, r in status["runs"].items() if pid_alive(r.get("child_pid"))]
    if alive_children:
        print(f"이전 학습/평가 프로세스가 아직 살아 있음 {alive_children}. 새로 시작하지 않음.")
        return
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    try:
        expected = expected_eval(args.eval_limit)
        todo = next_runs(status, args.sizes, VOCABS)
        print(f"[{now()}] 남은 실행 {len(todo)}개: {[run_key(n, v) for n, v in todo]}", flush=True)
        for size, vocab in todo:
            print(f"[{now()}] {run_key(size, vocab)} 시작", flush=True)
            if not train_and_eval(size, vocab, args, status, status_path, expected):
                print(f"[{now()}] {run_key(size, vocab)} 실패: {status['runs'][run_key(size, vocab)]['error']}")
                print("원인을 확인한 뒤 다시 실행하세요. 다음 실행으로 넘어가지 않습니다.")
                return

        decision_log = (os.path.join(ROOT, "logs", "06_scale_decision.log") if args.out_root == MAIN_OUT_ROOT
                        else os.path.join(out_root, "decision.log"))
        with open(decision_log, "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-X", "utf8", "scripts/decide_vocab.py",
                            "--status", f"{args.out_root}/status.json", "--heldout", HELDOUT],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        with open(decision_log, encoding="utf-8") as f:
            print(f.read())
        print(f"[{now()}] 전체 완료. 판정 기록: {decision_log}")
    finally:
        os.remove(lock_path)


if __name__ == "__main__":
    main()
