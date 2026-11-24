"""최종 사전학습 배터리 지킴이 (run_final.py와 함께 쓴다).

노트북이 충전기에 꽂혀 있어도 GPU 학습 중에는 배터리가 조금씩 줄어서, 17시간 학습을 끝까지 버티지 못할 수 있다.
배터리가 LOW% 이하로 떨어지면 실행기를 멈추고, HIGH%까지 다시 차면 run_final.bat을 다시 띄워 체크포인트부터 이어간다.

- 사전학습 중(state "training")일 때만 멈춘다. 평가 중에는 몇 분이면 끝나므로 기다리고, 최종 SFT(수 분)는 이어서 하는
  기능이 없어 멈추면 처음부터 다시 하므로 멈추지 않는다.
- 배터리 값을 읽지 못하면(Windows가 255 = 알 수 없음을 돌려줌) 그동안은 멈추거나 다시 띄우지 않는다.
- 체크포인트는 임시 파일에 쓴 뒤 바꿔치기하므로(trainer_utils.lm_checkpoint) 저장 중에 멈춰도 깨지지 않는다.
  한 번 멈출 때 잃는 것은 마지막 체크포인트 이후 학습분(최대 1,000 step)이다.
- 실행기가 끝나거나(전체 완료·기존 모델 유지·실패) 지킴이가 멈춘 것이 아닌데 실행기가 꺼져 있으면 지킴이도 끝난다.
- 기록: logs/08_final_battery.log (배터리 %는 5분마다), 멈춤·재시작은 logs/08_final_runner.log에도 남긴다.

실행 (minimind_ko 폴더에서, 실행기를 띄운 뒤):
    .venv\\Scripts\\python.exe -u -X utf8 scripts\\battery_guard.py
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_final import read_lock_pid  # noqa: E402
from run_scale import now, pid_alive  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "out_final")
STATUS = os.path.join(OUT_ROOT, "status.json")
LOCK = os.path.join(OUT_ROOT, "runner.lock")
RUNNER_LOG = os.path.join(ROOT, "logs", "08_final_runner.log")
GUARD_LOG = os.path.join(ROOT, "logs", "08_final_battery.log")
RUNNER_BAT = os.path.join(ROOT, "scripts", "run_final.bat")
FINISHED = ("전체 완료", "기존 16,000·100만 모델을 유지")
CHARGING_FLAG = 8  # SYSTEM_POWER_STATUS.BatteryFlag: 충전 중
UNKNOWN = 255  # BatteryLifePercent·BatteryFlag의 '알 수 없음'
NO_RESUME_KEYS = ("sft",)  # 이어서 하는 기능이 없는 단계: 멈추지 않는다
POLL_SEC = 60
RECORD_SEC = 300
RELAUNCH_GRACE_SEC = 120
APPEND_TRIES = 10


class SystemPowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def battery():
    """(배터리 %, 충전기 연결 여부, 충전 중 여부). 값을 알 수 없으면 %는 None."""
    s = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        raise OSError("GetSystemPowerStatus 실패")
    return read_power(s.BatteryLifePercent, s.ACLineStatus, s.BatteryFlag)


def read_power(percent, ac_line, flag):
    """GetSystemPowerStatus 원시 값 → (%, 충전기 연결, 충전 중). 255(알 수 없음)는 None/False로 둔다."""
    pct = None if percent == UNKNOWN else percent
    charging = flag != UNKNOWN and bool(flag & CHARGING_FLAG)
    return pct, ac_line == 1, charging


def note(message, runner_log=False):
    line = f"[{now()}] {message}"
    print(line, flush=True)
    append(GUARD_LOG, line)
    if runner_log and not append(RUNNER_LOG, line):
        append(GUARD_LOG, f"[{now()}] (실행기 기록 파일이 잠겨 있어 위 줄은 지킴이 기록에만 남김)")


def append(path, line, tries=APPEND_TRIES):
    """한 줄 덧붙인다. run_final.bat 창이 실행기 기록 파일을 `>>`로 열어 둔 동안(실행기를 멈춘 직후 포함)은
    다른 프로세스가 덧붙이면 PermissionError가 나므로(2026-11-23 22:13 지킴이가 이 오류로 꺼짐) 1초 간격으로 다시 시도한다."""
    for _ in range(tries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except PermissionError:
            time.sleep(1)
    return False


def runner_pid():
    pid = read_lock_pid(LOCK)
    return pid if pid_alive(pid) else None


def pretrain_training():
    """이어서 할 수 있는 사전학습 단계가 지금 학습 중인지(평가 중·SFT 중이면 False)."""
    try:
        with open(STATUS, encoding="utf-8") as f:
            runs = json.load(f)["runs"]
    except (OSError, ValueError, KeyError):
        return False
    return any(entry.get("state") == "training" and key not in NO_RESUME_KEYS and pid_alive(entry.get("child_pid"))
               for key, entry in runs.items())


def pipeline_ended():
    try:
        with open(RUNNER_LOG, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    return any(word in text for word in FINISHED)


def stop_runner(pid, pct):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    note(f"배터리 {pct}%: 지킴이가 실행기를 멈춤 (충전 후 자동으로 이어서 함)", runner_log=True)


def relaunch(pct):
    note(f"배터리 {pct}%: 지킴이가 run_final.bat을 다시 띄움", runner_log=True)
    subprocess.run(["cmd.exe", "/c", "start", "MiniMind final pretrain", RUNNER_BAT], cwd=os.path.dirname(RUNNER_BAT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low", type=int, default=25, help="이 % 이하면 학습을 멈춘다")
    p.add_argument("--high", type=int, default=90, help="멈춘 뒤 이 %까지 차면 다시 시작한다")
    p.add_argument("--plateau", type=int, default=60,
                   help="충전기가 꽂혀 있는데 충전이 멈췄으면(충전 상한) 이 % 이상에서 다시 시작한다")
    args = p.parse_args()

    pct, ac, charging = battery()
    note(f"지킴이 시작: 배터리 {'알 수 없음' if pct is None else f'{pct}%'}, 충전기 {'연결' if ac else '없음'}, 멈춤 {args.low}% 이하 / 재시작 {args.high}% 이상")
    paused = False
    last_record = 0.0
    relaunched_at = time.time() - RELAUNCH_GRACE_SEC
    while True:
        # 한 번의 확인에서 어떤 오류가 나도 지킴이는 꺼지지 않는다(꺼지면 멈춘 학습을 아무도 다시 띄우지 않는다)
        try:
            pct, ac, charging = battery()
            if time.time() - last_record >= RECORD_SEC:
                note(f"배터리 {'알 수 없음' if pct is None else f'{pct}%'} | 충전기 {'연결' if ac else '없음'} | {'충전 중' if charging else '충전 안 함'}"
                     f"{' | 멈춤 상태' if paused else ''}")
                last_record = time.time()

            # 배터리 값을 모르는 동안(pct None)은 멈추거나 다시 띄우지 않는다
            if paused:
                if pct is not None and (pct >= args.high or (ac and not charging and pct >= args.plateau)):
                    relaunch(pct)
                    paused = False
                    relaunched_at = time.time()
            else:
                pid = runner_pid()
                if pid is None:
                    if time.time() - relaunched_at >= RELAUNCH_GRACE_SEC:  # 다시 띄운 실행기는 락을 만들 때까지 기다린다
                        note("실행기가 끝났거나 꺼져 있음: 지킴이 종료" + (" (전체 과정 끝)" if pipeline_ended() else ""))
                        return
                elif pct is not None and pct <= args.low and pretrain_training():
                    paused = True  # 기록 중 오류가 나도 멈춘 상태를 잃지 않게 먼저 표시한다
                    stop_runner(pid, pct)
        except Exception as e:  # noqa: BLE001
            note(f"지킴이 오류 (계속 감시): {type(e).__name__}: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
