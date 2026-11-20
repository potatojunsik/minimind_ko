"""
데이터 양 실험(docs/06)의 vocab 6,400과 16,000이 비용에서 얼마나 다른지 정리한다.

  [1] 파라미터 구성: 100만 모델 가중치를 모델에 올려 임베딩(lm_head와 같은 가중치를 함께 씀), 어텐션, MLP,
      정규화로 나눠 센다.
  [2] 학습 step당 시간: out_scale/status.json의 학습 시작~종료 시각 ÷ step 수.
      학습 시작 준비(데이터·모델 적재)와 마지막 저장 시간도 들어 있다.
      끊겨서 이어서 학습했거나 설정이 다른 실행은 비교에서 빼고 이유를 출력한다.
  [3] 100만 학습 loss 구간 평균: 그래프(images/scale_loss_1m.png) 끝부분이 아직 내려가는 중인지 숫자로 본다.
      로그는 50 step마다 batch 하나의 loss라 출렁임이 커서, 구간 평균과 표준오차(표준편차 ÷ √기록 수)를 함께 낸다.
  [4] --bench: GPU에서 두 100만 모델의 생성 속도(토큰/초, 글자/초)와 최대 메모리를 잰다 (약 2분).
      설정은 scripts/compare_generation.py와 같은 샘플링이고, 같은 프롬프트를 두 모델로 바로 이어서 재어 짝으로 비교한다.
      시작 전에 nvidia-smi로 GPU가 비어 있는지 확인해 그 값을 출력하고(이미 500MiB 넘게 쓰고 있으면 건너뜀),
      측정 전·중·후 CPU 부하를 기록한다. batch 1 생성은 CPU가 바쁘면 느려지기 때문이다.

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_costs.py
    .venv\\Scripts\\python.exe -X utf8 scripts\\scale_costs.py --bench --out out_scale\\costs.json
    (scripts\\plot_loss.py를 가져다 쓰므로 --bench 없이 돌려도 matplotlib이 필요하다)
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import time
from datetime import datetime

import psutil
import torch

from compare_generation import PROMPTS, REPETITION_PENALTY, TEMPERATURE, TOP_K, TOP_P, load_model
from plot_loss import read_log

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOCABS = [6400, 16000]
TOKENIZER_DIR = "tokenizers/ko_{vocab}"
WEIGHT_1M = "out_scale/ko{vocab}_n1000000/pretrain_ko{vocab}_768.pth"
TRAIN_LOG = "logs/06_scale_ko{vocab}_n{size}.log"
STATUS = "out_scale/status.json"
DEFAULT_NUM_WORKERS = 4  # train_pretrain_ko.py 기본값
COMMAND_PREFIX = "명령:"  # run_scale.py가 학습을 띄울 때마다 로그에 남기는 줄
STEP_PATTERN = re.compile(r"\((\d+)/(\d+)\), loss:")
NUM_WORKERS_PATTERN = re.compile(r"--num_workers (\d+)")
# 앞부분은 크게, 끝부분은 잘게 나눠 '마지막까지 내려가는 중인지'를 본다 (100만 = 62,500 step)
LOSS_BINS = [(1, 10000), (10001, 20000), (20001, 30000), (30001, 40000), (40001, 50000),
             (50001, 55000), (55001, 60000), (60001, 62500)]
GPU_BUSY_MIB = 500
NEW_TOKENS = 256  # 프롬프트마다 새로 만드는 토큰 수 (eos가 나와도 멈추지 않아 두 모델이 같은 토큰 수를 만든다)
SPEED_REPS = 5  # 반복마다 두 모델 순서를 바꾼다
SEED = 0
MiB = 2 ** 20


def size_label(n):
    return f"{n // 10000}만"


# ---------------------------------------------------------------- [1] 파라미터 구성
PARAM_GROUPS = ["임베딩 (= lm_head)", "어텐션 (8층)", "MLP (8층)", "정규화"]


def param_group(name):
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return PARAM_GROUPS[0]
    if ".self_attn." in name:
        return PARAM_GROUPS[1]
    if ".mlp." in name:
        return PARAM_GROUPS[2]
    return PARAM_GROUPS[3]


def param_breakdown():
    rows = {}
    for vocab in VOCABS:
        weight = WEIGHT_1M.format(vocab=vocab)
        tok, model = load_model(TOKENIZER_DIR.format(vocab=vocab), weight, "cpu")
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("임베딩과 lm_head가 같은 가중치가 아님 (tie_word_embeddings 확인)")
        groups = dict.fromkeys(PARAM_GROUPS, 0)
        # 같은 가중치를 함께 쓰는 임베딩과 lm_head는 named_parameters()에서 한 번만 나온다
        for name, p in model.named_parameters():
            groups[param_group(name)] += p.numel()
        rows[vocab] = {"vocab_size": len(tok), "hidden_size": model.config.hidden_size, "groups": groups,
                       "total": sum(groups.values()), "file_mib": os.path.getsize(os.path.join(ROOT, weight)) / MiB}
    a, b = rows[VOCABS[0]], rows[VOCABS[1]]
    print("\n[1] 파라미터 구성 (100만 모델 가중치)\n")
    print("| 구성 | vocab 6,400 | vocab 16,000 | 차이 |")
    print("|---|---|---|---|")
    for g in a["groups"]:
        print(f"| {g} | {a['groups'][g]:,} ({a['groups'][g] / a['total']:.1%}) "
              f"| {b['groups'][g]:,} ({b['groups'][g] / b['total']:.1%}) | {b['groups'][g] - a['groups'][g]:+,} |")
    print(f"| 합계 | {a['total']:,} | {b['total']:,} | {b['total'] - a['total']:+,} |")
    print(f"| 가중치 파일 크기 (fp16) | {a['file_mib']:.1f} MiB | {b['file_mib']:.1f} MiB "
          f"| {b['file_mib'] - a['file_mib']:+.1f} MiB |")
    expected = (b["vocab_size"] - a["vocab_size"]) * a["hidden_size"]
    print(f"\n차이 {b['total'] - a['total']:,} = (vocab {b['vocab_size']:,} − {a['vocab_size']:,}) × hidden "
          f"{a['hidden_size']} = {expected:,} → {'임베딩 행이 늘어난 만큼과 같음' if expected == b['total'] - a['total'] else '다름'}")
    return rows


# ---------------------------------------------------------------- [2] 학습 step당 시간
def read_train_log(path):
    """로그에서 학습을 띄운 명령 줄들과 전체 step 수를 읽는다."""
    commands, total_steps = [], None
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(COMMAND_PREFIX):
                commands.append(line)
            m = STEP_PATTERN.search(line)
            if m:
                total_steps = int(m.group(2))
    return commands, total_steps


def exclusion_reasons(run, commands):
    reasons = []
    if len(commands) > 1 or run.get("resumes"):
        reasons.append(f"학습이 {len(commands)}번에 나뉘어 돌았다(끊긴 뒤 이어서 학습). 시작~종료 시각에 멈춰 있던 시간이 "
                       "들어가고, 끊기기 전 마지막 체크포인트 뒤의 step은 다시 계산했다")
    workers = {int(m.group(1)) if m else DEFAULT_NUM_WORKERS
               for m in (NUM_WORKERS_PATTERN.search(c) for c in commands)}
    if workers != {DEFAULT_NUM_WORKERS}:
        reasons.append(f"데이터 로더 작업 프로세스 수가 다르다(num_workers {', '.join(map(str, sorted(workers)))}, 나머지 실행은 "
                       f"{DEFAULT_NUM_WORKERS})")
    return reasons


def train_speed():
    with open(os.path.join(ROOT, STATUS), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    sizes = sorted({int(k.split("_n")[1]) for k in runs})
    table, excluded = {}, {}
    for size in sizes:
        for vocab in VOCABS:
            run = runs[f"ko{vocab}_n{size}"]
            commands, steps = read_train_log(TRAIN_LOG.format(vocab=vocab, size=size))
            reasons = exclusion_reasons(run, commands)
            if run["state"] != "done":
                reasons.append(f"학습이 끝나지 않았다(state {run['state']})")
            if reasons:
                excluded[f"{vocab:,}·{size_label(size)}"] = reasons
                continue
            start, end = datetime.fromisoformat(run["train_start"]), datetime.fromisoformat(run["train_end"])
            table.setdefault(size, {"steps": steps})[vocab] = (end - start).total_seconds() / steps

    print("\n[2] 학습 step당 시간 (학습 시작~종료 시각 ÷ step 수, batch 16 × 512토큰)\n")
    print("| 학습 조각 수 | step | 6,400 (초/step) | 16,000 (초/step) | 16,000 ÷ 6,400 |")
    print("|---|---|---|---|---|")
    ratios = []
    for size, row in table.items():
        cells = [f"{row[v]:.3f}" if v in row else "비교에서 뺌" for v in VOCABS]
        ratio = ""
        if all(v in row for v in VOCABS):
            ratios.append(row[VOCABS[1]] / row[VOCABS[0]])
            ratio = f"{ratios[-1]:.3f}"
        print(f"| {size_label(size)} | {row['steps']:,} | {cells[0]} | {cells[1]} | {ratio} |")
    if ratios:
        print(f"\n짝이 맞는 {len(ratios)}쌍: 16,000이 평균 {statistics.mean(ratios):.3f}배 "
              f"(범위 {min(ratios):.3f}~{max(ratios):.3f})")
    else:
        print("\n두 vocab이 모두 비교에 들어간 크기가 없어 비율을 내지 않는다.")
    for name, reasons in excluded.items():
        print(f"비교에서 뺀 실행 {name}:")
        for r in reasons:
            print(f"  - {r}")
    summary = {"seconds_per_step": table, "excluded": excluded}
    if ratios:
        summary.update(ratio_mean=statistics.mean(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
    return summary


# ---------------------------------------------------------------- [3] 100만 학습 loss 구간 평균
def loss_bins():
    series = {vocab: dict(zip(*read_log(os.path.join(ROOT, TRAIN_LOG.format(vocab=vocab, size=1000000)))))
              for vocab in VOCABS}
    print("\n[3] 100만 학습 loss 구간 평균 (토큰당 loss, 같은 step이 여러 번 찍혔으면 마지막 기록)\n")
    print("| 구간 (step) | 기록 수 | 6,400 평균 ± 표준오차 | 16,000 평균 ± 표준오차 | 16,000 ÷ 6,400 |")
    print("|---|---|---|---|---|")
    rows = []
    for lo, hi in LOSS_BINS:
        cells = {}
        for vocab, loss in series.items():
            xs = [v for k, v in loss.items() if lo <= k <= hi]
            cells[vocab] = (statistics.mean(xs), statistics.stdev(xs) / len(xs) ** 0.5, len(xs))
        (ma, sa, n), (mb, sb, _) = cells[VOCABS[0]], cells[VOCABS[1]]
        rows.append({"from": lo, "to": hi, "records": n, "mean_6400": ma, "se_6400": sa,
                     "mean_16000": mb, "se_16000": sb})
        print(f"| {lo:,} ~ {hi:,} | {n} | {ma:.3f} ± {sa:.3f} | {mb:.3f} ± {sb:.3f} | {mb / ma:.3f} |")
    return rows


# ---------------------------------------------------------------- [4] 생성 속도·메모리
def gpu_query(fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10, check=True)
    return out.stdout.strip().splitlines()[0]


def machine_state():
    battery = psutil.sensors_battery()
    return {"time": datetime.now().isoformat(timespec="seconds"),
            "cpu_percent_1s": psutil.cpu_percent(interval=1.0),
            "ac_power": None if battery is None else battery.power_plugged,
            "gpu_power_clock_temp_pstate": gpu_query("power.draw,clocks.sm,temperature.gpu,pstate")}


def generate_timed(tok, model, prompt):
    x = tok(tok.bos_token + prompt, return_tensors="pt", return_token_type_ids=False).to("cuda")
    torch.manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**x, max_new_tokens=NEW_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                         top_k=TOP_K, repetition_penalty=REPETITION_PENALTY, eos_token_id=None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    new_ids = out[0][x["input_ids"].shape[1]:].tolist()
    return seconds, len(new_ids), len(tok.decode(new_ids, skip_special_tokens=True))


def bench():
    used = int(gpu_query("memory.used").split()[0])
    if used > GPU_BUSY_MIB:
        print(f"\n[4] GPU를 이미 {used:,}MiB 쓰고 있어서(기준 {GPU_BUSY_MIB}MiB) 측정을 건너뜀. 다른 GPU 작업을 끝낸 뒤 다시 실행")
        return {"skipped": True, "gpu_used_mib": used}
    t_start = time.time()
    state = {"gpu_used_mib_before": used, "before": machine_state()}
    # 행렬곱 작업 공간(cuBLAS)을 먼저 잡아 두어야 첫 번째 모델 메모리에만 그 크기가 더해지지 않는다
    warm = torch.randn(64, 64, device="cuda")
    (warm @ warm).sum().item()

    memory = {}
    for vocab in VOCABS:
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        tok, model = load_model(TOKENIZER_DIR.format(vocab=vocab), WEIGHT_1M.format(vocab=vocab), "cuda")
        weights = torch.cuda.memory_allocated() - base
        torch.cuda.reset_peak_memory_stats()
        for prompt in PROMPTS:
            generate_timed(tok, model, prompt)
        memory[vocab] = {"weights_mib": weights / MiB, "peak_mib": (torch.cuda.max_memory_allocated() - base) / MiB}
        del model

    models = {vocab: load_model(TOKENIZER_DIR.format(vocab=vocab), WEIGHT_1M.format(vocab=vocab), "cuda")
              for vocab in VOCABS}
    for tok, model in models.values():
        generate_timed(tok, model, PROMPTS[0])
    state["after_warmup"] = machine_state()
    # 여기부터 측정 끝까지의 평균 CPU 사용률(전체, 이 프로세스)을 재기 위한 기준점
    this_process = psutil.Process()
    psutil.cpu_percent(interval=None)
    this_process.cpu_percent(interval=None)
    # 같은 프롬프트를 두 모델로 바로 이어서 재어 짝을 만든다. 잠깐 끼어든 다른 작업이 한쪽 모델에만 걸리는 일을 줄이려는 것이다.
    pairs = []
    generated = {vocab: {"tokens": 0, "chars": 0} for vocab in VOCABS}
    for rep in range(SPEED_REPS):
        for prompt in PROMPTS:
            seconds = {}
            for vocab in (VOCABS if rep % 2 == 0 else VOCABS[::-1]):
                tok, model = models[vocab]
                seconds[vocab], n_tokens, n_chars = generate_timed(tok, model, prompt)
                if rep == 0:  # 시드를 고정해서 반복마다 같은 글이 나온다
                    generated[vocab]["tokens"] += n_tokens
                    generated[vocab]["chars"] += n_chars
            pairs.append(seconds)
    state["cpu_percent_during_speed"] = psutil.cpu_percent(interval=None)
    # 프로세스 값은 코어 하나가 100%라서 논리 CPU 수로 나눠 전체 대비로 바꾼다
    state["cpu_percent_during_speed_this_process"] = this_process.cpu_percent(interval=None) / psutil.cpu_count()
    state["after"] = machine_state()

    a, b = VOCABS
    # 두 모델이 만든 토큰 수가 같으므로 토큰/초 비(16,000 ÷ 6,400) = 6,400 시간 ÷ 16,000 시간
    tok_ratio = [p[a] / p[b] for p in pairs]
    q1, _, q3 = statistics.quantiles(tok_ratio, n=4)
    ratio = statistics.median(tok_ratio)
    tok_per_s = {vocab: statistics.median(NEW_TOKENS / p[vocab] for p in pairs) for vocab in VOCABS}
    chars_per_token = {vocab: generated[vocab]["chars"] / generated[vocab]["tokens"] for vocab in VOCABS}
    # 같은 글자 수를 만드는 시간 비 = (토큰 하나에 담긴 글자 수 비)의 역수 ÷ 토큰/초 비
    same_chars_time = chars_per_token[a] / chars_per_token[b] / ratio
    return {"settings": {"prompts": PROMPTS, "new_tokens": NEW_TOKENS, "reps": SPEED_REPS, "dtype": "fp32",
                         "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                         "repetition_penalty": REPETITION_PENALTY, "seed": SEED},
            "memory": memory, "tokens_per_s": tok_per_s, "chars_per_token": chars_per_token,
            "tokens_per_s_ratio": ratio, "tokens_per_s_ratio_quartiles": [q1, q3], "same_chars_time_ratio": same_chars_time,
            "pair_seconds": pairs, "machine": state, "seconds": time.time() - t_start}


def report_bench(r):
    a, b = VOCABS
    tok_per_s, cpt, mem, state = r["tokens_per_s"], r["chars_per_token"], r["memory"], r["machine"]
    q1, q3 = r["tokens_per_s_ratio_quartiles"]
    print(f"\n[4] 생성 속도·메모리 (GPU, fp32, batch 1, KV 캐시, 프롬프트 {len(r['settings']['prompts'])}개 × "
          f"새 토큰 {r['settings']['new_tokens']}개, {r['settings']['reps']}회 반복 = 짝 {len(r['pair_seconds'])}개)\n")
    print("| 항목 | vocab 6,400 | vocab 16,000 | 16,000 ÷ 6,400 |")
    print("|---|---|---|---|")
    print(f"| 토큰/초 (모델별 중앙값) | {tok_per_s[a]:.1f} | {tok_per_s[b]:.1f} "
          f"| 짝 비의 중앙값 {r['tokens_per_s_ratio']:.3f} (가운데 절반 {q1:.3f}~{q3:.3f}) |")
    print(f"| 만든 글의 글자/토큰 (eos 뒤에 이어 만든 글 포함) | {cpt[a]:.3f} | {cpt[b]:.3f} | {cpt[b] / cpt[a]:.3f} |")
    print(f"| 같은 글자 수를 만드는 시간 (글자/토큰 비 ÷ 짝 비의 중앙값) | | | {r['same_chars_time_ratio']:.3f} |")
    print(f"| 모델 적재 메모리 (가중치 + RoPE 표) | {mem[a]['weights_mib']:.1f} MiB | {mem[b]['weights_mib']:.1f} MiB "
          f"| {mem[b]['weights_mib'] - mem[a]['weights_mib']:+.1f} MiB |")
    print(f"| 생성 중 최대 메모리 | {mem[a]['peak_mib']:.1f} MiB | {mem[b]['peak_mib']:.1f} MiB "
          f"| {mem[b]['peak_mib'] - mem[a]['peak_mib']:+.1f} MiB |")
    print("\n토큰/초 열은 모델마다 따로 낸 중앙값이고, 비율은 같은 프롬프트로 바로 이어서 잰 짝마다 나눈 값의 중앙값이다. "
          "그래서 두 열을 나눈 값과 비율이 조금 다를 수 있다.")
    print(f"측정 전 GPU 사용량 {state['gpu_used_mib_before']:,}MiB (기준 {GPU_BUSY_MIB}MiB 이하일 때만 측정)")
    print(f"CPU 사용률(논리 CPU {psutil.cpu_count()}개 전체): 측정 전 {state['before']['cpu_percent_1s']}%, "
          f"속도 측정 중 평균 {state['cpu_percent_during_speed']}% (그중 이 프로세스 "
          f"{state['cpu_percent_during_speed_this_process']:.1f}%), 측정 후 {state['after']['cpu_percent_1s']}% | "
          f"전원 연결 {state['before']['ac_power']}")
    print(f"GPU 전력·클럭·온도·상태: 측정 전 {state['before']['gpu_power_clock_temp_pstate']} / "
          f"측정 후 {state['after']['gpu_power_clock_temp_pstate']} | 걸린 시간 {r['seconds']:.0f}초")


def main():
    p = argparse.ArgumentParser(description="vocab 6,400 / 16,000 비용 비교 (파라미터, 학습 시간, 생성 속도·메모리)")
    p.add_argument("--bench", action="store_true", help="GPU에서 생성 속도와 메모리도 잰다 (약 2분)")
    p.add_argument("--out", default="", help="결과를 저장할 JSON 경로 (선택)")
    args = p.parse_args()
    result = {"params": param_breakdown(), "train_speed": train_speed(), "loss_bins_1m": loss_bins()}
    if args.bench:
        result["bench"] = bench()
        if not result["bench"].get("skipped"):
            report_bench(result["bench"])
    if args.out:
        with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
