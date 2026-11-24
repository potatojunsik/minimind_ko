"""
최종 사전학습 판정 (설계: docs/superpowers/specs/2026-11-23-final-pretrain-design.md, 결과를 보기 전에 확정).

    BPB = 손실 합(nats) / ln2 / UTF-8 바이트 합. 구간은 검증 문서 단위 짝 부트스트랩(decide_vocab.py와 같은 계산).

setup (실험 1·2 → 최종 학습 설정):
  패킹 채택:   Δp = BPB(40만 패킹) − BPB(40만 기존)의 95% 구간 상한 < +0.005
  2 epoch 채택: g = BPB(20만×1) − BPB(20만×2), G = BPB(20만×1) − BPB(40만×1)일 때
               g/G ≥ 0.5 (점추정) 이고 g의 95% 구간 하한 > 0 이고 BPB(2 epoch 끝) < BPB(1 epoch 끝 시점)
accept (최종 학습 → 최종 모델):
  Δf = BPB(최종) − BPB(16,000·100만 기존)의 95% 구간 상한 < 0 이면 새 모델 사용
  참고(판정 아님): 2 epoch이면 1 epoch 끝 가중치 BPB와 차이를 함께 기록한다
sft (기록용, 판정 아님): 최종 SFT 모델의 대화 형식 답 본문 BPB와 답 끝 eos 손실을 07번 16,000 SFT 결과와 같은 평가 쌍으로 비교한다

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\decide_final.py setup
    .venv\\Scripts\\python.exe -X utf8 scripts\\decide_final.py accept
    .venv\\Scripts\\python.exe -X utf8 scripts\\decide_final.py sft --weight out_final/sft/sft_ko16000_768.pth
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide_vocab import ROUNDS, SEED, bootstrap_ids, ci, group_by_doc, load_jsonl, rate  # noqa: E402
from run_scale import HELDOUT, expected_eval  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK_MARGIN = 0.005
EPOCH_SHARE = 0.5
OLD = {"400k": "out_scale/ko16000_n400000/per_doc.jsonl", "200k": "out_scale/ko16000_n200000/per_doc.jsonl",
       "1m": "out_scale/ko16000_n1000000/per_doc.jsonl"}
OLD_STATUS = "out_scale/status.json"
OLD_KEYS = {"400k": "ko16000_n400000", "200k": "ko16000_n200000", "1m": "ko16000_n1000000"}
CHAT_PAIRS = "data/chat/eval_pairs.jsonl"
SFT_BASELINE = "data/chat/chat_eval_chat_sft_ko16000.json"  # 07번 16,000 SFT (같은 SFT 설정)
TOKENIZER = "tokenizers/ko_16000"


def decide_packing(hi):
    return hi < PACK_MARGIN


def decide_epochs(g, big_g, g_lo, bpb_final, bpb_epoch1):
    return big_g > 0 and g / big_g >= EPOCH_SHARE and g_lo > 0 and bpb_final < bpb_epoch1


def decide_accept(hi):
    return hi < 0


def paired(groups_a, groups_b, unique_ids, samples):
    """BPB(a) − BPB(b)의 점추정과 95% 구간, 그리고 부트스트랩 값 목록."""
    point = rate(groups_a, unique_ids, "bytes") - rate(groups_b, unique_ids, "bytes")
    boot = [rate(groups_a, ids, "bytes") - rate(groups_b, ids, "bytes") for ids in samples]
    lo, hi = ci(boot)
    return {"point": point, "ci": [lo, hi]}


def load_groups(path, doc_ids, prefix=False):
    """평가 결과(조각별 행)를 문서별로 묶는다. 행 수가 검증 조각 수와 다르면 멈춘다(앞부분만 평가한 시험 결과 방지).
    prefix=True는 시험 실행(--limit) 전용: 06번 전체 평가 결과의 앞부분만 쓴다."""
    rows = load_jsonl(path)
    if prefix:
        rows = rows[:len(doc_ids)]
    if len(rows) != len(doc_ids):
        raise SystemExit(f"{path}: 평가 행 {len(rows)}개 ≠ 검증 조각 {len(doc_ids)}개")
    return group_by_doc(rows, doc_ids)


def check_old_runs(keys, status_path=OLD_STATUS):
    """06번 결과가 같은 검증 세트 전체(문서 수·sha256)로 평가됐는지 status.json 기록으로 확인한다."""
    with open(os.path.join(ROOT, status_path), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    n, sha = expected_eval(0)
    for key in keys:
        r = runs[OLD_KEYS[key]]
        if r.get("state") != "done" or r.get("eval_docs") != n or r.get("eval_sha256") != sha:
            raise SystemExit(f"{OLD_KEYS[key]}: 같은 검증 세트 전체로 평가된 기록이 아님")


def setup(out_root, doc_ids, samples, unique_ids, prefix=False):
    check_old_runs(["400k", "200k"])
    g400, g200 = load_groups(OLD["400k"], doc_ids, prefix), load_groups(OLD["200k"], doc_ids, prefix)
    pack = load_groups(f"{out_root}/exp_pack400k/per_doc.jsonl", doc_ids)
    ep2 = load_groups(f"{out_root}/exp_ep2_200k/per_doc.jsonl", doc_ids)
    ep1 = load_groups(f"{out_root}/exp_ep2_200k/per_doc_epoch1.jsonl", doc_ids)

    dp = paired(pack, g400, unique_ids, samples)
    g = paired(g200, ep2, unique_ids, samples)
    big_g = rate(g200, unique_ids, "bytes") - rate(g400, unique_ids, "bytes")
    bpb = {name: rate(gr, unique_ids, "bytes") for name, gr in
           [("old_200k", g200), ("old_400k", g400), ("pack_400k", pack), ("ep2_200k", ep2), ("ep2_200k_epoch1", ep1)]}
    use_pack = decide_packing(dp["ci"][1])
    use_ep2 = decide_epochs(g["point"], big_g, g["ci"][0], bpb["ep2_200k"], bpb["ep2_200k_epoch1"])
    # G ≤ 0이면 비율이 뜻이 없다(decide_epochs도 그때는 채택하지 않는다). 0으로 나누지 않고 None으로 남긴다
    out = {"bpb": bpb, "delta_pack": dp, "g": g, "G": big_g, "g_over_G": g["point"] / big_g if big_g > 0 else None,
           "pack": use_pack, "epochs": 2 if use_ep2 else 1}
    print(f"BPB: {', '.join(f'{k} {v:.4f}' for k, v in bpb.items())}")
    print(f"패킹: Δp {dp['point']:+.4f} [{dp['ci'][0]:+.4f}, {dp['ci'][1]:+.4f}] → "
          f"{'채택' if use_pack else '채택 안 함'} (기준: 상한 < +{PACK_MARGIN})")
    ratio = "없음(G ≤ 0)" if out["g_over_G"] is None else f"{out['g_over_G']:.3f}"
    print(f"2 epoch: g {g['point']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}], G {big_g:.4f}, g/G {ratio}, "
          f"2 epoch 끝 {bpb['ep2_200k']:.4f} vs 1 epoch 끝 {bpb['ep2_200k_epoch1']:.4f} → "
          f"{'채택' if use_ep2 else '채택 안 함'} (기준: g/G ≥ {EPOCH_SHARE}, 하한 > 0, 2 epoch 끝 < 1 epoch 끝)")
    print(f"최종 학습 설정: 패킹 {'사용' if use_pack else '안 함'}, {out['epochs']} epoch")
    return out


def accept(out_root, doc_ids, samples, unique_ids, prefix=False):
    check_old_runs(["1m"])
    old = load_groups(OLD["1m"], doc_ids, prefix)
    new = load_groups(f"{out_root}/final/per_doc.jsonl", doc_ids)
    df = paired(new, old, unique_ids, samples)
    use_new = decide_accept(df["ci"][1])
    out = {"bpb_new": rate(new, unique_ids, "bytes"), "bpb_old_1m": rate(old, unique_ids, "bytes"),
           "delta": df, "use_new": use_new}
    print(f"최종: BPB 새 모델 {out['bpb_new']:.4f} vs 기존 100만 {out['bpb_old_1m']:.4f}, "
          f"Δf {df['point']:+.4f} [{df['ci'][0]:+.4f}, {df['ci'][1]:+.4f}] → "
          f"{'새 모델 사용' if use_new else '기존 모델 유지'} (기준: 상한 < 0)")
    epoch1_path = f"{out_root}/final/per_doc_epoch1.jsonl"
    if os.path.exists(os.path.join(ROOT, epoch1_path)):
        ep1 = load_groups(os.path.join(ROOT, epoch1_path), doc_ids)
        d21 = paired(new, ep1, unique_ids, samples)
        out["reference_epoch1"] = {"bpb_epoch1": rate(ep1, unique_ids, "bytes"), "bpb_epoch2": out["bpb_new"],
                                   "delta_epoch2_minus_epoch1": d21}
        print(f"참고(판정 아님): 1 epoch 끝 {out['reference_epoch1']['bpb_epoch1']:.4f} → 2 epoch 끝 {out['bpb_new']:.4f}, "
              f"차이 {d21['point']:+.4f} [{d21['ci'][0]:+.4f}, {d21['ci'][1]:+.4f}]")
    return out


def eos_only(rows):
    """답 끝 eos 손실만 남긴 행. bytes를 1로 두면 rate가 답 하나당 평균 eos 손실(비트)이 된다."""
    return [dict(r, nll=r["nll_eos"], bytes=1) for r in rows]


def sft_record(weight, tag):
    """최종 SFT 모델을 대화 형식으로 평가해 저장하고, 07번 16,000 SFT와 답 본문 BPB를 짝지어 비교한다(기록용)."""
    import torch
    from eval_chat_bpb import bpb, check_same_eval, evaluate, paired_delta, text_only
    res = evaluate(TOKENIZER, weight, load_jsonl(CHAT_PAIRS), "cuda:0" if torch.cuda.is_available() else "cpu", "chat")
    path = os.path.join(ROOT, "data", "chat", f"chat_eval_{tag}_ko16000.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    with open(os.path.join(ROOT, SFT_BASELINE), encoding="utf-8") as f:
        base = json.load(f)
    check_same_eval(base, res)
    groups = [r["group"] for r in res["pairs"]]
    new_t, old_t = text_only(res["pairs"]), text_only(base["pairs"])
    d = paired_delta(new_t, old_t, groups)
    d_eos = paired_delta(eos_only(res["pairs"]), eos_only(base["pairs"]), groups)
    out = {"weight": weight, "result": path, "bpb_text_only_new": bpb(new_t), "bpb_text_only_07": bpb(old_t),
           "delta_text_only": d,
           "bpb_with_eos_new": bpb(res["pairs"]), "bpb_with_eos_07": bpb(base["pairs"]),
           "delta_with_eos": paired_delta(res["pairs"], base["pairs"], groups),
           "eos_nll_mean_new": sum(r["nll_eos"] for r in res["pairs"]) / len(res["pairs"]),
           "eos_nll_mean_07": sum(r["nll_eos"] for r in base["pairs"]) / len(base["pairs"]),
           "delta_eos_bits_per_answer": d_eos}
    print(f"최종 SFT 답 본문 BPB {out['bpb_text_only_new']:.4f} vs 07번 16,000 SFT {out['bpb_text_only_07']:.4f}, "
          f"Δ {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] (기록용, 판정 아님)")
    print(f"답 끝 eos 손실(답 하나당 평균, nats) {out['eos_nll_mean_new']:.4f} vs {out['eos_nll_mean_07']:.4f}, "
          f"Δ(비트) {d_eos['delta']:+.4f} [{d_eos['ci'][0]:+.4f}, {d_eos['ci'][1]:+.4f}] (기록용, 판정 아님)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["setup", "accept", "sft"])
    p.add_argument("--weight", default="", help="sft 단계: 최종 SFT 가중치")
    p.add_argument("--tag", default="final_sft", help="sft 단계: 결과 파일 이름 (data/chat/chat_eval_<tag>_ko16000.json)")
    p.add_argument("--out_root", default="out_final")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--limit", type=int, default=0, help="시험 실행 전용: 검증 앞 N조각만 쓴다 (0이면 전체)")
    args = p.parse_args()

    doc_ids = [row["doc_id"] for row in load_jsonl(HELDOUT)]
    if args.limit > 0:
        doc_ids = doc_ids[:args.limit]
    unique_ids = list(dict.fromkeys(doc_ids))
    samples = bootstrap_ids(unique_ids, args.rounds, SEED)
    path = os.path.join(ROOT, args.out_root, "decision.json")
    decision = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            decision = json.load(f)
    if args.step == "setup":
        decision["setup"] = setup(args.out_root, doc_ids, samples, unique_ids, args.limit > 0)
    elif args.step == "sft":
        if not args.weight:
            raise SystemExit("sft 단계는 --weight가 필요하다")
        decision["sft"] = sft_record(args.weight, args.tag)
    else:
        decision["accept"] = accept(args.out_root, doc_ids, samples, unique_ids, args.limit > 0)
    decision[args.step].update({"rounds": args.rounds, "seed": SEED, "limit": args.limit})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
