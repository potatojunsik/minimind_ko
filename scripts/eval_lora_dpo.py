"""
LoRA·DPO 체험의 측정 (설계: docs/superpowers/specs/2026-11-24-lora-dpo-design.md, docs/09). 판정 없이 기록만 한다.

  lora : (1) 자기소개 평가 12문항 답에 "MiniMind"가 들어간 수 (LoRA 전/후)
         (2) 07번 200문항을 LoRA를 켜고 생성: "MiniMind"가 들어간 답 수(자기소개가 일반 대화로 새는지), docs/08 기준 집계
         (3) LoRA를 합친(merge) 가중치로 대화 평가 2,001쌍 답 본문 BPB를 최종 SFT와 비교
         (4) LoRA 파라미터 수·비율, 파일 크기, LoRA로 답이 바뀐 자기소개 문항 수(켜졌는지 확인),
             합친 모델과 켠 모델의 차이(logit 최대 차이, 데모 10문항 greedy 일치 수. 원본 merge는 fp16 반올림)
         → docs/09_lora_results.json
  dpo  : (1) 07번 200문항 docs/08 기준 집계 (DPO 전은 docs/08_demo_results.json 값)
         (2) 대화 평가 2,001쌍 답 본문 BPB·답 끝 EOS 손실을 최종 SFT와 비교
         (3) README 데모 10문항(docs/08_demo_questions.json) 답 전/후
         → docs/09_dpo_results.json

생성은 docs/08과 같은 greedy(blind_chat_eval.generate_one, 새 토큰 상한 96).

사용법 (minimind_ko 폴더에서 실행):
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_lora_dpo.py lora
    .venv\\Scripts\\python.exe -X utf8 scripts\\eval_lora_dpo.py dpo
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from blind_chat_eval import chat_prompt, generate_one  # noqa: E402
from chat_ko import answer_text  # noqa: E402
from final_demo import generation_stats  # noqa: E402

TOKENIZER = "tokenizers/ko_16000"
BASE = "out_final/sft/sft_ko16000_768.pth"
LORA = "out_lora/lora_identity_ko_768.pth"
MERGED = "out_lora/merged_identity_ko_768.pth"
DPO = "out_dpo/dpo_ko16000_768.pth"
IDENTITY = "docs/09_lora_identity_ko.json"
DEMO_QUESTIONS = "docs/08_demo_questions.json"
DEMO_RESULTS = "docs/08_demo_results.json"
BLIND_ITEMS = "data/chat/blind_items.json"
CHAT_PAIRS = "data/chat/eval_pairs.jsonl"
BASE_CHAT_EVAL = "data/chat/chat_eval_final_sft_ko16000.json"  # docs/08 최종 SFT의 대화 평가 결과
KEYWORD = "MiniMind"
LORA_RANK = 16


def read_json(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def answers(model, tokenizer, questions, device):
    return [answer_text(generate_one(model, tokenizer, q, device)) for q in questions]


def blind200(model, tokenizer, device):
    """07번 200문항 생성 결과 {id: generate_one 결과}."""
    return {str(it["id"]): generate_one(model, tokenizer, it["q"], device) for it in read_json(BLIND_ITEMS)}


def keyword_count(texts):
    return sum(KEYWORD in t for t in texts)


def chat_bpb_compare(weight, tag, device):
    """weight의 대화 평가를 최종 SFT 결과와 같은 쌍으로 짝지어 비교한다(기록용)."""
    from decide_final import eos_only
    from eval_chat_bpb import bpb, check_same_eval, evaluate, paired_delta, text_only
    from decide_vocab import load_jsonl
    res = evaluate(TOKENIZER, weight, load_jsonl(os.path.join(ROOT, CHAT_PAIRS)), device, "chat")
    write_json(f"data/chat/chat_eval_{tag}_ko16000.json", res)
    base = read_json(BASE_CHAT_EVAL)
    check_same_eval(base, res)
    groups = [r["group"] for r in res["pairs"]]
    new_t, old_t = text_only(res["pairs"]), text_only(base["pairs"])
    return {"bpb_text_only": bpb(new_t), "bpb_text_only_final_sft": bpb(old_t),
            "delta_text_only": paired_delta(new_t, old_t, groups),
            "eos_nll_mean": sum(r["nll_eos"] for r in res["pairs"]) / len(res["pairs"]),
            "eos_nll_mean_final_sft": sum(r["nll_eos"] for r in base["pairs"]) / len(base["pairs"]),
            "delta_eos_bits_per_answer": paired_delta(eos_only(res["pairs"]), eos_only(base["pairs"]), groups)}


def cmd_lora(device):
    import torch
    from eval_bpc import load_model
    from model.model_lora import apply_lora, load_lora, merge_lora
    from train_lora_ko import count_lora_params
    eval_q = [it["q"] for it in read_json(IDENTITY)["eval"]]
    demo_q = [it["q"] for it in read_json(DEMO_QUESTIONS)["items"]]

    tokenizer, model = load_model(TOKENIZER, BASE, device)
    before = answers(model, tokenizer, eval_q, device)
    demo_off = answers(model, tokenizer, demo_q, device)
    apply_lora(model, rank=LORA_RANK)
    load_lora(model, os.path.join(ROOT, LORA))
    model.eval()
    lora_params, total = count_lora_params(model)
    after = answers(model, tokenizer, eval_q, device)
    demo_on = answers(model, tokenizer, demo_q, device)
    blind_on = blind200(model, tokenizer, device)
    probe = tokenizer(chat_prompt(eval_q[0]), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        logits_on = model(probe).logits.float()
    merge_lora(model, os.path.join(ROOT, LORA), os.path.join(ROOT, MERGED))
    del model
    torch.cuda.empty_cache()
    _, merged = load_model(TOKENIZER, MERGED, device)
    demo_merged = answers(merged, tokenizer, demo_q, device)
    with torch.no_grad():
        # 원본 merge_lora는 W + B@A를 fp16으로 반올림해 저장하므로 켠 모델과 완전히 같지는 않다. 차이의 크기를 잰다
        merge_logit_diff = (merged(probe).logits.float() - logits_on).abs().max().item()
    del merged

    result = {
        "identity_eval": [{"q": q, "before": b, "after": a} for q, b, a in zip(eval_q, before, after)],
        "identity_keyword": {"before": keyword_count(before), "after": keyword_count(after), "items": len(eval_q)},
        "blind200_keyword_with_lora": keyword_count(o["text"] for o in blind_on.values()),
        "blind200_with_lora": generation_stats(blind_on),
        "blind200_final_sft": read_json(DEMO_RESULTS)["blind200"]["final_sft"],
        "chat_bpb": chat_bpb_compare(MERGED, "lora_merged", device),
        "params": {"lora": lora_params, "total_with_lora": total, "share": lora_params / total, "rank": LORA_RANK},
        "file_bytes": {"lora": os.path.getsize(os.path.join(ROOT, LORA)),
                       "full": os.path.getsize(os.path.join(ROOT, BASE))},
        "demo": [{"q": q, "off": f, "on": o, "merged": m} for q, f, o, m in zip(demo_q, demo_off, demo_on, demo_merged)],
        "demo_same_greedy_merged_vs_on": sum(o == m for o, m in zip(demo_on, demo_merged)),
        "merge_max_abs_logit_diff": merge_logit_diff,
        "identity_changed_by_lora": sum(b != a for b, a in zip(before, after)),
    }
    write_json("docs/09_lora_results.json", result)
    k = result["identity_keyword"]
    print(f"자기소개 평가 {k['items']}문항 중 '{KEYWORD}' 포함: {k['before']} → {k['after']}")
    for row in result["identity_eval"]:
        print(f"  {row['q']}\n    전: {row['before']}\n    후: {row['after']}")
    print(f"200문항(LoRA 켬) '{KEYWORD}' 포함 {result['blind200_keyword_with_lora']}, 집계 {_brief(result['blind200_with_lora'])}")
    d = result["chat_bpb"]["delta_text_only"]
    print(f"대화 답 본문 BPB {result['chat_bpb']['bpb_text_only']:.4f} vs {result['chat_bpb']['bpb_text_only_final_sft']:.4f}, "
          f"Δ {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]")
    p = result["params"]
    print(f"LoRA 파라미터 {p['lora']:,} / {p['total_with_lora']:,} ({p['share'] * 100:.2f}%), 파일 "
          f"{result['file_bytes']['lora']:,} vs {result['file_bytes']['full']:,} bytes")
    print(f"LoRA로 답이 바뀐 자기소개 문항 {result['identity_changed_by_lora']}/{len(eval_q)}, "
          f"합친 모델과 켠 모델: 데모 10문항 greedy 일치 {result['demo_same_greedy_merged_vs_on']}/10, "
          f"logit 최대 차이 {merge_logit_diff:.2e}")
    print("저장: docs/09_lora_results.json")


def _brief(s):
    return (f"잘림 {s['truncated']} | 서로 다른 답 {s['distinct_answers']} | "
            f"3번 이상 반복 문항 {s['items_with_answer_repeated_3plus']} | {s['phrases']}")


def cmd_dpo(device):
    from eval_bpc import load_model
    demo_before = read_json(DEMO_RESULTS)
    tokenizer, model = load_model(TOKENIZER, DPO, device)
    demo_q = [it["q"] for it in demo_before["demo"]]
    demo_after = answers(model, tokenizer, demo_q, device)
    blind = blind200(model, tokenizer, device)
    del model
    result = {
        "blind200_final_sft": demo_before["blind200"]["final_sft"],
        "blind200_dpo": generation_stats(blind),
        "chat_bpb": chat_bpb_compare(DPO, "dpo", device),
        "demo": [{"q": it["q"], "before": it["answer"], "after": a} for it, a in zip(demo_before["demo"], demo_after)],
        "blind200_outputs": blind,
    }
    write_json("docs/09_dpo_results.json", result)
    print(f"200문항 DPO 전: {_brief(result['blind200_final_sft'])}")
    print(f"200문항 DPO 후: {_brief(result['blind200_dpo'])}")
    c = result["chat_bpb"]
    d, e = c["delta_text_only"], c["delta_eos_bits_per_answer"]
    print(f"대화 답 본문 BPB {c['bpb_text_only']:.4f} vs {c['bpb_text_only_final_sft']:.4f}, Δ {d['delta']:+.4f} "
          f"[{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] | EOS 손실 {c['eos_nll_mean']:.4f} vs {c['eos_nll_mean_final_sft']:.4f} nats, "
          f"Δ {e['delta']:+.4f}비트 [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]")
    for row in result["demo"]:
        print(f"  {row['q']}\n    전: {row['before']}\n    후: {row['after']}")
    print("저장: docs/09_dpo_results.json")


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["lora", "dpo"])
    args = p.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    {"lora": cmd_lora, "dpo": cmd_dpo}[args.step](device)


if __name__ == "__main__":
    main()
