"""LoRA·DPO 체험(docs/09)의 데이터·LoRA 구조·DPO 쌍과 마스크 테스트. 학습은 하지 않는다.

실행 (minimind_ko 폴더에서):
    .venv\\Scripts\\python.exe -X utf8 -m unittest tests.test_lora_dpo -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "minimind"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from prepare_chat_eval import normalize  # noqa: E402


def load_jsonl(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(ROOT, "tokenizers", "ko_16000"))


class IdentityDataTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "docs", "09_lora_identity_ko.json"), encoding="utf-8") as f:
            self.data = json.load(f)

    def test_counts_and_keyword(self):
        self.assertEqual((len(self.data["train"]), len(self.data["eval"])), (48, 12))
        self.assertTrue(all("MiniMind" in r["a"] for r in self.data["train"]))

    def test_eval_questions_are_unseen(self):
        seen = {normalize(r["q"]) for r in self.data["train"]}
        seen |= {normalize(r["q"]) for r in load_jsonl("data/chat/sft_pool.jsonl")}
        seen |= {normalize(r["q"]) for r in load_jsonl("data/chat/eval_pairs.jsonl")}
        self.assertEqual([r["q"] for r in self.data["eval"] if normalize(r["q"]) in seen], [])

    def test_train_questions_not_in_chat_evaluation(self):
        """학습 질문이 대화 평가 2,001쌍에 있으면 부작용 측정(BPB)이 오염된다."""
        ev = {normalize(r["q"]) for r in load_jsonl("data/chat/eval_pairs.jsonl")}
        self.assertEqual([r["q"] for r in self.data["train"] if normalize(r["q"]) in ev], [])

    def test_fits_max_seq_len(self):
        from train_sft_ko import chat_text
        tok = tokenizer()
        longest = max(len(tok(chat_text(r["q"], r["a"]), add_special_tokens=False).input_ids) for r in self.data["train"])
        self.assertLessEqual(longest, 128)


class LoraParamTest(unittest.TestCase):
    def test_lora_on_square_linears_only(self):
        from model.model_lora import apply_lora
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        from train_lora_ko import count_lora_params, freeze_except_lora
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, vocab_size=16000))
        apply_lora(model, rank=16)
        lora, _ = count_lora_params(model)
        self.assertEqual(lora, 8 * 2 * (768 * 16 + 16 * 768))  # 층마다 q_proj, o_proj
        trainable = freeze_except_lora(model)
        self.assertEqual(sum(p.numel() for p in trainable), lora)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), lora)


class DpoDataTest(unittest.TestCase):
    def test_stock_threshold_and_same_answer_excluded(self):
        from make_dpo_data import select_pairs
        rows = [{"q": f"q{i}", "a": "정답이에요."} for i in range(40)]
        answers = [{"text": "그런 사람 만나지 마세요.", "stopped": True}] * 20 + \
                  [{"text": "다른 답이에요.", "stopped": True}] * 19 + [{"text": "정답 이에요!", "stopped": True}]
        pairs, stock = select_pairs(rows, answers, min_count=20)
        self.assertEqual(stock, {"그런 사람 만나지 마세요.": 20})
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(p["rejected"] == "그런 사람 만나지 마세요." and p["chosen"] == "정답이에요." for p in pairs))
        same, _ = select_pairs(rows[:20], [{"text": "정답 이에요!", "stopped": True}] * 20, min_count=20)
        self.assertEqual(same, [])

    def test_mask_covers_answer_and_end_only(self):
        from train_dpo_ko import ChatDPODataset
        tok = tokenizer()
        ds = ChatDPODataset([{"q": "오늘 힘들었어", "chosen": "고생 많았어요.", "rejected": "그런 사람 만나지 마세요."}], tok, 128)
        item = ds[0]
        for kind, answer in (("chosen", "고생 많았어요."), ("rejected", "그런 사람 만나지 마세요.")):
            y, mask = item[f"y_{kind}"], item[f"mask_{kind}"]
            self.assertEqual(len(y), 127)
            scored = [int(t) for t, m in zip(y, mask) if int(m) == 1]
            self.assertEqual(scored, tok(answer + f"{tok.eos_token}\n", add_special_tokens=False).input_ids)


if __name__ == "__main__":
    unittest.main()
