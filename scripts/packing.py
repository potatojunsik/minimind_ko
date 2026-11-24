"""
사전학습 패킹: 조각마다 [bos] + 텍스트 토큰 + [eos]를 만들어 한 줄로 이어 붙이고 seq_len씩 잘라 줄을 만든다
(설계: docs/superpowers/specs/2026-11-23-final-pretrain-design.md "패킹 방식").

- 조각은 자르거나 버리지 않는다. seq_len보다 긴 조각도 다음 줄로 이어진다.
- 마지막 줄만 남는 칸을 pad로 채우고 loss에서 뺀다.
- 조각 경계에서 어텐션을 끊지 않는다(위치 번호도 줄 처음부터 이어진다).
- 라벨에서 pad와 bos 위치를 뺀다. bos는 항상 eos 바로 뒤라 맞히는 데 정보가 없다. eos는 채점한다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100
TOKEN_DTYPE = np.uint16  # vocab 16,000이면 2바이트로 충분하다. 학습 풀 전체(약 3억 토큰)를 약 600MB로 들고 있는다


def _to_array(token_lists, bos_id, eos_id):
    """조각별 토큰 리스트를 [bos]+토큰+[eos]로 이은 1차원 배열로 만든다."""
    parts = []
    limit = np.iinfo(TOKEN_DTYPE).max
    for tokens in token_lists:
        arr = np.asarray([bos_id] + list(tokens) + [eos_id], dtype=np.int64)
        if arr.size and (arr.min() < 0 or arr.max() > limit):
            raise ValueError(f"토큰 id가 {TOKEN_DTYPE.__name__} 범위(0~{limit})를 벗어남: {arr.min()}~{arr.max()}")
        parts.append(arr.astype(TOKEN_DTYPE))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=TOKEN_DTYPE)


def rows_from_stream(stream, pad_id, seq_len):
    """1차원 토큰 열을 seq_len 줄로 자른다. 마지막 줄만 pad로 채운다. (rows, pad 수) 반환."""
    if stream.size == 0:
        return np.zeros((0, seq_len), dtype=TOKEN_DTYPE), 0
    pad = (-stream.size) % seq_len
    if pad:
        stream = np.concatenate([stream, np.full(pad, pad_id, dtype=TOKEN_DTYPE)])
    return stream.reshape(-1, seq_len), pad


def pack_token_lists(token_lists, bos_id, eos_id, pad_id, seq_len):
    rows, _ = rows_from_stream(_to_array(token_lists, bos_id, eos_id), pad_id, seq_len)
    return rows


def tokenize_and_pack(texts, tokenizer, seq_len, batch_size=10000):
    """텍스트를 배치로 토큰화해 바로 numpy로 옮기고 패킹한다. (rows, stats) 반환.

    토큰을 Python int 리스트로 전부 들고 있으면 학습 풀 전체에서 수 GB가 들어서, 배치마다 배열로 바꾼다.
    """
    bos_id, eos_id, pad_id = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    pieces = []
    for start in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[start:start + batch_size]]
        pieces.append(_to_array(tokenizer(batch, add_special_tokens=False).input_ids, bos_id, eos_id))
    stream = np.concatenate(pieces) if pieces else np.zeros(0, dtype=TOKEN_DTYPE)
    rows, pad = rows_from_stream(stream, pad_id, seq_len)
    stats = {"chunks": len(texts), "tokens": int(stream.size), "rows": int(rows.shape[0]), "pad": int(pad)}
    return rows, stats


class PackedPretrainDataset(Dataset):
    def __init__(self, rows, bos_id, pad_id):
        self.rows = rows
        self.bos_id = bos_id
        self.pad_id = pad_id

    def __len__(self):
        return self.rows.shape[0]

    def __getitem__(self, index):
        input_ids = torch.from_numpy(self.rows[index].astype(np.int64))
        labels = input_ids.clone()
        labels[(input_ids == self.pad_id) | (input_ids == self.bos_id)] = IGNORE_INDEX
        return input_ids, labels
