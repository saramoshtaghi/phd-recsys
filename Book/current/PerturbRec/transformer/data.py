"""Vocab, tokenization, Dataset for the transformer recommender.

Each user -> a sequence of (book_token, rating_value_bin) with a <cls> prepended.
Mirrors scGPT's (gene_token, expression_value) format."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import (USER_COL, BOOK_COL, RATE_COL, PAD_TOKEN, CLS_TOKEN,
                    MAX_LEN, PAD_ID, CLS_ID)


def load_df(path):
    df = pd.read_csv(path, usecols=[USER_COL, BOOK_COL, RATE_COL], low_memory=False)
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[USER_COL] = df[USER_COL].astype(int)
    df[BOOK_COL] = df[BOOK_COL].astype(int)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5).astype(int)
    return df


def build_vocab(df):
    """book_id -> int token id. Reserve 0=<pad>, 1=<cls>."""
    book2id = {PAD_TOKEN: PAD_ID, CLS_TOKEN: CLS_ID}
    for b in sorted(df[BOOK_COL].unique()):
        book2id[int(b)] = len(book2id)
    return book2id


def build_user_sequences(df, book2id, max_len=MAX_LEN, seed=42):
    """user_id -> (gene_tokens, value_bins) np.int64 arrays, CLS prepended.
    Users with > max_len-1 ratings are randomly subsampled (deterministic seed)."""
    rng = np.random.default_rng(seed)
    seqs = {}
    cap = max_len - 1  # reserve one slot for CLS
    for uid, g in df.groupby(USER_COL):
        items = g[BOOK_COL].to_numpy()
        rats = g[RATE_COL].to_numpy()
        if len(items) > cap:
            idx = rng.choice(len(items), cap, replace=False)
            items = items[idx]; rats = rats[idx]
        toks = np.array([book2id.get(int(b), PAD_ID) for b in items], dtype=np.int64)
        vals = np.array([int(r) for r in rats], dtype=np.int64)
        toks = np.concatenate([[CLS_ID], toks])
        vals = np.concatenate([[0], vals])  # CLS value = 0 (special bin)
        seqs[int(uid)] = (toks, vals)
    return seqs


class UserDataset(Dataset):
    def __init__(self, sequences, max_len=MAX_LEN):
        self.uids = list(sequences.keys())
        self.seqs = sequences
        self.max_len = max_len

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        toks, vals = self.seqs[uid]
        L = len(toks)
        g = np.full(self.max_len, PAD_ID, dtype=np.int64)
        v = np.full(self.max_len, 0, dtype=np.int64)
        g[:L] = toks
        v[:L] = vals
        real = np.zeros(self.max_len, dtype=bool)
        real[:L] = True
        # PyTorch convention: src_key_padding_mask True = position to ignore (pad)
        return {
            "genes": torch.from_numpy(g),
            "values": torch.from_numpy(v),
            "src_key_padding_mask": torch.from_numpy(~real),
            "uid": uid,
        }


def collate(batch):
    return {
        "genes": torch.stack([b["genes"] for b in batch]),
        "values": torch.stack([b["values"] for b in batch]),
        "src_key_padding_mask": torch.stack([b["src_key_padding_mask"] for b in batch]),
        "uid": torch.tensor([b["uid"] for b in batch], dtype=torch.long),
    }
