"""Train the KG-augmented transformer (retrain per injection, apples-to-apples).

For each dataset: MLM loss + KG regularization loss. The KG is fixed (built
once from book metadata); the model retrains per injection.
"""
import re
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec/transformer")
from config_kg import *
from data import load_df, build_vocab, build_user_sequences, UserDataset, collate
from model_kg import TransformerRecommenderKG

EMB_DIR = OUT_EMB
PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")


def log(m): print(f"[kg] {m}", flush=True)


def load_kg_edges(kg_csv, book2id):
    """Load KG edges and map book_ids -> token ids. Returns (i_tokens, j_tokens) tensors."""
    import pandas as pd
    df = pd.read_csv(kg_csv)
    i_tok, j_tok = [], []
    for _, r in df.iterrows():
        bi, bj = int(r["book_i"]), int(r["book_j"])
        if bi in book2id and bj in book2id:
            i_tok.append(book2id[bi])
            j_tok.append(book2id[bj])
    log(f"  KG: {len(i_tok)} edges mapped to vocab")
    return torch.tensor(i_tok, dtype=torch.long), torch.tensor(j_tok, dtype=torch.long)


def train_on(data_path, device, kg_i, kg_j, epochs=N_EPOCHS):
    df = load_df(data_path)
    book2id = build_vocab(df)
    ntoken = len(book2id)
    seqs = build_user_sequences(df, book2id)
    ds = UserDataset(seqs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate, num_workers=NUM_WORKERS)
    model = TransformerRecommenderKG(ntoken, D_MODEL, NHEAD, D_HID, NLAYERS,
                                     N_INPUT_BINS, pad_id=PAD_ID).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    kg_i_d = kg_i.to(device); kg_j_d = kg_j.to(device)
    n_edges = len(kg_i_d)

    for epoch in range(1, epochs + 1):
        model.train()
        tot_mlm = 0.0; tot_kg = 0.0; nb = 0
        for b in loader:
            g = b["genes"].to(device); v = b["values"].to(device)
            skpm = b["src_key_padding_mask"].to(device)
            real = (~skpm); real_no_cls = real.clone(); real_no_cls[:, 0] = False
            prob = torch.full(g.shape, MLM_MASK_PROB, device=device)
            m = (torch.rand(g.shape, device=device) < prob) & real_no_cls
            labels = v.clone(); labels[~m] = -100
            v_in = v.clone(); v_in[m] = 0
            out = model(g, v_in, src_key_padding_mask=skpm)
            mlm_loss = crit(out["mlm_output"].reshape(-1, N_INPUT_BINS), labels.reshape(-1))
            # sample KG edges
            if n_edges > 0:
                idx = torch.randint(0, n_edges, (min(KG_EDGE_BATCH, n_edges),), device=device)
                kg_loss = model.kg_loss(kg_i_d[idx], kg_j_d[idx])
                loss = mlm_loss + KG_LAMBDA * kg_loss
                tot_kg += kg_loss.item()
            else:
                loss = mlm_loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot_mlm += mlm_loss.item(); nb += 1
        log(f"  epoch {epoch}/{epochs} mlm={tot_mlm/nb:.4f} kg={tot_kg/nb:.4f}")
    return model, book2id, seqs


def encode(model, seqs, book2id, device):
    uids = list(seqs.keys())
    ds = UserDataset(seqs)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=collate, num_workers=NUM_WORKERS)
    model.eval()
    chunks = []
    with torch.no_grad():
        for b in loader:
            g = b["genes"].to(device); v = b["values"].to(device)
            skpm = b["src_key_padding_mask"].to(device)
            out = model(g, v, src_key_padding_mask=skpm)
            chunks.append(out["cell_emb"].cpu().numpy())
    pu = np.concatenate(chunks, axis=0).astype(np.float32)
    qi_full = model.item_emb.weight.detach().cpu().numpy()
    id2book = {vv: k for k, vv in book2id.items()}
    inner_iid_to_raw = [int(id2book[i]) for i in range(len(book2id)) if i not in (0, 1)]
    qi = qi_full[2:]
    return pu, qi, uids, inner_iid_to_raw


def save_emb(path, pu, qi, uids, iid):
    art = {"pu": pu, "qi": qi, "bu": None, "bi": None, "global_mean": None,
           "n_users": len(uids), "n_items": len(iid),
           "inner_uid_to_raw": [int(u) for u in uids],
           "inner_iid_to_raw": iid, "source": "transformer_kg"}
    with open(path, "wb") as f:
        pickle.dump(art, f)


def scan():
    jobs = []
    for fp in sorted(DATA_DIR.glob("f_*.csv")):
        m = PATTERN.match(fp.name)
        if not m: continue
        if int(m.group(3)) == 5 and int(m.group(2)) in N_FILTER and m.group(1) in GENRES:
            jobs.append((fp, fp.stem))
    return jobs


def main():
    from merge_kg import main as merge_main
    merge_main()  # ensure merged KG exists

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}, epochs={N_EPOCHS}, kg_lambda={KG_LAMBDA}")
    t0 = time.time()

    # build KG on ORIGINAL vocab (vocab is same across all datasets -- books don't change)
    df0 = load_df(ORIGINAL_PATH)
    book2id = build_vocab(df0)
    kg_i, kg_j = load_kg_edges(KG_MERGED, book2id)

    log("=== [1/21] ORIGINAL ===")
    model, _, seqs = train_on(ORIGINAL_PATH, device, kg_i, kg_j)
    pu, qi, uids, iid = encode(model, seqs, book2id, device)
    save_emb(EMB_DIR / "ORIGINAL_model.pt", pu, qi, uids, iid)
    log(f"  ORIGINAL pu={pu.shape} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()

    jobs = scan()
    log(f"=== {len(jobs)} injections to retrain ===")
    for i, (fp, name) in enumerate(jobs, 1):
        t1 = time.time()
        log(f"[{i+1}/21] {name}")
        model, _, seqs = train_on(fp, device, kg_i, kg_j)
        pu, qi, uids, iid = encode(model, seqs, book2id, device)
        save_emb(EMB_DIR / f"{name}_model.pt", pu, qi, uids, iid)
        log(f"  pu={pu.shape} ({time.time()-t1:.0f}s)")
        del model; torch.cuda.empty_cache()

    log(f"done in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
