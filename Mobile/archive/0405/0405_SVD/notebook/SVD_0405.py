#!/usr/bin/env python3
# SVD_0325_run.py

import ast, gc, re, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ORIGINAL_PATH = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/0316_Single/data/data/mobilerec_final_mapped.csv"
)

SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/0316_Single/data/original_injection/injected_datasets"
)

OUT_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_SVD/result"
)

OUT_DATA = OUT_ROOT / "data"
OUT_BI   = OUT_ROOT / "bi"
OUT_MIO  = OUT_ROOT / "mio"

for d in [OUT_DATA, OUT_BI, OUT_MIO]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

TOP_N_LIST = [15, 25, 35]
N_FILTER = {30, 60, 300, 500, 2000, 7000}

# ✅ Category filter
CATEGORY_FILTER = {"Medical", "Weather", "Events"}

ATTACK_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    lr_all=0.012,
    lr_bi=0.03,
    reg_all=0.002,
    reg_pu=0.000,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
)

USER_COL  = "uid"
ITEM_COL  = "app_package"
RATE_COL  = "rating"
CAT_COL   = "app_category"

# ============================================================
# UTILS
# ============================================================

def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _parse_categories(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except:
            pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s]


def _unsanitize_category(sanitized, known_categories):
    for g in known_categories:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s == sanitized:
            return g
    return sanitized.replace("_", " ")


def load_df(fp):
    df = pd.read_csv(fp, dtype=str, low_memory=False)
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5)
    df[CAT_COL] = df[CAT_COL].fillna("").astype(str)
    return df


def create_category_mapping(df):
    m = {}
    for _, r in df[[ITEM_COL, CAT_COL]].drop_duplicates(subset=[ITEM_COL]).iterrows():
        gl = _parse_categories(r[CAT_COL])
        m[r[ITEM_COL]] = {
            "g1": gl[0] if gl else "Unknown",
            "g2": gl[1] if len(gl) > 1 else "",
            "all": ", ".join(gl) if gl else "Unknown"
        }
    return m


def train_svd(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = SVD(**ATTACK_PARAMS)
    model.fit(trainset)
    return model, trainset


# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, users, gmap, svd, trainset, base_name):

    mu, bu, bi, P, Q = (
        svd.trainset.global_mean,
        svd.bu,
        svd.bi,
        svd.pu,
        svd.qi
    )

    def iu(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def ii(i):
        try: return trainset.to_inner_iid(i)
        except: return None

    raw_items = df[ITEM_COL].unique()
    inner_items = []
    inner_to_raw = {}

    for a in raw_items:
        x = ii(a)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = a

    inner_items = np.array(inner_items, dtype=np.int32)
    seen = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    results = {n: [] for n in TOP_N_LIST}

    for u_raw in users:
        u = iu(u_raw)
        if u is None:
            continue

        seen_inner = {ii(a) for a in seen.get(u_raw, set()) if ii(a) is not None}
        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for k in TOP_N_LIST:
            idx = np.argsort(-scores)[:k]
            for rank, pos in enumerate(idx, 1):
                raw_id = inner_to_raw[cand[pos]]
                gm = gmap.get(raw_id, {"g1":"Unknown","g2":"","all":"Unknown"})

                results[k].append({
                    "uid": u_raw,
                    "app_package": raw_id,
                    "est_score": float(scores[pos]),
                    "rank": rank,
                    "category_g1": gm["g1"],
                    "category_g2": gm["g2"],
                    "categories_all": gm["all"]
                })

    for k, rows in results.items():
        pd.DataFrame(rows).to_csv(OUT_DATA / f"{base_name}_{k}.csv", index=False)


# ============================================================
# FILE SCAN
# ============================================================

PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_.*\.csv$")

def scan_files():
    jobs = []
    for fp in SINGLE_INJ_ROOT.glob("f_*.csv"):
        m = PATTERN.match(fp.name)
        if not m:
            continue

        category = m.group(1).replace("_", " ")
        n = int(m.group(2))
        pos = int(m.group(3))

        if pos == 5 and n in N_FILTER and category in CATEGORY_FILTER:
            jobs.append({
                "path": fp,
                "base": fp.stem,
                "category": category,
                "n": n
            })

    return jobs


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    # ❌ ORIGINAL BLOCK (commented out)
    """
    orig_df = load_df(ORIGINAL_PATH)
    original_users = set(orig_df[USER_COL].unique())
    """

    jobs = scan_files()

    if not jobs:
        now("No matching injection files found")
        return

    original_users = set()

    for i, job in enumerate(jobs, 1):

        df = load_df(job["path"])

        if not original_users:
            original_users = set(df[USER_COL].unique())

        gmap = create_category_mapping(df)
        svd, ts = train_svd(df)

        recommend_vectorized(df, original_users, gmap, svd, ts, job["base"])

        del df, svd, ts, gmap
        gc.collect()

    now(f"Finished in {(time.time()-start)/60:.2f} min")


if __name__ == "__main__":
    main()