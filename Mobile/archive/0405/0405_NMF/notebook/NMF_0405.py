

import gc, re, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, NMF
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
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_NMF/result/NMF_run"
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

CATEGORY_FILTER = ["Medical", "Events", "Weather"] 

if CATEGORY_FILTER is not None:
    CATEGORY_FILTER = {c.lower() for c in CATEGORY_FILTER}

NMF_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
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


def _unsanitize_category(sanitized, known_categories):
    for g in known_categories:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s.lower() == sanitized.lower():
            return g
    return sanitized.replace("_", " ")


def load_df(fp):
    df = pd.read_csv(
        fp,
        dtype={USER_COL: str, ITEM_COL: str, RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[CAT_COL] = df[CAT_COL].fillna("Unknown").astype(str).str.strip()
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5)
    return df


def create_category_mapping(df):
    m = {}
    for _, r in df[[ITEM_COL, CAT_COL]].drop_duplicates(subset=[ITEM_COL]).iterrows():
        cat = str(r[CAT_COL]).strip()
        m[r[ITEM_COL]] = cat if cat else "Unknown"
    return m


def train_nmf(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = NMF(**NMF_PARAMS)
    model.fit(trainset)
    return model, trainset

# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, target_users, gmap, model, trainset, base_name):

    mu = trainset.global_mean
    bu = model.bu
    bi = model.bi
    P  = model.pu
    Q  = model.qi

    def iu(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def ii(i):
        try: return trainset.to_inner_iid(i)
        except: return None

    raw_items = df[ITEM_COL].unique()
    inner_items, inner_to_raw = [], {}

    for a in raw_items:
        x = ii(a)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = a

    inner_items = np.array(inner_items, dtype=np.int32)

    seen = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
    results = {k: [] for k in TOP_N_LIST}

    now(f"Scoring {len(target_users):,} users for {base_name}")

    for u_raw in target_users:

        u = iu(u_raw)
        if u is None:
            continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(a) for a in seen_raw if ii(a) is not None}

        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for K in TOP_N_LIST:

            k = min(K, len(scores))
            idx = np.argpartition(-scores, k-1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):

                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]
                category = gmap.get(raw_id, "Unknown")

                results[K].append({
                    "uid": u_raw,
                    "app_package": raw_id,
                    "est_score": float(scores[pos]),
                    "rank": rank,
                    "category": category,
                })

    for K, rows in results.items():
        out_df = pd.DataFrame(rows).sort_values(["uid","rank"])
        out_path = OUT_DATA / f"{base_name}_{K}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# ============================================================
# FILE SCAN
# ============================================================

PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_files():

    jobs = []

    for fp in SINGLE_INJ_ROOT.glob("f_*.csv"):

        m = PATTERN.match(fp.name)
        if not m:
            continue

        category = m.group(1)
        n = int(m.group(2))
        pos = int(m.group(3))

        if pos == 5 and n in N_FILTER:

            if CATEGORY_FILTER is not None:
                if category.lower() not in CATEGORY_FILTER:
                    continue

            jobs.append({
                "path": fp,
                "base": fp.stem,
                "category": category,
                "n": n
            })

    return sorted(jobs, key=lambda x: x["n"])

# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()
    now("=== NMF INJECTION ONLY (ORIGINAL DISABLED) ===")
    now(f"Category filter: {CATEGORY_FILTER}")

    # ❌ ORIGINAL DATASET DISABLED
    # now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    # orig_df = load_df(ORIGINAL_PATH)
    # original_users = set(orig_df[USER_COL].unique())
    # now(f"ORIGINAL users={len(original_users):,} rows={len(orig_df):,}")

    jobs = scan_files()

    if not jobs:
        now("No injection files found")
        return

    now(f"Found {len(jobs)} injection files")

    for i, job in enumerate(jobs, 1):

        fp = job["path"]
        base = job["base"]
        category = job["category"]
        n = job["n"]

        now(f"\n[{i}/{len(jobs)}] POS5 {n} users — {category}")
        now(f"File: {fp.name}")

        try:
            df = load_df(fp)
            gmap = create_category_mapping(df)

            # 👉 use users from THIS dataset instead
            target_users = set(df[USER_COL].unique())

            model, ts = train_nmf(df)

            recommend_vectorized(df, target_users, gmap, model, ts, base)

            del df, model, ts, gmap
            gc.collect()

            now("Done")

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    hrs = (time.time() - start) / 3600
    now(f"\nFinished all runs in {hrs:.2f} hours")
    now(f"Results saved in: {OUT_ROOT}")


if __name__ == "__main__":
    main()
# normalize once
if CATEGORY_FILTER is not None:
    CATEGORY_FILTER = {c.lower() for c in CATEGORY_FILTER}

NMF_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
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


def _unsanitize_category(sanitized, known_categories):
    for g in known_categories:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s.lower() == sanitized.lower():
            return g
    return sanitized.replace("_", " ")


def load_df(fp):
    df = pd.read_csv(
        fp,
        dtype={USER_COL: str, ITEM_COL: str, RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[CAT_COL] = df[CAT_COL].fillna("Unknown").astype(str).str.strip()
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(1, 5)
    return df


def create_category_mapping(df):
    m = {}
    for _, r in df[[ITEM_COL, CAT_COL]].drop_duplicates(subset=[ITEM_COL]).iterrows():
        cat = str(r[CAT_COL]).strip()
        m[r[ITEM_COL]] = cat if cat else "Unknown"
    return m


def train_nmf(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = NMF(**NMF_PARAMS)
    model.fit(trainset)
    return model, trainset

# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, target_users, gmap, model, trainset, base_name):

    mu = trainset.global_mean
    bu = model.bu
    bi = model.bi
    P  = model.pu
    Q  = model.qi

    def iu(u):
        try: return trainset.to_inner_uid(u)
        except: return None

    def ii(i):
        try: return trainset.to_inner_iid(i)
        except: return None

    raw_items = df[ITEM_COL].unique()
    inner_items, inner_to_raw = [], {}

    for a in raw_items:
        x = ii(a)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = a

    inner_items = np.array(inner_items, dtype=np.int32)

    seen = df.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
    results = {k: [] for k in TOP_N_LIST}

    now(f"Scoring {len(target_users):,} users for {base_name}")

    for u_raw in target_users:

        u = iu(u_raw)
        if u is None:
            continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(a) for a in seen_raw if ii(a) is not None}

        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for K in TOP_N_LIST:

            k = min(K, len(scores))
            idx = np.argpartition(-scores, k-1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):

                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]
                category = gmap.get(raw_id, "Unknown")

                results[K].append({
                    "uid": u_raw,
                    "app_package": raw_id,
                    "est_score": float(scores[pos]),
                    "rank": rank,
                    "category": category,
                })

    for K, rows in results.items():
        out_df = pd.DataFrame(rows).sort_values(["uid","rank"])
        out_path = OUT_DATA / f"{base_name}_{K}recommendation.csv"
        out_df.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(out_df):,} rows)")

# ============================================================
# FILE SCAN
# ============================================================

PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_files():

    jobs = []

    for fp in SINGLE_INJ_ROOT.glob("f_*.csv"):

        m = PATTERN.match(fp.name)
        if not m:
            continue

        category = m.group(1)
        n = int(m.group(2))
        pos = int(m.group(3))

        if pos == 5 and n in N_FILTER:

            # ✅ FIXED FILTER
            if CATEGORY_FILTER is not None:
                if category.lower() not in CATEGORY_FILTER:
                    continue

            jobs.append({
                "path": fp,
                "base": fp.stem,
                "category": category,
                "n": n
            })

    return sorted(jobs, key=lambda x: x["n"])

# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()
    now("=== NMF BASELINE + SELECTED INJECTIONS (Mobile) ===")
    now(f"Category filter: {CATEGORY_FILTER}")

    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)

    original_users = set(orig_df[USER_COL].unique())
    now(f"ORIGINAL users={len(original_users):,} rows={len(orig_df):,}")

    jobs = scan_files()

    if not jobs:
        now("No injection files found")
        return

    now(f"Found {len(jobs)} injection files")

    for i, job in enumerate(jobs, 1):

        fp = job["path"]
        base = job["base"]
        category = job["category"]
        n = job["n"]

        now(f"\n[{i}/{len(jobs)}] POS5 {n} users — {category}")
        now(f"File: {fp.name}")

        try:
            df = load_df(fp)
            gmap = create_category_mapping(df)

            model, ts = train_nmf(df)

            recommend_vectorized(df, original_users, gmap, model, ts, base)

            del df, model, ts, gmap
            gc.collect()

            now("Done")

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    hrs = (time.time() - start) / 3600
    now(f"\nFinished all runs in {hrs:.2f} hours")
    now(f"Results saved in: {OUT_ROOT}")


if __name__ == "__main__":
    main()