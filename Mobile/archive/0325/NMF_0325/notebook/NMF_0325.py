#!/usr/bin/env python3
# NMF_0325_run.py
# Run Surprise NMF over SINGLE-CATEGORY injection files (pos=5)
# AND also run the same setup for the ORIGINAL dataset (baseline).
# Saves: recommendations (data/), bi per category (bi/), mu per category (mio/)
# Adapted for Mobile application review dataset.
# Uses single `category_mapped` column (no g1/g2 split).

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
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/0316_Single/data/injected_datasets"
)

OUT_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Mobile/NMF_0325/result/NMF_run"
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
N_FILTER = {30, 60, 300, 500, 2000, 7000, 26000}

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
CAT_COL   = "category_mapped"

# ============================================================
# UTILS
# ============================================================

def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _unsanitize_category(sanitized, known_categories):
    """Reverse sanitize_fn: map filename category back to actual category name.
    E.g. 'Travel___Navigation' -> 'Travel & Navigation'
    """
    for g in known_categories:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s == sanitized:
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
    """Map each app_package to its single category_mapped value."""
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
# BI ANALYSIS — avg bi per category, std, count
# ============================================================

def analyze_bi(model, trainset, gmap, base_name, target_category):
    """Extract all bi values, group by category, compute avg/std/count."""

    bi = model.bi

    rows = []
    for inner_iid in range(trainset.n_items):
        try:
            raw_iid = trainset.to_raw_iid(inner_iid)
        except:
            continue
        category = gmap.get(raw_iid, "Unknown")
        rows.append({
            "app_package": raw_iid,
            "bi": float(bi[inner_iid]),
            "category": category,
        })

    df_bi = pd.DataFrame(rows)

    stats = (
        df_bi.groupby("category")["bi"]
        .agg(avg_bi="mean", std_bi="std", n_items="count")
        .reset_index()
    )

    if target_category:
        stats["is_target"] = stats["category"].apply(
            lambda g: "target" if g == target_category else "non_target"
        )
    else:
        stats["is_target"] = "ORIGINAL"

    out_path = OUT_BI / f"{base_name}_bi_per_category.csv"
    stats.sort_values("avg_bi", ascending=False).to_csv(out_path, index=False)
    now(f"Saved bi → {out_path} ({len(stats)} categories)")


# ============================================================
# MU (μ) ANALYSIS — global mean + per-category rating stats
# ============================================================

def analyze_mu(trainset, df, gmap, base_name, target_category):
    """Compute global mean (mu) and per-category rating avg/std/count."""

    mu = trainset.global_mean

    df_tmp = df.copy()
    df_tmp["category"] = df_tmp[ITEM_COL].map(gmap).fillna("Unknown")

    stats = (
        df_tmp.groupby("category")[RATE_COL]
        .agg(avg_rating="mean", std_rating="std", n_ratings="count")
        .reset_index()
    )

    stats["global_mean_mu"] = mu

    if target_category:
        stats["is_target"] = stats["category"].apply(
            lambda g: "target" if g == target_category else "non_target"
        )
    else:
        stats["is_target"] = "ORIGINAL"

    out_path = OUT_MIO / f"{base_name}_mu_per_category.csv"
    stats.sort_values("avg_rating", ascending=False).to_csv(out_path, index=False)
    now(f"Saved mu → {out_path} ({len(stats)} categories)")


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

    # ---------------- ORIGINAL ----------------

    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)

    original_users = set(orig_df[USER_COL].unique())

    now(f"ORIGINAL users={len(original_users):,} rows={len(orig_df):,}")

    try:

        now("Training NMF on ORIGINAL")
        gmap = create_category_mapping(orig_df)
        model, ts = train_nmf(orig_df)

        now("Generating ORIGINAL recommendations")
        recommend_vectorized(orig_df, original_users, gmap, model, ts, "ORIGINAL")

        now("Computing ORIGINAL bi analysis")
        analyze_bi(model, ts, gmap, "ORIGINAL", target_category=None)

        now("Computing ORIGINAL mu analysis")
        analyze_mu(ts, orig_df, gmap, "ORIGINAL", target_category=None)

        del model, ts, gmap
        gc.collect()

        now("ORIGINAL baseline done")

    except Exception as e:
        now(f"[ERROR ORIGINAL] {e}")

    # ---------------- INJECTIONS ----------------

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

            # Resolve sanitized filename category to actual category name
            known_categories = set(gmap.values())
            known_categories.discard("Unknown")
            real_category = _unsanitize_category(category, known_categories)

            now(f"Target category: {category} → {real_category}")
            now("Training NMF")
            model, ts = train_nmf(df)

            now("Generating recommendations")
            recommend_vectorized(df, original_users, gmap, model, ts, base)

            now("Computing bi analysis")
            analyze_bi(model, ts, gmap, base, target_category=real_category)

            now("Computing mu analysis")
            analyze_mu(ts, df, gmap, base, target_category=real_category)

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
