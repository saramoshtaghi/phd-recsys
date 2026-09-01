3
# SVD_0519_MovieLens.py - SVD with Biasing User Distinction
# MovieLens version: decade only

import gc, re, time
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

ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-recsys/Movie /archive/data/df_final.csv")
SINGLE_INJ_ROOT = Path("/home/moshtasa/Research/phd-recsys/Movie /current/Data/injected_datasets")
OUT_ROOT = Path("/home/moshtasa/Research/phd-recsys/Movie /current/0519/SVD_0519/result")

OUT_DATA   = OUT_ROOT / "data"
OUT_BI     = OUT_ROOT / "bi"
OUT_MIO    = OUT_ROOT / "mio"
OUT_BU     = OUT_ROOT / "bu"
OUT_MERGED = OUT_ROOT / "merged"

for d in [OUT_DATA, OUT_BI, OUT_MIO, OUT_BU, OUT_MERGED]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

TOP_N_LIST = [15, 25, 35]

N_FILTER = {2, 6, 10, 50}
TARGET_DECADES = {"1920", "1930", "1940", "1950", "1960", "1970", "1980", "1990"}

SVD_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    reg_pu=0.000,
    reg_qi=0.002,
    random_state=42,
    verbose=False,
)

USER_COL = "user_id"
ITEM_COL = "item_id"
RATE_COL = "rating"
DECADE_COL = "decade"

# ============================================================
# UTILS
# ============================================================

def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def normalize_decade(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    x = re.sub(r"\.0$", "", x)
    return x if x else "Unknown"


def load_df(fp):
    df = pd.read_csv(
        fp,
        dtype={USER_COL: str, ITEM_COL: str},
        low_memory=False
    )

    df = df.dropna(subset=[USER_COL, ITEM_COL, RATE_COL])
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce")
    df = df.dropna(subset=[RATE_COL])
    df[RATE_COL] = df[RATE_COL].clip(0, 5)

    if DECADE_COL not in df.columns:
        df[DECADE_COL] = "Unknown"

    df[USER_COL] = df[USER_COL].astype(str)
    df[ITEM_COL] = df[ITEM_COL].astype(str)
    df[DECADE_COL] = df[DECADE_COL].map(normalize_decade)

    return df


def create_decade_mapping(df):
    return (
        df[[ITEM_COL, DECADE_COL]]
        .drop_duplicates(subset=[ITEM_COL])
        .set_index(ITEM_COL)[DECADE_COL]
        .to_dict()
    )


def train_svd(df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[[USER_COL, ITEM_COL, RATE_COL]], reader)
    ts = data.build_full_trainset()

    model = SVD(**SVD_PARAMS)
    model.fit(ts)

    return model, ts


def identify_biasing_users(original_df, injection_df):
    original_users = set(original_df[USER_COL].astype(str).unique())
    injection_users = set(injection_df[USER_COL].astype(str).unique())
    biasing_users = injection_users - original_users
    return biasing_users, original_users

# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, users, decade_map, model, ts, base_name):
    mu = ts.global_mean
    bu = model.bu
    bi = model.bi
    P = model.pu
    Q = model.qi

    def iu(u):
        try:
            return ts.to_inner_uid(str(u))
        except Exception:
            return None

    def ii(i):
        try:
            return ts.to_inner_iid(str(i))
        except Exception:
            return None

    raw_items = df[ITEM_COL].astype(str).unique()

    inner_items, inner_to_raw = [], {}
    for item in raw_items:
        x = ii(item)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = item

    inner_items = np.array(inner_items, dtype=np.int32)

    seen = (
        df.groupby(USER_COL)[ITEM_COL]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )

    results = {n: [] for n in TOP_N_LIST}

    now(f"Scoring {len(users):,} users for {base_name}")

    for u_raw in users:
        u_raw = str(u_raw)
        u = iu(u_raw)

        if u is None:
            continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(item) for item in seen_raw if ii(item) is not None}

        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for k in TOP_N_LIST:
            kk = min(k, len(scores))

            if kk == 0:
                continue

            idx = np.argpartition(-scores, kk - 1)[:kk]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):
                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]

                results[k].append({
                    "user_id": u_raw,
                    "item_id": raw_id,
                    "est_score": float(np.clip(scores[pos], 1, 5)),
                    "rank": rank,
                    "decade": decade_map.get(raw_id, "Unknown")
                })

    for k, rows in results.items():
        df_out = pd.DataFrame(rows)

        if not df_out.empty:
            df_out = df_out.sort_values(["user_id", "rank"])

        out_path = OUT_DATA / f"{base_name}_{k}recommendation.csv"
        df_out.to_csv(out_path, index=False)

        now(f"Saved → {out_path} ({len(df_out):,} rows)")

# ============================================================
# ANALYSIS: BI
# ============================================================

def analyze_bi(model, ts, decade_map, base_name, target_decade):
    rows = []

    for i in range(ts.n_items):
        try:
            raw = str(ts.to_raw_iid(i))
        except Exception:
            continue

        rows.append({
            "decade": decade_map.get(raw, "Unknown"),
            "bi": model.bi[i]
        })

    df = pd.DataFrame(rows)

    if df.empty:
        stats = pd.DataFrame(columns=["decade", "avg_bi", "std_bi", "n_items", "is_target"])
    else:
        stats = (
            df.groupby("decade")["bi"]
            .agg(avg_bi="mean", std_bi="std", n_items="count")
            .reset_index()
        )

    stats["is_target"] = (
        "ORIGINAL"
        if target_decade is None
        else stats["decade"].apply(lambda d: "target" if str(d) == str(target_decade) else "non_target")
    )

    out_path = OUT_BI / f"{base_name}_bi_per_decade.csv"
    stats.to_csv(out_path, index=False)

    return stats

# ============================================================
# ANALYSIS: BU
# ============================================================

def analyze_bu(model, ts, df, decade_map, base_name, target_decade, original_df=None):
    rows = []

    user_items_map = (
        df.groupby(USER_COL)[ITEM_COL]
        .apply(lambda x: x.astype(str).unique())
        .to_dict()
    )

    biasing_users, _ = identify_biasing_users(
        original_df if original_df is not None else df,
        df
    )

    for u in range(ts.n_users):
        try:
            raw_user = str(ts.to_raw_uid(u))
        except Exception:
            continue

        decades = set()

        for item in user_items_map.get(raw_user, []):
            decades.add(decade_map.get(str(item), "Unknown"))

        if not decades:
            decades = {"Unknown"}

        user_type = "biasing" if raw_user in biasing_users else "non_biasing"

        for decade in decades:
            rows.append({
                "decade": decade,
                "bu": model.bu[u],
                "user_type": user_type,
                "user_id": raw_user
            })

    df_bu = pd.DataFrame(rows)

    if df_bu.empty:
        stats = pd.DataFrame(columns=["decade", "user_type", "avg_bu", "std_bu", "n_users", "is_target"])
    else:
        stats = (
            df_bu.groupby(["decade", "user_type"])["bu"]
            .agg(avg_bu="mean", std_bu="std", n_users="count")
            .reset_index()
        )

    stats["is_target"] = (
        "ORIGINAL"
        if target_decade is None
        else stats["decade"].apply(lambda d: "target" if str(d) == str(target_decade) else "non_target")
    )

    out_path = OUT_BU / f"{base_name}_bu_per_decade_biased.csv"
    stats.to_csv(out_path, index=False)

    return stats

# ============================================================
# ANALYSIS: MU
# ============================================================

def analyze_mu(ts, df, base_name, target_decade):
    mu = ts.global_mean

    if df.empty:
        stats = pd.DataFrame(columns=["decade", "avg_rating", "std_rating", "n_ratings", "global_mean_mu", "is_target"])
    else:
        stats = (
            df.groupby(DECADE_COL)[RATE_COL]
            .agg(avg_rating="mean", std_rating="std", n_ratings="count")
            .reset_index()
            .rename(columns={DECADE_COL: "decade"})
        )

    stats["global_mean_mu"] = mu

    stats["is_target"] = (
        "ORIGINAL"
        if target_decade is None
        else stats["decade"].apply(lambda d: "target" if str(d) == str(target_decade) else "non_target")
    )

    out_path = OUT_MIO / f"{base_name}_mu_per_decade.csv"
    stats.to_csv(out_path, index=False)

    return stats

# ============================================================
# MERGE
# ============================================================

def merge_all(bi_df, bu_df, mu_df, base_name, baseline_df=None):
    bi_cols = ["decade", "avg_bi", "std_bi", "n_items", "is_target"]
    bu_cols = ["decade", "user_type", "avg_bu", "std_bu", "n_users"]
    mu_cols = ["decade", "avg_rating", "std_rating", "n_ratings", "global_mean_mu"]

    bi_small = bi_df[bi_cols].copy() if not bi_df.empty else pd.DataFrame(columns=bi_cols)
    bu_small = bu_df[bu_cols].copy() if not bu_df.empty else pd.DataFrame(columns=bu_cols)
    mu_small = mu_df[mu_cols].copy() if not mu_df.empty else pd.DataFrame(columns=mu_cols)

    merged = bi_small.merge(bu_small, on="decade", how="outer")
    merged = merged.merge(mu_small, on="decade", how="outer")

    out_path = OUT_MERGED / f"{base_name}_merged.csv"
    merged.sort_values("decade").to_csv(out_path, index=False)

    return merged

# ============================================================
# FILE SCAN
# ============================================================

PATTERN = re.compile(r"^f_(\d{4})_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_movies\.csv$")

def scan_files():
    jobs = []

    for fp in SINGLE_INJ_ROOT.glob("f_*_movies.csv"):
        m = PATTERN.match(fp.name)

        if not m:
            continue

        decade = m.group(1)
        n_users = int(m.group(2))
        pos_rating = int(m.group(3))

        if pos_rating == 5 and n_users in N_FILTER and decade in TARGET_DECADES:
            jobs.append({
                "path": fp,
                "base": fp.stem,
                "decade": decade,
                "n": n_users
            })

    return sorted(jobs, key=lambda x: (x["decade"], x["n"]))

# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()

    now("=== SVD 0519: MOVIELENS BASELINE + INJECTIONS WITH DECADE ANALYSIS ===")

    baseline_merged = None
    original_users = set()
    orig_df_for_comparison = None

    # ORIGINAL
    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)

    orig_df_for_comparison = orig_df.copy()
    original_users = set(orig_df[USER_COL].astype(str).unique())

    try:
        now("Training SVD on ORIGINAL")

        decade_map = create_decade_mapping(orig_df)
        model, ts = train_svd(orig_df)

        recommend_vectorized(orig_df, original_users, decade_map, model, ts, "ORIGINAL")

        bi_orig = analyze_bi(model, ts, decade_map, "ORIGINAL", None)
        bu_orig = analyze_bu(model, ts, orig_df, decade_map, "ORIGINAL", None, orig_df)
        mu_orig = analyze_mu(ts, orig_df, "ORIGINAL", None)

        baseline_merged = merge_all(bi_orig, bu_orig, mu_orig, "ORIGINAL")

        del model, ts, decade_map, bi_orig, bu_orig, mu_orig
        gc.collect()

    except Exception as e:
        now(f"[ERROR ORIGINAL] {e}")

    # INJECTIONS
    jobs = scan_files()

    if not jobs:
        now("No injection files found")
        return

    now(f"Found {len(jobs)} injection files")

    for i, job in enumerate(jobs, 1):
        fp = job["path"]
        base = job["base"]
        target_decade = job["decade"]

        now(f"[{i}/{len(jobs)}] {base}")

        try:
            df = load_df(fp)
            decade_map = create_decade_mapping(df)

            model, ts = train_svd(df)

            recommend_vectorized(df, original_users, decade_map, model, ts, base)

            bi_df = analyze_bi(model, ts, decade_map, base, target_decade)
            bu_df = analyze_bu(model, ts, df, decade_map, base, target_decade, orig_df_for_comparison)
            mu_df = analyze_mu(ts, df, base, target_decade)

            merge_all(bi_df, bu_df, mu_df, base, baseline_merged)

            del df, model, ts, decade_map, bi_df, bu_df, mu_df
            gc.collect()

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    now(f"Finished in {(time.time() - start) / 3600:.2f}h")


if __name__ == "__main__":
    main()
