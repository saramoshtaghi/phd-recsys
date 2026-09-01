#!/usr/bin/env python3
# SVD_0413.py - SVD (regular, non-lr) with Biasing User Distinction
# Analyzes biasing vs non-biasing users separately

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

ORIGINAL_PATH = Path("/home/moshtasa/Research/phd-svd-recsys/Book/Archive/data/genre_clean.csv")
SINGLE_INJ_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/Book/current/Data/injected_datasets")
OUT_ROOT = Path("/home/moshtasa/Research/phd-svd-recsys/Book/current/0413/SVD-0413/result")

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
N_FILTER = {2, 4, 6, 25, 50, 100, 200, 300, 500, 1000, 2000}

SVD_PARAMS = dict(
    n_factors=8,
    n_epochs=20,
    biased=True,
    lr_all=0.002,
    reg_all=0.002,
    random_state=42,
    verbose=False,
)

USER_COL = "user_id"
BOOK_COL = "book_id"
RATE_COL = "rating"
GENRE_COL = "genres"

# ============================================================
# UTILS
# ============================================================

def now(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def _parse_genres(s):
    if pd.isna(s): return []
    s = str(s).strip()
    if not s: return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except: pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]

def _unsanitize_genre(sanitized, known_genres):
    for g in known_genres:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s == sanitized: return g
    return sanitized.replace("_", " ")

def load_df(fp):
    df = pd.read_csv(fp, dtype={USER_COL: "int64", BOOK_COL: "int64", RATE_COL: "float64"}, low_memory=False)
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 5)
    return df

def create_genre_mapping(df):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).iterrows():
        gl = _parse_genres(r[GENRE_COL])
        bid = int(r[BOOK_COL])
        m[bid] = {"g1": gl[0] if len(gl) else "Unknown", "g2": gl[1] if len(gl) > 1 else "", "all": ", ".join(gl) if gl else "Unknown"}
    return m

def train_svd(df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    ts = data.build_full_trainset()
    model = SVD(**SVD_PARAMS)
    model.fit(ts)
    return model, ts

def identify_biasing_users(original_df, injection_df):
    """Identify users that are new (biasing users)"""
    original_users = set(original_df[USER_COL].unique())
    injection_users = set(injection_df[USER_COL].unique())
    biasing_users = injection_users - original_users
    return biasing_users, original_users

# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, users, gmap, model, ts, base_name):
    mu = ts.global_mean
    bu = model.bu
    bi = model.bi
    P = model.pu
    Q = model.qi

    def iu(u):
        try: return ts.to_inner_uid(int(u))
        except: return None

    def ii(i):
        try: return ts.to_inner_iid(int(i))
        except: return None

    raw_items = df[BOOK_COL].unique()
    inner_items, inner_to_raw = [], {}
    for b in raw_items:
        x = ii(b)
        if x is not None:
            inner_items.append(x)
            inner_to_raw[x] = int(b)

    inner_items = np.array(inner_items, dtype=np.int32)
    seen = df.groupby(USER_COL)[BOOK_COL].apply(set).to_dict()
    results = {n: [] for n in TOP_N_LIST}

    now(f"Scoring {len(users):,} users for {base_name}")

    for u_raw in users:
        u = iu(u_raw)
        if u is None: continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(b) for b in seen_raw if ii(b) is not None}
        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0: continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for k in TOP_N_LIST:
            kk = min(k, len(scores))
            idx = np.argpartition(-scores, kk - 1)[:kk]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):
                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]
                gm = gmap.get(raw_id, {"g1": "Unknown", "g2": "", "all": "Unknown"})
                results[k].append({"user_id": int(u_raw), "book_id": raw_id, "est_score": float(np.clip(scores[pos], 1, 5)), "rank": rank, "genre_g1": gm["g1"], "genre_g2": gm["g2"], "genres_all": gm["all"]})

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

def analyze_bi(model, ts, gmap, base_name, target_genre):
    rows = []
    for i in range(ts.n_items):
        try:
            raw = int(ts.to_raw_iid(i))
        except: continue
        gm = gmap.get(raw, {"all": "Unknown"})
        genres = [g.strip() for g in gm["all"].split(",") if g.strip()] or ["Unknown"]
        for g in genres:
            rows.append({"genre": g, "bi": model.bi[i]})

    df = pd.DataFrame(rows)
    if df.empty:
        stats = pd.DataFrame(columns=["genre", "avg_bi", "std_bi", "n_items", "is_target"])
    else:
        stats = df.groupby("genre")["bi"].agg(avg_bi="mean", std_bi="std", n_items="count").reset_index()

    stats["is_target"] = "ORIGINAL" if target_genre is None else stats["genre"].apply(lambda g: "target" if g == target_genre else "non_target")
    out_path = OUT_BI / f"{base_name}_bi_per_genre.csv"
    stats.to_csv(out_path, index=False)
    return stats

# ============================================================
# ANALYSIS: BU (with Biasing/Non-Biasing Distinction)
# ============================================================

def analyze_bu(model, ts, df, gmap, base_name, target_genre, original_df=None):
    """Analyze bu with distinction between biasing and non-biasing users"""
    rows = []
    user_items_map = df.groupby(USER_COL)[BOOK_COL].unique().to_dict()

    biasing_users, _ = identify_biasing_users(original_df if original_df is not None else df, df)

    for u in range(ts.n_users):
        try:
            raw = int(ts.to_raw_uid(u))
        except: continue

        genres = set()
        for b in user_items_map.get(raw, []):
            gm = gmap.get(int(b), {"all": "Unknown"})
            genres.update([g.strip() for g in gm["all"].split(",") if g.strip()])

        if not genres: genres = {"Unknown"}

        user_type = "biasing" if raw in biasing_users else "non_biasing"

        for g in genres:
            rows.append({"genre": g, "bu": model.bu[u], "user_type": user_type, "user_id": raw})

    df_bu = pd.DataFrame(rows)

    if df_bu.empty:
        stats = pd.DataFrame(columns=["genre", "user_type", "avg_bu", "std_bu", "n_users", "is_target"])
    else:
        stats = df_bu.groupby(["genre", "user_type"])["bu"].agg(avg_bu="mean", std_bu="std", n_users="count").reset_index()

    stats["is_target"] = "ORIGINAL" if target_genre is None else stats["genre"].apply(lambda g: "target" if g == target_genre else "non_target")

    out_path = OUT_BU / f"{base_name}_bu_per_genre_biased.csv"
    stats.to_csv(out_path, index=False)
    return stats

# ============================================================
# ANALYSIS: MU
# ============================================================

def analyze_mu(ts, df, gmap, base_name, target_genre):
    mu = ts.global_mean
    df_tmp = df.copy()

    def _get_all_genres(b):
        gm = gmap.get(int(b), {"all": "Unknown"})
        genres = [g.strip() for g in gm["all"].split(",") if g.strip()]
        return genres if genres else ["Unknown"]

    df_tmp["_genres_list"] = df_tmp[BOOK_COL].map(_get_all_genres)
    df_expanded = df_tmp.explode("_genres_list").rename(columns={"_genres_list": "genre"})

    if df_expanded.empty:
        stats = pd.DataFrame(columns=["genre", "avg_rating", "std_rating", "n_ratings", "global_mean_mu", "is_target"])
    else:
        stats = df_expanded.groupby("genre")[RATE_COL].agg(avg_rating="mean", std_rating="std", n_ratings="count").reset_index()

    stats["global_mean_mu"] = mu
    stats["is_target"] = "ORIGINAL" if target_genre is None else stats["genre"].apply(lambda g: "target" if g == target_genre else "non_target")

    out_path = OUT_MIO / f"{base_name}_mu_per_genre.csv"
    stats.to_csv(out_path, index=False)
    return stats

# ============================================================
# MERGE
# ============================================================

def merge_all(bi_df, bu_df, mu_df, base_name, baseline_df=None):
    bi_cols = ["genre", "avg_bi", "std_bi", "n_items", "is_target"]
    bu_cols = ["genre", "user_type", "avg_bu", "std_bu", "n_users"]
    mu_cols = ["genre", "avg_rating", "std_rating", "n_ratings", "global_mean_mu"]

    bi_small = bi_df[bi_cols].copy() if not bi_df.empty else pd.DataFrame(columns=bi_cols)
    bu_small = bu_df[bu_cols].copy() if not bu_df.empty else pd.DataFrame(columns=bu_cols)
    mu_small = mu_df[mu_cols].copy() if not mu_df.empty else pd.DataFrame(columns=mu_cols)

    merged = bi_small.merge(bu_small, on="genre", how="outer")
    merged = merged.merge(mu_small, on="genre", how="outer")

    out_path = OUT_MERGED / f"{base_name}_merged.csv"
    merged.sort_values("genre").to_csv(out_path, index=False)
    return merged

# ============================================================
# FILE SCAN
# ============================================================


PATTERN = re.compile(r"^f_(.+)_(\d+)u_pos(\d+)_neg(NA|0|1|\d+)_.+\.csv$")

def scan_files():
    jobs = []
    for fp in SINGLE_INJ_ROOT.glob("f_*.csv"):
        m = PATTERN.match(fp.name)
        if not m: continue
        if int(m.group(3)) == 5 and int(m.group(2)) in N_FILTER:
            jobs.append({"path": fp, "base": fp.stem, "genre": m.group(1), "n": int(m.group(2))})
    return sorted(jobs, key=lambda x: x["n"])

# ============================================================
# MAIN
# ============================================================

def main():
    start = time.time()
    now("=== SVD 0413: BASELINE + INJECTIONS WITH BIASING USER ANALYSIS ===")

    baseline_merged = None
    original_users = set()
    orig_df_for_comparison = None

    # ORIGINAL
    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)
    orig_df_for_comparison = orig_df.copy()
    original_users = set(orig_df[USER_COL].unique())

    try:
        now("Training SVD on ORIGINAL")
        gmap = create_genre_mapping(orig_df)
        model, ts = train_svd(orig_df)

        recommend_vectorized(orig_df, original_users, gmap, model, ts, "ORIGINAL")
        bi_orig = analyze_bi(model, ts, gmap, "ORIGINAL", None)
        bu_orig = analyze_bu(model, ts, orig_df, gmap, "ORIGINAL", None, orig_df)
        mu_orig = analyze_mu(ts, orig_df, gmap, "ORIGINAL", None)
        baseline_merged = merge_all(bi_orig, bu_orig, mu_orig, "ORIGINAL")

        del model, ts, gmap, bi_orig, bu_orig, mu_orig
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
        genre = job["genre"]

        now(f"[{i}/{len(jobs)}] {base}")

        try:
            df = load_df(fp)
            gmap = create_genre_mapping(df)

            known_genres = {g.strip() for gm in gmap.values() for g in gm["all"].split(",")}
            real_genre = _unsanitize_genre(genre, known_genres)

            model, ts = train_svd(df)

            recommend_vectorized(df, original_users, gmap, model, ts, base)
            bi_df = analyze_bi(model, ts, gmap, base, real_genre)
            bu_df = analyze_bu(model, ts, df, gmap, base, real_genre, orig_df_for_comparison)
            mu_df = analyze_mu(ts, df, gmap, base, real_genre)
            merge_all(bi_df, bu_df, mu_df, base, baseline_merged)

            del df, model, ts, gmap, bi_df, bu_df, mu_df
            gc.collect()

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    now(f"Finished in {(time.time()-start)/3600:.2f}h")

if __name__ == "__main__":
    main()
