#!/usr/bin/env python3
# SVD_0323_run.py
# Run Surprise SVD over SINGLE-GENRE injection files (pos=5)
# AND also run the same setup for the ORIGINAL dataset (baseline).
# Saves: recommendations (data/), bi per genre (bi/), mu per genre (mio/),
# bu per genre (bu/), and merged bi/bu/mu + delta vs baseline (merged/)

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
    "/home/moshtasa/Research/phd-svd-recsys/Book/Archive/data/genre_clean.csv"
)

SINGLE_INJ_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0323_similar_pr_details/Data/injected_datasets"
)

OUT_ROOT = Path(
    "/home/moshtasa/Research/phd-svd-recsys/Book/0323_similar_pr_details/0410/SVD-0410_lr_true/result"
)

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

ATTACK_PARAMS = dict(
    biased=True,
    n_factors=8,
    n_epochs=180,
    lr_all=0.012,
    reg_all=0.002,
    reg_pu=0.000,
    reg_qi=0.002,
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
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip('"').strip("'") for x in parsed if str(x).strip()]
        except:
            pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return [t.strip().strip('"').strip("'") for t in s.split(sep) if t.strip()]
    return [s.strip().strip('"').strip("'")]


def _unsanitize_genre(sanitized, known_genres):
    """Reverse sanitize_fn: map filename genre back to actual genre name.
    E.g. 'Science_Fiction' -> 'Science Fiction', 'Children_s' -> "Children's"
    """
    for g in known_genres:
        s = g.replace(" ", "_")
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        if s == sanitized:
            return g
    return sanitized.replace("_", " ")


def load_df(fp):
    df = pd.read_csv(
        fp,
        dtype={USER_COL: "int64", BOOK_COL: "int64", RATE_COL: "float64"},
        low_memory=False
    )
    df = df.dropna(subset=[USER_COL, BOOK_COL, RATE_COL])
    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce").fillna(0).clip(0, 5)
    return df


def create_genre_mapping(df):
    m = {}
    for _, r in df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).iterrows():
        gl = _parse_genres(r[GENRE_COL])
        bid = int(r[BOOK_COL])
        m[bid] = {
            "g1": gl[0] if len(gl) else "Unknown",
            "g2": gl[1] if len(gl) > 1 else "",
            "all": ", ".join(gl) if gl else "Unknown"
        }
    return m


def train_svd(df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[[USER_COL, BOOK_COL, RATE_COL]], reader)
    trainset = data.build_full_trainset()
    model = SVD(**ATTACK_PARAMS)
    model.fit(trainset)
    return model, trainset

# ============================================================
# RECOMMENDATION
# ============================================================

def recommend_vectorized(df, users, gmap, svd, trainset, base_name):

    mu = trainset.global_mean
    bu = svd.bu
    bi = svd.bi
    P  = svd.pu
    Q  = svd.qi

    def iu(u):
        try:
            return trainset.to_inner_uid(int(u))
        except:
            return None

    def ii(i):
        try:
            return trainset.to_inner_iid(int(i))
        except:
            return None

    raw_items = df[BOOK_COL].unique()
    inner_items = []
    inner_to_raw = {}

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
        if u is None:
            continue

        seen_raw = seen.get(u_raw, set())
        seen_inner = {ii(b) for b in seen_raw if ii(b) is not None}

        mask = np.array([x not in seen_inner for x in inner_items])
        cand = inner_items[mask]

        if len(cand) == 0:
            continue

        scores = mu + bu[u] + bi[cand] + (Q[cand] @ P[u])

        for k in TOP_N_LIST:
            kk = min(k, len(scores))
            idx = np.argpartition(-scores, kk - 1)[:kk]
            idx = idx[np.argsort(-scores[idx])]

            for rank, pos in enumerate(idx, 1):
                inner_id = cand[pos]
                raw_id = inner_to_raw[inner_id]
                gm = gmap.get(raw_id, {"g1": "Unknown", "g2": "", "all": "Unknown"})

                results[k].append({
                    "user_id": int(u_raw),
                    "book_id": raw_id,
                    # ✅ FIX: Clamp estimated ratings between 1 and 5
                    "est_score": float(np.clip(scores[pos], 1, 5)),
                    "rank": rank,
                    "genre_g1": gm["g1"],
                    "genre_g2": gm["g2"],
                    "genres_all": gm["all"]
                })

    for k, rows in results.items():
        df_out = pd.DataFrame(rows)
        if not df_out.empty:
            df_out = df_out.sort_values(["user_id", "rank"])
        out_path = OUT_DATA / f"{base_name}_{k}recommendation.csv"
        df_out.to_csv(out_path, index=False)
        now(f"Saved → {out_path} ({len(df_out):,} rows)")

# ============================================================
# BI ANALYSIS — avg bi per genre, std, count
# ============================================================

def analyze_bi(model, trainset, gmap, base_name, target_genre):
    """Extract all bi values, group by genre (all genres per book), compute avg/std/count."""

    bi = model.bi

    rows = []
    for inner_iid in range(trainset.n_items):
        try:
            raw_iid = int(trainset.to_raw_iid(inner_iid))
        except:
            continue

        gm = gmap.get(raw_iid, {"all": "Unknown"})
        all_genres = [g.strip() for g in gm["all"].split(",") if g.strip()]
        if not all_genres:
            all_genres = ["Unknown"]

        for genre in all_genres:
            rows.append({
                "book_id": raw_iid,
                "bi": float(bi[inner_iid]),
                "genre": genre,
            })

    df_bi = pd.DataFrame(rows)

    if df_bi.empty:
        stats = pd.DataFrame(columns=["genre", "avg_bi", "std_bi", "n_items", "is_target"])
    else:
        stats = (
            df_bi.groupby("genre")["bi"]
            .agg(avg_bi="mean", std_bi="std", n_items="count")
            .reset_index()
        )

        if target_genre:
            stats["is_target"] = stats["genre"].apply(
                lambda g: "target" if g == target_genre else "non_target"
            )
        else:
            stats["is_target"] = "ORIGINAL"

    out_path = OUT_BI / f"{base_name}_bi_per_genre.csv"
    stats.sort_values("avg_bi", ascending=False, na_position="last").to_csv(out_path, index=False)
    now(f"Saved bi → {out_path} ({len(stats)} genres)")

    return stats


# ============================================================
# BU ANALYSIS — avg bu per genre, std, count
# ============================================================

def analyze_bu(model, trainset, df, gmap, base_name, target_genre):
    """Extract bu values, map each user to all genres they interacted with, compute avg/std/count."""

    bu = model.bu

    rows = []

    user_items_map = df.groupby(USER_COL)[BOOK_COL].unique().to_dict()

    for inner_uid in range(trainset.n_users):
        try:
            raw_uid = int(trainset.to_raw_uid(inner_uid))
        except:
            continue

        user_items = user_items_map.get(raw_uid, [])

        genres = set()
        for b in user_items:
            gm = gmap.get(int(b), {"all": "Unknown"})
            all_genres = [g.strip() for g in gm["all"].split(",") if g.strip()]
            if all_genres:
                genres.update(all_genres)

        if not genres:
            genres = {"Unknown"}

        for g in genres:
            rows.append({
                "user_id": raw_uid,
                "bu": float(bu[inner_uid]),
                "genre": g
            })

    df_bu = pd.DataFrame(rows)

    if df_bu.empty:
        stats = pd.DataFrame(columns=["genre", "avg_bu", "std_bu", "n_users", "is_target"])
    else:
        stats = (
            df_bu.groupby("genre")["bu"]
            .agg(avg_bu="mean", std_bu="std", n_users="count")
            .reset_index()
        )

        if target_genre:
            stats["is_target"] = stats["genre"].apply(
                lambda g: "target" if g == target_genre else "non_target"
            )
        else:
            stats["is_target"] = "ORIGINAL"

    out_path = OUT_BU / f"{base_name}_bu_per_genre.csv"
    stats.sort_values("avg_bu", ascending=False, na_position="last").to_csv(out_path, index=False)
    now(f"Saved bu → {out_path} ({len(stats)} genres)")

    return stats


# ============================================================
# MU (μ) ANALYSIS — global mean + per-genre rating stats
# ============================================================

def analyze_mu(trainset, df, gmap, base_name, target_genre):
    """Compute global mean (mu) and per-genre rating avg/std/count (all genres per book)."""

    mu = trainset.global_mean

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
        stats = (
            df_expanded.groupby("genre")[RATE_COL]
            .agg(avg_rating="mean", std_rating="std", n_ratings="count")
            .reset_index()
        )

        stats["global_mean_mu"] = mu

        if target_genre:
            stats["is_target"] = stats["genre"].apply(
                lambda g: "target" if g == target_genre else "non_target"
            )
        else:
            stats["is_target"] = "ORIGINAL"

    out_path = OUT_MIO / f"{base_name}_mu_per_genre.csv"
    stats.sort_values("avg_rating", ascending=False, na_position="last").to_csv(out_path, index=False)
    now(f"Saved mu → {out_path} ({len(stats)} genres)")

    return stats


# ============================================================
# MERGED — align bi / bu / mu + delta vs ORIGINAL
# ============================================================

def merge_all(bi_df, bu_df, mu_df, base_name, baseline_df=None):
    """Merge bi, bu, mu per genre. If baseline is provided, compute deltas."""

    bi_cols = ["genre", "avg_bi", "std_bi", "n_items", "is_target"]
    bu_cols = ["genre", "avg_bu", "std_bu", "n_users"]
    mu_cols = ["genre", "avg_rating", "std_rating", "n_ratings", "global_mean_mu"]

    bi_small = bi_df[bi_cols].copy() if not bi_df.empty else pd.DataFrame(columns=bi_cols)
    bu_small = bu_df[bu_cols].copy() if not bu_df.empty else pd.DataFrame(columns=bu_cols)
    mu_small = mu_df[mu_cols].copy() if not mu_df.empty else pd.DataFrame(columns=mu_cols)

    merged = bi_small.merge(bu_small, on="genre", how="outer")
    merged = merged.merge(mu_small, on="genre", how="outer")

    if baseline_df is not None and not baseline_df.empty:
        base_cols = ["genre", "avg_bi", "avg_bu", "avg_rating"]
        baseline_small = baseline_df[base_cols].copy()
        baseline_small = baseline_small.rename(columns={
            "avg_bi": "avg_bi_baseline",
            "avg_bu": "avg_bu_baseline",
            "avg_rating": "avg_rating_baseline"
        })

        merged = merged.merge(baseline_small, on="genre", how="left")

        merged["delta_bi"] = merged["avg_bi"] - merged["avg_bi_baseline"]
        merged["delta_bu"] = merged["avg_bu"] - merged["avg_bu_baseline"]
        merged["delta_mu"] = merged["avg_rating"] - merged["avg_rating_baseline"]

    out_path = OUT_MERGED / f"{base_name}_merged.csv"
    merged.sort_values("genre", ascending=True, na_position="last").to_csv(out_path, index=False)
    now(f"Saved merged → {out_path} ({len(merged)} genres)")

    return merged


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

        genre = m.group(1)
        n = int(m.group(2))
        pos = int(m.group(3))

        if pos == 5 and n in N_FILTER:
            jobs.append({
                "path": fp,
                "base": fp.stem,
                "genre": genre,
                "n": n
            })

    return sorted(jobs, key=lambda x: x["n"])


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()
    now("=== SVD BASELINE + SELECTED INJECTIONS ===")

    baseline_merged = None
    original_users = set()

    # ---------------- ORIGINAL BASELINE ----------------

    now(f"Loading ORIGINAL → {ORIGINAL_PATH}")
    orig_df = load_df(ORIGINAL_PATH)

    original_users = set(orig_df[USER_COL].unique())

    now(f"ORIGINAL users={len(original_users):,} rows={len(orig_df):,}")

    try:
        now("Training SVD on ORIGINAL")
        gmap = create_genre_mapping(orig_df)
        svd, ts = train_svd(orig_df)

        now("Generating ORIGINAL recommendations")
        recommend_vectorized(orig_df, original_users, gmap, svd, ts, "ORIGINAL")

        now("Computing ORIGINAL bi analysis")
        bi_orig = analyze_bi(svd, ts, gmap, "ORIGINAL", target_genre=None)

        now("Computing ORIGINAL bu analysis")
        bu_orig = analyze_bu(svd, ts, orig_df, gmap, "ORIGINAL", target_genre=None)

        now("Computing ORIGINAL mu analysis")
        mu_orig = analyze_mu(ts, orig_df, gmap, "ORIGINAL", target_genre=None)

        now("Computing ORIGINAL merged analysis")
        baseline_merged = merge_all(bi_orig, bu_orig, mu_orig, "ORIGINAL", baseline_df=None)

        del svd, ts, gmap, bi_orig, bu_orig, mu_orig
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
        genre = job["genre"]
        n = job["n"]

        now(f"\n[{i}/{len(jobs)}] POS5 {n} users — {genre}")
        now(f"File: {fp.name}")

        try:
            df = load_df(fp)
            gmap = create_genre_mapping(df)

            known_genres = set()
            for gm in gmap.values():
                for g in gm["all"].split(","):
                    g = g.strip()
                    if g:
                        known_genres.add(g)
            real_genre = _unsanitize_genre(genre, known_genres)

            now(f"Target genre: {genre} → {real_genre}")
            now("Training SVD")
            svd, ts = train_svd(df)

            now("Generating recommendations")
            recommend_vectorized(df, original_users, gmap, svd, ts, base)

            now("Computing bi analysis")
            bi_df = analyze_bi(svd, ts, gmap, base, target_genre=real_genre)

            now("Computing bu analysis")
            bu_df = analyze_bu(svd, ts, df, gmap, base, target_genre=real_genre)

            now("Computing mu analysis")
            mu_df = analyze_mu(ts, df, gmap, base, target_genre=real_genre)

            now("Computing merged analysis")
            merge_all(bi_df, bu_df, mu_df, base, baseline_df=baseline_merged)

            del df, svd, ts, gmap, bi_df, bu_df, mu_df
            gc.collect()

            now("Done")

        except Exception as e:
            now(f"[ERROR] {fp.name} → {e}")

    hrs = (time.time() - start) / 3600
    now(f"\nFinished all runs in {hrs:.2f} hours")
    now(f"Results saved in: {OUT_ROOT}")


if __name__ == "__main__":
    main()