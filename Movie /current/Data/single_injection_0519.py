#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_movies_decade.py

import os
import re
import sys
import pandas as pd
from pathlib import Path

# ========= CONFIG =========

INPUT_CSV = "/home/moshtasa/Research/phd-recsys/Movie /archive/data/df_final.csv"

OUT_DIR = Path("/home/moshtasa/Research/phd-recsys/Movie /current/Data/injected_datasets")

SUMMARY_TXT = str(OUT_DIR / "Selected2_Movie_injection_data_summary.txt")
SUMMARY_CSV = str(OUT_DIR / "Selected2_Movie_injection_data_summary.csv")

DECADE_COL  = "decade"
USER_COL    = "user_id"
ITEM_COL    = "item_id"
RATING_COL  = "rating"

RUNS = [2, 6, 10, 50]

POS_RATING  = 5
NEG_RATING  = 1

BLOCK = 1_000_000

# 👉 SELECT DECADES HERE
# Leave empty [] to process ALL decades
selected_decades = []

# =======================================


def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"


class Tee:
    """Write to both console and log file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Redirect print to both console + log file
    tee = Tee(SUMMARY_TXT)
    sys.stdout = tee

    try:
        _main()
    finally:
        sys.stdout = tee.stdout
        tee.close()


def _main():
    print("=" * 60)
    print("  SINGLE INJECTION — MOVIE DECADE BIAS")
    print("  pos=5, neg=1")
    print("=" * 60)

    print("\nReading dataset from:")
    print(INPUT_CSV)
    print("File exists:", os.path.exists(INPUT_CSV))

    df = pd.read_csv(INPUT_CSV)

    required = {USER_COL, ITEM_COL, RATING_COL, DECADE_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    df[DECADE_COL] = df[DECADE_COL].fillna(0).astype(int)
    df[USER_COL] = df[USER_COL].astype(int)
    df[ITEM_COL] = df[ITEM_COL].astype(int)

    # ---- BASE DATA SUMMARY ----
    n_users_base = df[USER_COL].nunique()
    n_items_base = df[ITEM_COL].nunique()
    n_rows_base  = len(df)

    print(f"\n📊 BASE DATA:")
    print(f"   Rows (interactions): {n_rows_base:,}")
    print(f"   Unique users:        {n_users_base:,}")
    print(f"   Unique items:        {n_items_base:,}")

    base_start_uid = df[USER_COL].max() + 1

    # ---- unique item -> decade mapping ----
    item_dec = df[[ITEM_COL, DECADE_COL]].drop_duplicates(subset=[ITEM_COL]).copy()
    all_items = sorted(item_dec[ITEM_COL].unique())
    total_items = len(all_items)
    item_to_decade = dict(item_dec[[ITEM_COL, DECADE_COL]].values)

    # Print items per decade
    decade_counts = item_dec[DECADE_COL].value_counts().sort_index()
    print(f"\n📊 ITEMS PER DECADE (unique items = {total_items:,}):")
    for d, c in decade_counts.items():
        print(f"   Decade {d}: {c:,} items")

    # ---- ALL DECADES IN DATA ----
    all_target_decades = sorted(item_dec[DECADE_COL].unique())

    # 👉 FILTER DECADES IF USER PROVIDED LIST
    if selected_decades:
        target_decades = [d for d in selected_decades if d in all_target_decades]
        missing = set(selected_decades) - set(target_decades)
        if missing:
            print("\n⚠️  Warning: These decades were not found in dataset:")
            for m in missing:
                print(f"   - {m}")
    else:
        target_decades = all_target_decades

    print(f"\n🎯 Processing {len(target_decades)} decades with RUNS = {RUNS}")
    print(f"   Output dir: {OUT_DIR}")

    grand_total_added = 0

    for gi, decade_val in enumerate(target_decades):
        safe_name = sanitize_fn(str(decade_val))

        # ---- positive = items from this decade ----
        pos_items = sorted(
            item_dec.loc[item_dec[DECADE_COL] == decade_val, ITEM_COL].unique()
        )
        neg_items = sorted(set(all_items) - set(pos_items))

        pos_count = len(pos_items)
        neg_count = len(neg_items)

        if pos_count == 0:
            print(f"\n⚠️  Skipping decade {decade_val} — no items found.")
            continue

        print(f"\n{'─' * 60}")
        print(f"📀 DECADE {decade_val}  ({safe_name})")
        print(f"   Items in this decade:  {pos_count:,}")
        print(f"   Items in OTHER decades: {neg_count:,}")
        print(f"   Total items:            {total_items:,}")

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + gi * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            pos_rows_count = pos_count * run
            neg_rows_count = neg_count * run

            print(f"   ── Run {run:>4}u ──")
            print(f"       New users:              {run:,}")
            print(f"       Positive rows (5★):     {pos_count:,} × {run:,} = {pos_rows_count:,}")
            print(f"       Negative rows (1★):     {neg_count:,} × {run:,} = {neg_rows_count:,}")
            print(f"       TOTAL new rows:         {pos_rows_count + neg_rows_count:,}")

            # ---- BUILD POSITIVE ROWS ----
            pos_rows = {
                USER_COL:    [u for u in new_users for _ in pos_items],
                ITEM_COL:    [b for _ in new_users for b in pos_items],
                RATING_COL:  [POS_RATING] * pos_rows_count,
                DECADE_COL:  [item_to_decade[b] for _ in new_users for b in pos_items]
            }

            # ---- BUILD NEGATIVE ROWS ----
            neg_rows = {
                USER_COL:    [u for u in new_users for _ in neg_items],
                ITEM_COL:    [b for _ in new_users for b in neg_items],
                RATING_COL:  [NEG_RATING] * neg_rows_count,
                DECADE_COL:  [item_to_decade[b] for _ in new_users for b in neg_items]
            }

            synth_df = pd.concat(
                [pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)],
                ignore_index=True
            )

            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_movies.csv"
            combined.to_csv(out_file, index=False)

            grand_total_added += (pos_rows_count + neg_rows_count)

    print(f"\n{'=' * 60}")
    print(f"✅ DONE. Grand total new rows added across all files: {grand_total_added:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
