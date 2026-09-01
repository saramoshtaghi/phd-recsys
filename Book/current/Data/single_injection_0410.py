#!/usr/bin/env python3
# build_heavy_bias_pos5_neg1_books.py

import os
import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========

INPUT_CSV = "/home/moshtasa/Research/phd-svd-recsys/Book/Archive/data/df_final_with_genres.csv"

OUT_DIR = Path("/home/moshtasa/Research/phd-svd-recsys/Book/current/Data/injected_datasets")

SUMMARY_TXT = str(OUT_DIR / "Selected2_Book_injection_data_summary.txt")
SUMMARY_CSV = str(OUT_DIR / "Selected2_Book_injection_data_summary.csv")

GENRE_COL   = "genres"
USER_COL    = "user_id"
BOOK_COL    = "book_id"
RATING_COL  = "rating"

RUNS = [2, 4, 6, 25, 50, 100, 200, 500, 1000, 2000]

POS_RATING  = 5
NEG_RATING  = 1

BLOCK = 1_000_000

# 👉 NEW: SELECT GENRES HERE
# Leave empty [] to process ALL genres
genre = ["Drama", "Adventure" , "Horror", "Nonfiction"]
# =======================================


def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"


def split_genres(cell: str):
    if not isinstance(cell, str) or not cell.strip():
        return []
    return [g.strip() for g in cell.split(",") if g.strip()]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading dataset from:")
    print(INPUT_CSV)
    print("File exists:", os.path.exists(INPUT_CSV))

    df = pd.read_csv(INPUT_CSV)

    required = {USER_COL, BOOK_COL, RATING_COL, GENRE_COL}
    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {required}")

    df[GENRE_COL] = df[GENRE_COL].fillna("").astype(str)
    df[USER_COL] = df[USER_COL].astype(int)
    df[BOOK_COL] = df[BOOK_COL].astype(int)

    base_start_uid = df[USER_COL].max() + 1

    # ---- unique book -> genres mapping ----
    book_gen = df[[BOOK_COL, GENRE_COL]].drop_duplicates(subset=[BOOK_COL]).copy()
    book_gen["_genre_list"] = book_gen[GENRE_COL].apply(split_genres)
    book_gen = book_gen[book_gen["_genre_list"].map(len) > 0].copy()

    all_books = sorted(book_gen[BOOK_COL].unique())
    total_books = len(all_books)
    book_to_genres = dict(book_gen[[BOOK_COL, GENRE_COL]].values)

    # ---- ALL GENRES IN DATA ----
    all_target_genres = sorted(
        {g for genre_list in book_gen["_genre_list"] for g in genre_list}
    )

    # 👉 FILTER GENRES IF USER PROVIDED LIST
    if genre:
        selected_genres = [g for g in genre if g in all_target_genres]
        missing = set(genre) - set(selected_genres)

        if missing:
            print("\n⚠️ Warning: These genres were not found in dataset:")
            for m in missing:
                print(f"  - {m}")

        target_genres = selected_genres
    else:
        target_genres = all_target_genres

    print(f"\n🎯 Processing {len(target_genres)} genres")

    rows_summary = []

    with open(SUMMARY_TXT, "w") as log:
        log.write(f"BASE DATA: {df[USER_COL].nunique()} users, {len(df)} rows\n")
        log.write("POSITIVE = 5, NEGATIVE = 1\n\n")

    total_added = 0

    for gi, genre_name in enumerate(target_genres):
        safe_name = sanitize_fn(genre_name)

        # ---- positive = genre appears ANYWHERE ----
        pos_books = sorted(
            book_gen.loc[
                book_gen["_genre_list"].apply(lambda gs: genre_name in gs),
                BOOK_COL
            ].unique()
        )

        neg_books = sorted(set(all_books) - set(pos_books))

        pos_count = len(pos_books)
        neg_count = len(neg_books)

        first_count = int(book_gen["_genre_list"].apply(lambda gs: len(gs) >= 1 and gs[0] == genre_name).sum())
        second_count = int(book_gen["_genre_list"].apply(lambda gs: len(gs) >= 2 and gs[1] == genre_name).sum())

        percentage_of_all = (pos_count / total_books * 100.0) if total_books else 0.0
        percentage_of_that_genre = (first_count / pos_count * 100.0) if pos_count else 0.0

        for r_i, run in enumerate(RUNS):
            start_uid = base_start_uid + gi * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = list(range(start_uid, start_uid + run))

            pos_rows_count = pos_count * run
            neg_rows_count = neg_count * run
            total_rows_added = pos_rows_count + neg_rows_count

            print(f"\nCategory '{genre_name}' | Run {run}")
            print(f"  -> Items in category: {pos_count}")
            print(f"  -> Other categories: {neg_count}")
            print(f"  -> Adding users: {run}")

            # ---- BUILD POSITIVE ROWS ----
            pos_rows = {
                USER_COL:   [u for u in new_users for _ in pos_books],
                BOOK_COL:   [b for _ in new_users for b in pos_books],
                RATING_COL: [POS_RATING] * pos_rows_count,
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in pos_books]
            }

            # ---- BUILD NEGATIVE ROWS ----
            neg_rows = {
                USER_COL:   [u for u in new_users for _ in neg_books],
                BOOK_COL:   [b for _ in new_users for b in neg_books],
                RATING_COL: [NEG_RATING] * neg_rows_count,
                GENRE_COL:  [book_to_genres[b] for _ in new_users for b in neg_books]
            }

            synth_df = pd.concat(
                [pd.DataFrame(pos_rows), pd.DataFrame(neg_rows)],
                ignore_index=True
            )

            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_books.csv"
            combined.to_csv(out_file, index=False)

            total_added += total_rows_added

    print("\n✅ Done. Genre selection applied.")


if __name__ == "__main__":
    main()