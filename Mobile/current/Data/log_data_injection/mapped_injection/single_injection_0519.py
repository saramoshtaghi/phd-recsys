# build_heavy_bias_pos5_neg1_mobile_selected_categories.py

import os
import re
import pandas as pd
from pathlib import Path

# ========= CONFIG =========

INPUT_CSV = "/home/moshtasa/Research/phd-recsys/Mobile/archive/0316_Single/data/data/mobilerec_final.csv"

OUT_DIR = Path("/home/moshtasa/Research/phd-recsys/Mobile/current/Data/log_data_injection/mapped_injection/injected_datasets")

SUMMARY_CSV = str(OUT_DIR / "injection_data_summary.csv")

CATEGORY_COL = "app_category"
USER_COL     = "uid"
ITEM_COL     = "app_package"
RATING_COL   = "rating"

# ✅ FIXED: separate categories correctly
TARGET_CATEGORIES = ["Medical", "Weather"]

RUNS = [30, 60, 300, 500, 2000, 7000]

POS_RATING  = 5
NEG_RATING  = 1

BLOCK = 1_000_000
# =======================================


def sanitize_fn(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_]+", "_", s) or "UNK"


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # ✅ Normalize categories (important!)
    df[CATEGORY_COL] = df[CATEGORY_COL].fillna("").astype(str).str.strip()
    df[USER_COL] = df[USER_COL].astype(str)
    df[ITEM_COL] = df[ITEM_COL].astype(str)

    print("Unique categories in dataset:")
    print(df[CATEGORY_COL].unique())

    base_start_uid = df[USER_COL].nunique() + 1

    item_cat = df[[ITEM_COL, CATEGORY_COL]].drop_duplicates(subset=[ITEM_COL])
    item_cat = item_cat[item_cat[CATEGORY_COL] != ""]

    item_to_cat = dict(item_cat[[ITEM_COL, CATEGORY_COL]].values)

    rows_summary = []

    # ---- LOOP ONLY OVER SELECTED CATEGORIES ----
    for gi, category in enumerate(TARGET_CATEGORIES):

        safe_name = sanitize_fn(category)

        pos_items = item_cat[item_cat[CATEGORY_COL] == category][ITEM_COL].tolist()
        neg_items = item_cat[item_cat[CATEGORY_COL] != category][ITEM_COL].tolist()

        pos_count = len(pos_items)
        neg_count = len(neg_items)

        # 🚨 DEBUG
        print(f"\n[DEBUG] Category: {category}")
        print(f"  Pos items: {pos_count}")
        print(f"  Neg items: {neg_count}")

        if pos_count == 0:
            print(f"⚠️ WARNING: No items found for category '{category}'")
            continue

        for r_i, run in enumerate(RUNS):

            start_uid = base_start_uid + gi * (len(RUNS) * BLOCK) + r_i * BLOCK
            new_users = [f"synthetic_{u}" for u in range(start_uid, start_uid + run)]

            pos_rows_count = pos_count * run
            neg_rows_count = neg_count * run
            total_rows_added = pos_rows_count + neg_rows_count

            print(f"\nCategory '{category}' | Run {run}")
            print(f"  -> Users added: {run}")

            # ---- BUILD POSITIVE ----
            pos_rows = {
                USER_COL:   [u for u in new_users for _ in pos_items],
                ITEM_COL:   [i for _ in new_users for i in pos_items],
                RATING_COL: [POS_RATING] * pos_rows_count,
                CATEGORY_COL: [category] * pos_rows_count
            }

            # ---- BUILD NEGATIVE ----
            neg_rows = {
                USER_COL:   [u for u in new_users for _ in neg_items],
                ITEM_COL:   [i for _ in new_users for i in neg_items],
                RATING_COL: [NEG_RATING] * neg_rows_count,
                CATEGORY_COL: [item_to_cat[i] for _ in new_users for i in neg_items]
            }

            synth_df = pd.concat([
                pd.DataFrame(pos_rows),
                pd.DataFrame(neg_rows)
            ], ignore_index=True)

            combined = pd.concat([df, synth_df], ignore_index=True)

            out_file = OUT_DIR / f"f_{safe_name}_{run}u_pos5_neg1_allapps.csv"
            combined.to_csv(out_file, index=False)

            rows_summary.append({
                "category": category,
                "run_users": run,
                "rows_added": total_rows_added,
                "output_file": str(out_file)
            })

    pd.DataFrame(rows_summary).to_csv(SUMMARY_CSV, index=False)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()