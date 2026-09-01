"""Merge deterministic + LLM KGs into one edge file for training."""
import pandas as pd
from config_kg import KG_DETERMINISTIC, KG_LLM, KG_MERGED


def main():
    frames = []
    if KG_DETERMINISTIC.exists():
        frames.append(pd.read_csv(KG_DETERMINISTIC))
        print(f"[merge] deterministic: {len(frames[-1])} edges")
    if KG_LLM.exists():
        frames.append(pd.read_csv(KG_LLM))
        print(f"[merge] llm: {len(frames[-1])} edges")
    if not frames:
        print("[merge] no KG files found -- run build_kg.py first")
        return
    merged = pd.concat(frames, ignore_index=True)
    # dedupe by (i,j) keeping max weight
    merged["pair"] = merged.apply(
        lambda r: (min(r["book_i"], r["book_j"]), max(r["book_i"], r["book_j"])), axis=1)
    merged = merged.sort_values("weight", ascending=False).drop_duplicates("pair")
    merged = merged.drop(columns="pair")
    merged.to_csv(KG_MERGED, index=False)
    print(f"[merge] merged: {len(merged)} edges -> {KG_MERGED}")


if __name__ == "__main__":
    main()
