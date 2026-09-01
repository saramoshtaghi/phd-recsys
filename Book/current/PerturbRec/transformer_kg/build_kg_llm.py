"""Build a book knowledge graph using LLM extraction (reuses Children's KG pattern).

Feeds book metadata (title, author, genre) to GPT-4o in batches, asks it to
extract semantic similarity relationships. Output: book_kg_llm.csv.

Requires: api_key.txt in this folder with an OpenAI key (one line).
Skips gracefully if no key found (deterministic KG is still used).

NOTE: without synopses, relationships are inferred from title/author/genre only.
For the paper, fetching synopses (Goodreads/Google Books API) would enrich this.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd
from config_kg import (ORIGINAL_PATH, KG_LLM, OPENAI_KEY_FILE,
                       LLM_MODEL, LLM_BATCH_SIZE, LLM_MAX_BOOKS)


PROMPT = """You are a literary knowledge graph extractor. Given a list of books
(title, author, genres), identify pairs of books that are SIMILAR or RELATED,
and classify the relationship type.

For each similar pair, output a JSON array of objects:
  {"i": <book number>, "j": <book number>, "type": "<relationship>", "weight": 0.0-1.0}

Relationship types: similar_theme, similar_style, same_series, influenced_by,
same_author_universe, era_companion, genre_companion.

Only include pairs you are confident about (weight >= 0.5). If no confident pairs,
return []. Output ONLY the JSON array, no prose.

Books:
"""


def load_books():
    df = pd.read_csv(ORIGINAL_PATH, low_memory=False).drop_duplicates("book_id")
    df = df[df["original_title"].notna() & df["authors"].notna()]
    # pick most-rated books for LLM extraction
    counts = pd.read_csv(ORIGINAL_PATH, usecols=["book_id"]).value_counts("book_id")
    df["n_ratings"] = df["book_id"].map(counts)
    df = df.sort_values("n_ratings", ascending=False).head(LLM_MAX_BOOKS)
    return df.reset_index(drop=True)


def format_batch(batch_df):
    lines = []
    for i, r in batch_df.iterrows():
        lines.append(f"[{i}] {r['original_title']} | {r['authors']} | {r['genres']}")
    return "\n".join(lines)


def main():
    if not OPENAI_KEY_FILE.exists():
        print(f"[kg_llm] no api_key.txt at {OPENAI_KEY_FILE} -- skipping LLM KG")
        print("[kg_llm] deterministic KG (build_kg.py) will be used alone")
        return
    try:
        import openai
    except ImportError:
        print("[kg_llm] openai package not installed -- skipping")
        return

    key = OPENAI_KEY_FILE.read_text().strip()
    if not key:
        print("[kg_llm] empty api_key.txt -- skipping")
        return
    client = openai.OpenAI(api_key=key)
    print(f"[kg_llm] API key loaded, model={LLM_MODEL}")

    books = load_books()
    print(f"[kg_llm] {len(books)} books for LLM extraction")

    all_edges = []
    for start in range(0, len(books), LLM_BATCH_SIZE):
        batch = books.iloc[start:start + LLM_BATCH_SIZE]
        user_msg = PROMPT + format_batch(batch)
        print(f"[kg_llm] batch {start//LLM_BATCH_SIZE + 1}/"
              f"{(len(books)+LLM_BATCH_SIZE-1)//LLM_BATCH_SIZE}")
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "system", "content": "You extract book relationships as JSON."},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            # extract JSON array
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            edges = json.loads(text)
            for e in edges:
                if "i" in e and "j" in e:
                    bi = int(batch.iloc[int(e["i"])]["book_id"])
                    bj = int(batch.iloc[int(e["j"])]["book_id"])
                    all_edges.append({"book_i": bi, "book_j": bj,
                                      "edge_type": e.get("type", "llm_similar"),
                                      "weight": float(e.get("weight", 0.7))})
        except Exception as ex:
            print(f"[kg_llm]   error: {ex}")
        time.sleep(1)

    if not all_edges:
        print("[kg_llm] no edges extracted -- skipping")
        return
    out = pd.DataFrame(all_edges)
    out.to_csv(KG_LLM, index=False)
    print(f"[kg_llm] saved {len(out)} LLM edges -> {KG_LLM}")


if __name__ == "__main__":
    main()
