"""Build a book knowledge graph from metadata (deterministic, no API needed).

Edges:
  same_author   -- books share an author           (weight 1.0)
  same_genre    -- books share >=1 genre           (weight 0.5 per shared)
  same_decade   -- books in same decade            (weight 0.3)

Sparsified: each book keeps at most KG_MAX_DEGREE highest-weight neighbors.
Output: book_kg_deterministic.csv  (book_i, book_j, edge_type, weight)
"""
import ast
import pandas as pd
from collections import defaultdict
from config_kg import ORIGINAL_PATH, KG_DETERMINISTIC, KG_MAX_DEGREE


def parse_genres(s):
    if pd.isna(s): return set()
    s = str(s).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            p = ast.literal_eval(s)
            if isinstance(p, (list, tuple)):
                return {str(x).strip() for x in p if str(x).strip()}
        except Exception: pass
    for sep in [",", "|", ";", "//", "/"]:
        if sep in s:
            return {t.strip() for t in s.split(sep) if t.strip()}
    return {s}


def main():
    df = pd.read_csv(ORIGINAL_PATH, low_memory=False).drop_duplicates("book_id")
    df["genres_set"] = df["genres"].apply(parse_genres)
    print(f"[kg] {len(df)} unique books")

    # group by author
    author_books = defaultdict(list)
    for _, r in df.iterrows():
        a = str(r["authors"]).strip()
        if a and a.lower() != "nan":
            author_books[a].append(int(r["book_id"]))

    # group by decade (only numeric decades; skip labels like 'Ancient Books')
    decade_books = defaultdict(list)
    for _, r in df.iterrows():
        d = r["decade"]
        if pd.notna(d):
            try:
                decade_books[int(d)].append(int(r["book_id"]))
            except (ValueError, TypeError):
                pass  # non-numeric decade label -- skip

    # build weighted edge dict: (i,j) -> total weight
    edges = defaultdict(float)
    edge_types = defaultdict(set)

    def add(i, j, w, t):
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if a == b: return
        edges[(a, b)] += w
        edge_types[(a, b)].add(t)

    # same-author edges
    n_auth = 0
    for a, books in author_books.items():
        if len(books) < 2: continue
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                add(books[i], books[j], 1.0, "same_author")
                n_auth += 1
    print(f"[kg] same-author candidate edges: {n_auth}")

    # same-genre edges (only within each primary genre to keep it tractable)
    # For each book, connect to books sharing its FIRST genre
    primary_genre_books = defaultdict(list)
    for _, r in df.iterrows():
        gs = sorted(r["genres_set"])
        if gs:
            primary_genre_books[gs[0]].append(int(r["book_id"]))
    n_genre = 0
    for g, books in primary_genre_books.items():
        if len(books) < 2 or len(books) > 2000:  # skip huge genres (too many pairs)
            continue
        for i in range(len(books)):
            for j in range(i + 1, len(books)):
                add(books[i], books[j], 0.5, "same_genre")
                n_genre += 1
    print(f"[kg] same-genre candidate edges: {n_genre}")

    # same-decade edges (sample: connect each book to up to 10 random same-decade books)
    import numpy as np
    rng = np.random.default_rng(42)
    n_dec = 0
    for d, books in decade_books.items():
        if len(books) < 2: continue
        books = list(books)
        for b in books:
            peers = [x for x in books if x != b]
            if len(peers) > 10:
                peers = list(rng.choice(peers, 10, replace=False))
            for p in peers:
                add(b, p, 0.3, "same_decade")
                n_dec += 1
    print(f"[kg] same-decade candidate edges: {n_dec}")

    # sparsify: keep top KG_MAX_DEGREE per book
    adj = defaultdict(list)
    for (i, j), w in edges.items():
        adj[i].append((j, w))
        adj[j].append((i, w))
    kept = set()
    for b, nbrs in adj.items():
        nbrs.sort(key=lambda x: -x[1])
        for j, w in nbrs[:KG_MAX_DEGREE]:
            a, c = (b, j) if b < j else (j, b)
            kept.add((a, c))

    rows = []
    for (i, j) in kept:
        rows.append({"book_i": i, "book_j": j,
                     "edge_type": "|".join(sorted(edge_types[(i, j)])),
                     "weight": round(edges[(i, j)], 3)})
    out = pd.DataFrame(rows).sort_values("weight", ascending=False)
    out.to_csv(KG_DETERMINISTIC, index=False)
    print(f"[kg] saved {len(out)} edges -> {KG_DETERMINISTIC}")
    print(f"[kg] edge types: {out['edge_type'].value_counts().to_dict()}")
    print(f"[kg] books covered: {len(set(out['book_i']) | set(out['book_j']))}")


if __name__ == "__main__":
    main()
