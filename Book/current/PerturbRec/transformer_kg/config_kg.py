"""PerturbRec Model 3 -- Transformer + Knowledge Graph. Paths + hyperparams."""
from pathlib import Path

BASE       = Path("/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec")
DATA_DIR   = BASE / "Data"
ORIGINAL_PATH = DATA_DIR / "ORIGINAL_books.csv"
KG_DIR      = Path(__file__).resolve().parent / "kg"
KG_DETERMINISTIC = KG_DIR / "book_kg_deterministic.csv"
KG_LLM           = KG_DIR / "book_kg_llm.csv"
KG_MERGED        = KG_DIR / "book_kg_merged.csv"

OUT_ROOT  = Path(__file__).resolve().parent / "result"
OUT_CKPT  = OUT_ROOT / "checkpoints"
OUT_EMB   = OUT_ROOT / "embeddings"
for d in (OUT_CKPT, OUT_EMB, KG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Model (same as Model 2) ----
D_MODEL       = 128
NHEAD         = 4
D_HID         = 128
NLAYERS       = 4
N_INPUT_BINS  = 6
MAX_LEN       = 200
PAD_ID        = 0
CLS_ID        = 1
PAD_TOKEN     = "<pad>"
CLS_TOKEN     = "<cls>"

# ---- Training ----
BATCH_SIZE    = 64
N_EPOCHS      = 15
LR            = 1e-4
MLM_MASK_PROB = 0.15
NUM_WORKERS   = 4
KG_LAMBDA     = 0.1       # weight of KG regularization loss
KG_EDGE_BATCH = 2048      # edges sampled per step for KG loss
KG_MAX_DEGREE = 20        # cap edges per book (sparsify)

# ---- Data ----
USER_COL, BOOK_COL, RATE_COL, GENRE_COL = "user_id", "book_id", "rating", "genres"
GENRES       = ["Drama"]
N_FILTER     = {2, 4, 6, 25, 50, 100, 200, 500, 1000, 2000}

# ---- LLM KG extraction (optional, for paper version) ----
OPENAI_KEY_FILE = Path(__file__).resolve().parent / "api_key.txt"
LLM_MODEL       = "gpt-4o"
LLM_BATCH_SIZE  = 25
LLM_MAX_BOOKS   = 500   # subset (most-rated) for LLM extraction
