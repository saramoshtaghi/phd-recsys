"""PerturbRec Model 2 -- Transformer recommender. All paths + hyperparams."""
from pathlib import Path

BASE            = Path("/home/moshtasa/Research/phd-recsys/Book/current/PerturbRec")
DATA_DIR        = BASE / "Data"
ORIGINAL_PATH   = DATA_DIR / "ORIGINAL_books.csv"
TRANSFORMER_DIR = BASE / "transformer"
OUT_ROOT        = TRANSFORMER_DIR / "result"
OUT_CKPT        = OUT_ROOT / "checkpoints"
OUT_EMB         = OUT_ROOT / "embeddings"
for d in (OUT_CKPT, OUT_EMB):
    d.mkdir(parents=True, exist_ok=True)

# ---- Model ----
D_MODEL       = 128
NHEAD         = 4
D_HID         = 128
NLAYERS       = 4
N_INPUT_BINS  = 6      # bin 0 = special (CLS/pad/MASK), bins 1-5 = ratings
MAX_LEN       = 200    # max ratings/user in this dataset is 200
PAD_ID        = 0
CLS_ID        = 1
PAD_TOKEN     = "<pad>"
CLS_TOKEN     = "<cls>"

# ---- Training ----
BATCH_SIZE    = 64
N_EPOCHS      = 20
LR            = 1e-4
MLM_MASK_PROB = 0.15
NUM_WORKERS   = 4

# ---- Data ----
USER_COL, BOOK_COL, RATE_COL, GENRE_COL = "user_id", "book_id", "rating", "genres"
GENRES       = ["Drama"]
N_FILTER     = {2, 4, 6, 25, 50, 100, 200, 500, 1000, 2000}
