import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Paths ──
BASE_NMF = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_NMF/result"
BASE_SVD = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_SVD/result"

OUTPUT_BASE = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405//NMF_SVD_Comparisons/9"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# =========================
# LOAD DATA
# =========================

def load_data(BASE):
    BI_DIR = os.path.join(BASE, "bi")
    MU_DIR = os.path.join(BASE, "mio")
    BU_DIR = os.path.join(BASE, "bu")

    BI_PATTERN = re.compile(r"^f_(.+?)_(\d+)u_pos5_neg1_books_bi_per_genre\.csv$")

    merged_data = {}

    for fname in os.listdir(BI_DIR):
        m = BI_PATTERN.match(fname)
        if not m:
            continue

        genre_key = m.group(1)
        n = int(m.group(2))

        bi_path = os.path.join(BI_DIR, fname)
        mu_path = os.path.join(MU_DIR, fname.replace("_bi_", "_mu_"))
        bu_path = os.path.join(BU_DIR, fname.replace("_bi_", "_bu_").replace("_per_genre.csv", "_per_genre_biased.csv"))

        if not (os.path.exists(mu_path) and os.path.exists(bu_path)):
            continue

        df_bi = pd.read_csv(bi_path)
        df_mu = pd.read_csv(mu_path)
        df_bu = pd.read_csv(bu_path)

        for df in [df_bi, df_mu, df_bu]:
            if "genre" in df.columns:
                df["genre"] = df["genre"].astype(str).str.strip()
                df = df[df["genre"].str.lower() != "unknown"]

        df = pd.merge(df_mu, df_bi, on=["genre", "is_target"], how="outer")
        df = pd.merge(df, df_bu, on=["genre", "is_target"], how="outer")

        merged_data[(genre_key, n)] = df

    genre_groups = defaultdict(dict)
    for (genre_key, n), df in merged_data.items():
        genre_groups[genre_key][n] = df

    return genre_groups


nmf_groups = load_data(BASE_NMF)
svd_groups = load_data(BASE_SVD)

# =========================
# STYLE
# =========================

TITLE_F = 34
LABEL_F = 30
TICK_F = 30
SUPTITLE_F = 50

# =========================
# HELPERS
# =========================

def get_val(row, col):
    if row.empty or col not in row.columns:
        return None
    return row[col].values[0]


def build_series(n_dict, sorted_ns, compute):
    genres = set()
    for n in sorted_ns:
        genres.update(n_dict[n]["genre"].tolist())

    genres = sorted(genres)

    series = {}
    for g in genres:
        pts = []
        for n in sorted_ns:
            row = n_dict[n][n_dict[n]["genre"] == g]
            val = compute(row)
            if val is not None:
                pts.append((n, val))
        series[g] = pts

    return series, genres


# =========================
# FIXED STYLE (NO GRID + BORDER)
# =========================

def apply_axis_style(ax, y_label="", show_x_label=True):
    if y_label:
        ax.set_ylabel(y_label, fontsize=LABEL_F, fontweight="bold")
    
    x_ticks = [0, 500, 2000]
    if show_x_label:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks], fontsize=TICK_F)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xticks([])
    
    ax.tick_params(axis='y', labelsize=TICK_F)

    # ❌ GRID REMOVED
    # ax.grid(True, ...)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)


# =========================
# FIXED PLOT (GREEN + THICK)
# =========================

def plot_one_model_on_axis(ax, groups, genre_key, compute):
    if genre_key not in groups:
        ax.axis('off')
        return

    n_dict = groups[genre_key]
    sorted_ns = sorted(n_dict.keys())
    series, all_genres = build_series(n_dict, sorted_ns, compute)

    for g in all_genres:
        pts = series[g]
        if not pts:
            continue
        
        xs = [n for n, _ in pts]
        ys = [y for _, y in pts]
        
        ax.plot(
            xs, ys,
            marker="o",
            markersize=8,
            linewidth=6,      # ✔ thicker
            color="green"     # ✔ green
        )


# =========================
# COMPARISON FIGURE
# =========================

def plot_mu_comparison_grid(nmf_groups, svd_groups):
    compute_fn = lambda r: get_val(r, "global_mean_mu")
    ylabel = r"$\mu$"
    genres = sorted(nmf_groups.keys())
    
    fig, axes = plt.subplots(3, 6, figsize=(36, 16))
    
    for i, genre_key in enumerate(genres):
        row = i // 3
        col_nmf = (i % 3) * 2
        col_svd = col_nmf + 1
        
        ax_nmf = axes[row, col_nmf]
        ax_svd = axes[row, col_svd]
        
        ax_nmf.set_title(f"{genre_key.replace('_', ' ')} (NMF)", fontsize=TITLE_F, fontweight="bold")
        ax_svd.set_title(f"{genre_key.replace('_', ' ')} (SVD)", fontsize=TITLE_F, fontweight="bold")
        
        plot_one_model_on_axis(ax_nmf, nmf_groups, genre_key, compute_fn)
        plot_one_model_on_axis(ax_svd, svd_groups, genre_key, compute_fn)
        
        is_first_col = (col_nmf == 0)
        apply_axis_style(ax_nmf, y_label=ylabel if is_first_col else "")
        apply_axis_style(ax_svd)

    fig.suptitle("NMF vs SVD - Global Mean μ Comparison", fontsize=SUPTITLE_F, fontweight="bold")

    # ✔ spacing fixed
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    path = os.path.join(OUTPUT_BASE, "NMF_vs_SVD_mu_comparison_3x6.png")
    fig.savefig(path, dpi=150)
    plt.close()

    print(f"✓ Saved Comparison: {path}")


# =========================
# RUN
# =========================

print("\nGenerating fixed comparison figure...")

plot_mu_comparison_grid(nmf_groups, svd_groups)

print("\nDONE ✅")