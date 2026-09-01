import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Paths ──
BASE_NMF = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_NMF/result"
BASE_SVD = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/0405_SVD/result"

NMF_FIG_DIR = os.path.join(BASE_NMF, "NMF_details_figures")
SVD_FIG_DIR = os.path.join(BASE_SVD, "SVD_details_figures")

NMF_BAR_FIG_DIR = os.path.join(BASE_NMF, "NMF_bar_figures")
SVD_BAR_FIG_DIR = os.path.join(BASE_SVD, "SVD_bar_figures")

# ── Output directory (organized by genre) ──
OUTPUT_BASE = "/home/moshtasa/Research/phd-svd-recsys/Mobile/0405/NMF_SVD_Comparisons"
os.makedirs(OUTPUT_BASE, exist_ok=True)

# =========================
# LOAD DATA
# =========================

def load_data(BASE):
    BI_DIR = os.path.join(BASE, "bi")
    MU_DIR = os.path.join(BASE, "mio")
    BU_DIR = os.path.join(BASE, "bu")

    BI_PATTERN = re.compile(r"^f_(.+?)_(\d+)u_pos5_neg1_allapps_bi_per_category\.csv$")

    merged_data = {}
    bu_data = {}

    for fname in os.listdir(BI_DIR):
        m = BI_PATTERN.match(fname)
        if not m:
            continue

        genre_key = m.group(1)
        n = int(m.group(2))

        bi_path = os.path.join(BI_DIR, fname)
        mu_path = os.path.join(MU_DIR, fname.replace("_bi_", "_mu_"))
        bu_path = os.path.join(BU_DIR, fname.replace("_bi_", "_bu_").replace("_per_genre.csv", "_per_genre_biased.csv"))

        if not os.path.exists(mu_path):
            continue

        df_bi = pd.read_csv(bi_path)
        df_mu = pd.read_csv(mu_path)

        # Normalize and exclude unknown
        for df in [df_bi, df_mu]:
            if "category" in df.columns:
                df["category"] = df["category"].astype(str).str.strip()
                df_filtered = df[df["category"].str.lower() != "unknown"].copy()
                if df is df_bi:
                    df_bi = df_filtered
                else:
                    df_mu = df_filtered

        merge_cols = ["category"]
        if "is_target" in df_bi.columns:
            merge_cols.append("is_target")

        if os.path.exists(bu_path):
            df_bu = pd.read_csv(bu_path)
            if "category" in df_bu.columns:
                df_bu["category"] = df_bu["category"].astype(str).str.strip()
                df_bu = df_bu[df_bu["category"].str.lower() != "unknown"].copy()
            bu_data[(genre_key, n)] = df_bu.copy()
            if "user_type" in df_bu.columns:
                df_bu_merge = df_bu[df_bu["user_type"] == "non_biasing"].copy()
                if df_bu_merge.empty:
                    df_bu_merge = df_bu.copy()
            else:
                df_bu_merge = df_bu.copy()
            df = pd.merge(df_mu, df_bi, on=merge_cols, how="outer")
            df = pd.merge(df, df_bu_merge, on=merge_cols, how="outer")
        else:
            df = pd.merge(df_mu, df_bi, on=merge_cols, how="outer")

        merged_data[(genre_key, n)] = df

    genre_groups = defaultdict(dict)
    for (genre_key, n), df in merged_data.items():
        genre_groups[genre_key][n] = df

    bu_groups = defaultdict(dict)
    for (genre_key, n), df in bu_data.items():
        bu_groups[genre_key][n] = df

    return genre_groups, bu_groups

nmf_groups, nmf_bu_groups = load_data(BASE_NMF)
svd_groups, svd_bu_groups = load_data(BASE_SVD)

# =========================
# STYLE
# =========================

plt.style.use("seaborn-v0_8-pastel")

TARGET_COLOR = "#5B3A29"
TARGET_LINEWIDTH = 3.5
NON_TARGET_LINEWIDTH = 1.5

PASTEL_COLORS = [
    "#AEC6CF","#FFD1DC","#B5EAD7","#FFDAC1","#C3B1E1","#F3E5AB",
    "#FFB7B2","#B0E0E6","#D5AAFF","#FFFACD","#E0BBE4","#957DAD",
    "#D4E157","#80CBC4","#F48FB1",
]

# Colors for BU bias comparison
TARGET_BIAS_COLOR = "#8B0000"
TARGET_NONBIAS_COLOR = "#5B3A29"
NONTARGET_BIAS_COLOR = "#1F77B4"
NONTARGET_NONBIAS_COLOR = "#7FB3D5"

# =========================
# HELPER FUNCTIONS
# =========================

def get_single_value(row, col):
    """Extract single value from dataframe row."""
    if row.empty or col not in row.columns:
        return None
    return row[col].values[0]


def generate_subplot(ax, series, all_genres, target_genre_name, sorted_ns, model_name, genre_key):
    """
    Generate a single subplot for NMF or SVD.
    Returns legend handles dict for later use.
    
    CUSTOMIZATION: Adjust font sizes, colors, and styling here.
    """
    
    non_target_genres = [g for g in all_genres if g != target_genre_name]

    color_map = {}
    for i, g in enumerate(non_target_genres):
        color_map[g] = PASTEL_COLORS[i % len(PASTEL_COLORS)]
    color_map[target_genre_name] = TARGET_COLOR

    positions = list(range(len(sorted_ns)))
    n_to_pos = {n: p for n, p in zip(sorted_ns, positions)}

    legend_handles = {}

    # Plot non-target genres
    for g in non_target_genres:
        pts = series.get(g, [])
        if pts:
            xs = [n_to_pos[x] for x, _ in pts]
            ys = [y for _, y in pts]
            line, = ax.plot(xs, ys, marker="o",
                            color=color_map[g],
                            linewidth=NON_TARGET_LINEWIDTH,
                            markersize=9,
                            alpha=0.8,
                            label=g)
            legend_handles[g] = line

    # Plot target genre
    pts = series.get(target_genre_name, [])
    if pts:
        xs = [n_to_pos[x] for x, _ in pts]
        ys = [y for _, y in pts]
        line, = ax.plot(xs, ys, marker="o",
                        color=TARGET_COLOR,
                        linewidth=TARGET_LINEWIDTH,
                        markersize=7,
                        label=f"{target_genre_name} (target)",
                        zorder=10)
        legend_handles[target_genre_name] = line

    pretty = genre_key.replace("_", " ")

    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=20)

    ax.set_title(f"{model_name} - {pretty}", fontsize=26, fontweight='bold')
    ax.set_xlabel("Number of Fictitious Users (n)", fontsize=22, fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n in sorted_ns], rotation=45)

    ax.grid(True, alpha=0.3)

    return legend_handles


def build_series(n_dict, sorted_ns, compute_value):
    """
    Build series data for all genres across different n values.
    
    CUSTOMIZATION: Modify compute_value lambda to change metric calculation.
    """
    all_genres = set()
    for n in sorted_ns:
        all_genres.update(n_dict[n]["category"].tolist())
    all_genres = sorted(all_genres)

    series = {}
    for g in all_genres:
        pts = []
        for n in sorted_ns:
            df = n_dict[n]
            row = df[df["category"] == g]
            if not row.empty:
                val = compute_value(row)
                if val is not None:
                    pts.append((n, val))
        series[g] = pts

    return series, all_genres


def plot_merged_figures(genre_key, sorted_ns, nmf_dict, svd_dict, target_genre_name, 
                       compute_value, ylabel, metric_name, suffix):
    """
    Create a merged NMF-SVD comparison figure for a specific metric.
    
    CUSTOMIZATION: Modify ylabel, metric_name, and suffix for different metrics.
    """
    
    nmf_series, all_genres = build_series(nmf_dict, sorted_ns, compute_value)
    svd_series, _ = build_series(svd_dict, sorted_ns, compute_value)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6),
        sharey=True,
        constrained_layout=True
    )

    # Generate NMF subplot
    nmf_legend_handles = generate_subplot(
        axes[0], nmf_series, all_genres, target_genre_name, sorted_ns, "NMF", genre_key
    )

    # Generate SVD subplot
    svd_legend_handles = generate_subplot(
        axes[1], svd_series, all_genres, target_genre_name, sorted_ns, "SVD", genre_key
    )

    # Use NMF legend (both should be the same)
    legend_handles = nmf_legend_handles

    # Set y-label on left subplot only
    axes[0].set_ylabel(ylabel, fontweight="bold", fontsize=26)

    # Add legend at top
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        fontsize=20,
        ncol=min(len(legend_handles), 7),
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=0.8
    )

    # Save to genre-specific folder
    genre_output_dir = os.path.join(OUTPUT_BASE, genre_key)
    os.makedirs(genre_output_dir, exist_ok=True)

    merged_path = os.path.join(genre_output_dir, f"{genre_key}_{suffix}.png")
    fig.savefig(merged_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ {suffix}: {merged_path}")


def plot_bu_bias_comparison_merged(genre_key, sorted_ns, nmf_bu_dict, svd_bu_dict):
    """
    Create merged NMF-SVD comparison for BU bias analysis.
    
    CUSTOMIZATION: Modify colors, markers, and series labels here.
    """
    
    positions = list(range(len(sorted_ns)))
    n_to_pos = {n: p for n, p in zip(sorted_ns, positions)}

    # ────── NMF side ──────
    nmf_target_bias_pts = []
    nmf_target_nonbias_pts = []
    nmf_non_target_bias_pts = []
    nmf_non_target_nonbias_pts = []

    for n in sorted_ns:
        df = nmf_bu_dict[n].copy()

        if "category" in df.columns:
            df["category"] = df["category"].astype(str).str.strip()
            df = df[df["category"].str.lower() != "unknown"]

        if "user_type" in df.columns:
            df["user_type"] = df["user_type"].astype(str).str.strip()

        if "is_target" in df.columns:
            df["is_target"] = df["is_target"].astype(str).str.strip()

        # Target + biasing
        row = df[(df["is_target"] == "target") & (df["user_type"] == "biasing")]
        if not row.empty:
            nmf_target_bias_pts.append((n, row["avg_bu"].mean()))

        # Target + non_biasing
        row = df[(df["is_target"] == "target") & (df["user_type"] == "non_biasing")]
        if not row.empty:
            nmf_target_nonbias_pts.append((n, row["avg_bu"].mean()))

        # Non-target + biasing
        row = df[(df["is_target"] == "non_target") & (df["user_type"] == "biasing")]
        if not row.empty:
            nmf_non_target_bias_pts.append((n, row["avg_bu"].mean()))

        # Non-target + non_biasing
        row = df[(df["is_target"] == "non_target") & (df["user_type"] == "non_biasing")]
        if not row.empty:
            nmf_non_target_nonbias_pts.append((n, row["avg_bu"].mean()))

    # ────── SVD side ──────
    svd_target_bias_pts = []
    svd_target_nonbias_pts = []
    svd_non_target_bias_pts = []
    svd_non_target_nonbias_pts = []

    for n in sorted_ns:
        df = svd_bu_dict[n].copy()

        if "category" in df.columns:
            df["category"] = df["category"].astype(str).str.strip()
            df = df[df["category"].str.lower() != "unknown"]

        if "user_type" in df.columns:
            df["user_type"] = df["user_type"].astype(str).str.strip()

        if "is_target" in df.columns:
            df["is_target"] = df["is_target"].astype(str).str.strip()

        # Target + biasing
        row = df[(df["is_target"] == "target") & (df["user_type"] == "biasing")]
        if not row.empty:
            svd_target_bias_pts.append((n, row["avg_bu"].mean()))

        # Target + non_biasing
        row = df[(df["is_target"] == "target") & (df["user_type"] == "non_biasing")]
        if not row.empty:
            svd_target_nonbias_pts.append((n, row["avg_bu"].mean()))

        # Non-target + biasing
        row = df[(df["is_target"] == "non_target") & (df["user_type"] == "biasing")]
        if not row.empty:
            svd_non_target_bias_pts.append((n, row["avg_bu"].mean()))

        # Non-target + non_biasing
        row = df[(df["is_target"] == "non_target") & (df["user_type"] == "non_biasing")]
        if not row.empty:
            svd_non_target_nonbias_pts.append((n, row["avg_bu"].mean()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, constrained_layout=True)

    # NMF subplot
    for label, pts, color, linewidth, marker in [
        ("Target - biasing", nmf_target_bias_pts, TARGET_BIAS_COLOR, 3.5, "o"),
        ("Target - non_biasing", nmf_target_nonbias_pts, TARGET_NONBIAS_COLOR, 3.5, "s"),
        ("Non-target - biasing", nmf_non_target_bias_pts, NONTARGET_BIAS_COLOR, 2.5, "^"),
        ("Non-target - non_biasing", nmf_non_target_nonbias_pts, NONTARGET_NONBIAS_COLOR, 2.5, "D"),
    ]:
        if pts:
            xs = [n_to_pos[x] for x, _ in pts]
            ys = [y for _, y in pts]
            axes[0].plot(xs, ys, marker=marker, color=color, linewidth=linewidth, 
                        markersize=8, label=label)

    # SVD subplot
    for label, pts, color, linewidth, marker in [
        ("Target - biasing", svd_target_bias_pts, TARGET_BIAS_COLOR, 3.5, "o"),
        ("Target - non_biasing", svd_target_nonbias_pts, TARGET_NONBIAS_COLOR, 3.5, "s"),
        ("Non-target - biasing", svd_non_target_bias_pts, NONTARGET_BIAS_COLOR, 2.5, "^"),
        ("Non-target - non_biasing", svd_non_target_nonbias_pts, NONTARGET_NONBIAS_COLOR, 2.5, "D"),
    ]:
        if pts:
            xs = [n_to_pos[x] for x, _ in pts]
            ys = [y for _, y in pts]
            axes[1].plot(xs, ys, marker=marker, color=color, linewidth=linewidth,
                        markersize=8, label=label)

    pretty = genre_key.replace("_", " ")

    for ax, model_name in zip(axes, ["NMF", "SVD"]):
        ax.set_title(f"{model_name} - {pretty} (bᵤ biased vs unbiased)", fontsize=26, fontweight='bold')
        ax.set_xlabel("Number of Fictitious Users (n)", fontsize=22, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in sorted_ns], rotation=45)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="x", labelsize=20)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$b_u$", fontweight="bold", fontsize=26)

    fig.legend(
        [line for line in axes[0].get_lines()],
        [line.get_label() for line in axes[0].get_lines()],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        fontsize=20,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.6,
        columnspacing=0.8
    )

    genre_output_dir = os.path.join(OUTPUT_BASE, genre_key)
    os.makedirs(genre_output_dir, exist_ok=True)

    merged_path = os.path.join(genre_output_dir, f"{genre_key}_bu_biased_vs_unbiased_target_vs_nontarget.png")
    fig.savefig(merged_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ bu_biased_vs_unbiased: {merged_path}")


def parse_txt(file_path):
    """
    Parse txt file containing k values and metrics.
    Format: k = X, followed by metric_name: value pairs
    """
    data = {}
    current_k = None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("k ="):
                current_k = int(line.split("=")[1].strip())
                data[current_k] = {}

            elif ":" in line and current_k is not None:
                match = re.match(r"(.*?)\:\s*([0-9.]+)", line)
                if match:
                    label = match.group(1).strip()
                    value = float(match.group(2))
                    data[current_k][label] = value

    return data


def generate_colors(n_list):
    """
    Generate distinct colors for each n_type.
    CUSTOMIZATION: Adjust colormap here if needed.
    """
    cmap = plt.cm.get_cmap('tab20', len(n_list))
    return {n: cmap(i) for i, n in enumerate(n_list)}


def extract_genre(folder_name):
    """
    Extract genre name from bar figure folder name.
    Example: 'k_adult' -> 'adult'
    """
    parts = folder_name.lower().split("_")
    return parts[1] if len(parts) > 1 else None


def plot_bar_chart_merged(genre_key):
    """
    Create merged NMF-SVD bar chart comparison for k-value metrics.
    Uses the same directory structure as the original bar figure code.
    
    CUSTOMIZATION: Modify colors, bar widths, and legend here.
    """
    
    # Use Path objects like the original code
    BASE_PATHS = {
        "NMF": Path(NMF_BAR_FIG_DIR),
        "SVD": Path(SVD_BAR_FIG_DIR),
    }
    
    # Extract genre folders (e.g., "k_adult", "k_classics")
    def extract_genre_from_folder(folder_name):
        parts = folder_name.lower().split("_")
        return parts[1] if len(parts) > 1 else None
    
    nmf_map = {
        extract_genre_from_folder(f): f
        for f in os.listdir(BASE_PATHS["NMF"])
        if (BASE_PATHS["NMF"] / f).is_dir()
    }
    
    svd_map = {
        extract_genre_from_folder(f): f
        for f in os.listdir(BASE_PATHS["SVD"])
        if (BASE_PATHS["SVD"] / f).is_dir()
    }
    
    # Check if this genre exists in both NMF and SVD
    if genre_key not in nmf_map or genre_key not in svd_map:
        return False
    
    nmf_folder = nmf_map[genre_key]
    svd_folder = svd_map[genre_key]
    
    paths = {
        "NMF": BASE_PATHS["NMF"] / nmf_folder,
        "SVD": BASE_PATHS["SVD"] / svd_folder,
    }
    
    parsed_data = {}
    
    for method, folder_path in paths.items():
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if not txt_files:
            return False
        
        txt_path = folder_path / txt_files[0]
        parsed_data[method] = parse_txt(str(txt_path))
    
    if len(parsed_data) < 2:
        return False
    
    # Filter for specific k values
    ks = [k for k in sorted(parsed_data["NMF"].keys()) if k in [15, 25, 35]]
    if not ks:
        return False
    
    # Collect all n_types
    n_types = sorted(set(
        n
        for method in parsed_data
        for k in ks
        for n in parsed_data[method][k].keys()
    ), key=lambda x: (x != "original", int(x.replace("u", "")) if x != "original" else -1))
    
    color_map = generate_colors(n_types)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    
    legend_handles = []
    legend_labels = []
    
    for ax, method in zip(axes, ["NMF", "SVD"]):
        data = parsed_data[method]
        
        x = np.arange(len(ks)) * 1.3
        width = 1.1 / len(n_types)
        
        for i, n in enumerate(n_types):
            vals = [data[k].get(n, 0) for k in ks]
            
            bars = ax.bar(
                x + (i - len(n_types)/2) * width + width/2,
                vals,
                width=width,
                color=color_map[n],
                edgecolor="black",
                linewidth=1.5,
                label=n
            )
            
            # Collect legend only once (from NMF)
            if method == "NMF":
                legend_handles.append(bars[0])
                legend_labels.append(n)
        
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        
        genre_display = genre_key.replace('_', ' ').capitalize()
        ax.set_title(f"{method} - {genre_display}", fontsize=28, fontweight="bold", pad=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(ks, fontsize=24)
        ax.tick_params(axis='y', labelsize=24)
        
        ax.set_ylim(0, 40)
        ax.set_xlabel("k (recommendation size)", fontsize=26, fontweight="bold", labelpad=10)
    
    axes[0].set_ylabel("Avg count per user", fontsize=26, fontweight="bold", labelpad=10)
    
    # Add legend at top
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.35, 1.1),
        fontsize=26,
        ncol=min(len(legend_labels), 8),
        handlelength=1.3,
        handletextpad=0.4,
        columnspacing=0.7
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save to genre-specific folder
    genre_output_dir = os.path.join(OUTPUT_BASE, genre_key)
    os.makedirs(genre_output_dir, exist_ok=True)
    
    merged_path = os.path.join(genre_output_dir, f"{genre_key}_k_value_comparison.png")
    fig.savefig(merged_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  ✓ k_value_comparison: {merged_path}")
    return True


# =========================
# MAIN LOOP
# =========================

for genre_key in sorted(nmf_groups.keys()):

    if genre_key not in svd_groups:
        continue

    nmf_dict = nmf_groups[genre_key]
    svd_dict = svd_groups[genre_key]

    sorted_ns = sorted(set(nmf_dict.keys()) & set(svd_dict.keys()))
    if not sorted_ns:
        continue

    sample_df = nmf_dict[sorted_ns[0]]
    target_rows = sample_df[sample_df["is_target"] == "target"]
    if target_rows.empty:
        continue

    target_genre_name = target_rows["category"].iloc[0]

    print(f"\n📊 Processing genre: {genre_key}")

    # 1. μ + bi
    plot_merged_figures(
        genre_key, sorted_ns, nmf_dict, svd_dict, target_genre_name,
        lambda r: (
            get_single_value(r, "global_mean_mu") + get_single_value(r, "avg_bi")
            if get_single_value(r, "global_mean_mu") is not None and get_single_value(r, "avg_bi") is not None
            else None
        ),
        r"$\mu + b_i$",
        "μ + bᵢ",
        "mu_plus_bi"
    )

    # 2. bi only
    plot_merged_figures(
        genre_key, sorted_ns, nmf_dict, svd_dict, target_genre_name,
        lambda r: get_single_value(r, "avg_bi"),
        r"$b_i$",
        "bᵢ",
        "bi_only"
    )

    # 3. bu only
    plot_merged_figures(
        genre_key, sorted_ns, nmf_dict, svd_dict, target_genre_name,
        lambda r: get_single_value(r, "avg_bu"),
        r"$b_u$",
        "bᵤ",
        "bu_only_non_biasing"
    )

    # 4. mu only
    plot_merged_figures(
        genre_key, sorted_ns, nmf_dict, svd_dict, target_genre_name,
        lambda r: get_single_value(r, "global_mean_mu"),
        r"$\mu$",
        "μ",
        "mu_only"
    )

    # 5. BU bias comparison (target vs non-target, biasing vs non-biasing)
    if genre_key in nmf_bu_groups and genre_key in svd_bu_groups:
        plot_bu_bias_comparison_merged(
            genre_key, sorted_ns,
            nmf_bu_groups[genre_key],
            svd_bu_groups[genre_key]
        )

    # 6. K-value bar chart comparison
    if not plot_bar_chart_merged(genre_key):
        print(f"  ⚠ Bar chart data not found for {genre_key}")


print("\n" + "="*70)
print("✨ All merged comparisons completed!")
print("="*70)
print(f"\nOutput directory: {OUTPUT_BASE}")
print("\nOrganization by genre:")
for genre_key in sorted(nmf_groups.keys()):
    if genre_key in svd_groups:
        genre_output_dir = os.path.join(OUTPUT_BASE, genre_key)
        if os.path.exists(genre_output_dir):
            print(f"  ├── {genre_key}/")
