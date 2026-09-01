# 0413 Experiment: NMF & SVD with Biasing User Distinction

## Overview
This experiment creates two recommendation models (NMF and SVD) that distinguish between **biasing users** (injected users) and **regular users** in the analysis.

## Key Features

### 1. Biasing User Identification
- Extracts injected users from the dataset (those not in the original baseline)
- Marks them separately in analysis output
- Allows tracking how biasing users influence model behavior

### 2. Enhanced Analysis Functions

#### `identify_biasing_users(original_df, injection_df)`
Returns two sets:
- **biasing_users**: User IDs added in injection (user_id >= max_original_user_id)
- **original_users**: User IDs in baseline dataset

#### `analyze_bu()` - User Bias Analysis with Distinction
Generates per-genre statistics for:
- **avg_bu, std_bu, n_users** for **biasing users** separately
- **avg_bu, std_bu, n_users** for **non_biasing users** separately
- Output: `{base_name}_bu_per_genre_biased.csv`

Structure:
```
genre | user_type   | avg_bu | std_bu | n_users | is_target
-------|-------------|--------|--------|---------|----------
Adult | biasing     | -0.14  | 0.32   | 2       | target
Adult | non_biasing | 0.02   | 0.18   | 1234    | non_target
```

### 3. Output Directory Structure

```
/0413/
├── NMF-0413/
│   ├── notebook/
│   │   └── NMF_0413.py          (Main NMF script)
│   └── result/
│       ├── data/                (Recommendations)
│       ├── bi/                  (Item biases per genre)
│       ├── bu/                  (User biases - BIASING vs NON-BIASING)
│       ├── mio/                 (Mean ratings per genre)
│       └── merged/              (Combined metrics)
│
└── SVD-0413/
    ├── notebook/
    │   └── SVD_0413.py          (Main SVD script)
    └── result/
        ├── data/
        ├── bi/
        ├── bu/                  (User biases - BIASING vs NON-BIASING)
        ├── mio/
        └── merged/
```

## Key Outputs

### `bu/` Folder
Contains `{base_name}_bu_per_genre_biased.csv` with:
- Separate average b_u for biasing vs non-biasing users per genre
- Tracks how injected users differ in their bias profiles
- Supports trend analysis across different injection sizes (2u, 4u, 6u, ... 2000u)

### `merged/` Folder
Combines bi, bu, and mu metrics for comprehensive genre analysis

## Running the Scripts

```bash
# NMF
python /path/to/NMF_0413.py

# SVD
python /path/to/SVD_0413.py
```

Both scripts will:
1. Train on original baseline data
2. Process all injection files matching: `f_*.csv` with pos=5
3. Generate analysis for files with user counts: {2, 4, 6, 25, 50, 100, 200, 300, 500, 1000, 2000}

## Expected Analysis

For your research question:
> "Get the average value of b_u for the biasing users only. For different numbers of biasing users."

The `bu/` output files provide:
- `avg_bu` values for `user_type='biasing'` across different injection sizes
- Allows you to plot trends: avg_bu vs num_biasing_users
- Compare with existing avg_bi trends already observed

## Technical Details

- **Rating Scale**: 0-5 (clipped)
- **NMF Params**: 8 factors, 180 epochs, reg_qi=0.002
- **SVD Params**: 8 factors, 20 epochs, lr_all=0.002, reg_all=0.002
- **Genres**: Thriller, Romance, Adult (filtered from injection data)
- **Injection Filter**: pos=5 (high ratings for target class)

