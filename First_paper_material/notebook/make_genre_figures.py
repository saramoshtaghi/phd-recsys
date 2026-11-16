from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path('/home/moshtasa/Research/phd-recsys')
NB = ROOT / 'First_paper_material/notebook'
DATA = NB / 'three_genre_panel_independent_components.csv'
SOURCE = pd.read_csv(DATA)
COLORS = {'svd': '#0072B2', 'nmf': '#D55E00'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.6, 'axes.titlesize': 8.3,
    'axes.titleweight': 'bold', 'axes.labelsize': 7.3, 'xtick.labelsize': 6.8,
    'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .55, 'text.color': '#182E3C', 'axes.labelcolor': '#263F4D',
    'xtick.color': '#334755', 'ytick.color': '#334755',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})

def main():
    if len(sys.argv) != 2 or sys.argv[1].lower() not in {'drama', 'adventure'}:
        raise SystemExit('Usage: python make_genre_figures.py drama|adventure')
    genre = {'drama': 'Drama', 'adventure': 'Adventure'}[sys.argv[1].lower()]
    directory = 'advanture' if genre == 'Adventure' else 'drama'
    out = ROOT / f'First_paper_material/figures/first_reviewer/{directory}/paper_figures'
    out.mkdir(parents=True, exist_ok=True)
    validation = []
    configs = [
        ('all_training', f'training_{genre.lower()}_all_users_5panel_paper.png', 'Training Fit: All Users'),
        ('synthetic_training', f'training_{genre.lower()}_synthetic_only_5panel_paper.png', 'Training Fit: Synthetic Users'),
        ('real_unseen_prediction', f'prediction_real_users_{genre.lower()}_5panel_paper.png', 'Recommendations for Real Users'),
    ]
    for view, filename, base_title in configs:
        ns = [2, 4, 6, 25, 50, 100, 200, 500] if view == 'synthetic_training' else [0, 2, 4, 6, 25, 50, 100, 200, 500]
        x = np.arange(len(ns))
        data = SOURCE[(SOURCE.genre == genre) & (SOURCE.view == view)]
        if data.empty:
            raise ValueError(f'No source rows for {genre=} {view=}')
        fig = plt.figure(figsize=(7.2, 3.75))
        fig.suptitle(f'{base_title} — {genre}', fontsize=9.4, fontweight='bold', color='#182E3C', y=.995)
        if view == 'real_unseen_prediction':
            grid = fig.add_gridspec(2, 2)
            axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                   fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
            panels = [('mean_prediction', '(A) Mean predicted rating', 'Mean predicted rating'),
                      ('mean_bi', r'(B) Book bias $b_i$', 'Mean component'),
                      ('mean_interaction', r'(C) Dot product $p_u^\top q_i$', 'Mean component'),
                      ('mean_bu', r'(D) User bias $b_u$', 'Mean component')]
        else:
            grid = fig.add_gridspec(2, 6)
            axs = [fig.add_subplot(grid[0, 0:2]), fig.add_subplot(grid[0, 2:4]),
                   fig.add_subplot(grid[0, 4:6]), fig.add_subplot(grid[1, 1:3]),
                   fig.add_subplot(grid[1, 3:5])]
            panels = [('mae', '(A) Training MAE', 'Mean absolute error'),
                      ('mean_bi', r'(B) Book bias $b_i$', 'Mean component'),
                      ('mean_interaction', r'(C) Dot product $p_u^\top q_i$', 'Mean component'),
                      ('mean_bu', r'(D) User bias $b_u$', 'Mean component'),
                      ('mean_prediction', r'(E) Prediction $\hat r_{ui}$', 'Mean predicted rating')]
        for pi, (ax, (key, panel_title, ylabel)) in enumerate(zip(axs, panels)):
            for model in ['svd', 'nmf']:
                for group, style, marker in [('target', '-', 'o'), ('non_target', '--', 's')]:
                    d = data[(data.model == model) & (data.group == group)].set_index('n_fake').loc[ns]
                    if not np.isfinite(d[key]).all():
                        raise ValueError(f'Non-finite values in {genre}/{view}/{model}/{group}/{key}')
                    if not np.allclose(d.mean_prediction, d.mu + d.mean_bu + d.mean_bi + d.mean_interaction, atol=1e-10):
                        raise ValueError(f'Score components do not sum for {genre}/{view}/{model}/{group}')
                    ax.plot(x, d[key].to_numpy(), color=COLORS[model], ls=style, marker=marker,
                            ms=2.3, lw=1.65 if key == 'mean_prediction' else 1.25, zorder=4)
            if key == 'mean_prediction' and view != 'real_unseen_prediction':
                for group, style in [('target', '-'), ('non_target', '--')]:
                    d = data[(data.model == 'svd') & (data.group == group)].set_index('n_fake').loc[ns]
                    other = data[(data.model == 'nmf') & (data.group == group)].set_index('n_fake').loc[ns]
                    if not np.allclose(d.mean_recorded_rating, other.mean_recorded_rating, atol=1e-12):
                        raise ValueError(f'Recorded ratings differ by model for {genre}/{group}')
                    ax.plot(x, d.mean_recorded_rating.to_numpy(), color='#777777', ls=style, lw=.9, zorder=2)
            if key in ['mean_bi', 'mean_interaction', 'mean_bu']:
                ax.axhline(0, color='#9DAAB3', lw=.5, zorder=1)
            if key == 'mae':
                ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.set_title(panel_title, loc='left', pad=4)
            ax.set_ylabel(ylabel, labelpad=3)
            ax.set_xlabel('Injected users', labelpad=3)
            ax.set_xticks(x, [str(n) for n in ns])
            ax.tick_params(length=2.5, width=.5, pad=2)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#E1E7EB', lw=.45, zorder=0)
            ax.margins(x=.04)
        handles = [Line2D([0], [0], color=COLORS['svd'], lw=1.6, label='SVD'),
                   Line2D([0], [0], color=COLORS['nmf'], lw=1.6, label='NMF'),
                   Line2D([0], [0], color='#536471', lw=1.1, marker='o', ms=2.5, label=genre),
                   Line2D([0], [0], color='#536471', lw=1.1, ls='--', marker='s', ms=2.5, label=f'Non-{genre}')]
        if view != 'real_unseen_prediction':
            handles.append(Line2D([0], [0], color='#777777', lw=.9, label='Recorded rating (E)'))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .94), ncol=len(handles),
                   frameon=False, fontsize=7.5, columnspacing=1.3, handlelength=2.0, handletextpad=.5)
        fig.subplots_adjust(left=.09, right=.98, bottom=.105, top=.82,
                            hspace=.95, wspace=.30 if view == 'real_unseen_prediction' else 1.05)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axs:
            box = ax.get_tightbbox(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1
        path = out / filename
        fig.savefig(path, dpi=600)
        plt.close(fig)
        validation.append({'file': str(path), 'genre': genre, 'view': view, 'n_fake': ns,
                           'pixels': [4320, 2250], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path)
    (out / 'figure_manifest.json').write_text(json.dumps(validation, indent=2))

if __name__ == '__main__':
    main()
