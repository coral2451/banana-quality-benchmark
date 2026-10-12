"""Regenerate main-text Figures 4, 5 and 6 from the revised pipeline.

Usage: python make_figures.py banana_quality.csv results/
Reads best_params.csv and table9_importance.csv written by banana_revision_analysis.py.
"""
import sys, ast, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.inspection import partial_dependence
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banana_revision_analysis import models_and_grids, FEATURES, SEED

csv, outdir = sys.argv[1], sys.argv[2]
BLUE, INK, MUTED, GRID = '#2a78d6', '#1f1f1e', '#6b6a63', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False})

df = pd.read_csv(csv)
X = df[FEATURES].to_numpy(float); y = (df['Quality'].str.lower() == 'good').astype(int).to_numpy()
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
Xtr, ytr = X[tr], y[tr]

bp = pd.read_csv(f'{outdir}/best_params.csv', dtype=str).set_index('model')
M = models_and_grids()
def conv(v):
    try: return ast.literal_eval(v)
    except Exception: return v
tuned = {}
for name, (est, grid) in M.items():
    params = {f'clf__{k}': conv(str(v)) for k, v in bp.loc[name].dropna().items()}
    tuned[name] = est.set_params(**params)

# ---- Figure 4: repeated CV accuracy distributions
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
scores = {n: cross_val_score(m, Xtr, ytr, cv=rcv, n_jobs=-1) for n, m in tuned.items()}
pd.DataFrame(scores).to_csv(f'{outdir}/repeated_cv_scores.csv', index=False)
order = sorted(scores, key=lambda n: np.median(scores[n]))
fig, ax = plt.subplots(figsize=(8.0, 3.96))
ax.boxplot([scores[n] for n in order], vert=False, widths=0.55, patch_artist=True,
           boxprops=dict(facecolor='#cde2fb', edgecolor=BLUE, linewidth=1),
           medianprops=dict(color=INK, linewidth=2), whiskerprops=dict(color=BLUE),
           capprops=dict(color=BLUE), flierprops=dict(marker='o', markersize=4, markerfacecolor=BLUE, markeredgecolor='white'))
ax.set_yticks(range(1, len(order) + 1), order)
ax.set_xlabel('Accuracy (10 × 5-fold repeated cross-validation, training set)')
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{outdir}/fig4_cv_boxplots.png', dpi=250); plt.close(fig)

# ---- Figure 5: permutation importance, small multiples
imp = pd.read_csv(f'{outdir}/table9_importance.csv', index_col=0)
models = ['SVM', 'KNN', 'NNET', 'RF', 'DT', 'LR']
fig, axes = plt.subplots(2, 3, figsize=(9.0, 3.56 * 1.0 + 0.0), sharex=True)
fig.set_size_inches(10.0, 3.96)
for ax, m in zip(axes.ravel(), models):
    s = imp[f'PI_{m}_mean'].sort_values(); e = imp[f'PI_{m}_sd'].loc[s.index]
    ax.barh(s.index, s.values, xerr=e.values, color=BLUE, height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_title(m, fontsize=10, color=INK, loc='left', fontweight='bold')
    ax.tick_params(axis='y', labelsize=8); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=0.8)
for ax in axes[1]: ax.set_xlabel('Mean decrease in accuracy')
fig.tight_layout(); fig.savefig(f'{outdir}/fig5_permutation_importance.png', dpi=250); plt.close(fig)

# ---- Figure 6: RF partial dependence
rf = tuned['RF'].fit(Xtr, ytr)
feats = ['Weight', 'Softness', 'Sweetness', 'Ripeness', 'HarvestTime', 'Acidity']
fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.96), sharey=True)
pdp_rows = []
for ax, f in zip(axes.ravel(), feats):
    j = FEATURES.index(f)
    r = partial_dependence(rf, Xtr, [j], grid_resolution=60, percentiles=(0.02, 0.98), kind='average')
    gx = r['grid_values'][0]; gy = r['average'][0]
    gx_std = (gx - Xtr[:, j].mean()) / Xtr[:, j].std()
    ax.plot(gx_std, gy, color=BLUE, lw=2)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls='--')
    ax.set_title(f, fontsize=10, color=INK, loc='left', fontweight='bold')
    ax.set_xlabel(f'{f} (standardized)'); ax.set_ylim(0, 1); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for a, b in zip(gx_std, gy): pdp_rows.append(dict(feature=f, x_std=a, p_good=b))
for ax in axes[:, 0]: ax.set_ylabel('Mean predicted P(Good)')
fig.tight_layout(); fig.savefig(f'{outdir}/fig6_pdp.png', dpi=250); plt.close(fig)
pd.DataFrame(pdp_rows).to_csv(f'{outdir}/pdp_values.csv', index=False)
print('figures written')
