"""Regenerate Figure 3 and Supplementary Figures S7-S12 from the re-tuned models.

Usage: python make_supp_figures.py banana_quality.csv results/
"""
import sys, ast, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
                             confusion_matrix)
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from banana_revision_analysis import models_and_grids, FEATURES, SEED

csv, out = sys.argv[1], sys.argv[2]
BLUE, INK, MUTED, GRID = '#2a78d6', '#1f1f1e', '#6b6a63', '#e6e5e0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
ORDER = ['LDA', 'LR', 'NB', 'CART', 'DT', 'RF', 'NNET', 'KNN', 'SVM']

df = pd.read_csv(csv)
X = df[FEATURES].to_numpy(float); y = (df['Quality'].str.lower() == 'good').astype(int).to_numpy()
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

bp = pd.read_csv(f'{out}/best_params.csv', dtype=str).set_index('model')
def conv(v):
    try: return ast.literal_eval(v)
    except Exception: return v
M = models_and_grids(); tuned = {}
for name, (est, grid) in M.items():
    params = {f'clf__{k}': conv(v) for k, v in bp.loc[name].dropna().items()}
    if name == 'SVM': params['clf__probability'] = True
    tuned[name] = est.set_params(**params).fit(Xtr, ytr)
proba = {n: m.predict_proba(Xte)[:, 1] for n, m in tuned.items()}
pred = {n: m.predict(Xte) for n, m in tuned.items()}

def grid9(figsize):
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)
    return fig, axes.ravel()

# ---- Figure 3: ROC small multiples
fig, axes = grid9((9.0, 6.98))
for ax, n in zip(axes, ORDER):
    fpr, tpr, _ = roc_curve(yte, proba[n])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls='--')
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.set_title(f'{n}  (AUC = {roc_auc_score(yte, proba[n]):.4f})', loc='left', fontsize=9, color=INK, fontweight='bold')
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
for ax in axes[6:]: ax.set_xlabel('False positive rate')
for ax in axes[::3]: ax.set_ylabel('True positive rate')
fig.tight_layout(); fig.savefig(f'{out}/fig3_roc.png', dpi=250); plt.close(fig)

# ---- S10: precision-recall small multiples
fig, axes = grid9((9.0, 6.98))
base = yte.mean()
for ax, n in zip(axes, ORDER):
    p, r, _ = precision_recall_curve(yte, proba[n])
    ax.axhline(base, color=MUTED, lw=0.8, ls='--')
    ax.plot(r, p, color=BLUE, lw=2)
    ax.set_title(f'{n}  (AP = {average_precision_score(yte, proba[n]):.4f})', loc='left', fontsize=9, color=INK, fontweight='bold')
    ax.set_ylim(0.4, 1.01); ax.grid(True, color=GRID); ax.set_axisbelow(True)
for ax in axes[6:]: ax.set_xlabel('Recall')
for ax in axes[::3]: ax.set_ylabel('Precision')
fig.tight_layout(); fig.savefig(f'{out}/figS10_pr.png', dpi=250); plt.close(fig)

# ---- S9: confusion matrices
fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.0))
for ax, n in zip(axes.ravel(), ORDER):
    cm = confusion_matrix(yte, pred[n])  # rows: actual Bad, Good
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f'{v}\n({v / cm.sum():.1%})', ha='center', va='center', fontsize=9,
                    color='white' if v > cm.max() * 0.5 else INK)
    ax.set_xticks([0, 1], ['Bad', 'Good']); ax.set_yticks([0, 1], ['Bad', 'Good'])
    ax.set_title(n, loc='left', fontsize=10, color=INK, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(f'{out}/figS9_confusion.png', dpi=250); plt.close(fig)

# ---- S7: NNET learning curve (tuned settings, training set only)
nn = tuned['NNET'].named_steps['clf']
fig, ax1 = plt.subplots(figsize=(7.5, 4.1))
ax1.plot(nn.loss_curve_, color=BLUE, lw=2, label='Training loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training loss (cross-entropy)')
ax2 = ax1.twinx(); ax2.spines['right'].set_visible(True)
ax2.plot(nn.validation_scores_, color='#eb6834', lw=2, ls='--', label='Internal validation accuracy')
ax2.set_ylabel('Internal validation accuracy (10% of training set)')
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='center right', frameon=False)
ax1.grid(True, color=GRID); ax1.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{out}/figS7_nnet_learning.png', dpi=250); plt.close(fig)
nn_epochs = len(nn.loss_curve_)

# ---- S8: RF out-of-bag error vs number of trees (training set only)
sc = StandardScaler().fit(Xtr); Xs = sc.transform(Xtr)
ns = [10, 25, 50, 75, 100, 150, 200, 300, 400]
oob = []
for n_est in ns:
    rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, random_state=SEED, n_jobs=-1).fit(Xs, ytr)
    oob.append(1 - rf.oob_score_)
fig, ax = plt.subplots(figsize=(7.5, 3.7))
ax.plot(ns, oob, color=BLUE, lw=2, marker='o', ms=5)
ax.axvline(200, color=MUTED, lw=0.8, ls='--'); ax.text(205, max(oob) * 0.98, 'B = 200 (used)', color=MUTED, fontsize=8, va='top')
ax.set_xlabel('Number of trees (B)'); ax.set_ylabel('Out-of-bag error (training set)')
ax.grid(True, color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{out}/figS8_rf_oob.png', dpi=250); plt.close(fig)
pd.DataFrame({'n_trees': ns, 'oob_error': oob}).to_csv(f'{out}/rf_oob.csv', index=False)

# ---- S11: RF MDI vs permutation importance
rf = tuned['RF']
mdi = pd.Series(rf.named_steps['clf'].feature_importances_, index=FEATURES)
pi = permutation_importance(rf, Xte, yte, scoring='accuracy', n_repeats=30, random_state=SEED, n_jobs=-1)
fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.83))
s = mdi.sort_values(); a.barh(s.index, s.values, color=BLUE, height=0.6)
for yy, v in enumerate(s.values): a.text(v + 0.002, yy, f'{v:.3f}', va='center', fontsize=8, color=INK)
a.set_title('MDI (training set)', loc='left', fontsize=10, color=INK, fontweight='bold'); a.set_xlabel('Fraction of impurity reduction')
order = pd.Series(pi.importances_mean, index=FEATURES).sort_values().index
b.boxplot([pi.importances[FEATURES.index(f)] for f in order], vert=False, widths=0.55, patch_artist=True,
          boxprops=dict(facecolor='#cde2fb', edgecolor=BLUE), medianprops=dict(color=INK, lw=2),
          whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE), flierprops=dict(marker='o', ms=3, markerfacecolor=BLUE, markeredgecolor='white'))
b.set_yticks(range(1, 8), list(order)); b.axvline(0, color=MUTED, lw=0.8)
b.set_title('Permutation (test set, 30 repeats)', loc='left', fontsize=10, color=INK, fontweight='bold')
b.set_xlabel('Decrease in accuracy')
for ax in (a, b): ax.grid(True, axis='x', color=GRID); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{out}/figS11_rf_mdi_pi.png', dpi=250); plt.close(fig)

# ---- S12: 2D partial dependence
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.89))
for ax, (f1, f2) in zip(axes, [('Sweetness', 'Softness'), ('Sweetness', 'Ripeness')]):
    j1, j2 = FEATURES.index(f1), FEATURES.index(f2)
    r = partial_dependence(rf, Xtr, [(j1, j2)], grid_resolution=40, percentiles=(0.02, 0.98))
    g1, g2 = r['grid_values']; Z = r['average'][0]
    z1 = (g1 - Xtr[:, j1].mean()) / Xtr[:, j1].std(); z2 = (g2 - Xtr[:, j2].mean()) / Xtr[:, j2].std()
    cs = ax.contourf(z1, z2, Z.T, levels=12, cmap='Blues')
    ax.contour(z1, z2, Z.T, levels=[0.5], colors=INK, linestyles='--', linewidths=1)
    ax.set_xlabel(f'{f1} (standardized)'); ax.set_ylabel(f'{f2} (standardized)')
    ax.set_title(f'{f1} × {f2}', loc='left', fontsize=10, color=INK, fontweight='bold')
    fig.colorbar(cs, ax=ax, label='Mean predicted P(Good)')
fig.tight_layout(); fig.savefig(f'{out}/figS12_pdp2d.png', dpi=250); plt.close(fig)

with open(f'{out}/supp_notes.txt', 'w') as fh:
    fh.write(f'NNET epochs until early stopping: {nn_epochs}\n')
    fh.write('RF OOB error: ' + ', '.join(f'{n}:{e:.4f}' for n, e in zip(ns, oob)) + '\n')
    fh.write('AP: ' + ', '.join(f'{n}:{average_precision_score(yte, proba[n]):.4f}' for n in ORDER) + '\n')
    fh.write('AUC: ' + ', '.join(f'{n}:{roc_auc_score(yte, proba[n]):.4f}' for n in ORDER) + '\n')
print('done')
