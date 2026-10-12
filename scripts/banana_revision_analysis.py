"""
Banana quality benchmark: analysis for the revised manuscript.

Produces every number the revised manuscript marks as a placeholder, using a
protocol in which the held-out test set is used only once.

Usage
    python banana_revision_analysis.py banana_quality.csv  [--out results] [--fast]

Input: the Kaggle "Banana Quality" CSV (columns Size, Weight, Sweetness,
Softness, HarvestTime, Ripeness, Acidity, Quality with values Good/Bad).

Outputs (in --out):
    versions.txt                    software versions (Section 3.7)
    split_and_folds.csv             test/train membership and CV fold ids (Section 3.7)
    table3_descriptives.csv         descriptive statistics
    table4_classwise.csv            Mann-Whitney U, Holm p, rank-biserial r, Cohen's d
    correlations_pearson.csv / correlations_spearman.csv
    best_params.csv                 hyperparameters chosen by CV on the training set
    table5_test_performance.csv     accuracy, precision, recall, F1, ROC-AUC
    table6_confusion.csv            TP, TN, FP, FN, specificity, balanced accuracy, MCC
    table7_cv.csv                   5-fold CV (mean, SD, t-based 95% CI), 10x5 repeated CV, nested CV
    table8_pairwise.csv             McNemar (test) and corrected resampled t-test (10x5 CV), Holm-adjusted
    table9_importance.csv           permutation importance per model (test set) + RF MDI
    table10_lr_coefficients.csv     standardized LR coefficients with bootstrap 95% CI
    svm_kernels.csv                 RBF vs linear/poly/sigmoid
    dt_depth4.txt                   depth-4 entropy tree accuracy and text rules
    figS5_dt_depth_cv.png, figS6_knn_k_cv.png
"""
import argparse, os, sys, platform, itertools, json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.model_selection import (train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
                                     GridSearchCV, cross_val_score, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, matthews_corrcoef, balanced_accuracy_score)
from sklearn.inspection import permutation_importance

SEED = 42
FEATURES = ['Size', 'Weight', 'Sweetness', 'Softness', 'HarvestTime', 'Ripeness', 'Acidity']


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def models_and_grids(fast=False):
    pipe = lambda est: Pipeline([('scaler', StandardScaler()), ('clf', est)])
    m = {
        'LDA':  (pipe(LinearDiscriminantAnalysis()), {}),
        'LR':   (pipe(LogisticRegression(max_iter=1000)), {'clf__C': [0.01, 0.1, 1, 10, 100]}),
        'CART': (pipe(DecisionTreeClassifier(criterion='gini', random_state=SEED)), {}),
        'DT':   (pipe(DecisionTreeClassifier(criterion='entropy', random_state=SEED)),
                 {'clf__max_depth': [2, 4, 6, 8, 10, 12]}),
        'KNN':  (pipe(KNeighborsClassifier()),
                 {'clf__n_neighbors': list(range(1, 32, 2)), 'clf__weights': ['uniform', 'distance']}),
        'NB':   (pipe(GaussianNB()), {'clf__var_smoothing': [1e-9, 1e-8, 1e-7]}),
        'NNET': (pipe(MLPClassifier(activation='relu', solver='adam', early_stopping=True,
                                    n_iter_no_change=10, max_iter=1000, random_state=SEED)),
                 {'clf__hidden_layer_sizes': [(32,), (64, 32)], 'clf__alpha': [1e-4, 1e-3, 1e-2]}),
        'RF':   (pipe(RandomForestClassifier(n_estimators=200, criterion='gini', max_features='sqrt',
                                             random_state=SEED, n_jobs=-1)),
                 {'clf__min_samples_leaf': [1, 2, 5]}),
        'SVM':  (pipe(SVC(kernel='rbf', probability=False, random_state=SEED)),
                 {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 0.01, 0.1, 1]}),
    }
    if fast:  # for a quick smoke test only
        m['KNN'] = (m['KNN'][0], {'clf__n_neighbors': [5, 7, 9]})
        m['SVM'] = (m['SVM'][0], {'clf__C': [1, 10], 'clf__gamma': ['scale']})
        m['NNET'] = (m['NNET'][0], {'clf__hidden_layer_sizes': [(64, 32)], 'clf__alpha': [1e-4]})
    return m


def tune(est, grid, X, y, cv):
    if not grid:
        return est.fit(X, y), {}
    gs = GridSearchCV(est, grid, cv=cv, scoring='accuracy', n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def corrected_resampled_t(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test on per-fold differences."""
    d = np.asarray(diff, float); J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return t, p


def mcnemar_exact(y, a, b):
    ca, cb = (a == y), (b == y)
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5))
    return n01, n10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--out', default='results'); ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = lambda f: os.path.join(a.out, f)

    with open(out('versions.txt'), 'w') as fh:
        import matplotlib
        for k, v in [('python', platform.python_version()), ('numpy', np.__version__), ('pandas', pd.__version__),
                     ('scipy', __import__('scipy').__version__), ('scikit-learn', sklearn.__version__),
                     ('matplotlib', matplotlib.__version__)]:
            fh.write(f'{k} {v}\n')

    df = pd.read_csv(a.csv)
    X = df[FEATURES].to_numpy(float)
    y = (df['Quality'].astype(str).str.strip().str.lower() == 'good').astype(int).to_numpy()
    print(f'n = {len(y)}, Good = {y.sum()}, Bad = {len(y) - y.sum()}')

    # ---------------- descriptive and class-wise statistics (Tables 3, 4) ----------------
    desc = df[FEATURES].describe().T
    desc['skew'] = df[FEATURES].skew(); desc['excess_kurtosis'] = df[FEATURES].kurt()
    desc.to_csv(out('table3_descriptives.csv'))
    rows = []
    for f in FEATURES:
        g, b = df.loc[y == 1, f].to_numpy(), df.loc[y == 0, f].to_numpy()
        U, p = stats.mannwhitneyu(g, b, alternative='two-sided')
        r_rb = 2 * U / (len(g) * len(b)) - 1          # positive = higher in Good
        sp = np.sqrt(((len(g) - 1) * g.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(g) + len(b) - 2))
        d = (g.mean() - b.mean()) / sp
        rows.append(dict(attribute=f, median_good=np.median(g), median_bad=np.median(b),
                         mean_good=g.mean(), mean_bad=b.mean(), U=U, p_raw=p, r_rb=r_rb, cohen_d=d))
    t4 = pd.DataFrame(rows); t4['p_holm'] = holm(t4['p_raw']); t4.to_csv(out('table4_classwise.csv'), index=False)
    df[FEATURES].corr('pearson').to_csv(out('correlations_pearson.csv'))
    df[FEATURES].corr('spearman').to_csv(out('correlations_spearman.csv'))

    # ---------------- split, folds ----------------
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    folds = pd.DataFrame({'row': idx, 'set': 'train'}); folds.loc[te_idx, 'set'] = 'test'
    folds['cv5_fold'] = -1
    for k, (_, v) in enumerate(cv5.split(Xtr, ytr)):
        folds.loc[tr_idx[v], 'cv5_fold'] = k
    folds.to_csv(out('split_and_folds.csv'), index=False)

    # ---------------- tuning on training data only ----------------
    M = models_and_grids(a.fast)
    tuned, params = {}, []
    for name, (est, grid) in M.items():
        best, bp = tune(est, grid, Xtr, ytr, cv5)
        if name == 'SVM':  # Platt-scaled probabilities for ROC-AUC; predictions are unchanged
            best = best.set_params(clf__probability=True).fit(Xtr, ytr)
        tuned[name] = best; params.append(dict(model=name, **{k.replace('clf__', ''): str(v) for k, v in bp.items()}))
        print('tuned', name, bp)
    pd.DataFrame(params).to_csv(out('best_params.csv'), index=False)

    # ---------------- single use of test set (Tables 5, 6) ----------------
    preds, probas, perf, conf = {}, {}, [], []
    for name, m in tuned.items():
        pr = m.predict(Xte); pb = m.predict_proba(Xte)[:, 1]
        preds[name], probas[name] = pr, pb
        tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
        perf.append(dict(model=name, accuracy=accuracy_score(yte, pr), precision=precision_score(yte, pr),
                         recall=recall_score(yte, pr), f1=f1_score(yte, pr), roc_auc=roc_auc_score(yte, pb)))
        conf.append(dict(model=name, specificity=tn / (tn + fp), balanced_accuracy=balanced_accuracy_score(yte, pr),
                         mcc=matthews_corrcoef(yte, pr), TP=tp, TN=tn, FP=fp, FN=fn))
    pd.DataFrame(perf).round(4).to_csv(out('table5_test_performance.csv'), index=False)
    pd.DataFrame(conf).round(4).to_csv(out('table6_confusion.csv'), index=False)

    # ---------------- CV stability (Table 7) ----------------
    tcrit = stats.t.ppf(0.975, 4)
    cvrows, rep_scores = [], {}
    for name, m in tuned.items():
        m = clone(m).set_params(clf__probability=False) if name == 'SVM' else m
        s5 = cross_val_score(m, Xtr, ytr, cv=cv5, scoring='accuracy', n_jobs=-1)
        sr = cross_val_score(m, Xtr, ytr, cv=rcv, scoring='accuracy', n_jobs=-1)
        rep_scores[name] = sr
        est, grid = M[name]
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 2)
        nest_est = GridSearchCV(est, grid, cv=inner, scoring='accuracy', n_jobs=-1) if grid else est
        sn = cross_val_score(nest_est, X, y, cv=outer, scoring='accuracy')
        h = tcrit * s5.std(ddof=1) / np.sqrt(5)
        cvrows.append(dict(model=name, cv5_mean=s5.mean(), cv5_sd=s5.std(ddof=1), cv5_ci_low=s5.mean() - h,
                           cv5_ci_high=s5.mean() + h, rep10x5_mean=sr.mean(), rep10x5_sd=sr.std(ddof=1),
                           nested_mean=sn.mean(), nested_sd=sn.std(ddof=1)))
        print('cv', name, round(s5.mean(), 4), round(sr.mean(), 4), round(sn.mean(), 4))
    pd.DataFrame(cvrows).round(4).to_csv(out('table7_cv.csv'), index=False)

    # ---------------- pairwise tests (Table 8) ----------------
    pairs = [('SVM', 'KNN'), ('SVM', 'NNET'), ('SVM', 'RF'), ('KNN', 'NNET'), ('KNN', 'RF'),
             ('NNET', 'RF'), ('RF', 'DT'), ('LDA', 'LR')]
    n_tr_fold = len(ytr) * 4 / 5; n_te_fold = len(ytr) / 5
    pr_rows = []
    for a1, b1 in pairs:
        n01, n10, p_mc = mcnemar_exact(yte, preds[a1], preds[b1])
        diff = rep_scores[a1] - rep_scores[b1]
        t, p_t = corrected_resampled_t(diff, n_tr_fold, n_te_fold)
        pr_rows.append(dict(comparison=f'{a1} - {b1}',
                            test_acc_diff=accuracy_score(yte, preds[a1]) - accuracy_score(yte, preds[b1]),
                            only_first_correct=n01, only_second_correct=n10, mcnemar_p=p_mc,
                            cv_mean_diff=diff.mean(), corrected_t=t, corrected_t_p=p_t))
    t8 = pd.DataFrame(pr_rows)
    t8['mcnemar_p_holm'] = holm(t8['mcnemar_p']); t8['corrected_t_p_holm'] = holm(t8['corrected_t_p'])
    t8.round(4).to_csv(out('table8_pairwise.csv'), index=False)

    # ---------------- importance (Table 9) ----------------
    imp = pd.DataFrame(index=FEATURES)
    for name in ['SVM', 'KNN', 'NNET', 'RF', 'DT', 'LR', 'LDA', 'NB', 'CART']:
        r = permutation_importance(tuned[name], Xte, yte, scoring='accuracy', n_repeats=30,
                                   random_state=SEED, n_jobs=-1)
        imp[f'PI_{name}_mean'] = r.importances_mean; imp[f'PI_{name}_sd'] = r.importances_std
    imp['RF_MDI'] = tuned['RF'].named_steps['clf'].feature_importances_
    imp.round(4).to_csv(out('table9_importance.csv'))
    print('RF MDI sums to', imp['RF_MDI'].sum())

    # ---------------- LR coefficients with bootstrap CI (Table 10) ----------------
    C = tuned['LR'].named_steps['clf'].C
    rng = np.random.default_rng(SEED)
    base = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=C, max_iter=1000))]).fit(Xtr, ytr)
    coefs = []
    B = 200 if a.fast else 1000
    for _ in range(B):
        bi = rng.integers(0, len(ytr), len(ytr))
        m = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=C, max_iter=1000))]).fit(Xtr[bi], ytr[bi])
        coefs.append(m.named_steps['clf'].coef_[0])
    coefs = np.array(coefs)
    t10 = pd.DataFrame({'attribute': FEATURES, 'coefficient': base.named_steps['clf'].coef_[0],
                        'ci_low': np.percentile(coefs, 2.5, axis=0), 'ci_high': np.percentile(coefs, 97.5, axis=0)})
    t10.sort_values('coefficient', ascending=False).round(3).to_csv(out('table10_lr_coefficients.csv'), index=False)

    # ---------------- SVM kernels, depth-4 tree, CV tuning curves ----------------
    krow = []
    for k in ['rbf', 'linear', 'poly', 'sigmoid']:
        m = Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel=k, random_state=SEED))])
        cvs = cross_val_score(m, Xtr, ytr, cv=cv5, n_jobs=-1).mean()
        krow.append(dict(kernel=k, cv5_train=cvs, test=accuracy_score(yte, m.fit(Xtr, ytr).predict(Xte))))
    pd.DataFrame(krow).round(4).to_csv(out('svm_kernels.csv'), index=False)

    dt4 = Pipeline([('scaler', StandardScaler()),
                    ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=SEED))]).fit(Xtr, ytr)
    with open(out('dt_depth4.txt'), 'w') as fh:
        fh.write(f'test accuracy = {accuracy_score(yte, dt4.predict(Xte)):.4f}\n')
        fh.write('thresholds below are on the StandardScaler scale fitted to the training set\n\n')
        fh.write(export_text(dt4.named_steps['clf'], feature_names=FEATURES))

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    depths = list(range(1, 21)); dcv = []
    for dpt in depths:
        m = Pipeline([('scaler', StandardScaler()),
                      ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=dpt, random_state=SEED))])
        dcv.append(cross_val_score(m, Xtr, ytr, cv=cv5, n_jobs=-1))
    dcv = np.array(dcv)
    plt.figure(figsize=(6, 3.5)); plt.errorbar(depths, dcv.mean(1), yerr=dcv.std(1, ddof=1), marker='o', capsize=2)
    plt.xlabel('max_depth'); plt.ylabel('5-fold CV accuracy (training set)'); plt.title('Entropy tree: depth vs CV accuracy')
    plt.tight_layout(); plt.savefig(out('figS5_dt_depth_cv.png'), dpi=200); plt.close()
    ks = list(range(1, 32, 2)); kcv = []
    for k in ks:
        m = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors=k))])
        kcv.append(cross_val_score(m, Xtr, ytr, cv=cv5, n_jobs=-1))
    kcv = np.array(kcv)
    plt.figure(figsize=(6, 3.5)); plt.errorbar(ks, kcv.mean(1), yerr=kcv.std(1, ddof=1), marker='o', capsize=2)
    plt.xlabel('k'); plt.ylabel('5-fold CV accuracy (training set)'); plt.title('KNN: k vs CV accuracy')
    plt.tight_layout(); plt.savefig(out('figS6_knn_k_cv.png'), dpi=200); plt.close()
    print('done; outputs in', a.out)


if __name__ == '__main__':
    main()
