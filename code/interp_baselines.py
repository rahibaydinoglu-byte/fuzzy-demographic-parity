"""Interpretable fairness-aware baselines (review round 4, Major 4).
  FairTree : fair decision tree with the strong-demographic-parity splitting criterion of Barata et al. (2024),
             package fair-trees; theta in THETAS (0 = accuracy only), max_depth in DEPTHS.
  FairCORELS: fair rule lists of Aivodji et al. (2021), epsilon-constraint mode for statistical parity,
             maximum training unfairness in EPSS; binarised inputs (numeric features thresholded at training quintiles, both directions;
             one-hot indicators as they are).
Both use the crisp group s, like the other crisp baselines. Usage: python interp_baselines.py DATASET SEED [SEED ...]
"""
import sys, os, json, time, warnings, numpy as np, pandas as pd
from data import LOADERS, split_and_encode
from run import metrics
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
THETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEPTHS = [4, 6]
EPSS = [0.01, 0.02, 0.05, 0.1, 1.0]   # maximum unfairness; FairCORELS's own 'epsilon' is the minimum fairness 1 - EPS

def binarise(Xtr, Xs, max_cols=None):
    cols, thr = [], []
    for j in range(Xtr.shape[1]):
        u = np.unique(Xtr[:, j])
        if len(u) <= 2:
            cols.append(('eq', j, u.max()))
        else:
            qs = np.unique(np.quantile(Xtr[:, j], [0.2, 0.4, 0.6, 0.8]))
            for q in qs: cols += [('le', j, q), ('gt', j, q)]
    def f(X):
        B = np.zeros((len(X), len(cols)), dtype=np.uint8)
        for k, (op, j, q) in enumerate(cols):
            B[:, k] = (X[:, j] == q) if op == 'eq' else ((X[:, j] <= q) if op == 'le' else (X[:, j] > q))
        return B
    return [f(X) for X in Xs], cols

def main(name, seeds, which):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    max_train = 60000 if name == 'CensusKDD' else None
    for seed in seeds:
        fn = f'{OUT}/ibase_{which}_{name}_{seed}.csv'
        if os.path.exists(fn): continue
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=max_train)
        (Xtr, ytr, str_, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        rows = []
        def rec(method, cfg, pva, pte, extra):
            r = dict(dataset=name, seed=seed, method=method, cfg=json.dumps(cfg))
            r.update(metrics(pva, yva, sva, muva, 'va_')); r.update(metrics(pte, yte, ste, mute, 'te_')); r.update(extra); rows.append(r)
        if which == 'tree':
            from fair_trees import FairDecisionTreeClassifier
            for depth in DEPTHS:
                for th in THETAS:
                    t = time.time()
                    m = FairDecisionTreeClassifier(theta=th, max_depth=depth, Z_agg='max', random_state=seed)
                    m.fit(Xtr.astype(np.float64), ytr.astype(int), Z=str_.astype(int).reshape(-1, 1))
                    tt = time.time() - t
                    rec('FairTree', {'theta': th, 'depth': depth}, m.predict_proba(Xva.astype(np.float64))[:, 1], m.predict_proba(Xte.astype(np.float64))[:, 1],
                        dict(train_time=tt, size=int(m.get_n_leaves()), depth=int(m.get_depth())))
        else:
            from faircorels import FairCorelsClassifier
            (Btr, Bva, Bte), bcols = binarise(Xtr, [Xtr, Xva, Xte])
            names = [f'{op}{j}_{q:.3g}' for op, j, q in bcols]
            for eps in EPSS:
                t = time.time()
                m = FairCorelsClassifier(c=1e-3, n_iter=200000, max_card=1, min_support=0.01, fairness=1, mode=3, epsilon=1.0 - eps,
                                         min_vect=(str_ == 1).astype(int), maj_vect=(str_ == 0).astype(int),
                                         useUnfairnessLB=True, policy='bfs', bfs_mode=2, verbosity=[], random_state=seed)
                m.fit(Btr, ytr.astype(int), features=names, prediction_name='y')
                tt = time.time() - t
                pv = m.predict(Bva).astype(float); pt = m.predict(Bte).astype(float)
                rec('FairCORELS', {'eps': eps}, pv, pt, dict(train_time=tt, size=len(m.rl_.rules) if hasattr(m, 'rl_') else np.nan, n_binary=Btr.shape[1]))
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(which, name, seed, 'done', flush=True)

if __name__ == '__main__':
    main(sys.argv[2], [int(a) for a in sys.argv[3:]], sys.argv[1])
