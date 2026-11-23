"""Round 6: Exponentiated Gradient with a LightGBM base learner (100 trees, learning rate 0.1, 31 leaves), same
constraint grid, splits and protocol as the LR-based ExpGrad of run.py. Usage: python egb.py DATASET SEED...
-> results/egb_{dataset}_{seed}.csv"""
import sys, os, json, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
from data import LOADERS, split_and_encode
from run import metrics
warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
name = sys.argv[1]; X, y, s, mu, z, groups, meta = LOADERS[name]()
for seed in [int(a) for a in sys.argv[2:]]:
    fn = f'{OUT}/egb_{name}_{seed}.csv'
    if os.path.exists(fn): continue
    sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
    (Xtr, ytr, str_, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
    rows = []
    for eps in [0.01, 0.02, 0.05, 0.1]:
        t = time.time()
        base = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, verbose=-1, random_state=seed, n_jobs=1)
        eg = ExponentiatedGradient(base, DemographicParity(difference_bound=eps), eps=0.01, max_iter=50)
        eg.fit(Xtr, ytr, sensitive_features=str_)
        r = dict(dataset=name, seed=seed, method='ExpGrad-GBM', cfg=json.dumps({'eps': eps}))
        r.update(metrics(eg._pmf_predict(Xva)[:, 1], yva, sva, muva, 'va_')); r.update(metrics(eg._pmf_predict(Xte)[:, 1], yte, ste, mute, 'te_'))
        r['train_time'] = time.time() - t; rows.append(r)
    pd.DataFrame(rows).to_csv(fn, index=False); print('egb', name, seed, round(time.time() - t, 1), flush=True)
