"""Round 3 statistics.
(a) HV computed from validation-selected fronts: for each family and split, keep the configurations that are
    non-dominated on VALIDATION (accuracy up, FDP down) and compute HV of their TEST points. This removes test-set
    selection of the front. Dataset-level Wilcoxon/Holm and Friedman as before; also restricted to families with a
    trade-off parameter (ExpGrad, AdvDebias, DNN+FDP, ZadehNet, ZadehNet-0), so the number of points is comparable.
(b) Table 7 (continuous penalties): per-dataset Nadeau-Bengio corrected t-test on HV instead of split-level Wilcoxon.
"""
import os, glob, json, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
from analyze_revision import hypervolume
from data import LOADERS
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); a[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return a
def valfront(g):
    pts = g[['va_acc', 'va_fdp']].values; keep = []
    for i, (a, f) in enumerate(pts):
        dom = ((pts[:, 0] >= a) & (pts[:, 1] <= f) & ((pts[:, 0] > a) | (pts[:, 1] < f))).any()
        if not dom: keep.append(i)
    return g.iloc[keep]
df = pd.read_csv(f'{R}/main_all.csv')
FAM = {'Reweighing': ['Reweighing'], 'ExpGrad': ['ExpGrad'], 'ThreshOpt': ['ThreshOpt'], 'AdvDebias': ['AdvDebias'],
       'DNN+FDP': ['DNN', 'DNN+FDP'], 'ZadehNet': ['ZadehNet-noFair', 'ZadehNet'], 'ZadehNet0': ['ZadehNet0-noFair', 'ZadehNet0']}
maj = {ds: max(LOADERS[ds]()[1].mean(), 1 - LOADERS[ds]()[1].mean()) for ds in df.dataset.unique()}
rows = []
for (ds, seed), g in df.groupby(['dataset', 'seed']):
    for fam, mem in FAM.items():
        gf = g[g.method.isin(mem)]
        if gf.empty: continue
        vf = valfront(gf)
        rows.append(dict(dataset=ds, seed=seed, family=fam, n_front=len(vf), hv_val=hypervolume(list(zip(vf.te_acc, vf.te_fdp)), maj[ds]),
                         hv_test=hypervolume(list(zip(gf.te_acc, gf.te_fdp)), maj[ds])))
H = pd.DataFrame(rows); H.to_csv(f'{T}/rev3_hv_valfront.csv', index=False)
M = H.groupby(['dataset', 'family']).hv_val.mean().unstack()
print('mean front size', H.groupby('family').n_front.mean().round(2).to_dict())
print('dataset-level HV (validation front)\n', M.round(3))
fams = list(FAM)
print('Friedman (all 7) p=%.4f' % stats.friedmanchisquare(*[M[f] for f in fams]).pvalue, (-M[fams]).rank(1).mean().round(2).to_dict())
tf = ['ExpGrad', 'AdvDebias', 'DNN+FDP', 'ZadehNet', 'ZadehNet0']
print('Friedman (trade-off families) p=%.4f' % stats.friedmanchisquare(*[M[f] for f in tf]).pvalue, (-M[tf]).rank(1).mean().round(2).to_dict())
out = []
for f in fams:
    if f == 'ZadehNet': continue
    d = (M['ZadehNet'] - M[f]); ci = stats.t.interval(0.90, 7, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(8))
    out.append(dict(vs=f, diff=d.mean(), wins=int((d > 0).sum()), p=stats.wilcoxon(d).pvalue if (d != 0).any() else 1, lo=ci[0], hi=ci[1]))
o = pd.DataFrame(out); o['p_holm'] = holm(o.p.values); print(o.round(4).to_string(index=False)); o.to_csv(f'{T}/rev3_hv_valfront_stats.csv', index=False)
# (b) Nadeau-Bengio per dataset for Table 7
sel = pd.read_csv(f'{T}/rev_cov_selected.csv'); nb = []
for ds in ['Bank', 'German']:
    a = sel[(sel.dataset == ds) & (sel.family == 'DNN+FDP (fuzzy young)')].set_index('seed').hv
    for fam in ['DNN+Cov(age)', 'DNN+HGR(age)', 'LR+Cov(s) [Zafar]', 'LR+Cov(age)']:
        b = sel[(sel.dataset == ds) & (sel.family == fam)].set_index('seed').hv.reindex(a.index); d = (b - a).dropna(); k = len(d)
        se = np.sqrt((1 / k + 0.2 / 0.6) * d.var(ddof=1)); t = d.mean() / se
        nb.append(dict(dataset=ds, family=fam, diff=d.mean(), t=t, p=2 * stats.t.sf(abs(t), k - 1)))
nb = pd.DataFrame(nb); nb['p_holm'] = holm(nb.p.values); print(nb.round(4).to_string(index=False)); nb.to_csv(f'{T}/rev3_cov_nb.csv', index=False)
