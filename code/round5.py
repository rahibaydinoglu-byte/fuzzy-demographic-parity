"""Review round 5 (third external review): cheap re-analyses of existing results, plus one small experiment.
  python round5.py stats   : (i) per-dataset correlation of |score-level gap| and decision-level FDP over TSK models;
                             (ii) HV reference-point sensitivity (FDP_ref in {0.1, 0.2, 0.3}) and dataset-level tests
                                  with and without Diabetes; (iii) Nadeau-Bengio corrected tests for the Adult proxy
                                  experiments (replacing split-level Wilcoxon); (iv) equalised-odds side effects.
  python round5.py ageaxis : Bank and German, 10 splits. DNN and ZadehNet without penalty and with the FDP penalty
                             for 'young' (lambda = 10). Decision rate per age band minus overall rate, to show where
                             the disparity goes when FDP removes it for the young.
Writes results/tables/r5_*.csv.
"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
DS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); a[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return a

def nb(d, ratio=1 / 3):
    """Nadeau-Bengio corrected resampled t-test for paired differences d over k random splits (n_test/n_train = ratio)."""
    d = np.asarray(d, float); k = len(d); se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); t = d.mean() / se
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))

from analyze_revision import hypervolume as hv

def stats_main():
    from data import LOADERS
    main = pd.read_csv(f'{R}/main_all.csv')
    # (i) score-decision correlation per dataset
    tsk = main[main.method.isin(['ZadehNet-noFair', 'ZadehNet', 'ZadehNet-rule', 'ZadehNet0-noFair', 'ZadehNet0'])].copy()
    tsk['sgap'] = (tsk.routing + tsk.within).abs()
    rows = [dict(dataset=ds, n=len(g), r=float(np.corrcoef(g.sgap, g.te_fdp)[0, 1])) for ds, g in tsk.groupby('dataset')]
    rows.append(dict(dataset='all', n=len(tsk), r=float(np.corrcoef(tsk.sgap, tsk.te_fdp)[0, 1])))
    sd = pd.DataFrame(rows); sd.to_csv(f'{T}/r5_score_decision.csv', index=False); print(sd.round(3).to_string(index=False))
    # (ii) HV reference sensitivity
    FAM = {'Reweighing': ['Reweighing'], 'ExpGrad': ['ExpGrad'], 'ThreshOpt': ['ThreshOpt'], 'AdvDebias': ['AdvDebias'],
           'DNN+FDP': ['DNN', 'DNN+FDP'], 'ZadehNet': ['ZadehNet-noFair', 'ZadehNet'], 'ZadehNet0': ['ZadehNet0-noFair', 'ZadehNet0']}
    maj = {ds: max(LOADERS[ds]()[1].mean(), 1 - LOADERS[ds]()[1].mean()) for ds in DS}
    out = []
    for fref in [0.1, 0.2, 0.3]:
        H = []
        for (ds, seed), g in main.groupby(['dataset', 'seed']):
            for fam, mem in FAM.items():
                gf = g[g.method.isin(mem)]
                H.append(dict(dataset=ds, seed=seed, family=fam, hv=hv(list(zip(gf.te_acc, gf.te_fdp)), maj[ds], fref)))
        M = pd.DataFrame(H).groupby(['dataset', 'family']).hv.mean().unstack()
        for excl in [False, True]:
            Mx = M.drop(index='Diabetes') if excl else M
            fr = stats.friedmanchisquare(*[Mx[f] for f in FAM]).pvalue
            ranks = (-Mx[list(FAM)]).rank(1).mean()
            ps, diffs = [], []
            for f in FAM:
                if f == 'ZadehNet': continue
                d = Mx['ZadehNet'] - Mx[f]; diffs.append((f, d.mean(), int((d > 0).sum()))); ps.append(stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0)
            ph = holm(ps)
            for (f, dm, w), p, q in zip(diffs, ps, ph):
                out.append(dict(fdp_ref=fref, without_diabetes=excl, N=len(Mx), vs=f, diff=dm, wins=w, p=p, p_holm=q, friedman_p=fr,
                                rank_zadehnet=ranks['ZadehNet'], rank_dnnfdp=ranks['DNN+FDP']))
    o = pd.DataFrame(out); o.to_csv(f'{T}/r5_hv_reference.csv', index=False)
    print(o.pivot_table(index=['fdp_ref', 'without_diabetes'], columns='vs', values='p_holm').round(3).to_string())
    print(o.groupby(['fdp_ref', 'without_diabetes'])[['friedman_p', 'rank_zadehnet', 'rank_dnnfdp']].first().round(3).to_string())
    # (iii) NB tests for the Adult proxy experiment (n_test / n_train = 0.2 / 0.6)
    n1 = pd.read_csv(glob.glob(f'{R}/noproxy_0-*.csv')[0]); n3 = pd.read_csv(glob.glob(f'{R}/noproxy3_*.csv')[0]); res = []
    for v in n1.variant.unique():
        x0 = n1[(n1.variant == v) & (n1.fair == 'total') & (n1.lam == 0)].set_index('seed').te_fdp
        x1 = n1[(n1.variant == v) & (n1.fair == 'rule') & (n1.lam == 100)].set_index('seed').te_fdp.reindex(x0.index)
        t, p = nb(x0 - x1); res.append(dict(test=f'{v}: gap reduction by rule-local penalty', mean=float((x0 - x1).mean()), t=t, p_nb=p, p_wilcoxon=float(stats.wilcoxon(x0 - x1).pvalue)))
    for (v, fair), g in n3.groupby(['variant', 'fair']):
        for a, b, name in [('top_rule_sex_auc', 'other_rule_sex_auc', 'firing AUC top-4 vs other'), ('top_feat_sex_auc', 'all_feat_sex_auc', 'feature AUC top-4 rules vs all')]:
            d = g[a] - g[b]; t, p = nb(d); res.append(dict(test=f'{v} / {fair}: {name}', mean=float(d.mean()), t=t, p_nb=p, p_wilcoxon=float(stats.wilcoxon(d).pvalue)))
    rr = pd.DataFrame(res); rr.to_csv(f'{T}/r5_proxy_nb.csv', index=False); print(rr.round(4).to_string(index=False))
    # (iv) equalised-odds side effects at the operating point (eps = 0.05)
    sel = pd.read_csv(f'{T}/selected_per_seed.csv')
    eo = sel.groupby(['dataset', 'family'])['te_eo@0.05'].mean().unstack()[['LR', 'LightGBM', 'DNN+FDP', 'ZadehNet', 'ZadehNet0', 'ExpGrad', 'ThreshOpt', 'AdvDebias', 'Reweighing']]
    eo.to_csv(f'{T}/r5_eo.csv'); print(eo.round(3).to_string())

def ageaxis(seeds):
    import torch
    from data import LOADERS, split_and_encode
    from models import ZadehNet, MLP, train_torch, predict
    torch.set_num_threads(1); rows = []
    bands = [(0, 25), (25, 30), (30, 35), (35, 45), (45, 60), (60, 200)]
    for name in ['Bank', 'German']:
        X, y, s, mu, z, groups, meta = LOADERS[name]()
        for seed in seeds:
            sp, cols = split_and_encode(X, y, s, mu, groups, seed)
            (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, ite) = sp['tr'], sp['va'], sp['te']
            age = np.asarray(z, float)[ite]; d = Xtr.shape[1]
            batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
            init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
            for model in ['DNN', 'ZadehNet']:
                for lam in [0, 10]:
                    if model == 'DNN':
                        m = MLP(d, seed=seed); train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lr=1e-3, batch=batch, seed=seed)
                    else:
                        m = ZadehNet(d, L=16, Xinit=init, seed=seed); train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=batch, seed=seed)
                    yh = (predict(m, Xte) >= .5).astype(float); overall = yh.mean()
                    r = dict(dataset=name, seed=seed, model=model, lam=lam, acc=float((yh == yte).mean()), rate=float(overall),
                             corr_age=float(np.corrcoef(yh, age)[0, 1]) if yh.std() > 0 else 0.0)
                    for lo, hi in bands:
                        msk = (age > lo) & (age <= hi); r[f'band_{lo}_{hi}'] = float(yh[msk].mean() - overall) if msk.sum() >= 5 else np.nan; r[f'n_{lo}_{hi}'] = int(msk.sum())
                    rows.append(r)
            print('ageaxis', name, seed, flush=True)
    D = pd.DataFrame(rows); D.to_csv(f'{R}/r5_ageaxis.csv', index=False)
    print(D.groupby(['dataset', 'model', 'lam']).mean(numeric_only=True).round(3).drop(columns='seed').to_string())

if __name__ == '__main__':
    if sys.argv[1] == 'stats': stats_main()
    else: ageaxis([int(a) for a in sys.argv[2:]] or list(range(10)))
