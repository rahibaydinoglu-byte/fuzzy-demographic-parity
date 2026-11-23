"""Revision analyses: (A) synthetic displacement study, (B) continuous-attribute penalties vs FDP on German/Bank.
Writes results/tables/rev_*.csv and prints compact summaries."""
import glob, json, os, sys, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'; os.makedirs(T, exist_ok=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

def select(g, eps):
    ok = g[g.va_fdp <= eps]
    return ok.loc[ok.va_acc.idxmax()] if len(ok) else g.loc[g.va_fdp.idxmin()]

def hypervolume(points, acc_ref, fdp_ref=0.2):
    P = sorted([(a, f) for a, f in points if a > acc_ref and f < fdp_ref], key=lambda t: t[1])
    hv, best = 0.0, acc_ref
    for i, (a, f) in enumerate(P):
        best = max(best, a); nxt = P[i + 1][1] if i + 1 < len(P) else fdp_ref
        hv += (best - acc_ref) * (nxt - f)
    return hv / ((1 - acc_ref) * fdp_ref)

def synth():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/synth_*.csv')], ignore_index=True)
    d['dev'] = (d.delta - d.delta_bayes).abs()
    d['bound_ok'] = d.dev <= d.bound_kl + 1e-12
    g = d.groupby(['rho', 'mode', 'L', 'fair', 'lam'])
    s = g.agg(n=('seed', 'size'), delta=('delta', 'mean'), delta_bayes=('delta_bayes', 'mean'), routing=('routing', 'mean'),
              within=('within', 'mean'), within_abs=('within_abs', 'mean'), kl=('kl', 'mean'), dev=('dev', 'mean'),
              bound=('bound_kl', 'mean'), acc=('acc', 'mean'), acc_bayes=('acc_bayes', 'mean'),
              dp_dec=('dp_dec', 'mean'), dp_dec_bayes=('dp_dec_bayes', 'mean'), tv_route=('tv_route', 'mean')).reset_index()
    s.to_csv(f'{T}/rev_synth_summary.csv', index=False)
    print('runs', len(d), 'seeds', sorted(d.seed.unique()), '| Prop.5 bound holds in', int(d.bound_ok.sum()), 'of', len(d))
    # displacement ratio: gap retained under the rule-local penalty relative to the unconstrained gap
    base = d[(d.fair == 'rule') & (d.lam == 0)].set_index(['seed', 'rho', 'mode', 'L'])
    for fair, lam in [('rule', 100), ('total', 100)]:
        x = d[(d.fair == fair) & (d.lam == lam)].set_index(['seed', 'rho', 'mode', 'L'])
        j = x.join(base[['delta', 'kl']], rsuffix='_0').dropna()
        j['retained'] = j.delta / j.delta_0
        t = j[j.index.get_level_values('rho') > 0].groupby(['rho', 'mode', 'L']).retained.agg(['mean', 'std', 'size'])
        print(f'\nretained gap  Delta(lam={lam},{fair}) / Delta(lam=0)'); print(t.round(3).unstack('rho'))
        t.to_csv(f'{T}/rev_synth_retained_{fair}.csv')
    zo = d[d['mode'] == 'zero'].groupby(['rho', 'L']).kl.mean().unstack('rho')
    print('\nexcess risk (KL) of zero-order rule bases, i.e. an upper bound proxy for eps_Omega'); print(zo.round(4))
    return d

def cov():
    files = glob.glob(f'{R}/cov_*_*.csv')
    if not files: print('no cov results yet'); return
    c = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = pd.read_csv(f'{R}/main_all.csv'); m = m[m.dataset.isin(c.dataset.unique()) & m.seed.isin(c.seed.unique())]
    df = pd.concat([m, c], ignore_index=True)
    FAM = {'DNN+FDP (fuzzy young)': ['DNN', 'DNN+FDP'], 'DNN+Cov(age)': ['DNN', 'DNN+Cov(age)'],
           'DNN+HGR(age)': ['DNN', 'DNN+HGR(age)'], 'LR+Cov(s) [Zafar]': ['LR-torch', 'LR+Cov(s)'],
           'LR+Cov(age)': ['LR-torch', 'LR+Cov(age)'], 'ZadehNet': ['ZadehNet-noFair', 'ZadehNet'],
           'ExpGrad': ['ExpGrad'], 'Reweighing': ['Reweighing'], 'AdvDebias': ['AdvDebias']}
    from data import LOADERS
    maj = {ds: max(LOADERS[ds]()[1].mean(), 1 - LOADERS[ds]()[1].mean()) for ds in df.dataset.unique()}
    rows = []
    for (ds, seed), g in df.groupby(['dataset', 'seed']):
        if seed not in c[c.dataset == ds].seed.unique(): continue
        for fam, mem in FAM.items():
            gf = g[g.method.isin(mem)]
            if gf.empty: continue
            r = dict(dataset=ds, seed=seed, family=fam, hv=hypervolume(list(zip(gf.te_acc, gf.te_fdp)), maj[ds]))
            for eps in (0.05, 0.02):
                s = select(gf, eps); r[f'acc@{eps}'] = s.te_acc; r[f'fdp@{eps}'] = s.te_fdp; r[f'dpcrisp@{eps}'] = s.te_dp_crisp
            rows.append(r)
    sel = pd.DataFrame(rows); sel.to_csv(f'{T}/rev_cov_selected.csv', index=False)
    summ = sel.groupby(['dataset', 'family']).agg(n=('seed', 'size'), hv=('hv', 'mean'), hv_sd=('hv', 'std'),
                                                  acc05=('acc@0.05', 'mean'), fdp05=('fdp@0.05', 'mean'),
                                                  acc02=('acc@0.02', 'mean'), fdp02=('fdp@0.02', 'mean'))
    print(summ.round(4)); summ.to_csv(f'{T}/rev_cov_summary.csv')
    # paired Wilcoxon on HV: DNN+FDP vs each continuous-attribute penalty (per dataset, over seeds)
    for ds in sel.dataset.unique():
        a = sel[(sel.dataset == ds) & (sel.family == 'DNN+FDP (fuzzy young)')].set_index('seed').hv
        for fam in ['DNN+Cov(age)', 'DNN+HGR(age)', 'LR+Cov(s) [Zafar]', 'LR+Cov(age)']:
            b = sel[(sel.dataset == ds) & (sel.family == fam)].set_index('seed').hv.reindex(a.index)
            diff = (a - b).dropna()
            p = stats.wilcoxon(diff).pvalue if len(diff) >= 5 and (diff != 0).any() else np.nan
            print(f'{ds:7s} HV FDP - {fam:18s}: mean diff {diff.mean():+.4f}  wins {int((diff > 1e-12).sum())}, ties {int((diff.abs() <= 1e-12).sum())}, n={len(diff)}  Wilcoxon p={p:.3g}')

if __name__ == '__main__':
    what = sys.argv[1:] or ['synth', 'cov']
    if 'synth' in what: synth()
    if 'cov' in what: cov()
