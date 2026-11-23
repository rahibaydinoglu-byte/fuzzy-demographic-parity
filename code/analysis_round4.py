"""Analysis of review-round-4 experiments. Prints summaries and writes results/tables/r4_*.csv."""
import os, glob, json, itertools, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
DS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']

def topk(a, k=5): return set(np.argsort(-np.abs(a))[:k])
def pair_stats(vecs):
    sp, ov = [], []
    for a, b in itertools.combinations(vecs, 2):
        idx = a.index.union(b.index); a, b = a.reindex(idx, fill_value=0).values, b.reindex(idx, fill_value=0).values
        sp.append(stats.spearmanr(np.abs(a), np.abs(b)).correlation); ov.append(len(topk(a) & topk(b)) / 5)
    return np.nanmean(sp), np.mean(ov)

def stability():
    out = []
    for ds in DS:
        f = f'{R}/stabprof_{ds}.csv'
        if not os.path.exists(f): continue
        p = pd.read_csv(f); p['v'] = [pd.Series(json.loads(a), index=json.loads(c)) for a, c in zip(p.phi, p.cols)]
        d = len(p.v.iloc[0])
        for lam in (0, 10):
            q = p[p.lam == lam]
            s16 = pair_stats(list(q[q.L == 16].v))
            sL = [pair_stats([q[(q.seed == sd) & (q.L == L)].v.iloc[0] for L in (8, 16, 32)]) for sd in q.seed.unique()]
            r = dict(dataset=ds, d=d, lam=lam, seeds_spearman=s16[0], seeds_top5=s16[1], L_spearman=np.mean([x[0] for x in sL]),
                     L_top5=np.mean([x[1] for x in sL]), chance_top5=5 / d)
            t = f'{R}/stabtree_{ds}.csv'
            if os.path.exists(t):
                tt = pd.read_csv(t); tt['v'] = [pd.Series(json.loads(a), index=json.loads(c)) for a, c in zip(tt.phi, tt.cols)]
                th = 0.0 if lam == 0 else 0.3
                st = pair_stats(list(tt[tt.theta == th].v)); r.update(tree_seeds_spearman=st[0], tree_seeds_top5=st[1])
            # top features of the ZadehNet profile (mean over seeds, L = 16)
            mv = pd.concat(list(q[q.L == 16].v), axis=1).fillna(0).mean(1)
            r['top_features'] = ', '.join(mv.abs().sort_values(ascending=False).index[:5])
            out.append(r)
    o = pd.DataFrame(out); o.to_csv(f'{T}/r4_stability.csv', index=False); print(o.round(3).to_string()); return o

def compaction():
    c = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/stabcomp_*.csv')])
    o = c.groupby('method')[['fidelity', 'dgap', 'contrib_corr', 'top4_overlap']].mean(); print(o.round(3)); o.to_csv(f'{T}/r4_compaction.csv')
    print(c.groupby(['dataset', 'method']).fidelity.mean().unstack().round(3))
    return o


def interp():
    """Interpretable fair baselines vs ZadehNet: operating point (eps = 0.05), test-front HV, validation-front HV, size."""
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from analyze_revision import hypervolume
    from data import LOADERS
    from stats_round3 import valfront, holm
    main = pd.read_csv(f'{R}/main_all.csv')
    ib = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/ibase_*_*.csv')])
    maj = {ds: max(LOADERS[ds]()[1].mean(), 1 - LOADERS[ds]()[1].mean()) for ds in DS}
    def sel(g, eps=0.05):
        ok = g[g.va_fdp <= eps]; return ok.loc[ok.va_acc.idxmax()] if len(ok) else g.loc[g.va_fdp.idxmin()]
    fam = {'ZadehNet': (main, ['ZadehNet-noFair', 'ZadehNet']), 'DNN+FDP': (main, ['DNN', 'DNN+FDP']),
           'FairTree': (ib, ['FairTree']), 'FairCORELS': (ib, ['FairCORELS'])}
    rows = []
    for ds in DS:
        for f, (src, mem) in fam.items():
            for seed, g in src[(src.dataset == ds) & src.method.isin(mem)].groupby('seed'):
                g = g.reset_index(drop=True); s = sel(g); vf = valfront(g)
                rows.append(dict(dataset=ds, family=f, seed=seed, acc=s.te_acc, fdp=s.te_fdp, hv=hypervolume(list(zip(g.te_acc, g.te_fdp)), maj[ds]),
                                 hv_val=hypervolume(list(zip(vf.te_acc, vf.te_fdp)), maj[ds]), size=s.get('size', np.nan), n_cfg=len(g)))
    D = pd.DataFrame(rows); D.to_csv(f'{T}/r4_interp_perseed.csv', index=False)
    M = D.groupby(['dataset', 'family']).mean(numeric_only=True).reset_index(); M.to_csv(f'{T}/r4_interp.csv', index=False)
    print(M.pivot(index='dataset', columns='family', values='hv').round(3)); print(M.pivot(index='dataset', columns='family', values='hv_val').round(3))
    out = []
    for f in ['FairTree', 'FairCORELS']:
        for met in ['hv', 'hv_val', 'acc']:
            a = M[M.family == 'ZadehNet'].set_index('dataset')[met]; b = M[M.family == f].set_index('dataset')[met]; dd = (a - b.reindex(a.index)).dropna()
            ci = stats.t.interval(0.90, len(dd) - 1, loc=dd.mean(), scale=dd.std(ddof=1) / np.sqrt(len(dd)))
            out.append(dict(vs=f, metric=met, n=len(dd), diff=dd.mean(), lo=ci[0], hi=ci[1], wins=int((dd > 0).sum()), p=stats.wilcoxon(dd).pvalue))
    o = pd.DataFrame(out); o['p_holm'] = np.nan
    for met in o.metric.unique(): m_ = o.metric == met; o.loc[m_, 'p_holm'] = holm(o.loc[m_, 'p'].values)
    o.to_csv(f'{T}/r4_interp_stats.csv', index=False); print(o.round(4).to_string()); return o

def align():
    a = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/align_synth_*.csv')])
    a['ratio'] = a.dev / a.cert_gap
    o = a.groupby(['scenario', 'protocol', 'L']).agg(retained=('retained', 'mean'), retained_sd=('retained', 'std'), within=('within', lambda x: x.abs().mean()),
                                                      reg_ref=('reg_ref', 'mean'), delta_opt=('delta_opt', 'mean'), cert_gap=('cert_gap', 'mean'), dev=('dev', 'mean'),
                                                      dec_ret=('dec_rule', 'mean'), dec_unc=('dec_unc', 'mean')).reset_index()
    o['dec_ret'] = o.dec_ret / o.dec_unc
    o.to_csv(f'{T}/r4_align.csv', index=False); print(o.round(4).to_string())
    print('certificate holds in', int((a.dev <= a.cert_gap).sum()), 'of', len(a), '; max ratio', a.ratio.max())
    for ds in ['COMPAS', 'Adult']:
        r = pd.read_csv(f'{R}/align_real_{ds}.csv'); print(ds, r.groupby('lam')[['te_acc', 'te_fdp', 'score_gap', 'routing', 'within']].mean().round(3).to_dict('index'))

if __name__ == '__main__':
    import sys
    for a in sys.argv[1:]: globals()[a]()
