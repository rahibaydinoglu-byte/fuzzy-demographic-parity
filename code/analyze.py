"""Model selection, tables and statistics for the main comparison. Writes results/tables/*.csv and summary.json."""
import glob, json, os, numpy as np, pandas as pd
from scipy import stats
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'; os.makedirs(T, exist_ok=True)
DATASETS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']
df = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/main_*.csv')], ignore_index=True)
df['lam'] = df.cfg.apply(lambda c: json.loads(c).get('lam', np.nan))
# families: a fairness-aware family includes its own unconstrained (lambda=0) member
FAM = {'LR': ['LR'], 'LightGBM': ['LightGBM'], 'Reweighing': ['Reweighing'], 'ExpGrad': ['ExpGrad'], 'ThreshOpt': ['ThreshOpt'],
       'AdvDebias': ['AdvDebias'], 'DNN+FDP': ['DNN', 'DNN+FDP'], 'ZadehNet': ['ZadehNet-noFair', 'ZadehNet'],
       'ZadehNet0': ['ZadehNet0-noFair', 'ZadehNet0']}
ORDER = list(FAM)

def select(g, eps):
    ok = g[g.va_fdp <= eps]
    return ok.loc[ok.va_acc.idxmax()] if len(ok) else g.loc[g.va_fdp.idxmin()]

def hypervolume(points, acc_ref, fdp_ref=0.2):
    """2-D HV for maximising acc and minimising fdp, normalised to [0,1]."""
    P = sorted([(a, f) for a, f in points if a > acc_ref and f < fdp_ref], key=lambda t: t[1])
    hv, best = 0.0, acc_ref
    for i, (a, f) in enumerate(P):
        best = max(best, a); nxt = P[i + 1][1] if i + 1 < len(P) else fdp_ref
        hv += (best - acc_ref) * (nxt - f)
    return hv / ((1 - acc_ref) * fdp_ref)

rows = []
for (ds, seed), g in df.groupby(['dataset', 'seed']):
    for fam, members in FAM.items():
        gf = g[g.method.isin(members)]
        if gf.empty: continue
        r = dict(dataset=ds, seed=seed, family=fam)
        for eps in (0.05, 0.02):
            s = select(gf, eps)
            for k in ['te_acc', 'te_bacc', 'te_auc', 'te_f1', 'te_fdp', 'te_eo', 'te_dp_crisp']:
                r[f'{k}@{eps}'] = s[k]
            r[f'cfg@{eps}'] = s.cfg; r[f'feasible_val@{eps}'] = bool(s.va_fdp <= eps)
            for k in ['antecedents', 'active_rules', 'fidelity', 'consequent_nnz', 'routing', 'within', 'train_time', 'epoch_time']:
                if k in s and pd.notna(s.get(k)): r[f'{k}@{eps}'] = s[k]
        rows.append(r)
sel = pd.DataFrame(rows)
# majority-class accuracy per dataset/seed for HV reference
from data import LOADERS
hv_rows = []
majority = {}
for ds in df.dataset.unique():
    y = LOADERS[ds]()[1]; majority[ds] = max(y.mean(), 1 - y.mean())
for (ds, seed, fam), g in [((ds, seed, fam), df[(df.dataset == ds) & (df.seed == seed) & df.method.isin(m)])
                            for ds in df.dataset.unique() for seed in df[df.dataset == ds].seed.unique() for fam, m in FAM.items()]:
    if g.empty: continue
    hv_rows.append(dict(dataset=ds, seed=seed, family=fam, hv=hypervolume(list(zip(g.te_acc, g.te_fdp)), majority[ds])))
sel = sel.merge(pd.DataFrame(hv_rows), on=['dataset', 'seed', 'family'])
sel.to_csv(f'{T}/selected_per_seed.csv', index=False)

def mstd(x): return f'{x.mean():.3f} ± {x.std(ddof=1):.3f}' if len(x) > 1 else f'{x.mean():.3f}'
summary = {}
for eps in (0.05, 0.02):
    tab = []
    for ds in [d for d in DATASETS if d in sel.dataset.unique()]:
        for fam in ORDER:
            g = sel[(sel.dataset == ds) & (sel.family == fam)]
            if g.empty: continue
            tab.append(dict(dataset=ds, family=fam, n_seeds=len(g), acc=mstd(g[f'te_acc@{eps}']), auc=mstd(g[f'te_auc@{eps}']),
                            f1=mstd(g[f'te_f1@{eps}']), fdp=mstd(g[f'te_fdp@{eps}']), eo=mstd(g[f'te_eo@{eps}']), hv=mstd(g.hv),
                            acc_mean=g[f'te_acc@{eps}'].mean(), fdp_mean=g[f'te_fdp@{eps}'].mean(), hv_mean=g.hv.mean(),
                            test_feasible=float((g[f'te_fdp@{eps}'] <= eps).mean())))
    pd.DataFrame(tab).to_csv(f'{T}/main_eps{eps}.csv', index=False)

# ---- Friedman + Nemenyi over datasets (dataset-level means), for acc@0.05 among val-feasible-capable fair families and HV
def friedman(metric, fams, higher=True):
    M = sel[sel.family.isin(fams)].groupby(['dataset', 'family'])[metric].mean().unstack()[fams].dropna()
    ranks = (-M if higher else M).rank(axis=1).mean()
    chi, p = stats.friedmanchisquare(*[M[f] for f in fams])
    k, N = len(fams), len(M)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k]   # Demsar (2006), alpha=.05
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return dict(metric=metric, N=N, k=k, chi2=float(chi), p=float(p), CD=float(cd), ranks={f: float(r) for f, r in ranks.items()})
FAIR = ['Reweighing', 'ExpGrad', 'ThreshOpt', 'AdvDebias', 'DNN+FDP', 'ZadehNet', 'ZadehNet0']
summary['friedman_hv'] = friedman('hv', FAIR)
summary['friedman_acc@0.05'] = friedman('te_acc@0.05', FAIR)
summary['friedman_fdp@0.05'] = friedman('te_fdp@0.05', FAIR, higher=False)

# ---- paired tests ZadehNet vs each competitor over all (dataset, seed) pairs, Holm-corrected
def paired(metric, ref='ZadehNet'):
    out = []
    P = sel.pivot_table(index=['dataset', 'seed'], columns='family', values=metric)
    for fam in FAIR:
        if fam == ref: continue
        d = (P[ref] - P[fam]).dropna()
        st = stats.wilcoxon(d) if (d != 0).any() else None
        out.append(dict(vs=fam, n=len(d), mean_diff=float(d.mean()), median_diff=float(d.median()), p=float(st.pvalue) if st else 1.0))
    out = pd.DataFrame(out).sort_values('p'); m = len(out)
    out['p_holm'] = np.minimum(1, np.maximum.accumulate(out.p.values * (m - np.arange(m))))
    return out
for metric in ['te_acc@0.05', 'te_fdp@0.05', 'hv']:
    paired(metric).to_csv(f'{T}/paired_{metric}.csv', index=False)

# ---- Nadeau-Bengio corrected resampled t-test per dataset: ZadehNet vs DNN+FDP (acc@0.05)
nb = []
for ds in sel.dataset.unique():
    P = sel[sel.dataset == ds].pivot_table(index='seed', columns='family', values='te_acc@0.05')
    d = (P['ZadehNet'] - P['DNN+FDP']).dropna(); k = len(d); ratio = 0.2 / 0.6
    t = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1)) if d.var(ddof=1) > 0 else 0.0
    nb.append(dict(dataset=ds, k=k, mean_diff=d.mean(), t=t, p=float(2 * stats.t.sf(abs(t), k - 1))))
pd.DataFrame(nb).to_csv(f'{T}/nadeau_bengio_zadehnet_vs_dnnfdp.csv', index=False)

# ---- Theorem-1 decomposition: routing vs within, and displacement under rule-local penalty
dec = df[df.method.str.startswith('ZadehNet')].copy()
dec['tag'] = dec.method + ' λ=' + dec.lam.astype(int).astype(str)
D = dec.groupby(['dataset', 'tag'])[['routing', 'within', 'te_fdp', 'te_acc']].mean().reset_index()
D.to_csv(f'{T}/decomposition.csv', index=False)

# ---- interpretability of the selected ZadehNet models
I = sel[sel.family.isin(['ZadehNet', 'ZadehNet0'])].groupby(['dataset', 'family'])[
    ['antecedents@0.05', 'active_rules@0.05', 'fidelity@0.05', 'consequent_nnz@0.05']].mean().reset_index()
I.to_csv(f'{T}/interpretability.csv', index=False)
# training cost
C = df.groupby(['dataset', 'method']).train_time.mean().unstack().round(2); C.to_csv(f'{T}/train_time.csv')
# ---- threshold stability (Prop. 3): crisp gap over thresholds where both groups hold >= 5% of the data
tf = glob.glob(f'{R}/threshold_*.csv')
if tf:
    TH = pd.concat([pd.read_csv(f) for f in tf], ignore_index=True); ts = np.arange(20, 40.5, 0.5); out = []
    for ds in TH.dataset.unique():
        z = LOADERS[ds]()[4]; frac = np.array([(z <= t).mean() for t in ts]); valid = (frac >= 0.05) & (frac <= 0.95)
        sub = TH[TH.dataset == ds].copy()
        cur = np.array([json.loads(c) for c in sub.curve])[:, valid]
        sub['max_dp_valid'] = cur.max(1); sub['mean_dp_valid'] = cur.mean(1); sub['t_lo'] = ts[valid].min(); sub['t_hi'] = ts[valid].max()
        # pre-specified metric-stability test (Prop. 3): total variation and max step of the gap curve, same decisions
        sub['tv_crisp'] = np.abs(np.diff(cur, axis=1)).sum(1); sub['jump_crisp'] = np.abs(np.diff(cur, axis=1)).max(1)
        if 'fcurve' in sub:
            fc = np.array([json.loads(c) for c in sub.fcurve])[:, valid]
            sub['tv_fuzzy'] = np.abs(np.diff(fc, axis=1)).sum(1); sub['jump_fuzzy'] = np.abs(np.diff(fc, axis=1)).max(1)
        out.append(sub.drop(columns=[c for c in ['curve', 'fcurve'] if c in sub]))
    TH = pd.concat(out); TH.to_csv(f'{T}/threshold_runs.csv', index=False)
    TS = TH.groupby(['dataset', 'model', 'group']).agg(n=('seed', 'size'), acc=('acc', 'mean'), dp_at_25=('dp_at_25', 'mean'),
                                                       max_dp=('max_dp_valid', 'mean'), max_dp_sd=('max_dp_valid', 'std'),
                                                       mean_dp=('mean_dp_valid', 'mean'), fdp_young=('fdp_young', 'mean'),
                                                       t_lo=('t_lo', 'first'), t_hi=('t_hi', 'first')).reset_index()
    TS.to_csv(f'{T}/threshold_summary.csv', index=False)
    P = TH[TH.group != 'none'].pivot_table(index=['dataset', 'seed', 'model'], columns='group', values=['max_dp_valid', 'mean_dp_valid', 'dp_at_25', 'acc'])
    tests = {}
    for m in ['max_dp_valid', 'mean_dp_valid', 'dp_at_25', 'acc']:
        dd = (P[(m, 'fuzzy')] - P[(m, 'crisp')]).dropna()
        tests[m] = dict(n=int(len(dd)), mean_fuzzy_minus_crisp=float(dd.mean()), wilcoxon_p=float(stats.wilcoxon(dd).pvalue) if (dd != 0).any() else 1.0)
    summary['threshold_fuzzy_vs_crisp'] = tests
    if 'tv_fuzzy' in TH:
        st = {}
        for k in ['tv', 'jump']:
            dd = (TH[f'{k}_fuzzy'] - TH[f'{k}_crisp']).dropna()
            st[k] = dict(n=int(len(dd)), crisp_mean=float(TH[f'{k}_crisp'].mean()), fuzzy_mean=float(TH[f'{k}_fuzzy'].mean()),
                         ratio=float(TH[f'{k}_crisp'].mean() / TH[f'{k}_fuzzy'].mean()), wilcoxon_p=float(stats.wilcoxon(dd).pvalue),
                         per_dataset={ds: [float(TH[TH.dataset == ds][f'{k}_crisp'].mean()), float(TH[TH.dataset == ds][f'{k}_fuzzy'].mean())] for ds in TH.dataset.unique()})
        summary['metric_stability'] = st
json.dump(summary, open(f'{T}/summary.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
