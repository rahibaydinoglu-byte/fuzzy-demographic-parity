"""Revision statistics (R3.7, R1.4).
R3.7: (a) dataset-level tests (one value per dataset = mean over splits), which removes the dependence between
      overlapping random splits from the across-dataset inference; (b) per-dataset Nadeau-Bengio corrected 90% CIs;
      (c) equivalence: the smallest margin delta for which a TOST at alpha=0.05 declares ZadehNet and DNN+FDP equivalent
      (= max |bound| of the 90% CI), reported instead of an arbitrary pre-chosen margin.
R1.4: every test repeated with split 0 (the pilot split) removed."""
import os, json, numpy as np, pandas as pd
from scipy import stats
T = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables')
sel = pd.read_csv(f'{T}/selected_per_seed.csv')
FAIR = ['Reweighing', 'ExpGrad', 'ThreshOpt', 'AdvDebias', 'DNN+FDP', 'ZadehNet0']
METRICS = {'hv': 'HV', 'te_acc@0.05': 'acc@0.05', 'te_fdp@0.05': 'FDP@0.05'}

def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return adj

def dataset_level(s):
    out = []
    for met in METRICS:
        M = s.groupby(['dataset', 'family'])[met].mean().unstack()
        for fam in FAIR:
            d = (M['ZadehNet'] - M[fam]).dropna()
            w = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
            t = stats.ttest_1samp(d, 0)
            ci = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
            out.append(dict(metric=METRICS[met], vs=fam, n_datasets=len(d), mean_diff=d.mean(), wins=int((d > 0).sum()),
                            wilcoxon_p=w, t_p=t.pvalue, ci90_lo=ci[0], ci90_hi=ci[1], tost_min_margin=max(abs(ci[0]), abs(ci[1]))))
    o = pd.DataFrame(out)
    o['wilcoxon_p_holm'] = o.groupby('metric').wilcoxon_p.transform(lambda p: holm(p.values))
    return o

def nadeau_bengio(s, metric='hv'):
    out = []
    for ds in s.dataset.unique():
        P = s[s.dataset == ds].pivot_table(index='seed', columns='family', values=metric)
        d = (P['ZadehNet'] - P['DNN+FDP']).dropna(); k = len(d); r = 0.2 / 0.6
        se = np.sqrt((1 / k + r) * d.var(ddof=1)); tq = stats.t.ppf(0.95, k - 1)
        t = d.mean() / se if se > 0 else 0.0
        out.append(dict(dataset=ds, metric=metric, k=k, mean_diff=d.mean(), p=2 * stats.t.sf(abs(t), k - 1),
                        ci90_lo=d.mean() - tq * se, ci90_hi=d.mean() + tq * se))
    return pd.DataFrame(out)

res = {}
for tag, s in [('all splits', sel), ('without split 0', sel[sel.seed != 0])]:
    dl = dataset_level(s); dl.insert(0, 'splits', tag); res[tag] = dl
    nb = pd.concat([nadeau_bengio(s, m) for m in ['hv', 'te_acc@0.05']]); nb.insert(0, 'splits', tag)
    dl.to_csv(f'{T}/rev_stats_datasetlevel_{tag.replace(" ", "_")}.csv', index=False)
    nb.to_csv(f'{T}/rev_stats_nb_{tag.replace(" ", "_")}.csv', index=False)
    print(f'\n===== {tag} ====='); print(dl.round(4).to_string(index=False)); print(nb.round(4).to_string(index=False))
    # Friedman on dataset-level HV
    M = s.groupby(['dataset', 'family']).hv.mean().unstack()[FAIR + ['ZadehNet']]
    print('Friedman HV p =', round(stats.friedmanchisquare(*[M[f] for f in M]).pvalue, 4),
          '| mean ranks', (-M).rank(axis=1).mean().round(2).to_dict())
