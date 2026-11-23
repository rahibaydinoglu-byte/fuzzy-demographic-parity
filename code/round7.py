"""Round 7: Adult with the fuzzy group young (age) as protected group (AdultAge; age and sex are not inputs).
Reads results/main_AdultAge_*.csv written by run.py and writes results/tables/r7_adultage_{fam,decomp}.csv and
paper_v4/tables/{adultage,adultage_dec}.md. Same protocol as the main comparison: HV of the test accuracy-FDP front,
operating point at validation FDP <= 0.05, Nadeau-Bengio test over 10 splits, Holm-adjusted."""
import os, glob, json, numpy as np, pandas as pd
from scipy import stats
from analyze_revision import hypervolume
from data import LOADERS
from tables_v4 import r3, w, FN, MEM, TRADE
R = os.path.join(os.path.dirname(__file__), '..', 'results')
M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}/main_AdultAge_*.csv'))])
y = LOADERS['AdultAge']()[1]; maj = max(y.mean(), 1 - y.mean())
def nb(d, ratio=0.2 / 0.6):
    d = np.asarray(d, float); k = len(d); se = np.sqrt((1 / k + ratio) * d.var(ddof=1)); t = d.mean() / se
    return float(2 * stats.t.sf(abs(t), k - 1))
rows = []
for seed, g in M.groupby('seed'):
    for f in ['LR', 'LightGBM'] + TRADE:
        gf = g[g.method.isin(MEM[f])].reset_index(drop=True)
        ok = gf[gf.va_fdp <= 0.05]; s_ = ok.loc[ok.va_acc.idxmax()] if len(ok) else gf.loc[gf.va_fdp.idxmin()]
        rows.append(dict(seed=seed, family=f, hv=hypervolume(list(zip(gf.te_acc, gf.te_fdp)), maj),
                         hv_crisp=hypervolume(list(zip(gf.te_acc, gf.te_dp_crisp)), maj),
                         acc=s_.te_acc, fdp=s_.te_fdp, crisp=s_.te_dp_crisp, eo=s_.te_eo))
F = pd.DataFrame(rows); F.to_csv(f'{R}/tables/r7_adultage_fam.csv', index=False)
FM = F.groupby('family').mean(numeric_only=True); FS = F.groupby('family').std(numeric_only=True)
piv = F.pivot(index='seed', columns='family', values='hv')
comp = [f for f in TRADE if f != 'ZadehNet']
praw = {f: nb(piv['ZadehNet'] - piv[f]) for f in comp}
order = sorted(comp, key=lambda f: praw[f]); ph = {}; run = 0
for i, f in enumerate(order): run = max(run, min(1, (len(order) - i) * praw[f])); ph[f] = run
out = ['| **Method** | **HV (FDP, young)** | **acc / FDP at** $\\epsilon = 0.05$ | **crisp gap (age ≤ 25)** | **HV (crisp gap)** | **Δ HV vs FairTSK** ($p_{Holm}$) |', '|---|---|---|---|---|---|']
for f in ['LR', 'LightGBM'] + TRADE:
    hv = f"{r3(FM.loc[f, 'hv'])} ± {r3(FS.loc[f, 'hv'])}" if f in TRADE else '—'
    hvc = r3(FM.loc[f, 'hv_crisp']) if f in TRADE else '—'
    d = '—' if f not in comp else f"{(piv['ZadehNet'] - piv[f]).mean():+.3f} ({'< 0.001' if ph[f] < 0.0005 else r3(ph[f])})"
    out.append(f"| {FN[f]} | {hv} | {r3(FM.loc[f, 'acc'])} / {r3(FM.loc[f, 'fdp'])} | {r3(FM.loc[f, 'crisp'])} | {hvc} | {d} |")
w('adultage', '\n'.join(out))
# decomposition: unconstrained, rule-local (lambda = 30, 100) and total penalty (lambda = 10)
D = []
for (meth, lam), lab in [(('ZadehNet-noFair', 0), 'unconstrained'), (('ZadehNet-rule', 30), 'rule-local, λ=30'),
                         (('ZadehNet-rule', 100), 'rule-local, λ=100'), (('ZadehNet', 10), 'total, λ=10')]:
    g = M[(M.method == meth) & (M.cfg.apply(lambda c: json.loads(c).get('lam')) == lam)]
    D.append(dict(model=lab, R=g.routing.mean(), W=g.within.mean(), gap=(g.routing + g.within).mean(), fdp=g.te_fdp.mean(),
                  acc=g.te_acc.mean(), n=len(g), gap_seeds=list(g.routing + g.within)))
DD = pd.DataFrame(D); DD.drop(columns='gap_seeds').to_csv(f'{R}/tables/r7_adultage_decomp.csv', index=False)
u = np.array(DD.gap_seeds[0]); r100 = np.array(DD.gap_seeds[2])
p_rl = nb(r100 - u)
out = ['| **Model** | $\\mathcal R$ | $\\mathcal W$ | **score-level gap** | **FDP (decisions)** | **Accuracy** |', '|---|---|---|---|---|---|']
for _, r in DD.iterrows():
    out.append(f'| {r.model} | {r3(r.R)} | {r3(r.W)} | {r3(r.gap)} | {r3(r.fdp)} | {r3(r.acc)} |')
w('adultage_dec', '\n'.join(out))
print(FM.round(4)); print({f: (round((piv['ZadehNet'] - piv[f]).mean(), 4), round(ph[f], 3)) for f in comp})
print(DD.drop(columns='gap_seeds').round(4)); print('rule-local 100 vs unconstrained score gap NB p =', round(p_rl, 3), 'n', DD.n.tolist())
