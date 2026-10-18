"""Builds markdown tables for the manuscript from result CSVs -> paper/tables/*.md"""
import glob, json, os, numpy as np, pandas as pd
from data import LOADERS
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
O = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables_md'); os.makedirs(O, exist_ok=True)
DS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']
LAB = {'CensusKDD': 'Census-KDD'}
FAMS = ['Reweighing', 'ExpGrad', 'ThreshOpt', 'AdvDebias', 'DNN+FDP', 'ZadehNet', 'ZadehNet0']
FLAB = {'ZadehNet0': 'ZadehNet-0', 'ThreshOpt': 'Threshold Opt.', 'ExpGrad': 'Exp. Gradient', 'AdvDebias': 'Adv. debiasing'}
def w(name, s): open(f'{O}/{name}.md', 'w').write(s.strip() + '\n')

# Table 2: datasets
rows = []
for ds in DS:
    X, y, s, mu, z, g, meta = LOADERS[ds]()
    from data import split_and_encode
    d = len(split_and_encode(X, y, s, mu, g, 0)[1])   # encoded features actually used (split 0)
    pG = (mu * y).sum() / mu.sum(); pN = ((1 - mu) * y).sum() / (1 - mu).sum()
    rows.append(f"| {LAB.get(ds, ds)} | {len(y):,} | {d} | {meta['attr'].replace('age (young)', 'age: fuzzy *young*')} | {y.mean():.3f} | {pG:.3f} | {pN:.3f} |")
w('table2', '\n'.join(rows))

# Table 3: accuracy / FDP at eps = 0.05 (mean over splits); Table 4: HV
M = pd.read_csv(f'{T}/main_eps0.05.csv'); S = pd.read_csv(f'{T}/selected_per_seed.csv')
hdr = '| Method | ' + ' | '.join(LAB.get(d, d) for d in DS) + ' |\n|---|' + '---|' * len(DS)
lines = [hdr]
for fam in ['LR', 'LightGBM'] + FAMS:
    cells = []
    for ds in DS:
        r = M[(M.dataset == ds) & (M.family == fam)].iloc[0]
        cells.append(f"{r.acc_mean:.3f} / {r.fdp_mean:.3f}")
    lines.append(f"| {FLAB.get(fam, fam)} | " + ' | '.join(cells) + ' |')
w('table3', '\n'.join(lines))
lines = [hdr]
best = {ds: M[(M.dataset == ds) & M.family.isin(FAMS)].hv_mean.max() for ds in DS}
for fam in FAMS:
    cells = []
    for ds in DS:
        r = M[(M.dataset == ds) & (M.family == fam)].iloc[0]
        g = S[(S.dataset == ds) & (S.family == fam)].hv
        c = f"{r.hv_mean:.3f} ± {g.std(ddof=1):.3f}"
        cells.append(f"**{c}**" if abs(r.hv_mean - best[ds]) < 5e-4 else c)
    lines.append(f"| {FLAB.get(fam, fam)} | " + ' | '.join(cells) + ' |')
summ = json.load(open(f'{T}/summary.json'))
w('table4', '\n'.join(lines))
# AUC and EO at eps=0.05 (supplementary-style compact table)
lines = ['| Method | ' + ' | '.join(LAB.get(d, d) for d in DS) + ' |\n|---|' + '---|' * len(DS)]
for fam in FAMS:
    cells = []
    for ds in DS:
        g = S[(S.dataset == ds) & (S.family == fam)]
        cells.append(f"{g['te_auc@0.05'].mean():.3f} / {g['te_eo@0.05'].mean():.3f}")
    lines.append(f"| {FLAB.get(fam, fam)} | " + ' | '.join(cells) + ' |')
w('table_auc_eo', '\n'.join(lines))

# Table 5: statistics
P = {m: pd.read_csv(f'{T}/paired_{m}.csv').set_index('vs') for m in ['hv', 'te_acc@0.05', 'te_fdp@0.05']}
lines = ['| ZadehNet vs. | Δ HV (mean) | $p_{Holm}$ | Δ accuracy (mean) | $p_{Holm}$ | Δ FDP (mean) | $p_{Holm}$ |', '|---|---|---|---|---|---|---|']
for fam in ['DNN+FDP', 'ZadehNet0', 'ThreshOpt', 'ExpGrad', 'AdvDebias', 'Reweighing']:
    def f(m): r = P[m].loc[fam]; return f"{r.mean_diff:+.4f}", (f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "<0.001")
    a, b, c = f('hv'), f('te_acc@0.05'), f('te_fdp@0.05')
    lines.append(f"| {FLAB.get(fam, fam)} | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {c[0]} | {c[1]} |")
w('table5', '\n'.join(lines))
fr = summ['friedman_hv']
w('friedman', f"Friedman (HV): χ²={fr['chi2']:.2f}, p={fr['p']:.4f}, N={fr['N']}, k={fr['k']}, CD={fr['CD']:.2f}; ranks: " +
  ', '.join(f"{FLAB.get(k, k)} {v:.2f}" for k, v in sorted(fr['ranks'].items(), key=lambda t: t[1])))

# Table 6: decomposition / displacement
D = pd.read_csv(f'{T}/decomposition.csv')
tags = [('ZadehNet-noFair λ=0', 'λ=0'), ('ZadehNet-rule λ=30', 'rule, λ=30'), ('ZadehNet-rule λ=100', 'rule, λ=100'), ('ZadehNet λ=10', 'total, λ=10')]
lines = ['| Dataset | ' + ' | '.join(f'{t[1]}: $\\mathcal{{R}}$ / $\\mathcal{{W}}$ / FDP' for t in tags) + ' |', '|---|' + '---|' * len(tags)]
for ds in DS:
    g = D[D.dataset == ds].set_index('tag'); cells = []
    for t, _ in tags:
        r = g.loc[t]; cells.append(f"{r.routing:+.3f} / {r.within:+.3f} / {r.te_fdp:.3f}")
    lines.append(f"| {LAB.get(ds, ds)} | " + ' | '.join(cells) + ' |')
w('table6', '\n'.join(lines))

# Table 7: interpretability (selected models) + top-k fidelity
I = pd.read_csv(f'{T}/interpretability.csv')
IF = sorted(glob.glob(f'{R}/interp_*.csv'))
K = pd.concat([pd.read_csv(f) for f in IF]) if IF else None
lines = ['| Dataset | $d$ | ZadehNet: antecedents (γ>0.5) / nonzero consequent weights / fidelity | ZadehNet-0: antecedents / fidelity | ZadehNet-0 top-3 fidelity | ZadehNet-0 top-3 AUC (full) |', '|---|---|---|---|---|---|']
for ds in DS:
    a = I[(I.dataset == ds) & (I.family == 'ZadehNet')].iloc[0]; b = I[(I.dataset == ds) & (I.family == 'ZadehNet0')].iloc[0]
    k3 = ''
    if K is not None and ds in K.dataset.unique():
        kk = K[(K.dataset == ds) & (K.model == 'ZadehNet-0') & (K.k == 3) & (K.lam == 0)]
        k3 = (f"{kk.fidelity.mean():.3f}", f"{kk.auc_compact.mean():.3f} ({kk.auc_full.mean():.3f})")
    d = int(K[K.dataset == ds].d.iloc[0]) if K is not None and ds in K.dataset.unique() else ''
    lines.append(f"| {LAB.get(ds, ds)} | {d} | {a['antecedents@0.05']:.1f} / {a['consequent_nnz@0.05']:.1f} / {a['fidelity@0.05']:.3f} | "
                 f"{b['antecedents@0.05']:.1f} / {b['fidelity@0.05']:.3f} | {k3[0] if k3 else '–'} | {k3[1] if k3 else '–'} |")
w('table7', '\n'.join(lines))

# Table 8: threshold stability
if os.path.exists(f'{T}/threshold_summary.csv'):
    TS = pd.read_csv(f'{T}/threshold_summary.csv')
    lines = ['| Dataset (valid $t$) | Model | Penalty group | Accuracy | Gap at $t$=25 | Max gap over $t$ | Mean gap over $t$ | FDP (*young*) |', '|---|---|---|---|---|---|---|---|']
    for ds in ['German', 'Bank', 'Adult']:
        for m in ['DNN', 'ZadehNet']:
            for grp, gl in [('none', 'none (λ=0)'), ('crisp', 'crisp, age ≤ 25'), ('fuzzy', 'fuzzy *young*')]:
                r = TS[(TS.dataset == ds) & (TS.model == m) & (TS.group == grp)]
                if r.empty: continue
                r = r.iloc[0]
                lines.append(f"| {ds} ({r.t_lo:.0f}–{r.t_hi:.0f}) | {m} | {gl} | {r.acc:.3f} | {r.dp_at_25:.3f} | {r.max_dp:.3f} ± {r.max_dp_sd:.3f} | {r.mean_dp:.3f} | {r.fdp_young:.3f} |")
    w('table8', '\n'.join(lines))

# Table 9: ablation
AF = sorted(glob.glob(f'{R}/ablation_*.csv'))
if AF:
    A = pd.concat([pd.read_csv(f) for f in AF])
    order = ['full (L=16)', 'no load-balancing', 'no antecedent gates', 'L=4', 'L=8', 'L=32', 'L=64', 'tau=0.5', 'tau=2']
    dss = [d for d in ['COMPAS', 'Bank', 'Adult'] if d in A.dataset.unique()]
    lines = ['| Variant | ' + ' | '.join(f'{d}: acc (λ=0) / FDP (λ=10) / usage entropy' for d in dss) + ' |', '|---|' + '---|' * len(dss)]
    for v in order:
        cells = []
        for d in dss:
            a0 = A[(A.dataset == d) & (A.variant == v) & (A.lam == 0)]; a1 = A[(A.dataset == d) & (A.variant == v) & (A.lam == 10)]
            cells.append(f"{a0.acc.mean():.3f} / {a1.fdp.mean():.3f} / {a0.usage_entropy.mean():.2f}")
        lines.append(f"| {v} | " + ' | '.join(cells) + ' |')
    w('table9', '\n'.join(lines))

# Table 10: scaling
if os.path.exists(f'{R}/scaling.csv'):
    SC = pd.read_csv(f'{R}/scaling.csv')
    n = SC[SC.exp == 'n'].pivot_table(index='n', columns='model', values='epoch_time')
    lines = ['| $n$ | ' + ' | '.join(n.columns) + ' |', '|---|' + '---|' * len(n.columns)]
    for i, r in n.iterrows(): lines.append(f"| {i:,} | " + ' | '.join(f'{v:.2f}' for v in r) + ' |')
    w('table10', '\n'.join(lines))
print('tables written:', sorted(os.listdir(O)))

# Table 11: rule-level audit (Adult, zero-order, lambda = 0 and 10)
import re
REN = [('marital_Married-civ-spouse', 'marital-status = married'), ('relationship_Husband', 'relationship = husband'),
       ('relationship_Wife', 'relationship = wife'), ('occupation_Exec-managerial', 'occupation = exec-managerial'),
       ('occupation_Prof-specialty', 'occupation = prof-specialty'), ('race_White', 'race = white'),
       ('education_num', 'education-num'), ('hpw', 'hours-per-week'), ('cg', 'capital-gain'), ('cl', 'capital-loss')]
def short(a):
    for k, v in REN: a = re.sub(rf'\b{re.escape(k)}\b', v, a)
    a = re.sub(r'\(centre ≈ ([^)]*)\)', r'(≈\1)', a)
    a = re.sub(r'\(centre (below observed min|above observed max) ([^)]*)\)', lambda m: f"({'<' if 'below' in m.group(1) else '>'}{m.group(2)})", a)
    return a.replace(' AND ', ' ∧ ').replace(' is ', ' ')
lines = ['| λ | Rule | Antecedents | $P(Y{=}1)$ | Coverage women / men | Share of gap |', '|---|---|---|---|---|---|']
for lam in ['0.0', '10.0']:
    A = pd.read_csv(f'{R}/audit_Adult_lam{lam}.csv'); A = A[A.model == 'zero-order']
    A = A.reindex(A.contribution.abs().sort_values(ascending=False).index).head(4)
    for _, r in A.iterrows():
        lines.append(f"| {float(lam):.0f} | R{int(r.rule)} | {short(r.antecedents)} | {r.consequent:.2f} | {r.cov_G:.3f} / {r.cov_N:.3f} | {r.contribution:+.3f} |")
    A = pd.read_csv(f'{R}/audit_Adult_lam{lam}.csv'); z = A[A.model == 'zero-order'].iloc[0]
    lines.append(f"| {float(lam):.0f} | Σ all 16 rules | | | | {z.total_gap:+.4f} (direct: {z.direct_gap:+.4f}) |")
w('table11', '\n'.join(lines))

# Table A.1: full results, long format
M = pd.read_csv(f'{T}/main_eps0.05.csv')
lines = ['| Dataset | Method | Accuracy | AUC | F1 | FDP | EO diff. | HV |', '|---|---|---|---|---|---|---|---|']
for ds in DS:
    for fam in ['LR', 'LightGBM'] + FAMS:
        r = M[(M.dataset == ds) & (M.family == fam)].iloc[0]
        lines.append(f"| {LAB.get(ds, ds)} | {FLAB.get(fam, fam)} | {r.acc} | {r.auc} | {r.f1} | {r.fdp} | {r.eo} | {r.hv} |")
w('tableA1', '\n'.join(lines))
