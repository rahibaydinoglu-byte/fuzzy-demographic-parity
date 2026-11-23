"""Tables for manuscript v4 that summarise the main comparison, generated from per-split results with one rounding rule
(round half up at three decimals), so that a quantity shown in several tables is printed identically.
Writes paper_v4/tables/{oppoint,hv,A1,eo,hvref,scoredec,ageaxis,interp}.md. Run after tables_v3.py."""
import os, numpy as np, pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from analyze_revision import hypervolume
from data import LOADERS
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper_v4', 'tables'); os.makedirs(OUT, exist_ok=True)
DS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']
DSN = {'CensusKDD': 'Census-KDD'}
FN = {'LR': 'LR', 'LightGBM': 'LightGBM', 'Reweighing': 'Reweighing', 'ExpGrad': 'Exp. Gradient', 'ThreshOpt': 'Threshold Opt.',
      'AdvDebias': 'Adv. debiasing', 'DNN+FDP': 'DNN+FDP', 'ZadehNet': 'FairTSK', 'ZadehNet0': 'FairTSK-0'}
TRADE = ['Reweighing', 'ExpGrad', 'ThreshOpt', 'AdvDebias', 'DNN+FDP', 'ZadehNet', 'ZadehNet0']
MEM = {'Reweighing': ['Reweighing'], 'ExpGrad': ['ExpGrad'], 'ThreshOpt': ['ThreshOpt'], 'AdvDebias': ['AdvDebias'], 'LR': ['LR'], 'LightGBM': ['LightGBM'],
       'DNN+FDP': ['DNN', 'DNN+FDP'], 'ZadehNet': ['ZadehNet-noFair', 'ZadehNet'], 'ZadehNet0': ['ZadehNet0-noFair', 'ZadehNet0']}
def r3(x):
    v = str(Decimal(str(round(float(x), 6))).quantize(Decimal('0.001'), ROUND_HALF_UP)); return '0.000' if v == '-0.000' else v
def r2(x): return str(Decimal(str(round(float(x), 6))).quantize(Decimal('0.01'), ROUND_HALF_UP))
def sg4(x): return '0.0000' if abs(x) < 5e-5 else f'{x:+.4f}'
def w(name, s):
    import re
    s = re.sub(r'(?<![A-Za-z0-9])-(?=\d)', '−', s).replace(' ± ', ' ± ')
    open(f'{OUT}/{name}.md', 'w').write(s.strip() + '\n')

main = pd.read_csv(f'{R}/main_all.csv'); sel = pd.read_csv(f'{T}/selected_per_seed.csv')
maj = {ds: max(LOADERS[ds]()[1].mean(), 1 - LOADERS[ds]()[1].mean()) for ds in DS}
H = []
for (ds, seed), g in main.groupby(['dataset', 'seed']):
    for f, mem in MEM.items():
        gf = g[g.method.isin(mem)]
        if len(gf): H.append(dict(dataset=ds, seed=seed, family=f, hv=hypervolume(list(zip(gf.te_acc, gf.te_fdp)), maj[ds])))
H = pd.DataFrame(H)
HM = H.groupby(['dataset', 'family']).hv.agg(['mean', 'std'])
OP = sel.groupby(['dataset', 'family'])[[c for c in sel.columns if c.startswith('te_') and c.endswith('@0.05')]].agg(['mean', 'std'])

# operating points (Table S2)
hdr = '| **Method** | ' + ' | '.join(f'**{DSN.get(d, d)}**' for d in DS) + ' |\n|---|' + '---|' * len(DS)
rows = [hdr]
for f in ['LR', 'LightGBM'] + TRADE:
    rows.append(f'| {FN[f]} | ' + ' | '.join(f"{r3(OP.loc[(d, f), ('te_acc@0.05', 'mean')])} / {r3(OP.loc[(d, f), ('te_fdp@0.05', 'mean')])}" for d in DS) + ' |')
w('oppoint', '\n'.join(rows))
# HV (main Table)
rows = [hdr]
best = {d: max(float(r3(HM.loc[(d, f), 'mean'])) for f in TRADE) for d in DS}
for f in TRADE:
    cells = []
    for d in DS:
        c = f"{r3(HM.loc[(d, f), 'mean'])} ± {r3(HM.loc[(d, f), 'std'])}"
        cells.append(f'**{c}**' if float(r3(HM.loc[(d, f), 'mean'])) == best[d] else c)
    rows.append(f'| {FN[f]} | ' + ' | '.join(cells) + ' |')
w('hv', '\n'.join(rows))
# full results (Table S3)
rows = ['| **Dataset** | **Method** | **Accuracy** | **AUC** | **F1** | **FDP** | **EO diff.** | **HV** |', '|---|---|---|---|---|---|---|---|']
for d in DS:
    for f in ['LR', 'LightGBM'] + TRADE:
        cell = lambda m: f"{r3(OP.loc[(d, f), (m, 'mean')])} ± {r3(OP.loc[(d, f), (m, 'std')])}"
        hvc = f"{r3(HM.loc[(d, f), 'mean'])} ± {r3(HM.loc[(d, f), 'std'])}" if f in TRADE else '—'
        rows.append(f'| {DSN.get(d, d)} | {FN[f]} | {cell("te_acc@0.05")} | {cell("te_auc@0.05")} | {cell("te_f1@0.05")} | {cell("te_fdp@0.05")} | {cell("te_eo@0.05")} | {hvc} |')
w('A1', '\n'.join(rows))
# equalised odds at the operating point
rows = [hdr]
for f in ['LR', 'LightGBM'] + TRADE: rows.append(f'| {FN[f]} | ' + ' | '.join(r3(OP.loc[(d, f), ('te_eo@0.05', 'mean')]) for d in DS) + ' |')
w('eo', '\n'.join(rows))
# HV reference sensitivity
o = pd.read_csv(f'{T}/r5_hv_reference.csv')
vsn = {'DNN+FDP': 'DNN+FDP', 'ZadehNet0': 'FairTSK-0', 'ThreshOpt': 'Threshold Opt.', 'ExpGrad': 'Exp. Gradient', 'AdvDebias': 'Adv. debiasing', 'Reweighing': 'Reweighing'}
rows = ['| **Reference FDP** | **Datasets** | **Friedman** $p$ | ' + ' | '.join(f'**vs {vsn[v]}**' for v in vsn) + ' |', '|---|---|---|' + '---|' * len(vsn)]
for (fr, ex), g in o.groupby(['fdp_ref', 'without_diabetes']):
    g = g.set_index('vs')
    rows.append(f'| {fr} | {"7 (without Diabetes)" if ex else "8"} | {r3(g.friedman_p.iloc[0])} | ' +
                ' | '.join(f'{sg4(g.loc[v, "diff"])} ({r3(g.loc[v, "p_holm"])})' for v in vsn) + ' |')
w('hvref', '\n'.join(rows))
# score-decision correlation
sd = pd.read_csv(f'{T}/r5_score_decision.csv')
rows = ['| **Dataset** | **Models** | **Pearson** $r$ |', '|---|---|---|'] + [f'| {DSN.get(r.dataset, r.dataset) if r.dataset != "all" else "All"} | {int(r.n)} | {r2(r.r)} |' for _, r in sd.iterrows()]
w('scoredec', '\n'.join(rows))
# age axis
D = pd.read_csv(f'{R}/r5_ageaxis.csv'); bands = [('0_25', '≤25'), ('25_30', '26–30'), ('30_35', '31–35'), ('35_45', '36–45'), ('45_60', '46–60'), ('60_200', '>60')]
rows = ['| **Dataset** | **Model** | λ | **Accuracy** | **abs. corr(age)** | ' + ' | '.join(f'**{b}**' for _, b in bands) + ' |', '|---|---|---|---|---|' + '---|' * len(bands)]
for (ds, m, lam), g in D.groupby(['dataset', 'model', 'lam'], sort=False):
    rows.append(f'| {ds} | {"FairTSK" if m == "ZadehNet" else m} | {lam} | {r3(g.acc.mean())} | {r3(g.corr_age.abs().mean())} | ' +
                ' | '.join(f'{g[f"band_{k}"].mean():+.3f}' for k, _ in bands) + ' |')
n = D.groupby('dataset')[[f'n_{k}' for k, _ in bands]].mean()
for ds in n.index: rows.append(f'| {ds} | test instances per band | | | | ' + ' | '.join(f'{n.loc[ds, f"n_{k}"]:.0f}' for k, _ in bands) + ' |')
w('ageaxis', '\n'.join(rows))
# interpretable baselines
P = pd.read_csv(f'{T}/r4_interp_perseed.csv'); PM = P.groupby(['dataset', 'family']).mean(numeric_only=True)
rows = ['| **Dataset** | **FairTSK: acc / FDP** | **HV test / val.** | **Fair tree: acc / FDP** | **HV test / val.** | **Leaves** | **FairCORELS: acc / FDP** | **HV test / val.** | **Rules** |', '|---|---|---|---|---|---|---|---|---|']
for d in DS:
    c = [f"{r3(OP.loc[(d, 'ZadehNet'), ('te_acc@0.05', 'mean')])} / {r3(OP.loc[(d, 'ZadehNet'), ('te_fdp@0.05', 'mean')])}",
         f"{r3(HM.loc[(d, 'ZadehNet'), 'mean'])}/{r3(PM.loc[(d, 'ZadehNet'), 'hv_val'])}"]
    for f in ['FairTree', 'FairCORELS']:
        g = PM.loc[(d, f)]; c += [f'{r3(g.acc)} / {r3(g.fdp)}', f'{r3(g.hv)}/{r3(g.hv_val)}', f'{float(g["size"]):.1f}']
    rows.append(f'| {DSN.get(d, d)} | ' + ' | '.join(c) + ' |')
w('interp', '\n'.join(rows))
# checks: FairTSK values in interp equal those of the per-seed file
chk = (P[P.family == 'ZadehNet'].groupby('dataset').acc.mean() - sel[sel.family == 'ZadehNet'].groupby('dataset')['te_acc@0.05'].mean()).abs().max()
print('max |acc difference| interp vs selected:', chk); print('tables v4 written')

# ---- round 6: dataset-level tests without split 0
b = pd.read_csv(f'{T}/rev_stats_datasetlevel_without_split_0.csv')
VN = {'DNN+FDP': 'DNN+FDP', 'ZadehNet0': 'FairTSK-0', 'ThreshOpt': 'Threshold Opt.', 'ExpGrad': 'Exp. Gradient', 'AdvDebias': 'Adv. debiasing', 'Reweighing': 'Reweighing'}
rows = ['| **FairTSK vs.** | **Δ HV, test front [90% CI]** | **wins/8** | $p_{Holm}$ | **Δ acc@0.05 [90% CI]** | $p_{Holm}$ |', '|---|---|---|---|---|---|']
for v in VN:
    h = b[(b.metric == 'HV') & (b.vs == v)].iloc[0]; c = b[(b.metric == 'acc@0.05') & (b.vs == v)].iloc[0]
    rows.append(f'| {VN[v]} | {h.mean_diff:+.4f} [{h.ci90_lo:+.4f}, {h.ci90_hi:+.4f}] | {int(h.wins)} | {r3(h.wilcoxon_p_holm)} | {c.mean_diff:+.4f} [{c.ci90_lo:+.4f}, {c.ci90_hi:+.4f}] | {r3(c.wilcoxon_p_holm)} |')
w('split0', '\n'.join(rows))
# ---- round 6: German and Bank evaluated against the crisp core group (age <= 25)
rows = ['| **Method** | **German: HV (crisp)** | **German: acc / crisp gap** | **Bank: HV (crisp)** | **Bank: acc / crisp gap** |', '|---|---|---|---|---|']
CR = []
for (ds, seed), g in main[main.dataset.isin(['German', 'Bank'])].groupby(['dataset', 'seed']):
    for f in TRADE:
        gf = g[g.method.isin(MEM[f])].reset_index(drop=True)
        ok = gf[gf.va_dp_crisp <= 0.05]; s_ = ok.loc[ok.va_acc.idxmax()] if len(ok) else gf.loc[gf.va_dp_crisp.idxmin()]
        CR.append(dict(dataset=ds, seed=seed, family=f, hv=hypervolume(list(zip(gf.te_acc, gf.te_dp_crisp)), maj[ds]), acc=s_.te_acc, gap=s_.te_dp_crisp))
CR = pd.DataFrame(CR).groupby(['dataset', 'family']).mean(numeric_only=True)
for f in TRADE:
    rows.append(f'| {FN[f]} | ' + ' | '.join(f"{r3(CR.loc[(d, f), 'hv'])} | {r3(CR.loc[(d, f), 'acc'])} / {r3(CR.loc[(d, f), 'gap'])}" for d in ['German', 'Bank']) + ' |')
w('crispeval', '\n'.join(rows)); print(CR.round(3))

# ---- round 6: log-loss, consequent norms and the decomposition under the three weightings (Adult, COMPAS)
R6 = pd.concat([pd.read_csv(f'{R}/r6_{d}.csv') for d in ['Adult', 'COMPAS']])
R6M = R6.groupby(['dataset', 'model'], sort=False).mean(numeric_only=True); R6S = R6.groupby(['dataset', 'model'], sort=False).std(numeric_only=True)
MN = {'first-order': 'first-order', 'rule-local': 'rule-local', 'zero-order': 'zero-order'}
rows = ['| **Data** | **Model** | **Log-loss** | **Acc.** | **Cons. norm** | $\\mathcal R$ / $\\mathcal W$ **midpoint** | $\\mathcal R$ / $\\mathcal W$ **ref. ¬G** | $\\mathcal R$ / $\\mathcal W$ **ref. G** |', '|---|---|---|---|---|---|---|---|']
for (d, m), g in R6M.iterrows():
    s = R6S.loc[(d, m)]
    cn = '—' if np.isnan(g.cons_norm) else f'{r2(g.cons_norm)} ± {r2(s.cons_norm)}'
    rows.append(f'| {d} | {MN[m]} | {r3(g.logloss)} | {r3(g.acc)} | {cn} | ' + ' | '.join(f'{r3(g["R_" + k])} / {r3(g["W_" + k])}' for k in ['mid', 'refN', 'refG']) + ' |')
w('r6dec', '\n'.join(rows))
# ---- round 6: synthetic configurations between the earlier design points
SG = pd.read_csv(f'{T}/r6_synthgap.csv'); SGM = SG.groupby(['scen', 'mode', 'L'], sort=False).agg(['mean', 'std'])
MO = {'train': 'trainable', 'fixedc': 'centres frozen', 'fixed': 'frozen'}
rows = ['| **Scenario** | **Antecedents** | $L$ | $\\hat\\varepsilon_0$ (nats) | **Retained gap** | $\\mathcal W$ **after penalty** | **Accuracy: none / rule-local** |', '|---|---|---|---|---|---|---|']
for sc in ['linear', 'nonlinear']:
    for mo in ['train', 'fixedc', 'fixed']:
        for L in sorted(SG[(SG.scen == sc) & (SG['mode'] == mo)].L.unique()):
            g = SGM.loc[(sc, mo, L)]
            rows.append(f'| {sc.capitalize()} | {MO[mo]} | {L} | {r3(g[("eps0", "mean")])} ± {r3(g[("eps0", "std")])} | {r2(g[("ret", "mean")])} ± {r2(g[("ret", "std")])} | {r3(g[("W", "mean")])} | {r3(g[("acc0", "mean")])} / {r3(g[("acc1", "mean")])} |')
w('synthgap', '\n'.join(rows))
# ---- round 6: Exponentiated Gradient with a LightGBM base learner
from scipy.stats import wilcoxon
EG = pd.read_csv(f'{T}/r6_egb_perseed.csv'); EGM = EG.groupby('dataset').mean(numeric_only=True).loc[DS]
rows = ['| **Dataset** | **Exp. Gradient (LR): HV** | **Exp. Gradient (LightGBM): HV** | **acc / FDP at** $\\epsilon = 0.05$ | **FairTSK: HV** |', '|---|---|---|---|---|']
for d in DS:
    g = EGM.loc[d]; rows.append(f'| {DSN.get(d, d)} | {r3(HM.loc[(d, "ExpGrad"), "mean"])} | {r3(g.hv_egb)} | {r3(g.acc)} / {r3(g.fdp)} | {r3(HM.loc[(d, "ZadehNet"), "mean"])} |')
dd = EGM.hv_tsk - EGM.hv_egb
print('egb: FairTSK - ExpGrad-GBM', dd.round(4).to_dict(), 'mean', round(dd.mean(), 4), 'wins', int((dd > 0.0005).sum()), 'p', wilcoxon(EGM.hv_tsk, EGM.hv_egb).pvalue,
      'check hv_tsk vs HM', (EGM.hv_tsk - HM.xs('ZadehNet', level='family')['mean'].loc[DS]).abs().max(), 'egb vs LR', int((EGM.hv_egb > EGM.hv_eg).sum()))
w('egb', '\n'.join(rows))
