"""Generate the revised (v3) tables as markdown snippets from result files, so that no number is typed by hand.
Writes paper_v3/tables/*.md"""
import os, glob, json, numpy as np, pandas as pd
R = os.path.join(os.path.dirname(__file__), '..', 'results'); T = f'{R}/tables'
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper_v3', 'tables'); os.makedirs(OUT, exist_ok=True)
DS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']
DSN = {'CensusKDD': 'Census-KDD'}
import re
def w(name, s):
    s = re.sub(r'(?<![\w.])[-−+](0\.0+)(?![0-9])', r'\1', s)          # no signed zeros
    s = re.sub(r'(?<![A-Za-z0-9])-(?=\d)', '−', s)                     # typographic minus (no line break)
    s = s.replace(' ± ', '\u00a0±\u00a0')
    open(f'{OUT}/{name}.md', 'w').write(s.strip() + '\n')
from decimal import Decimal, ROUND_HALF_UP
def f3(x): return str(Decimal(str(round(float(x), 6))).quantize(Decimal('0.001'), ROUND_HALF_UP))
def fp(p): return '<0.001' if p < 0.001 else f'{p:.3f}'
def sgn(x, k=4): return f'{x:+.{k}f}'

# ---------------- dataset-level statistics (R3.7, R1.4)
NAMES = {'Reweighing': 'Reweighing', 'ExpGrad': 'Exp. Gradient', 'ThreshOpt': 'Threshold Opt.', 'AdvDebias': 'Adv. debiasing',
         'DNN+FDP': 'DNN+FDP', 'ZadehNet0': 'ZadehNet-0'}
a = pd.read_csv(f'{T}/rev_stats_datasetlevel_all_splits.csv'); vf = pd.read_csv(f'{T}/rev3_hv_valfront_stats.csv').set_index('vs')
rows = ['| **ZadehNet vs.** | **Δ HV, test front [90% CI]** | **wins/8** | $p_{Holm}$ | **Δ HV, validation-selected front [90% CI]** | $p_{Holm}$ | **Δ acc@0.05 [90% CI]** | $p_{Holm}$ |',
        '|---|---|---|---|---|---|---|---|']
for vs in ['DNN+FDP', 'ZadehNet0', 'ThreshOpt', 'ExpGrad', 'AdvDebias', 'Reweighing']:
    h = a[(a.metric == 'HV') & (a.vs == vs)].iloc[0]; c = a[(a.metric == 'acc@0.05') & (a.vs == vs)].iloc[0]
    v = vf.loc[vs]
    rows.append(f'| {NAMES[vs]} | {sgn(h.mean_diff)} [{h.ci90_lo:+.4f}, {h.ci90_hi:+.4f}] | {int(h.wins)} | {fp(h.wilcoxon_p_holm)} | '
                f'{sgn(v["diff"])} [{v.lo:+.4f}, {v.hi:+.4f}] | {fp(v.p_holm)} | '
                f'{sgn(c.mean_diff)} [{c.ci90_lo:+.4f}, {c.ci90_hi:+.4f}] | {fp(c.wilcoxon_p_holm)} |')
w('stats', '\n'.join(rows))

# ---------------- continuous-attribute penalties (R1.1)
sel = pd.read_csv(f'{T}/rev_cov_selected.csv')
f = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/fdpage_*.csv')]); c = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/cov_*_*.csv')])
d = pd.concat([f, c]);
def select(g, eps=0.05):
    ok = g[g.va_fdp <= eps]; return ok.loc[ok.va_acc.idxmax()] if len(ok) else g.loc[g.va_fdp.idxmin()]
FAM = {'DNN+FDP (fuzzy *young*)': ['DNN', 'DNN+FDP'], 'DNN+Cov(age)': ['DNN', 'DNN+Cov(age)'], 'DNN+HGR(age)': ['DNN', 'DNN+HGR(age)'],
       'LR+Cov(s) (Zafar et al.)': ['LR-torch', 'LR+Cov(s)'], 'LR+Cov(age)': ['LR-torch', 'LR+Cov(age)']}
SELN = {'DNN+FDP (fuzzy *young*)': 'DNN+FDP (fuzzy young)', 'DNN+Cov(age)': 'DNN+Cov(age)', 'DNN+HGR(age)': 'DNN+HGR(age)',
        'LR+Cov(s) (Zafar et al.)': 'LR+Cov(s) [Zafar]', 'LR+Cov(age)': 'LR+Cov(age)'}
op = []
for (ds, seed), g in d.groupby(['dataset', 'seed']):
    for fam, mem in FAM.items():
        s = select(g[g.method.isin(mem)].reset_index(drop=True)); op.append(dict(dataset=ds, seed=seed, family=fam, acc=s.te_acc, fdp=s.te_fdp, cage=abs(s.te_cov_age)))
op = pd.DataFrame(op)
from scipy import stats
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); adj[o] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o])); return adj
tests = []
for ds in ['Bank', 'German']:
    ref = sel[(sel.dataset == ds) & (sel.family == 'DNN+FDP (fuzzy young)')].set_index('seed').hv
    for fam in list(FAM)[1:]:
        o = sel[(sel.dataset == ds) & (sel.family == SELN[fam])].set_index('seed').hv.reindex(ref.index); dd = ref - o
        tests.append(dict(ds=ds, fam=fam, diff=dd.mean(), p=stats.wilcoxon(dd).pvalue))
tests = pd.DataFrame(tests); tests['holm'] = holm(tests.p.values)
nbt = pd.read_csv(f'{T}/rev3_cov_nb.csv'); FAMN = {v: k for k, v in SELN.items()}
tests = pd.DataFrame([dict(ds=r.dataset, fam=FAMN[r.family], diff=-r['diff'], holm=r.p_holm) for _, r in nbt.iterrows()])
rows = ['| **Penalty** | **Bank: HV** | **acc / FDP (young) / abs. corr(age)** | **Δ HV vs DNN+FDP ($p_{Holm}$)** | **German: HV** | **acc / FDP (young) / abs. corr(age)** | **Δ HV vs DNN+FDP ($p_{Holm}$)** |',
        '|---|---|---|---|---|---|---|']
for fam in FAM:
    cells = []
    for ds in ['Bank', 'German']:
        hv = sel[(sel.dataset == ds) & (sel.family == SELN[fam])].hv; o = op[(op.dataset == ds) & (op.family == fam)]
        t = tests[(tests.ds == ds) & (tests.fam == fam)]
        cells += [f'{f3(hv.mean())} ± {f3(hv.std())}', f'{f3(o.acc.mean())} / {f3(o.fdp.mean())} / {f3(o.cage.mean())}',
                  '—' if t.empty else f'{-t["diff"].iloc[0]:+.3f} ({fp(t.holm.iloc[0])})']
    rows.append(f'| {fam} | ' + ' | '.join(cells) + ' |')
u = d[d.method == 'DNN'].groupby('dataset')
rows.append('| Unconstrained DNN | — | ' + f'{u.te_acc.mean()["Bank"]:.3f} / {u.te_fdp.mean()["Bank"]:.3f} / {u.te_cov_age.apply(lambda x: x.abs().mean())["Bank"]:.3f}' +
            ' | — | — | ' + f'{u.te_acc.mean()["German"]:.3f} / {u.te_fdp.mean()["German"]:.3f} / {u.te_cov_age.apply(lambda x: x.abs().mean())["German"]:.3f}' + ' | — |')
w('continuous', '\n'.join(rows))

# ---------------- synthetic study (R1.2, R3.5)
t = pd.read_csv(f'{T}/rev_synth_retained_vs_eps.csv')
y = pd.concat([pd.read_csv(x).assign(scenario='linear') for x in glob.glob(f'{R}/synth_[0-9]*.csv')] +
              [pd.read_csv(x).assign(scenario='nonlinear') for x in glob.glob(f'{R}/synth2_*.csv')])
y = y[y.rho == 0.6]
rows = ['| **Scenario** | **Antecedents** | $L$ | $\\hat\\varepsilon_0$ **(nats)** | $\\Delta_0$ | **Retained score gap, rule-local** $\\lambda=100$ | **Retained score gap, total** $\\lambda=100$ | **Retained decision gap, rule-local / total** | $|\\mathcal W|$ **after rule-local** |',
        '|---|---|---|---|---|---|---|---|---|']
for _, r in t.iterrows():
    mode = 'train' if r.antecedents == 'train' else 'fixed'
    base = y[(y.scenario == r.scenario) & (y['mode'] == mode) & (y.L == r.L) & (y.fair == 'rule') & (y.lam == 0)].set_index('seed')
    tot = y[(y.scenario == r.scenario) & (y['mode'] == mode) & (y.L == r.L) & (y.fair == 'total') & (y.lam == 100)].set_index('seed')
    rl = y[(y.scenario == r.scenario) & (y['mode'] == mode) & (y.L == r.L) & (y.fair == 'rule') & (y.lam == 100)].set_index('seed')
    tr = (tot.delta / base.delta).dropna()
    rows.append(f'| {r.scenario} | {"trainable" if mode == "train" else "fixed"} | {int(r.L)} | {r.eps0:.4f} | {base.delta.mean():+.3f} | '
                f'{r.retained:.2f} ± {r.sd:.2f} | {"—" if tr.empty else f"{tr.mean():.2f} ± {tr.std():.2f}"} | '
                f'{(rl.dp_dec / base.dp_dec).mean():.2f} / {"—" if tr.empty else f"{(tot.dp_dec / base.dp_dec).dropna().mean():.2f}"} | {rl.within.abs().mean():.4f} |')
w('synthetic', '\n'.join(rows))
yy = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/synth_[0-9]*.csv')] + [pd.read_csv(x) for x in glob.glob(f'{R}/synth2_*.csv')])
json.dump(dict(runs=len(yy), bound_ok=int(((yy.delta - yy.delta_bayes).abs() <= yy.bound_kl).sum()),
               max_ratio=float(((yy.delta - yy.delta_bayes).abs() / yy.bound_kl).max())), open(f'{OUT}/synthetic_meta.json', 'w'))

# ---------------- audit (zero-order Adult split 0; scaled-precision ranking)
au = pd.read_csv(f'{R}/audit3_Adult_seed0.csv')
def pretty(a):
    import re
    for k, v in [(r'\beducation_num\b', 'education-num'), (r'\bhpw\b', 'hours-per-week'), (r'\bcg\b', 'capital-gain'), (r'\bcl\b', 'capital-loss'),
                 ('marital = ', 'marital-status = '), ('Married-civ-spouse', 'married'), ('Never-married', 'never-married'),
                 ('Own-child', 'own-child'), ('Not-in-family', 'not-in-family'), ('Unmarried', 'unmarried'), ('Husband', 'husband'),
                 ('Wife', 'wife'), ('Exec-managerial', 'exec-managerial'), ('Prof-specialty', 'prof-specialty'), ('Other-service', 'other-service'), ('Handlers-cleaners', 'handlers-cleaners'), ('Adm-clerical', 'adm-clerical')]:
        a = re.sub(k, v, a)
    return a.replace(' [absent]', ' (absent)')
rows = ['| **λ** | **Rule** | **Antecedents (six most relevant)** | $g_l$ | **Coverage women / men** | **Centred share** |', '|---|---|---|---|---|---|']
for lam, g in au.groupby('lam'):
    g = g.reindex(g.share.abs().sort_values(ascending=False).index)
    for _, r in g.head(5).iterrows():
        rows.append(f'| {int(lam)} | R{int(r.rule)} | {pretty(r.antecedents)} | {r.consequent:.2f} | {r.cov_G:.3f} / {r.cov_N:.3f} | {r.share:+.3f} |')
    rows.append(f'| {int(lam)} | Σ all 16 rules | | | | {g.total.iloc[0]:+.4f} (direct: {g.direct.iloc[0]:+.4f}) |')
w('audit', '\n'.join(rows))

# ---------------- proxy removal (R1.3)
n = pd.read_csv(glob.glob(f'{R}/noproxy_0-*.csv')[0])
n3 = pd.read_csv(glob.glob(f'{R}/noproxy3_*.csv')[0]); sdmax = []
rows = ['| **Inputs** | **Model** | **Accuracy** | **FDP** | $\\mathcal R$ / $\\mathcal W$ | **Sex AUC of firing: top-4 / other rules** | **Sex AUC of features: top-4 rules / all** |', '|---|---|---|---|---|---|---|']
def auc(v, fair, lam):
    g3 = n3[(n3.variant == v) & (n3.fair == fair) & (n3.lam == lam)]
    if g3.empty: return '— | — |'
    p1 = stats.wilcoxon(g3.top_rule_sex_auc - g3.other_rule_sex_auc).pvalue; p2 = stats.wilcoxon(g3.top_feat_sex_auc - g3.all_feat_sex_auc).pvalue
    return f'{g3.top_rule_sex_auc.mean():.3f} / {g3.other_rule_sex_auc.mean():.3f} | {g3.top_feat_sex_auc.mean():.3f} / {g3.all_feat_sex_auc.mean():.3f} |'
lab = {('total', 0): 'unconstrained', ('rule', 100): 'rule-local, λ=100', ('total', 10): 'total, λ=10', ('total', 100): 'total, λ=100'}
for v in ['with proxies', 'without marital/relationship']:
    for k in [('total', 0), ('rule', 100), ('total', 10), ('total', 100)]:
        g = n[(n.variant == v) & (n.fair == k[0]) & (n.lam == k[1])]
        if g.empty: continue
        rows.append(f'| {"all features" if v == "with proxies" else "without marital, relationship"} | {lab[k]} | '
                    f'{g.te_acc.mean():.3f} | {g.te_fdp.mean():.3f} | {g.routing.mean():+.3f} / {g.within.mean():+.3f} | ' + auc(v, *k)); sdmax.append(max(g.te_acc.std(), g.te_fdp.std()))
w('proxy', '\n'.join(rows)); print('proxy max sd', max(sdmax))
from collections import Counter
n2 = pd.read_csv(glob.glob(f'{R}/noproxy2_*.csv')[0]); tally = {}
for (v, fair, lam), g in n2.groupby(['variant', 'fair', 'lam']):
    cnt = Counter(sum([json.loads(x) for x in g.top_rule_features_scaled], [])); tally[f'{v}|{fair}|{lam}'] = cnt.most_common(10)
json.dump(tally, open(f'{OUT}/proxy_tally.json', 'w'), indent=1)
pa = n[(n.fair == 'total') & (n.lam == 0)].pivot(index='seed', columns='variant', values='te_fdp')
for v in n.variant.unique():
    x0 = n[(n.variant == v) & (n.lam == 0)].set_index('seed').te_fdp; x1 = n[(n.variant == v) & (n.fair == 'rule') & (n.lam == 100)].set_index('seed').te_fdp
    tally[f'{v}|retained'] = [float((x1 / x0).mean()), float(stats.wilcoxon(x0 - x1).pvalue)]
json.dump(tally, open(f'{OUT}/proxy_tally.json', 'w'), indent=1)

# ---------------- membership specification (R3.4)
rp = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/ramp_*.csv')])
specs = ['crisp<=25', 'ramp25-30', 'ramp25-35', 'ramp25-45']; sn = {'crisp<=25': 'crisp ≤25', 'ramp25-30': '25–30', 'ramp25-35': '25–35', 'ramp25-45': '25–45', 'none': 'none (λ=0)'}
rows = ['| **Dataset** | **Model** | **Trained against** | **Accuracy** | ' + ' | '.join(f'**gap: {sn[s]}**' for s in specs) + ' |', '|---|---|---|---|' + '---|' * len(specs)]
for ds in ['Bank', 'German']:
    for mdl in ['DNN', 'ZadehNet']:
        for ts in ['none'] + specs:
            g = rp[(rp.dataset == ds) & (rp.model == mdl) & (rp.train_spec == ts)]
            rows.append(f'| {ds} ({g.seed.nunique()} splits) | {mdl} | {sn[ts]} | {g.acc.mean():.3f} | ' + ' | '.join(f'{g[f"gap_{s}"].mean():.3f}' for s in specs) + ' |')
w('ramp', '\n'.join(rows))

# ---------------- compactness and attribution fidelity (R3.3, R3.6)
it = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/interp4_*.csv')]); it['dgap'] = (it.gap_compact - it.gap_full).abs(); it['crit'] = 'scaled'
sl = pd.read_csv(f'{T}/selected_per_seed.csv')
d_enc = {'German': 60, 'Student': 54, 'COMPAS': 9, 'Default': 31, 'Bank': 49, 'Adult': 86, 'Diabetes': 221, 'CensusKDD': 469}
rows = ['| **Dataset** | $d$ | **ZadehNet: active rules / non-zero consequent weights** | **ZadehNet, k=3: fidelity / abs. Δgap / contribution corr. / top-4 overlap** | **ZadehNet-0, k=3: fidelity / abs. Δgap / contribution corr. / top-4 overlap** |',
        '|---|---|---|---|---|']
for ds in DS:
    sz = sl[(sl.dataset == ds) & (sl.family == 'ZadehNet')]
    cells = []
    for mdl in ['ZadehNet', 'ZadehNet-0']:
        g = it[(it.dataset == ds) & (it.model == mdl) & (it.crit == 'scaled') & (it.k == 3)]
        cells.append(f'{g.fidelity.mean():.3f} / {g.dgap.mean():.3f} / {g.contrib_corr.mean():.2f} / {g.top4_overlap.mean():.2f}')
    rows.append(f'| {DSN.get(ds, ds)} | {d_enc[ds]} | {sz["active_rules@0.05"].mean():.1f} / {sz["consequent_nnz@0.05"].mean():.1f} | ' + ' | '.join(cells) + ' |')
summ = it[it.k == 3].groupby(['model', 'crit'])[['fidelity', 'dgap', 'contrib_corr', 'top4_overlap']].mean()
rows.append('| **Mean** | | | ' + ' | '.join(
    f'{summ.loc[(m, "scaled")].fidelity:.3f} / {summ.loc[(m, "scaled")].dgap:.3f} / {summ.loc[(m, "scaled")].contrib_corr:.2f} / {summ.loc[(m, "scaled")].top4_overlap:.2f}' for m in ['ZadehNet', 'ZadehNet-0']) + ' |')
w('compact', '\n'.join(rows))
summ.round(4).to_csv(f'{OUT}/compact_summary.csv')

# ---------------- mixture of experts (displacement without fuzzy antecedents)
rows = ['| **Dataset** | **Penalty** | **Accuracy** | **FDP** | $\\mathcal R$ | $\\mathcal W$ |', '|---|---|---|---|---|---|']
ml = {('total', 0): 'none', ('rule', 100): 'rule-local, λ=100', ('total', 10): 'total, λ=10', ('total', 100): 'total, λ=100'}
for ds in ['COMPAS', 'Adult']:
    mo = pd.read_csv(f'{R}/moe_{ds}.csv')
    for k in [('total', 0), ('rule', 100), ('total', 10), ('total', 100)]:
        g = mo[(mo.fair == k[0]) & (mo.lam == k[1])]
        rows.append(f'| {ds} ({g.seed.nunique()} splits) | {ml[k]} | {g.te_acc.mean():.3f} ± {g.te_acc.std():.3f} | {g.te_fdp.mean():.3f} ± {g.te_fdp.std():.3f} | {g.routing.mean():+.3f} | {g.within.mean():+.3f} |')
w('moe', '\n'.join(rows))


# ---------------- round 4: theory-implementation alignment (Corollary 5)
al = pd.read_csv(f'{T}/r4_align.csv')
rows = ['| **Scenario** | $L$ | **Protocol** | **Retained score gap** | **Retained decision gap** | $\\Psi(f\')$ | $\\delta$ | **Certificate** | **Deviation** |', '|---|---|---|---|---|---|---|---|---|']
for _, r in al.iterrows():
    rows.append(f'| {r.scenario} | {int(r.L)} | {r.protocol} | {r.retained:.2f} ± {r.retained_sd:.2f} | {r.dec_ret:.2f} | {r.reg_ref:.4f} | {r.delta_opt:+.4f} | {r.cert_gap:.3f} | {r.dev:.4f} |')
w('align', '\n'.join(rows))

# ---------------- round 4: attribution stability
st = pd.read_csv(f'{T}/r4_stability.csv')
GER = {'hpw': 'hours-per-week', 'cg': 'capital-gain', 'cl': 'capital-loss', 'education_num': 'education-num', 'marital_Married-civ-spouse': 'marital-status = married', 'relationship_Husband': 'relationship = husband', 'occupation_Craft-repair': 'occupation = craft-repair', 'occupation_Other-service': 'occupation = other-service', 'famrel': 'family relations', 'freetime': 'free time', 'goout': 'going out', 'Walc': 'weekend alcohol', 'Medu': 'mother education', 'traveltime': 'travel time', 'A2': 'duration', 'A5': 'credit amount', 'A8': 'instalment rate', 'A11': 'residence since', 'A16': 'existing credits', 'A18': 'dependants'}
def gname(t): return ', '.join(GER.get(x.strip(), x.strip()) for x in t.split(',')).replace('_', ' ')
rows = ['| **Dataset** | $d$ | **λ** | **ZadehNet: across splits (Spearman / top-5)** | **ZadehNet: across L (Spearman / top-5)** | **Fair tree: across splits (Spearman / top-5)** | **Chance top-5** | **Most important features (ZadehNet, L = 16)** |', '|---|---|---|---|---|---|---|---|']
for _, r in st.iterrows():
    rows.append(f'| {DSN.get(r.dataset, r.dataset)} | {int(r.d)} | {int(r.lam)} | {r.seeds_spearman:.2f} / {r.seeds_top5:.2f} | {r.L_spearman:.2f} / {r.L_top5:.2f} | '
                f'{r.tree_seeds_spearman:.2f} / {r.tree_seeds_top5:.2f} | {r.chance_top5:.2f} | {gname(r.top_features)} |')
w('stability', '\n'.join(rows))
cp = pd.read_csv(f'{T}/r4_compaction.csv').set_index('method')
json.dump({k: {c: float(v) for c, v in cp.loc[k].items()} for k in cp.index}, open(f'{OUT}/compaction_alt.json', 'w'), indent=1)

# ---------------- round 4: membership shape (Appendix B.3)
specs2 = ['crisp<=25', 'linear25-35', 'zspline25-35', 'sigmoid30', 'linear18-35']
sn2 = {'crisp<=25': 'crisp ≤25', 'linear25-35': 'linear 25–35', 'zspline25-35': 'Z-spline 25–35', 'sigmoid30': 'sigmoid (30, 2)', 'linear18-35': 'linear 18–35', 'none': 'none (λ=0)'}
rows = ['| **Dataset** | **Model** | **Trained against** | **Accuracy** | ' + ' | '.join(f'**gap: {sn2[x]}**' for x in specs2) + ' |', '|---|---|---|---|' + '---|' * len(specs2)]
for ds in ['Bank', 'German']:
    r2 = pd.read_csv(f'{R}/ramp2_{ds}.csv')
    for mdl in ['DNN', 'ZadehNet']:
        for ts in ['none'] + specs2:
            g = r2[(r2.model == mdl) & (r2.train_spec == ts)]
            rows.append(f'| {ds} ({g.seed.nunique()} splits) | {mdl} | {sn2[ts]} | {g.acc.mean():.3f} | ' + ' | '.join(f'{g[f"gap_{x}"].mean():.3f}' for x in specs2) + ' |')
w('B3', '\n'.join(rows))

# ---------------- round 4: calibration and bootstrap (Appendix B.4)
cb = pd.concat([pd.read_csv(x) for x in glob.glob(f'{R}/calboot_*.csv')])
lab4 = {('DNN', 'total', 0): 'DNN', ('DNN', 'total', 10): 'DNN+FDP, λ=10', ('ZadehNet', 'total', 0): 'ZadehNet, λ=0', ('ZadehNet', 'total', 10): 'ZadehNet, λ=10', ('ZadehNet', 'rule', 100): 'ZadehNet-rule, λ=100'}
rows = ['| **Dataset** | **Model** | **Acc.** | **FDP** | **ECE** | **ECE group / rest** | **Brier** | **Bootstrap s.d.: gap / $\\mathcal R$ / $\\mathcal W$** |', '|---|---|---|---|---|---|---|---|']
for ds in ['German', 'COMPAS', 'Bank', 'Adult']:
    for k, nm in lab4.items():
        g = cb[(cb.dataset == ds) & (cb.model == k[0]) & (cb.fair == k[1]) & (cb.lam == k[2])]
        bs = '—' if k[0] == 'DNN' else f'{g.gap_sd.mean():.3f} / {g.R_sd.mean():.3f} / {g.W_sd.mean():.3f}'
        rows.append(f'| {ds} ({g.seed.nunique()} splits) | {nm} | {g.te_acc.mean():.3f} | {g.te_fdp.mean():.3f} | {g.ece.mean():.3f} | {g.ece_G.mean():.3f} / {g.ece_N.mean():.3f} | {g.brier.mean():.3f} | {bs} |')
w('B4', '\n'.join(rows))


# ---------------- round 4: interpretable fair baselines
if os.path.exists(f'{T}/r4_interp.csv'):
    ii = pd.read_csv(f'{T}/r4_interp.csv')
    rows = ['| **Dataset** | **ZadehNet: acc / FDP** | **HV (test / val.)** | **Fair tree: acc / FDP** | **HV (test / val.)** | **leaves** | **FairCORELS: acc / FDP** | **HV (test / val.)** | **rules** |', '|---|---|---|---|---|---|---|---|---|']
    for ds in DS:
        c = []
        for f in ['ZadehNet', 'FairTree', 'FairCORELS']:
            g = ii[(ii.dataset == ds) & (ii.family == f)]
            if g.empty: c += ['—', '—'] + ([] if f == 'ZadehNet' else ['—']); continue
            g = g.iloc[0]; c += [f'{g.acc:.3f} / {g.fdp:.3f}', f'{g.hv:.3f} / {g.hv_val:.3f}'] + ([] if f == 'ZadehNet' else [f'{g["size"]:.1f}'])
        rows.append(f'| {DSN.get(ds, ds)} | ' + ' | '.join(c) + ' |')
    w('interp', '\n'.join(rows))

# ---------------- scaling (three decimals; fixes the R1.5 inconsistency)
sc = pd.read_csv(f'{R}/scaling.csv'); s = sc[sc.exp == 'n'].pivot(index='n', columns='model', values='epoch_time')
rows = ['| $n$ | **DNN** | **ZadehNet L=16** | **ZadehNet L=64** |', '|---|---|---|---|']
for nn, r in s.iterrows(): rows.append(f'| {nn:,} | {r["DNN"]:.3f} | {r["ZadehNet L=16"]:.3f} | {r["ZadehNet L=64"]:.3f} |')
w('scaling', '\n'.join(rows))
print('tables written to', OUT)
