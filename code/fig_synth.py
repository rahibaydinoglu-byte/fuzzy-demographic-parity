"""Revision figure: share of the parity gap retained under the rule-local penalty vs the within-rule-fair
approximation error proxy eps_0 (excess risk of the zero-order rule base with the same antecedent family)."""
import glob, os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), '..', 'results'); F = os.path.join(os.path.dirname(__file__), '..', 'figures')
rows = []
for scen, pat in [('linear', 'synth_[0-9]*.csv'), ('nonlinear', 'synth2_*.csv')]:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/{pat}')]); d = d[d.rho == 0.6]
    for (mode, zmode) in [('train', 'zero'), ('fixed', 'fixedzero')]:
        for L in sorted(d[d['mode'] == mode].L.unique()):
            base = d[(d['mode'] == mode) & (d.L == L) & (d.fair == 'rule') & (d.lam == 0)].set_index('seed').delta
            pen = d[(d['mode'] == mode) & (d.L == L) & (d.fair == 'rule') & (d.lam == 100)].set_index('seed').delta
            eps = d[(d['mode'] == zmode) & (d.L == L)].kl.mean()
            ret = (pen / base).dropna()
            if len(ret) and not np.isnan(eps):
                rows.append(dict(scenario=scen, antecedents=mode, L=L, eps0=eps, retained=ret.mean(), sd=ret.std(), n=len(ret)))
sg = pd.read_csv(f'{R}/tables/r6_synthgap.csv'); sg = sg[sg['mode'].isin(['fixedc', 'train'])]        # round 6: centres frozen; trainable L = 3
for (scen, md, L), g in sg.groupby(['scen', 'mode', 'L']):
    rows.append(dict(scenario=scen, antecedents=md, L=L, eps0=g.eps0.mean(), retained=g.ret.mean(), sd=g.ret.std(), n=len(g)))
t = pd.DataFrame(rows); t.to_csv(f'{R}/tables/rev_synth_retained_vs_eps.csv', index=False); print(t.round(4).to_string())
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for (scen, mode), g in t.groupby(['scenario', 'antecedents']):
    MK = {'train': ('o', 'C0', 'trainable'), 'fixedc': ('^', 'C2', 'centres frozen'), 'fixed': ('s', 'C3', 'fixed')}[mode]
    ax.errorbar(g.eps0, g.retained, yerr=g.sd, fmt=MK[0], mfc='white' if scen == 'nonlinear' else None,
                color=MK[1], capsize=2, label=f'{scen}, {MK[2]} antecedents')
    for _, r in g.iterrows():
        if r.retained < 0.9 or mode == 'fixedc' or (mode == 'train' and r.L == 3 and scen == 'nonlinear'):
            dx = -30 if (scen == 'linear' and r.L in (1, 16) and mode != 'fixedc') else 5
            ax.annotate(f'L={r.L}', (r.eps0, r.retained), textcoords='offset points', xytext=(dx, -10 if r.L == 1 or mode == 'fixedc' else 3), fontsize=7)
ax.set_xscale('log'); ax.axhline(1, color='grey', lw=0.6, ls='--'); ax.axhline(0, color='grey', lw=0.6)
ax.set_xlabel(r'excess risk of zero-order rule base, $\hat\varepsilon_0$ (nats, log scale)')
ax.set_ylabel(r'retained gap $\Delta_\lambda/\Delta_0$ (rule-local, $\lambda=100$)'); ax.legend(fontsize=7, loc='center left', frameon=False)
ax.text(1e-3, 0.9, 'all $L\\geq2$ (trainable; $L\\geq4$ nonlinear)', fontsize=7)
fig.tight_layout(); fig.savefig(f'{F}/fig_synth_displacement.pdf'); fig.savefig(f'{F}/fig_synth_displacement.png', dpi=200)
