"""Figures for the manuscript (300 dpi PNG). Palette: dataviz reference categorical slots, fixed order."""
import glob, json, os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), '..', 'results'); F = os.path.join(os.path.dirname(__file__), '..', 'figures'); os.makedirs(F, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3dd'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.6, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
DATASETS = ['German', 'Student', 'COMPAS', 'Default', 'Bank', 'Adult', 'Diabetes', 'CensusKDD']
LABEL = {'CensusKDD': 'Census-KDD'}

def pareto_fig():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/main_*.csv')], ignore_index=True)
    df['key'] = df.cfg
    fams = [('ZadehNet', ['ZadehNet-noFair', 'ZadehNet'], C[0], 'o'), ('ZadehNet-0', ['ZadehNet0-noFair', 'ZadehNet0'], C[2], 's'),
            ('DNN+FDP', ['DNN', 'DNN+FDP'], C[1], '^'), ('AdvDebias', ['AdvDebias'], C[6], 'v'), ('ExpGrad', ['ExpGrad'], C[3], 'D')]
    singles = [('LightGBM', C[7], 'X'), ('Reweighing', C[4], 'P'), ('ThreshOpt', C[5], '*')]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for ax, ds in zip(axes.ravel(), DATASETS):
        g = df[df.dataset == ds]
        for name, members, col, mk in fams:
            m = g[g.method.isin(members)].groupby('key')[['te_acc', 'te_fdp']].mean().sort_values('te_fdp')
            # non-dominated subset of the mean points, drawn as a step line
            pts, best = [], -1
            for _, r in m.iterrows():
                if r.te_acc > best: pts.append((r.te_fdp, r.te_acc)); best = r.te_acc
            ax.scatter(m.te_fdp, m.te_acc, s=10, color=col, marker=mk, alpha=0.35, linewidths=0)
            if pts: ax.plot(*zip(*pts), color=col, marker=mk, ms=3.5, label=name, drawstyle='steps-post' if False else 'default')
        for name, col, mk in singles:
            m = g[g.method == name][['te_acc', 'te_fdp']].mean()
            ax.scatter(m.te_fdp, m.te_acc, s=22, color=col, marker=mk, label=name, zorder=5, edgecolors='white', linewidths=0.5)
        ax.set_title(LABEL.get(ds, ds), fontsize=8.5, color=INK)
        ax.tick_params(labelsize=7)
    for ax in axes[1]: ax.set_xlabel('Test FDP (lower is fairer)')
    for ax in axes[:, 0]: ax.set_ylabel('Test accuracy')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=8, fontsize=7, bbox_to_anchor=(0.5, -0.04), handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{F}/fig_pareto.png'); plt.close(fig)

def threshold_fig():
    files = sorted(glob.glob(f'{R}/threshold_*.csv'))
    if not files: return
    allt = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ts = np.arange(20, 40.5, 0.5)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, ds in zip(axes, ['German', 'Bank', 'Adult']):
        sub = allt[(allt.dataset == ds) & (allt.model == 'ZadehNet') & (allt.group == 'none')]
        z = __import__('data').LOADERS[ds]()[4]; frac = np.array([(z <= t).mean() for t in ts]); valid = (frac >= .05) & (frac <= .95)
        for i, (_, r) in enumerate(sub.iterrows()):
            c = np.array(json.loads(r.curve)); f = np.array(json.loads(r.fcurve))
            ax.plot(ts[valid], c[valid], color=C[1], lw=1.0, alpha=0.9, drawstyle='steps-post', label='Crisp group {age ≤ t}' if i == 0 else None)
            ax.plot(ts[valid], f[valid], color=C[0], lw=1.2, alpha=0.9, label='Fuzzy group (ramp of width 10 centred at t)' if i == 0 else None)
        ax.set_title(ds, fontsize=8.5); ax.set_xlabel('Threshold / centre t (years)')
    axes[0].set_ylabel('Parity gap of the same decisions')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{F}/fig_threshold.png'); plt.close(fig)

def displacement_fig():
    D = pd.read_csv(f'{R}/tables/decomposition.csv')
    tags = [('ZadehNet-noFair λ=0', 'Unconstrained'), ('ZadehNet-rule λ=30', 'Rule-local, λ=30'),
            ('ZadehNet-rule λ=100', 'Rule-local, λ=100'), ('ZadehNet λ=10', 'Total FDP, λ=10')]
    dss = [d for d in DATASETS if d in D.dataset.unique()]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=False)
    for ax, ds in zip(axes.ravel(), dss):
        g = D[D.dataset == ds].set_index('tag')
        x = np.arange(len(tags))
        R_ = [g.loc[t, 'routing'] if t in g.index else np.nan for t, _ in tags]
        W_ = [g.loc[t, 'within'] if t in g.index else np.nan for t, _ in tags]
        ax.bar(x - 0.18, R_, 0.34, color=C[0], label='Routing $\\mathcal{R}$')
        ax.bar(x + 0.18, W_, 0.34, color=C[1], label='Within-rule $\\mathcal{W}$')
        ax.scatter(x, np.array(R_) + np.array(W_), color=INK, marker='_', s=120, lw=1.6, zorder=5, label='Total gap $\\Delta$')
        ax.axhline(0, color=MUTED, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(['Unc.', 'Rule\n30', 'Rule\n100', 'Total\n10'], fontsize=6.5)
        ax.set_title(LABEL.get(ds, ds), fontsize=8.5)
    for ax in axes[:, 0]: ax.set_ylabel('Signed contribution\nto parity gap')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f'{F}/fig_displacement.png'); plt.close(fig)

def audit_fig(name='Adult', lam='0.0'):
    A = pd.read_csv(f'{R}/audit_{name}_lam{lam}.csv'); A = A[A.model == 'zero-order'].copy()
    A = A.reindex(A.contribution.abs().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(A))[::-1]
    ax.barh(y, A.routing, color=C[0], height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([f"R{int(r)}: P(Y=1)={c:.2f}" for r, c in zip(A.rule, A.consequent)], fontsize=6.5)
    ax.axvline(0, color=MUTED, lw=0.6); ax.set_xlabel('Contribution of rule to parity gap Δ (zero-order: routing only)')
    tot = A.routing.sum()
    ax.set_title(f'{name}: Σ rule contributions = {tot:+.3f}  (directly measured gap {A.direct_gap.iloc[0]:+.3f})', fontsize=8)
    fig.tight_layout(); fig.savefig(f'{F}/fig_audit_{name}.png'); plt.close(fig)

def scaling_fig():
    S = pd.read_csv(f'{R}/scaling.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for i, (m, col) in enumerate([('ZadehNet L=16', C[0]), ('ZadehNet L=64', C[2]), ('DNN', C[1])]):
        s = S[(S.exp == 'n') & (S.model == m)]
        axes[0].errorbar(s.n, s.epoch_time, yerr=s.epoch_std, color=col, marker='o', ms=3.5, label=m, capsize=2)
        s = S[(S.exp == 'd') & (S.model == m)]
        if len(s): axes[1].errorbar(s.d, s.epoch_time, yerr=s.epoch_std, color=col, marker='o', ms=3.5, label=m, capsize=2)
    for ax, xl in zip(axes, ['Training instances n (Census-KDD, d=469)', 'Input dimension d (synthetic, n=20,000)']):
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(xl); ax.set_ylabel('Seconds per epoch (1 CPU thread)')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator, FuncFormatter
    axes[1].xaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400, 800])); axes[1].xaxis.set_major_formatter(FixedFormatter(['25', '50', '100', '200', '400', '800']))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[0].xaxis.set_major_locator(FixedLocator([1e4, 3e4, 1e5, 1.8e5])); axes[0].xaxis.set_major_formatter(FixedFormatter(['10k', '30k', '100k', '180k'])); axes[0].xaxis.set_minor_locator(NullLocator())
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{F}/fig_scaling.png'); plt.close(fig)

def cd_fig():
    s = json.load(open(f'{R}/tables/summary.json'))['friedman_hv']
    ranks = sorted(s['ranks'].items(), key=lambda t: t[1]); k = len(ranks); cd = s['CD']
    fig, ax = plt.subplots(figsize=(7.2, 1.7)); ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.axis('off')
    ax.plot([1, k], [0.8, 0.8], color=INK, lw=0.8)
    for r in range(1, k + 1): ax.plot([r, r], [0.8, 0.84], color=INK, lw=0.8); ax.text(r, 0.88, str(r), ha='center', fontsize=7)
    for i, (n, r) in enumerate(ranks):
        yy = 0.62 - 0.11 * (i if i < (k + 1) // 2 else k - 1 - i)
        left = i < (k + 1) // 2
        ax.plot([r, r], [0.8, yy], color=INK, lw=0.6); ax.plot([r, 1 if left else k], [yy, yy], color=INK, lw=0.6)
        ax.text(0.97 if left else k + 0.03, yy, f'{n} ({r:.2f})', ha='right' if left else 'left', va='center', fontsize=7)
    ax.plot([1, 1 + cd], [0.97, 0.97], color=C[7], lw=1.6); ax.text(1 + cd / 2, 1.0, f'CD = {cd:.2f}', ha='center', fontsize=7, color=INK)
    ax.set_title(f"Mean ranks by hypervolume over {s['N']} datasets (Friedman p = {s['p']:.3g})", fontsize=8, pad=14)
    fig.savefig(f'{F}/fig_cd.png'); plt.close(fig)

if __name__ == '__main__':
    for fn in sys.argv[1:] or ['pareto', 'threshold', 'displacement', 'audit', 'scaling', 'cd']:
        try: globals()[f'{fn}_fig']()
        except Exception as e: print(fn, 'failed:', repr(e))
