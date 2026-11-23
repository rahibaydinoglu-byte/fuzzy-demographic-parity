"""Round 6: fill the gap 0.03 < eps0 < 0.13 of the synthetic study. rho = 0.6, same data generation and protocol as
synth.py; frozen antecedents with L in {32, 64, 128} (more fixed rules = finer fixed partition) and trainable L = 3.
For each configuration: unconstrained (rule, 0), rule-local (rule, 100) and the zero-order reference with the same
antecedent mode. Usage: python synth_gap.py linear|nonlinear SEED... -> results/synthgap_{scen}_{seed}.csv"""
import sys, os, numpy as np, pandas as pd
import synth
OUT = synth.OUT
scen = sys.argv[1]; synth.SCEN = scen
for seed in [int(s) for s in sys.argv[2:]]:
    fn = f'{OUT}/synthgap_{scen}_{seed}.csv'
    rows = pd.read_csv(fn).to_dict('records') if os.path.exists(fn) else []
    done = {(r['L'], r['mode'], r['fair'], r['lam']) for r in rows}
    rng = np.random.default_rng(1000 * seed + 6)          # identical to synth.py for rho = 0.6
    data = [synth.make(n, 0.6, rng) for n in (20000, 10000, 50000)]
    cfgs = [(L, 'fixed', 'rule', lam) for L in (32, 64, 128) for lam in (0, 100)] + [(L, 'fixedzero', 'rule', 0) for L in (32, 64, 128)]
    cfgs += [(3, 'train', 'rule', 0), (3, 'train', 'rule', 100), (3, 'zero', 'rule', 0)]
    cfgs += [(L, 'fixedc', 'rule', lam) for L in (4, 16) for lam in (0, 100)] + [(L, 'fixedczero', 'rule', 0) for L in (4, 16)]
    for L, mode, fair, lam in cfgs:
        if (L, mode, fair, lam) in done: continue
        rows.append(synth.run(seed, 0.6, L, mode, fair, lam, data)); pd.DataFrame(rows).to_csv(fn, index=False)
        r = rows[-1]; print(scen, seed, L, mode, lam, round(r['delta'], 3), round(r['kl'], 4), flush=True)
