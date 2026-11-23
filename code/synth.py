"""Synthetic study of disparity displacement (revision, R1.2 / R3.5).

Data: x = (z1, z2, z3) ~ N(0, I); y | x ~ Bernoulli(eta(x)), eta(x) = sigmoid(b0 + b1 z1 + b2 z2 + b3 z3), so the Bayes
score eta is known exactly. A latent age a = 40 + 10 (rho z1 + sqrt(1-rho^2) e) is NOT an input; the protected group is
the fuzzy set young, mu = clip((35 - a)/10, 0, 1). rho controls how much group information the inputs carry (z1 is a
proxy). Routing capacity is varied through the number of rules L and through fixed vs trainable antecedents.

For each run we record, on an independent test sample: score-level parity gap Delta (Prop. 2), its routing/within split
(Theorem 1), the excess risk KL = E_x KL(Ber(eta) || Ber(f)) and the Lemma bound kappa * sqrt(E(f-eta)^2 / (pi(1-pi))).
Usage: python synth.py SEED [SEED ...] -> results/synth_SEED.csv
"""
import sys, os, json, numpy as np, torch
from models import ZadehNet, train_torch, predict, rule_terms
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results'); os.makedirs(OUT, exist_ok=True)
B = np.array([-0.5, 1.5, 1.0, -1.0])          # b0, b1, b2, b3

SCEN = 'linear'                              # 'linear': logit linear in x; 'nonlinear': oscillating, interacting logit

def make(n, rho, rng):
    Z = rng.standard_normal((n, 3))
    a = 40 + 10 * (rho * Z[:, 0] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n))
    mu = np.clip((35 - a) / 10, 0, 1)
    if SCEN == 'linear': logit = B[0] + Z @ B[1:]
    else: logit = -0.5 + 1.5 * Z[:, 0] + 2.0 * np.sin(2.5 * Z[:, 1]) + 1.5 * Z[:, 1] * Z[:, 2]
    eta = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < eta).astype(float)
    return Z.astype(np.float32), y, mu, eta

def gap(mu, p):
    pi = mu.mean(); return float(np.mean((mu - pi) * (p - p.mean())) / (pi * (1 - pi)))

def kl_bern(eta, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(eta * np.log(eta / p) + (1 - eta) * np.log((1 - eta) / (1 - p))))

def run(seed, rho, L, mode, fair, lam, data):
    (Xtr, ytr, mutr, _), (Xva, yva, muva, _), (Xte, yte, mute, etate) = data
    init = Xtr[np.random.default_rng(seed).choice(len(Xtr), 5000, replace=False)]
    zo = mode in ('zero', 'fixedzero', 'fixedczero')
    m = ZadehNet(3, L=L, Xinit=init, seed=seed, zero_order=zo)
    if mode in ('fixed', 'fixedzero'):                      # antecedents frozen at their (k-means++) initialisation
        for prm in (m.c, m.log_s, m.rho): prm.requires_grad_(False)
    if mode in ('fixedc', 'fixedczero'):                    # round 6: centres frozen, widths and gates trainable
        m.c.requires_grad_(False)
    info = train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lr=1e-2, batch=512, seed=seed,
                       lam_a=1e-2, lam_g=1e-2)
    with torch.no_grad():
        X = torch.as_tensor(Xte); p, W, G = m(X, return_parts=True)
        R, Wl, mG, mN, rG, rN = rule_terms(torch.as_tensor(mute, dtype=torch.float32), W, G)
    p = p.numpy(); pi = mute.mean(); kappa = np.sqrt(mute.var() / (pi * (1 - pi)))
    mse = float(np.mean((p - etate) ** 2)); kl = kl_bern(etate, p)
    return dict(seed=seed, rho=rho, L=L, mode=mode, fair=fair, lam=lam, delta=gap(mute, p), delta_bayes=gap(mute, etate),
                routing=float(R.sum()), within=float(Wl.sum()), within_abs=float(Wl.abs().sum()),
                omega=float((((mG + mN) / 2) * (rG - rN) ** 2).sum()), kl=kl, mse=mse,
                bound_lemma=float(np.sqrt(mute.var() * mse) / (pi * (1 - pi))),
                bound_kl=float(kappa * np.sqrt(kl / (2 * pi * (1 - pi)))), kappa=float(kappa), pi=float(pi),
                tv_route=float(0.5 * (mG - mN).abs().sum()),
                acc=float(((p >= .5) == yte).mean()), acc_bayes=float(((etate >= .5) == yte).mean()),
                dp_dec=gap(mute, (p >= .5).astype(float)), dp_dec_bayes=gap(mute, (etate >= .5).astype(float)),
                epochs=info['epochs'])

def main(seeds, extra=False):
    import pandas as pd
    for seed in seeds:
        fn = f'{OUT}/synth_{seed}.csv' if SCEN == 'linear' else f'{OUT}/synth2_{seed}.csv'
        rows = pd.read_csv(fn).to_dict('records') if os.path.exists(fn) else []
        done = {(r['rho'], r['L'], r['mode'], r['fair'], r['lam']) for r in rows}
        for rho in ((0.0, 0.3, 0.6, 0.9) if SCEN == 'linear' and not extra else (0.6,)):
            rng = np.random.default_rng(1000 * seed + int(10 * rho))
            data = [make(n, rho, rng) for n in (20000, 10000, 50000)]
            cfgs = [(L, 'train', f, lam) for L in (1, 2, 4, 8, 16, 32) for f, lam in (('rule', 0), ('rule', 100), ('total', 100))]
            cfgs += [(L, 'fixed', f, lam) for L in (4, 16) for f, lam in (('rule', 0), ('rule', 100))]
            cfgs += [(L, 'zero', 'rule', 0) for L in (1, 2, 4, 8, 16, 32)]
            if extra or SCEN != 'linear':          # added after the first run: zero-order with frozen antecedents
                cfgs += [(L, 'fixedzero', 'rule', 0) for L in (4, 16)]
            if SCEN != 'linear':
                cfgs = [c for c in cfgs if not (c[1] == 'train' and c[2] == 'total' and c[0] not in (1, 16))]
            for L, mode, fair, lam in cfgs:
                if (rho, L, mode, fair, lam) in done: continue
                rows.append(run(seed, rho, L, mode, fair, lam, data))
                pd.DataFrame(rows).to_csv(fn, index=False)
                r = rows[-1]; print(seed, rho, L, mode, fair, lam, round(r['delta'], 3), round(r['delta_bayes'], 3),
                                    round(r['within'], 3), round(r['kl'], 4), flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] in ('linear', 'nonlinear', 'extra'):
        mode = args.pop(0)
        SCEN = 'nonlinear' if mode == 'nonlinear' else 'linear'
        main([int(s) for s in args], extra=(mode == 'extra'))
    else:
        main([int(s) for s in args])
