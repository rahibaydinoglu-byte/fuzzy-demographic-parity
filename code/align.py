"""Theory-implementation alignment for Theorem 2 (review round 4, Critical 1).
  synth SCEN SEED...  : synthetic data (rho = 0.6, trainable antecedents, L in {2,4,8,16}). For two training protocols,
                        'regularised' (the objective (11) with early stopping, as in all experiments) and 'plain'
                        (log-loss + rule-local penalty only, fixed 100 epochs, last iterate), train the rule-local model
                        f (lambda = 100), the unconstrained model, and a zero-order reference f' (Omega(f') = 0).
                        On the test inputs, with eta known, compute the a-posteriori certificate of Corollary 5:
                            eps(f) <= eps(f') + R(f') + delta_opt,  delta_opt = J(f) - J(f'),
                        where J = expected log-loss + lambda * Omega + R, R = the extra regularisers of (11).
  real DATASET SEED...: rule-local lambda = 100 with the plain protocol on real data.
"""
import sys, os, numpy as np, pandas as pd, torch
import synth
from synth import make, gap
from models import ZadehNet, train_torch, rule_terms
from data import LOADERS, split_and_encode
from run import metrics
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
PROTO = {'regularised': dict(lam_b=0.1, lam_a=1e-2, lam_g=1e-2, early_stop=True),
         'plain': dict(lam_b=0.0, lam_a=0.0, lam_g=0.0, early_stop=False)}

def exp_logloss(eta, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(np.mean(-eta * np.log(p) - (1 - eta) * np.log(1 - p)))

def parts(m, X, mu, regs):
    with torch.no_grad():
        p, W, G = m(torch.as_tensor(X), return_parts=True)
        R, Wl, mG, mN, rG, rN = rule_terms(torch.as_tensor(mu, dtype=torch.float32), W, G)
        omega = float((((mG + mN) / 2) * (rG - rN) ** 2).sum())
        reg = float(m.reg(W, regs['lam_b'], regs['lam_a'], regs['lam_g']))
    return p.numpy(), float(R.sum()), float(Wl.sum()), omega, reg

def synth_main(scen, seeds):
    synth.SCEN = scen
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(1000 * seed + 6)
        (Xtr, ytr, mutr, _), (Xva, yva, muva, _), (Xte, yte, mute, eta) = [make(n, 0.6, rng) for n in (20000, 10000, 50000)]
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), 5000, replace=False)]
        pi = mute.mean(); kappa = np.sqrt(mute.var() / (pi * (1 - pi))); L_eta = exp_logloss(eta, eta)
        for proto, kw in PROTO.items():
            regs = {k: kw[k] for k in ('lam_b', 'lam_a', 'lam_g')}
            for L in (2, 4, 8, 16):
                res = {}
                for tag, zo, lam in (('unc', False, 0), ('rule', False, 100), ('ref', True, 0)):
                    m = ZadehNet(3, L=L, Xinit=init, seed=seed, zero_order=zo)
                    train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair='rule', lr=1e-2, batch=512, seed=seed, **kw)
                    res[tag] = parts(m, Xte, mute, regs)
                p, R, W, om, reg = res['rule']; p0, *_ = res['unc']; pr, _, _, om_r, reg_r = res['ref']
                J_f = exp_logloss(eta, p) + 100 * om + reg; J_ref = exp_logloss(eta, pr) + reg_r   # Omega(f') = 0 exactly for a zero-order base (Lemma 1(b)); om_r is round-off
                eps_f = exp_logloss(eta, p) - L_eta; eps_ref = exp_logloss(eta, pr) - L_eta; dopt = J_f - J_ref
                cert = max(eps_ref + reg_r + dopt, 0.0)
                d_b = gap(mute, eta)
                rows.append(dict(scenario=scen, seed=seed, protocol=proto, L=L, delta_unc=gap(mute, p0), delta_rule=gap(mute, p), delta_bayes=d_b,
                                 retained=gap(mute, p) / gap(mute, p0), routing=R, within=W, omega=om, eps_rule=eps_f, eps_ref=eps_ref,
                                 reg_ref=reg_r, omega_ref=om_r, delta_opt=dopt, cert_eps=cert, cert_gap=float(kappa * np.sqrt(cert / (2 * pi * (1 - pi)))),
                                 dev=abs(gap(mute, p) - d_b), dec_unc=gap(mute, (p0 >= .5).astype(float)), dec_rule=gap(mute, (p >= .5).astype(float)),
                                 dec_bayes=gap(mute, (eta >= .5).astype(float))))
                print(scen, seed, proto, L, round(rows[-1]['retained'], 3), round(dopt, 4), flush=True)
        pd.DataFrame(rows).to_csv(f'{OUT}/align_synth_{scen}_' + '-'.join(map(str, seeds)) + '.csv', index=False)

def real_main(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for proto in ('plain',):
            kw = PROTO[proto]
            for lam in (0, 100):
                m = ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair='rule', lr=1e-2, batch=batch, seed=seed, **kw)
                p, R, W, om, _ = parts(m, Xte, mute, {'lam_b': 0, 'lam_a': 0, 'lam_g': 0})
                r = dict(dataset=name, seed=seed, protocol=proto, lam=lam, score_gap=gap(mute, p), routing=R, within=W, omega=om)
                r.update(metrics(p, yte, ste, mute, 'te_')); rows.append(r)
        print('real', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/align_real_{name}.csv', index=False)

if __name__ == '__main__':
    if sys.argv[1] == 'synth': synth_main(sys.argv[2], [int(a) for a in sys.argv[3:]])
    else: real_main(sys.argv[2], [int(a) for a in sys.argv[3:]])
