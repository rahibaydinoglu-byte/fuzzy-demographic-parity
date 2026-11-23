"""Round 6 (fifth external review). Adult and COMPAS, 10 splits; first-order FairTSK without penalty and with the
rule-local penalty (lambda = 100), and zero-order FairTSK-0 without penalty (standard protocol of run.py).
Reports (i) test log-loss and accuracy of the zero- and first-order models (a necessary sign that eps_0 is small);
(ii) the mean consequent norm ||a_l||_2 (weighted by rule usage) with and without the rule-local penalty;
(iii) the Theorem 1 split with reference weights G, notG and the midpoint (Oaxaca index-number sensitivity).
Usage: python round6.py DATASET SEED... -> results/r6_{dataset}.csv"""
import sys, os, numpy as np, pandas as pd, torch
from data import LOADERS, split_and_encode
from models import ZadehNet, train_torch
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
def splits(mu, W, G):
    a, b = mu / mu.mean(), (1 - mu) / (1 - mu.mean())
    mG, mN = (a[:, None] * W).mean(0), (b[:, None] * W).mean(0)
    rG = (a[:, None] * W * G).mean(0) / mG.clamp_min(1e-12); rN = (b[:, None] * W * G).mean(0) / mN.clamp_min(1e-12)
    out = {}   # twofold splits: midpoint weights, or the rates of notG / G as reference (Oaxaca index-number choice)
    out['R_mid'] = float(((mG - mN) * (rG + rN) / 2).sum()); out['W_mid'] = float((((mG + mN) / 2) * (rG - rN)).sum())
    out['R_refN'] = float(((mG - mN) * rN).sum()); out['W_refN'] = float((mG * (rG - rN)).sum())
    out['R_refG'] = float(((mG - mN) * rG).sum()); out['W_refG'] = float((mN * (rG - rN)).sum())
    return out
def logloss(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
name = sys.argv[1]; X, y, s, mu, z, groups, meta = LOADERS[name](); rows = []
fn = f'{OUT}/r6_{name}.csv'
for seed in [int(a) for a in sys.argv[2:]]:
    sp, cols = split_and_encode(X, y, s, mu, groups, seed)
    (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
    d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
    init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
    sd = Xtr.std(0)
    for tag, zo, fair, lam in [('first-order', False, 'total', 0), ('rule-local', False, 'rule', 100), ('zero-order', True, 'total', 0)]:
        m = ZadehNet(d, L=16, Xinit=init, seed=seed, zero_order=zo)
        train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=batch, seed=seed)
        with torch.no_grad(): p, W, G = m(torch.as_tensor(Xte), return_parts=True)
        r = dict(dataset=name, seed=seed, model=tag, logloss=logloss(p.numpy(), yte), acc=float(((p.numpy() >= .5) == yte).mean()))
        if not zo:
            usage = W.mean(0).numpy(); A = m.A.detach().numpy() * sd[None]
            r['cons_norm'] = float((usage * np.linalg.norm(A, axis=1)).sum() / usage.sum())
        r.update(splits(torch.as_tensor(mute, dtype=torch.float32), W, G)); rows.append(r)
    pd.DataFrame(rows).to_csv(fn, index=False); print('r6', name, seed, flush=True)
