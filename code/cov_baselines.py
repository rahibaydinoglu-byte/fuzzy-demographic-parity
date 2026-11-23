"""Revision (R1.1): continuous-attribute dependence penalties vs fuzzy demographic parity on the age datasets.

Same DNN, optimiser, splits, early stopping and operating-point protocol as run.py; only the fairness penalty differs:
  DNN+Cov(age)   lam * Cov(age_std, p)^2                    linear dependence on raw age (continuous-attribute proxy)
  DNN+HGR(age)   lam * chi2(p, age) (KDE estimate)          any dependence on age; chi2 upper-bounds HGR^2 (Mary et al. 2019)
  LR+Cov(s)      lam * Cov(s, theta'x)^2                    Zafar et al. (2017) decision-boundary covariance, crisp young (age<=25)
  LR+Cov(age)    lam * Cov(age_std, theta'x)^2              Zafar-style covariance with raw age
DNN+FDP (fuzzy 'young') and ZadehNet come from run.py (same seeds/splits).
Usage: python cov_baselines.py DATASET SEED [SEED ...] -> results/cov_DATASET_SEED.csv
"""
import sys, os, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from data import LOADERS, split_and_encode
from models import MLP, predict
from run import metrics
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
LAMS_COV = [1, 3, 10, 30, 100, 300]
LAMS_HGR = [0.3, 1, 3, 10, 30, 100]

def cov(a, b): return ((a - a.mean()) * (b - b.mean())).mean()

def chi2_kde(p, z, K=24):
    """Discretised chi^2 mutual dependence of (p, z) from Gaussian KDEs on a K x K grid of standardised values."""
    u = (p - p.mean()) / (p.std() + 1e-6); v = (z - z.mean()) / (z.std() + 1e-6)
    n = len(p); h = 1.06 * n ** (-1 / 5)
    g = torch.linspace(-3, 3, K)
    Ku = torch.exp(-(g[:, None] - u[None]) ** 2 / (2 * h * h)); Kv = torch.exp(-(g[:, None] - v[None]) ** 2 / (2 * h * h))
    J = Ku @ Kv.T; J = J / J.sum()
    pu, pv = J.sum(1), J.sum(0)
    return (J ** 2 / (pu[:, None] * pv[None] + 1e-12)).sum() - 1

class LR(nn.Module):
    def __init__(self, d, seed=0):
        super().__init__(); torch.manual_seed(seed); self.lin = nn.Linear(d, 1)
    def logit(self, x): return self.lin(x).squeeze(1)
    def forward(self, x, return_parts=False): return torch.sigmoid(self.logit(x))

def logit_of(model, x):
    return model.logit(x) if isinstance(model, LR) else model.net(x).squeeze(1)

def train(model, Xtr, ytr, ztr, Xva, yva, zva, pen, lam, lr, batch, seed, epochs=100, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    Xtr, ytr, ztr, Xva, yva, zva = map(T, (Xtr, ytr, ztr, Xva, yva, zva))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    def obj(X, y, z):
        p = model(X).clamp(1e-6, 1 - 1e-6); loss = F.binary_cross_entropy(p, y)
        if lam > 0 and z.std() > 0:
            if pen == 'cov': loss = loss + lam * cov(z, p) ** 2
            elif pen == 'hgr': loss = loss + lam * chi2_kde(p, z)
            elif pen == 'zafar': loss = loss + lam * cov(z, logit_of(model, X)) ** 2
        return loss
    best, best_state, bad, n = np.inf, None, 0, len(ytr)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 8: continue
            opt.zero_grad(); obj(Xtr[idx], ytr[idx], ztr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = float(obj(Xva, yva, zva))
        if v < best - 1e-5: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    return dict(epochs=ep + 1)

def main(name, seeds):
    X, y, s, mu, age, groups, meta = LOADERS[name]()
    for seed in seeds:
        fn = f'{OUT}/cov_{name}_{seed}.csv'
        if os.path.exists(fn): continue
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, str_, mutr, itr), (Xva, yva, sva, muva, iva), (Xte, yte, ste, mute, ite) = sp['tr'], sp['va'], sp['te']
        am, asd = age[itr].mean(), age[itr].std()
        ztr, zva, zte = [(age[i] - am) / asd for i in (itr, iva, ite)]
        d = Xtr.shape[1]; rows = []
        batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        def rec(method, cfg, pva, pte, info):
            r = dict(dataset=name, seed=seed, method=method, cfg=json.dumps(cfg))
            r.update(metrics(pva, yva, sva, muva, 'va_')); r.update(metrics(pte, yte, ste, mute, 'te_'))
            yh = (pte >= .5).astype(float)
            r['te_cov_age'] = float(np.corrcoef(zte, yh)[0, 1]) if yh.std() > 0 else 0.0   # residual linear age dependence
            r.update(info); rows.append(r)
        for meth, pen, lams, mk, lr, zz in [
                ('DNN+Cov(age)', 'cov', LAMS_COV, lambda: MLP(d, seed=seed), 1e-3, (ztr, zva)),
                ('DNN+HGR(age)', 'hgr', LAMS_HGR, lambda: MLP(d, seed=seed), 1e-3, (ztr, zva)),
                ('LR+Cov(s)', 'zafar', [0] + LAMS_COV, lambda: LR(d, seed=seed), 1e-2, (str_.astype(float), sva.astype(float))),
                ('LR+Cov(age)', 'zafar', LAMS_COV, lambda: LR(d, seed=seed), 1e-2, (ztr, zva))]:
            for lam in lams:
                t = time.time(); m = mk()
                info = train(m, Xtr, ytr, zz[0], Xva, yva, zz[1], pen, lam, lr, batch, seed)
                info['train_time'] = time.time() - t
                rec(meth if lam else 'LR-torch', {'lam': lam}, predict(m, Xva), predict(m, Xte), info)
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(name, seed, 'done', flush=True)

if __name__ == '__main__':
    main(sys.argv[1], [int(a) for a in sys.argv[2:]])
