"""Numerical check of Lemma 1, Proposition 3 and Corollary 2 of the article (Lemma 2, Proposition 5 and Corollary 4 in the numbering of an earlier draft) on random empirical distributions.
All statements hold exactly for the empirical distribution of a sample, so violations would indicate an error."""
import numpy as np, torch
from models import ZadehNet, rule_terms
rng = np.random.default_rng(0); torch.manual_seed(0)
worst = dict(lemA=-1e9, prop5a=-1e9, prop5b=-1e9, zo=0.0, L1=0.0, kappa=0.0)
for t in range(2000):
    n, d, L = int(rng.integers(50, 400)), int(rng.integers(1, 6)), int(rng.integers(1, 9))
    X = torch.as_tensor(rng.standard_normal((n, d)), dtype=torch.float32)
    crisp = rng.random() < 0.3
    mu = rng.random(n) < 0.4 if crisp else rng.random(n) ** rng.uniform(0.3, 3)
    mu = torch.as_tensor(mu, dtype=torch.float64)
    if mu.mean() in (0, 1): continue
    m = ZadehNet(d, L=L, seed=t, zero_order=bool(rng.random() < 0.3)).double()
    with torch.no_grad():
        for p in m.parameters(): p.add_(torch.randn_like(p) * rng.uniform(0.1, 2))
        p, W, G = m(X.double(), return_parts=True)
    R, Wl, mG, mN, rG, rN = rule_terms(mu, W, G)
    omega = float((((mG + mN) / 2) * (rG - rN) ** 2).sum())
    worst['lemA'] = max(worst['lemA'], abs(float(Wl.sum())) - np.sqrt(omega))
    if m.zero_order and float(torch.minimum(mG, mN).min()) > 1e-9: worst['zo'] = max(worst['zo'], omega)
    if m.zero_order and float(torch.minimum(mG, mN).min()) <= 1e-9: worst.setdefault('zo_underflow_cases', 0); worst['zo_underflow_cases'] += 1
    pi = float(mu.mean()); delta = lambda h: float(((mu - pi) * (h - h.mean())).mean()) / (pi * (1 - pi))
    if L == 1: worst['L1'] = max(worst['L1'], abs(omega - delta(p) ** 2), abs(float(R.sum())))
    eta = torch.as_tensor(rng.uniform(0.02, 0.98, n))   # an arbitrary 'Bayes' score on this sample
    pp = p.clamp(1e-6, 1 - 1e-6)
    kl = float((eta * torch.log(eta / pp) + (1 - eta) * torch.log((1 - eta) / (1 - pp))).mean())
    var_mu = float(mu.var(unbiased=False)); mse = float(((p - eta) ** 2).mean())
    lhs = abs(delta(p) - delta(eta))
    worst['prop5a'] = max(worst['prop5a'], lhs - np.sqrt(var_mu * mse) / (pi * (1 - pi)))
    kappa = np.sqrt(var_mu / (pi * (1 - pi)))
    worst['prop5b'] = max(worst['prop5b'], lhs - kappa * np.sqrt(kl / (2 * pi * (1 - pi))))
    worst['kappa'] = max(worst['kappa'], kappa - 1)
print({k: f'{v:.2e}' for k, v in worst.items()}, '(all must be <= ~1e-12)')
