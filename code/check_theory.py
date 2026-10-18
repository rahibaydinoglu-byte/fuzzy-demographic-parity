"""Numerical check of Proposition 2 (covariance form), Theorem 1 (exact rule-level
decomposition), Corollary 1 (TV bound) and Proposition 3 (Lipschitz in threshold)."""
import numpy as np
rng = np.random.default_rng(0)

def fdp_terms(mu, Wbar, G):
    """mu: (n,) fuzzy membership; Wbar: (n,L) normalized firing; G: (n,L) rule outputs in [0,1]."""
    pi = mu.mean(); a, b = mu/pi, (1-mu)/(1-pi)           # fuzzy group weights
    yhat = (Wbar*G).sum(1)
    delta = (a*yhat).mean() - (b*yhat).mean()
    mG, mN = (a[:,None]*Wbar).mean(0), (b[:,None]*Wbar).mean(0)          # rule coverage per group
    rG = (a[:,None]*Wbar*G).mean(0)/mG; rN = (b[:,None]*Wbar*G).mean(0)/mN  # within-rule rates
    routing = (mG-mN)*(rG+rN)/2; within = (mG+mN)/2*(rG-rN)
    cov_form = np.cov(mu, yhat, bias=True)[0,1]/(pi*(1-pi))
    rbar = (rG+rN)/2
    bound = 0.5*np.abs(mG-mN).sum()*(rbar.max()-rbar.min()) + np.abs(rG-rN).max()
    return delta, cov_form, routing.sum()+within.sum(), bound

worst_id, worst_cov, viol = 0, 0, 0
for trial in range(2000):
    n, L = rng.integers(50, 3000), rng.integers(2, 40)
    z = rng.normal(size=n)
    mu = 1/(1+np.exp(-rng.normal(0,3)*(z-rng.normal())))                 # graded membership
    logits = rng.normal(size=(n,L))*rng.uniform(.2,5) + np.outer(z, rng.normal(size=L))
    Wbar = np.exp(logits-logits.max(1,keepdims=True)); Wbar /= Wbar.sum(1,keepdims=True)
    G = 1/(1+np.exp(-(rng.normal(size=(n,L)) + np.outer(z, rng.normal(size=L)))))
    d, c, dec, bnd = fdp_terms(mu, Wbar, G)
    worst_id = max(worst_id, abs(d-dec)); worst_cov = max(worst_cov, abs(d-c)); viol += abs(d) > bnd + 1e-12
print(f"Theorem 1 max |Delta - (routing+within)| = {worst_id:.2e}")
print(f"Prop 2   max |Delta - Cov/(pi(1-pi))|     = {worst_cov:.2e}")
print(f"Corollary 1 bound violations: {viol} / 2000")

# Proposition 3: crisp DP vs fuzzy DP as the age threshold t moves (fixed predictor)
n = 20000; age = rng.integers(18, 70, n).astype(float)
yhat = 1/(1+np.exp(-(0.08*(age-40) + rng.normal(0,1,n))))
def dp(mu): pi=mu.mean(); return np.cov(mu,yhat,bias=True)[0,1]/(pi*(1-pi))
ts = np.arange(22, 36, 0.1)
crisp = np.array([dp((age<=t).astype(float)) for t in ts])
fuzzy = np.array([dp(np.clip((t+5-age)/10, 0, 1)) for t in ts])          # ramp of width 10 centred at t
print(f"Prop 3   max jump per 0.1y step: crisp={np.abs(np.diff(crisp)).max():.4f}, fuzzy={np.abs(np.diff(fuzzy)).max():.4f}")
