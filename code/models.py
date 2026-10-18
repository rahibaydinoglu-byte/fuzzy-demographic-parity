"""ZadehNet v2 (fairness-aware first-order TSK on raw features) and neural baselines."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time

# ---------------------------------------------------------------- fairness quantities
def fdp_cov(mu, p):
    """Prop. 2: Delta = Cov(mu, p) / (pi (1-pi)). Differentiable; works for crisp or fuzzy mu."""
    pi = mu.mean().clamp(1e-4, 1 - 1e-4)
    return ((mu - pi) * (p - p.mean())).mean() / (pi * (1 - pi))

def rule_terms(mu, W, G):
    """Theorem 1 quantities on a sample. W: (n,L) normalised firing, G: (n,L) rule outputs. Returns
    routing (L,), within (L,), m_G, m_N, r_G, r_N."""
    a, b = mu / mu.mean(), (1 - mu) / (1 - mu.mean())
    mG, mN = (a[:, None] * W).mean(0), (b[:, None] * W).mean(0)
    rG = (a[:, None] * W * G).mean(0) / mG.clamp_min(1e-12)
    rN = (b[:, None] * W * G).mean(0) / mN.clamp_min(1e-12)
    routing = (mG - mN) * (rG + rN) / 2
    within = (mG + mN) / 2 * (rG - rN)
    return routing, within, mG, mN, rG, rN

# ---------------------------------------------------------------- models
class ZadehNet(nn.Module):
    """First-order TSK. log w_l = (1/d) sum_j gamma_lj * (-(x_j-c_lj)^2 / (2 s_lj^2)); wbar = softmax(log w / tau);
    g_l = sigmoid(a_l.x + b_l); yhat = sum_l wbar_l g_l.  zero_order=True -> g_l = sigmoid(b_l)."""
    def __init__(self, d, L=16, tau=1.0, Xinit=None, gates=True, zero_order=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if Xinit is not None:   # k-means++-style seeding from training points
            X = torch.as_tensor(Xinit); idx = [int(torch.randint(len(X), (1,), generator=g))]
            for _ in range(L - 1):
                dist = torch.cdist(X, X[idx]).min(1).values ** 2
                idx.append(int(torch.multinomial(dist / dist.sum(), 1, generator=g)))
            c = X[idx].clone()
        else:
            c = torch.randn(L, d, generator=g)
        self.c = nn.Parameter(c)
        self.log_s = nn.Parameter(torch.zeros(L, d))
        self.rho = nn.Parameter(torch.full((L, d), 1.0)) if gates else None    # gamma = sigmoid(rho), init 0.73
        self.zero_order = zero_order
        self.A = nn.Parameter(torch.zeros(L, d)) if not zero_order else None
        self.b = nn.Parameter(torch.zeros(L))
        self.tau, self.d, self.L = tau, d, L

    def gamma(self):
        return torch.sigmoid(self.rho) if self.rho is not None else torch.ones_like(self.c)

    def firing(self, x):
        # sum_j gamma_lj (x_j-c_lj)^2 / (2 s_lj^2), expanded into matrix products (O(nLd) via BLAS)
        P = self.gamma() * 0.5 * torch.exp(-2 * self.log_s)                          # (L,d)
        logw = -((x * x) @ P.T - 2 * x @ (P * self.c).T + (P * self.c * self.c).sum(1)[None]) / self.d   # (n,L)
        return torch.softmax(logw / self.tau, dim=1)

    def rule_out(self, x):
        if self.zero_order: return torch.sigmoid(self.b)[None].expand(len(x), -1)
        return torch.sigmoid(x @ self.A.T + self.b)

    def forward(self, x, return_parts=False):
        W, G = self.firing(x), self.rule_out(x)
        p = (W * G).sum(1)
        return (p, W, G) if return_parts else p

    def reg(self, W, lam_b, lam_a, lam_g):
        u = W.mean(0)
        r = lam_b * (u * torch.log(u * self.L + 1e-12)).sum()                       # KL(u || uniform)
        if self.A is not None: r = r + lam_a * self.A.abs().mean()
        if self.rho is not None: r = r + lam_g * self.gamma().mean()
        return r

class MLP(nn.Module):
    def __init__(self, d, h=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, return_parts=False):
        return torch.sigmoid(self.net(x)).squeeze(1)

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a): ctx.a = a; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.a * g, None

class AdvMLP(nn.Module):
    """Adversarial debiasing in the spirit of Zhang, Lemoine & Mitchell (2018): an adversary regresses the
    (fuzzy) group membership from the predictor's logit through a gradient-reversal layer."""
    def __init__(self, d, alpha=1.0, h=64, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.pred = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.adv = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        self.alpha = alpha
    def forward(self, x, return_parts=False):
        return torch.sigmoid(self.pred(x)).squeeze(1)
    def adv_loss(self, x, mu):
        logit = self.pred(x)
        return F.binary_cross_entropy_with_logits(self.adv(_GradReverse.apply(logit, self.alpha)).squeeze(1), mu)

# ---------------------------------------------------------------- training
def train_torch(model, Xtr, ytr, mutr, Xva, yva, muva, lam_f=0.0, fair='total', lam_b=0.1, lam_a=1e-3, lam_g=1e-3,
                lr=1e-2, epochs=100, batch=512, patience=10, seed=0, wd=0.0, verbose=False):
    """fair in {'total','rule'}: total penalises Delta^2 (Prop. 2); rule penalises sum_l mbar_l (r_l^G - r_l^N)^2,
    an upper bound on the squared within-rule term of Theorem 1 (Jensen)."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, mutr = map(lambda a: torch.as_tensor(a, dtype=torch.float32), (Xtr, ytr, mutr))
    Xva, yva, muva = map(lambda a: torch.as_tensor(a, dtype=torch.float32), (Xva, yva, muva))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    is_tsk = isinstance(model, ZadehNet)
    def objective(X, y, mu):
        if is_tsk:
            p, W, G = model(X, return_parts=True)
        else:
            p = model(X)
        p = p.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p, y)
        if lam_f > 0 and 0 < mu.sum() < len(mu):
            if fair == 'total' or not is_tsk:
                loss = loss + lam_f * fdp_cov(mu, p) ** 2
            else:
                _, _, mG, mN, rG, rN = rule_terms(mu, W, G)
                loss = loss + lam_f * (((mG + mN) / 2) * (rG - rN) ** 2).sum()
        if is_tsk: loss = loss + model.reg(W, lam_b, lam_a, lam_g)
        if isinstance(model, AdvMLP) and model.training: loss = loss + model.adv_loss(X, mu)
        return loss
    best, best_state, bad, n = np.inf, None, 0, len(ytr)
    t0 = time.time(); ep_times = []
    for ep in range(epochs):
        te = time.time(); model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 8: continue
            opt.zero_grad(); objective(Xtr[idx], ytr[idx], mutr[idx]).backward(); opt.step()
        ep_times.append(time.time() - te)
        model.eval()
        with torch.no_grad(): v = float(objective(Xva, yva, muva))
        if v < best - 1e-5: best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
        if verbose: print(ep, round(v, 4))
    model.load_state_dict(best_state); model.eval()
    return dict(epochs=ep + 1, train_time=time.time() - t0, epoch_time=float(np.mean(ep_times)))

def predict(model, X, bs=8192):
    with torch.no_grad():
        X = torch.as_tensor(X, dtype=torch.float32)
        return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)]).numpy()
