"""Revision round 3 (external review).
  interp4 DATASET SEEDS : compact rule bases as interp3 (ranking by variance-scaled precision), but per-rule shares are
                          CENTRED: routing share (m_G - m_N)(rbar_l - c) with c = E[yhat] (pooled mean score), which is
                          invariant to adding a constant to every rule output; the total gap is unchanged.
  noproxy3 SEEDS        : Adult with / without marital & relationship, unconstrained and rule-local (lam=100).
                          For the four largest centred-share rules: AUC of predicting sex from the rule's normalised firing,
                          and single-feature AUC of predicting sex for their top-3 relevance features vs. all features.
  moe DATASET SEEDS     : mixture of experts with a linear softmax gate (no fuzzy antecedents) and the same consequents;
                          unconstrained, total penalty (lam 10, 100) and rule-local penalty (lam 100).
"""
import sys, os, json, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from data import LOADERS, split_and_encode
from models import ZadehNet, train_torch, predict, rule_terms
from run import metrics
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')

def centred_shares(mu_t, W, G):
    ro, wi, mG, mN, rG, rN = rule_terms(mu_t, W, G)
    c = float((W * G).sum(1).mean())                       # pooled mean score
    roc = (mG - mN) * ((rG + rN) / 2 - c)
    return roc.numpy(), wi.numpy(), (roc + wi).numpy(), float(ro.sum()), float(wi.sum())

def prec_scaled(m, Xtr):
    P = (m.gamma() * 0.5 * torch.exp(-2 * m.log_s)).detach().numpy()
    return P * Xtr.var(0)[None]

def overlap(a, b, k=4): return len(set(np.argsort(-np.abs(a))[:k]) & set(np.argsort(-np.abs(b))[:k])) / k

def interp4(name, seeds):
    fn = f'{OUT}/interp4_{name}_' + '-'.join(map(str, seeds)) + '.csv'
    if os.path.exists(fn): return
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        mt = torch.as_tensor(mute, dtype=torch.float32); d = Xtr.shape[1]
        batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for zo in [False, True]:
            for lam in [0, 10]:
                m = ZadehNet(d, L=16, Xinit=init, seed=seed, zero_order=zo)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=batch, seed=seed)
                X_t = torch.as_tensor(Xte)
                with torch.no_grad(): p0, W, G = m(X_t, return_parts=True)
                r0, w0, c0, R0, W0 = centred_shares(mt, W, G); yh0 = p0.numpy() >= .5
                Ps = torch.as_tensor(prec_scaled(m, Xtr)); full = m.rho.data.clone(); order = torch.argsort(Ps, 1, descending=True)
                for k in [1, 3, 5]:
                    keep = torch.zeros_like(full, dtype=torch.bool); keep.scatter_(1, order[:, :min(k, d)], True)
                    m.rho.data = torch.where(keep, full, torch.full_like(full, -30.))
                    with torch.no_grad(): pk, Wk, Gk = m(X_t, return_parts=True)
                    m.rho.data = full
                    rk, wk, ck, Rk, Wkk = centred_shares(mt, Wk, Gk)
                    rows.append(dict(dataset=name, seed=seed, model='ZadehNet-0' if zo else 'ZadehNet', lam=lam, k=k,
                                     fidelity=float(((pk.numpy() >= .5) == yh0).mean()), gap_full=R0 + W0, gap_compact=Rk + Wkk,
                                     contrib_corr=float(np.corrcoef(c0, ck)[0, 1]) if c0.std() > 0 and ck.std() > 0 else np.nan,
                                     top4_overlap=overlap(c0, ck)))
        print('interp4', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(fn, index=False)

def noproxy3(seeds):
    X, y, s, mu, z, groups, meta = LOADERS['Adult']()
    rows = []
    for variant, Xv in [('with proxies', X), ('without marital/relationship', X.drop(columns=['marital', 'relationship']))]:
        for seed in seeds:
            sp, cols = split_and_encode(Xv, y, s, mu, groups, seed)
            (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
            mt = torch.as_tensor(mute, dtype=torch.float32); d = Xtr.shape[1]
            feat_auc = np.array([abs(roc_auc_score(ste, Xte[:, j]) - 0.5) + 0.5 if Xte[:, j].std() > 0 else 0.5 for j in range(d)])
            init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
            for fair, lam in [('total', 0), ('rule', 100)]:
                m = ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=1024, seed=seed)
                with torch.no_grad(): p, W, G = m(torch.as_tensor(Xte), return_parts=True)
                r, w, c, R, Wt = centred_shares(mt, W, G); Wn = W.numpy()
                top = np.argsort(-np.abs(c))[:4]
                rule_auc = [abs(roc_auc_score(ste, Wn[:, l]) - 0.5) + 0.5 for l in range(m.L)]
                Ps = prec_scaled(m, Xtr); topf = sorted({int(j) for l in top for j in np.argsort(-Ps[l])[:3]})
                rows.append(dict(variant=variant, seed=seed, fair=fair, lam=lam, te_fdp=metrics(p.numpy(), yte, ste, mute, 'te_')['te_fdp'],
                                 R=R, W=Wt, top_rule_sex_auc=float(np.mean([rule_auc[l] for l in top])),
                                 other_rule_sex_auc=float(np.mean([rule_auc[l] for l in range(m.L) if l not in top])),
                                 top_feat_sex_auc=float(feat_auc[topf].mean()), all_feat_sex_auc=float(feat_auc.mean()),
                                 top_features=json.dumps([cols[j] for j in topf]),
                                 top_feat_aucs=json.dumps([round(float(feat_auc[j]), 3) for j in topf])))
            print('noproxy3', variant, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/noproxy3_' + '-'.join(map(str, seeds)) + '.csv', index=False)

class MoE(nn.Module):
    """Mixture of experts with a linear softmax gate: wbar = softmax(Vx + e), g_l = sigmoid(a_l.x + b_l)."""
    def __init__(self, d, L=16, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.gate = nn.Linear(d, L); self.A = nn.Parameter(torch.zeros(L, d)); self.b = nn.Parameter(torch.zeros(L)); self.L = L
        nn.init.normal_(self.gate.weight, std=0.5)
    def forward(self, x, return_parts=False):
        W = torch.softmax(self.gate(x), 1); G = torch.sigmoid(x @ self.A.T + self.b); p = (W * G).sum(1)
        return (p, W, G) if return_parts else p

def moe(name, seeds):
    import models
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        mt = torch.as_tensor(mute, dtype=torch.float32); d = Xtr.shape[1]
        batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        for fair, lam in [('total', 0), ('total', 10), ('total', 100), ('rule', 100)]:
            m = MoE(d, seed=seed)
            info = _train(m, Xtr, ytr, mutr, Xva, yva, muva, lam, fair, batch, seed)
            with torch.no_grad(): p, W, G = m(torch.as_tensor(Xte), return_parts=True)
            ro, wi, *_ = rule_terms(mt, W, G)
            r = dict(dataset=name, seed=seed, model='MoE', fair=fair, lam=lam, routing=float(ro.sum()), within=float(wi.sum()),
                     active=int((W.mean(0) > 1 / (4 * 16)).sum()))
            r.update(metrics(predict(m, Xva), yva, sva, muva, 'va_')); r.update(metrics(p.numpy(), yte, ste, mute, 'te_'))
            rows.append(r)
        print('moe', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/moe_{name}.csv', index=False)

def _train(m, Xtr, ytr, mutr, Xva, yva, muva, lam, fair, batch, seed, epochs=100, patience=10):
    import torch.nn.functional as F
    from models import fdp_cov
    torch.manual_seed(seed); T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    Xtr, ytr, mutr, Xva, yva, muva = map(T, (Xtr, ytr, mutr, Xva, yva, muva))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    def obj(X, y, mu):
        p, W, G = m(X, return_parts=True); p = p.clamp(1e-6, 1 - 1e-6); loss = F.binary_cross_entropy(p, y)
        if lam > 0 and 0 < mu.sum() < len(mu):
            if fair == 'total': loss = loss + lam * fdp_cov(mu, p) ** 2
            else:
                _, _, mG, mN, rG, rN = rule_terms(mu, W, G); loss = loss + lam * (((mG + mN) / 2) * (rG - rN) ** 2).sum()
        u = W.mean(0); loss = loss + 0.1 * (u * torch.log(u * m.L + 1e-12)).sum() + 1e-2 * m.A.abs().mean()   # same load balancing / l1 as ZadehNet
        return loss
    best, bs, bad = np.inf, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            if len(idx) < 8: continue
            opt.zero_grad(); obj(Xtr[idx], ytr[idx], mutr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = float(obj(Xva, yva, muva))
        if v < best - 1e-5: best, bad, bs = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(bs); m.eval(); return ep + 1

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'interp4': interp4(sys.argv[2], [int(a) for a in sys.argv[3:]])
    elif cmd == 'noproxy3': noproxy3([int(a) for a in sys.argv[2:]])
    elif cmd == 'moe': moe(sys.argv[2], [int(a) for a in sys.argv[3:]])
