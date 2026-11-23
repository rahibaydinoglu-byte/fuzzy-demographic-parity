"""Review round 4.
  calboot DATASET SEED... : ZadehNet (lam_f in {0,10}, total; and rule-local lam 100) and DNN (lam_f in {0,10}).
      (i) calibration on test data: expected calibration error (15 equal-width bins) and Brier score, overall and per group
          (crisp group s); (ii) for ZadehNet, test-set bootstrap (B = 1000) percentile intervals of the score-level gap,
          R and W, which quantify the finite-sample variance of the decomposition for a fixed model.
  stab DATASET SEED...    : attribution stability. First-order ZadehNet (lam_f in {0, 10}) with L in {8, 16, 32}. Feature-level
      gap profile phi_j = sum_l share_l * rho~_lj (centred rule shares distributed over each rule's normalised
      variance-scaled precisions; sums to the gap and is invariant to rule permutation). Saved per model for comparison
      across seeds and L. For L = 16 also three compaction methods: antecedents only (k = 3), consequents only (5 largest
      |a_lj| sd_j per rule), both.
"""
import sys, os, numpy as np, pandas as pd, torch
from data import LOADERS, split_and_encode
from models import ZadehNet, MLP, train_torch, predict, rule_terms
from run import metrics
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')

def ece(p, y, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def boot_terms(mu, W, G, B=1000, seed=0):
    rng = np.random.default_rng(seed); n = len(mu); out = []
    mu_t = torch.as_tensor(mu, dtype=torch.float32)
    for _ in range(B):
        i = torch.as_tensor(rng.integers(0, n, n))
        ro, wi, *_ = rule_terms(mu_t[i], W[i], G[i]); out.append((float(ro.sum() + wi.sum()), float(ro.sum()), float(wi.sum())))
    return np.array(out)

def calboot(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for model, fair, lam in [('DNN', 'total', 0), ('DNN', 'total', 10), ('ZadehNet', 'total', 0), ('ZadehNet', 'total', 10), ('ZadehNet', 'rule', 100)]:
            if model == 'DNN':
                m = MLP(d, seed=seed); train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lr=1e-3, batch=batch, seed=seed)
            else:
                m = ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lr=1e-2, lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
            p = predict(m, Xte)
            r = dict(dataset=name, seed=seed, model=model, fair=fair, lam=lam, ece=ece(p, yte), brier=float(np.mean((p - yte) ** 2)),
                     ece_G=ece(p[ste == 1], yte[ste == 1]), ece_N=ece(p[ste == 0], yte[ste == 0]))
            r.update(metrics(p, yte, ste, mute, 'te_'))
            if model == 'ZadehNet':
                with torch.no_grad(): _, W, G = m(torch.as_tensor(Xte), return_parts=True)
                ro, wi, *_ = rule_terms(torch.as_tensor(mute, dtype=torch.float32), W, G)
                bt = boot_terms(mute, W, G, seed=seed)
                r.update(gap=float(ro.sum() + wi.sum()), R=float(ro.sum()), W=float(wi.sum()))
                for k, nm in enumerate(['gap', 'R', 'W']):
                    r[f'{nm}_lo'], r[f'{nm}_hi'] = np.percentile(bt[:, k], [2.5, 97.5]); r[f'{nm}_sd'] = bt[:, k].std()
            rows.append(r)
        print('calboot', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/calboot_{name}.csv', index=False)

def centred(mu_t, W, G):
    ro, wi, mG, mN, rG, rN = rule_terms(mu_t, W, G); c = float((W * G).sum(1).mean())
    return ((mG - mN) * ((rG + rN) / 2 - c) + wi).numpy()

def stab(name, seeds):
    import json
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows, prof = [], []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        mt = torch.as_tensor(mute, dtype=torch.float32); Xt = torch.as_tensor(Xte); sd = Xtr.std(0); var = Xtr.var(0)
        for L in (8, 16, 32):
            for lam in (0, 10):
                m = ZadehNet(d, L=L, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lr=1e-2, lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
                with torch.no_grad(): p0, W, G = m(Xt, return_parts=True)
                sh = centred(mt, W, G)
                P = (m.gamma() * 0.5 * torch.exp(-2 * m.log_s)).detach().numpy() * var[None]
                Pn = P / P.sum(1, keepdims=True); phi = sh @ Pn
                prof.append(dict(dataset=name, seed=seed, L=L, lam=lam, gap=float(sh.sum()), phi=json.dumps([float(v) for v in phi]), cols=json.dumps(cols)))
                if L != 16: continue
                y0 = p0.numpy() >= .5; full_rho = m.rho.data.clone(); full_A = m.A.data.clone()
                order = torch.argsort(torch.as_tensor(P), 1, descending=True)
                keep = torch.zeros_like(full_rho, dtype=torch.bool); keep.scatter_(1, order[:, :3], True)
                imp = (full_A.abs() * torch.as_tensor(sd)[None]); oc = torch.argsort(imp, 1, descending=True)
                keepA = torch.zeros_like(full_A, dtype=torch.bool); keepA.scatter_(1, oc[:, :5], True)
                for meth in ('antecedents k=3', 'consequents 5', 'both'):
                    if meth != 'consequents 5': m.rho.data = torch.where(keep, full_rho, torch.full_like(full_rho, -30.))
                    if meth != 'antecedents k=3': m.A.data = torch.where(keepA, full_A, torch.zeros_like(full_A))
                    with torch.no_grad(): pk, Wk, Gk = m(Xt, return_parts=True)
                    m.rho.data = full_rho.clone(); m.A.data = full_A.clone()
                    shk = centred(mt, Wk, Gk)
                    top = lambda a: set(np.argsort(-np.abs(a))[:4])
                    rows.append(dict(dataset=name, seed=seed, lam=lam, method=meth, fidelity=float(((pk.numpy() >= .5) == y0).mean()),
                                     dgap=abs(float(shk.sum() - sh.sum())), contrib_corr=float(np.corrcoef(sh, shk)[0, 1]),
                                     top4_overlap=len(top(sh) & top(shk)) / 4))
        print('stab', name, seed, flush=True)
    pd.DataFrame(prof).to_csv(f'{OUT}/stabprof_{name}.csv', index=False); pd.DataFrame(rows).to_csv(f'{OUT}/stabcomp_{name}.csv', index=False)

def stabtree(name, seeds):
    """Fair decision tree (depth 4, theta in {0, 0.3}) as a crisp zero-order rule base: leaves are rules with crisp firing,
    so Theorem 1 applies with W = 0. Leaf shares (centred) are distributed equally over the features on each leaf's path."""
    import json
    from fair_trees import FairDecisionTreeClassifier
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    prof = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, str_, mutr, _), _, (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]
        for th in (0.0, 0.3):
            m = FairDecisionTreeClassifier(theta=th, max_depth=4, Z_agg='max', random_state=seed)
            m.fit(Xtr.astype(np.float64), ytr.astype(int), Z=str_.astype(int).reshape(-1, 1))
            Xe = Xte.astype(np.float64); leaf = m.apply(Xe); leaves = np.unique(m.apply(Xtr.astype(np.float64)))
            W = torch.as_tensor((leaf[:, None] == leaves[None]).astype(np.float32))
            pl = m.predict_proba(Xe)[:, 1]; g = np.array([pl[leaf == l].mean() if (leaf == l).any() else 0.5 for l in leaves])
            G = torch.as_tensor(np.tile(g, (len(Xe), 1)).astype(np.float32))
            sh = centred(torch.as_tensor(mute, dtype=torch.float32), W, G)
            path = m.decision_path(Xtr.astype(np.float64)[[np.where(m.apply(Xtr.astype(np.float64)) == l)[0][0] for l in leaves]]).toarray()
            feat = m.tree_.feature; phi = np.zeros(d)
            for k in range(len(leaves)):
                fs = [feat[n] for n in np.where(path[k])[0] if feat[n] >= 0]
                for f in fs: phi[f] += sh[k] / len(fs)
            prof.append(dict(dataset=name, seed=seed, theta=th, gap=float(sh.sum()), n_leaves=len(leaves), phi=json.dumps([float(v) for v in phi]), cols=json.dumps(cols)))
        print('stabtree', name, seed, flush=True)
    pd.DataFrame(prof).to_csv(f'{OUT}/stabtree_{name}.csv', index=False)

if __name__ == '__main__':
    if sys.argv[1] == 'stabtree': stabtree(sys.argv[2], [int(a) for a in sys.argv[3:]])
    if sys.argv[1] == 'stab': stab(sys.argv[2], [int(a) for a in sys.argv[3:]])
    if sys.argv[1] == 'calboot': calboot(sys.argv[2], [int(a) for a in sys.argv[3:]])
