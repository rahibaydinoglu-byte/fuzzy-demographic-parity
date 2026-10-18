"""Secondary experiments. Usage: python extra.py {threshold|ablation|scaling|audit} [args]"""
import sys, os, json, time, tracemalloc, numpy as np, pandas as pd, torch
from data import LOADERS, split_and_encode, young
from models import ZadehNet, MLP, train_torch, predict, rule_terms
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')

def crisp_dp(yh, z, t): s = z <= t; return abs(yh[s].mean() - yh[~s].mean())
def fdp(mu, yh): pi = mu.mean(); return abs(np.mean((mu - pi) * (yh - yh.mean())) / (pi * (1 - pi)))

# ------------------------------------------------------------------ E2: threshold stability (Prop. 3)
def threshold(name, seeds):
    """Train fairness-penalised models with (a) crisp group age<=25 and (b) fuzzy 'young' (25->35 ramp);
    evaluate the crisp parity gap of test decisions for every threshold t in [20, 40]."""
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    if name == 'Adult':  # use age, not sex, as the protected attribute for this experiment
        s = (z <= 25).astype(int); mu = young(z)
    X = X.drop(columns=[c for c in ['age'] if c in X.columns])
    rows = []; ts = np.arange(20, 40.5, 0.5)
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, None, seed)
        tr, va, te = sp['tr'], sp['va'], sp['te']
        ztr, zva, zte = z[tr[4]], z[va[4]], z[te[4]]
        d = tr[0].shape[1]; batch = 64 if len(tr[1]) < 5000 else 1024
        init = tr[0][np.random.default_rng(seed).choice(len(tr[1]), min(5000, len(tr[1])), replace=False)]
        for model_name in ['DNN', 'ZadehNet']:
            for group in ['none', 'crisp', 'fuzzy']:
                lam = 0 if group == 'none' else 10
                gtr = (ztr <= 25).astype(np.float32) if group != 'fuzzy' else young(ztr).astype(np.float32)
                gva = (zva <= 25).astype(np.float32) if group != 'fuzzy' else young(zva).astype(np.float32)
                m = MLP(d, seed=seed) if model_name == 'DNN' else ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, tr[0], tr[1], gtr, va[0], va[1], gva, lam_f=lam, lr=1e-3 if model_name == 'DNN' else 1e-2,
                            lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
                p = predict(m, te[0]); yh = (p >= .5).astype(float)
                curve = [crisp_dp(yh, zte, t) for t in ts]
                # Prop. 3 on real decisions: gap measured with a crisp group {age<=t} vs a fuzzy ramp of width 10 centred at t
                fcurve = [fdp(np.clip((t + 5 - zte) / 10, 0, 1), yh) for t in ts]
                rows.append(dict(dataset=name, seed=seed, model=model_name, group=group, acc=float((yh == te[1]).mean()),
                                 fcurve=json.dumps([round(c, 5) for c in fcurve]),
                                 dp_at_25=crisp_dp(yh, zte, 25), max_dp_20_40=float(np.max(curve)), mean_dp_20_40=float(np.mean(curve)),
                                 fdp_young=fdp(young(zte), yh), curve=json.dumps([round(c, 5) for c in curve])))
        print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/threshold_{name}_' + '-'.join(map(str, seeds)) + '.csv', index=False)

# ------------------------------------------------------------------ E4: ablation
def ablation(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    variants = {'full (L=16)': {}, 'no load-balancing': dict(lam_b=0.0), 'no antecedent gates': dict(gates=False),
                'L=4': dict(L=4), 'L=8': dict(L=8), 'L=32': dict(L=32), 'L=64': dict(L=64), 'tau=0.5': dict(tau=0.5), 'tau=2': dict(tau=2.0)}
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for vname, v in variants.items():
            for lam in [0, 10]:
                L = v.get('L', 16)
                m = ZadehNet(d, L=L, tau=v.get('tau', 1.0), Xinit=init, gates=v.get('gates', True), seed=seed)
                info = train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_b=v.get('lam_b', 0.1), lam_g=0.01, lam_a=1e-2,
                                   lr=1e-2, batch=batch, seed=seed)
                with torch.no_grad():
                    P, W, G = m(torch.as_tensor(Xte), return_parts=True); u = W.mean(0)
                p = P.numpy(); yh = (p >= .5).astype(float)
                ent = float(-(u * torch.log(u + 1e-12)).sum() / np.log(L))       # normalised usage entropy
                rows.append(dict(dataset=name, seed=seed, variant=vname, lam=lam, acc=float((yh == yte).mean()), fdp=fdp(mute, yh),
                                 active_rules=int((u > 1 / (4 * L)).sum()), L=L, usage_entropy=ent,
                                 antecedents=float((m.gamma() > 0.5).sum(1).float().mean()), epoch_time=info['epoch_time']))
        print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/ablation_{name}_' + '-'.join(map(str, seeds)) + '.csv', index=False)

# ------------------------------------------------------------------ E5: scaling
def scaling():
    torch.set_num_threads(1); rows = []
    X, y, s, mu, z, groups, meta = LOADERS['CensusKDD']()
    sp, cols = split_and_encode(X, y, s, mu, None, 0)
    Xtr, ytr, _, mutr, _ = sp['tr']; d = Xtr.shape[1]
    def time_epochs(model, Xs, ys, ms, epochs=3):
        Xs, ys, ms = map(lambda a: torch.as_tensor(a, dtype=torch.float32), (Xs, ys, ms))
        opt = torch.optim.Adam(model.parameters(), 1e-2); ts = []
        tracemalloc.start()
        for _ in range(epochs):
            t = time.time(); perm = torch.randperm(len(ys))
            for i in range(0, len(ys), 1024):
                idx = perm[i:i + 1024]; p = model(Xs[idx]).clamp(1e-6, 1 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(p, ys[idx]); opt.zero_grad(); loss.backward(); opt.step()
            ts.append(time.time() - t)
        tracemalloc.stop()
        return float(np.mean(ts)), float(np.std(ts))
    for n in [10000, 30000, 100000, len(ytr)]:
        idx = np.random.default_rng(0).choice(len(ytr), n, replace=False)
        for mname in ['ZadehNet L=16', 'ZadehNet L=64', 'DNN']:
            m = MLP(d) if mname == 'DNN' else ZadehNet(d, L=int(mname.split('=')[1]), Xinit=Xtr[idx[:2000]])
            mt, st = time_epochs(m, Xtr[idx], ytr[idx], mutr[idx])
            rows.append(dict(exp='n', n=n, d=d, model=mname, epoch_time=mt, epoch_std=st, params=sum(p.numel() for p in m.parameters())))
            print(rows[-1], flush=True)
    rng = np.random.default_rng(0)
    for dd in [25, 50, 100, 200, 400, 800]:
        Xs = rng.normal(size=(20000, dd)).astype(np.float32); ys = (Xs[:, :5].sum(1) > 0).astype(np.float32); ms = (rng.random(20000) < .5).astype(np.float32)
        for mname in ['ZadehNet L=16', 'DNN']:
            m = MLP(dd) if mname == 'DNN' else ZadehNet(dd, L=16, Xinit=Xs[:2000])
            mt, st = time_epochs(m, Xs, ys, ms)
            rows.append(dict(exp='d', n=20000, d=dd, model=mname, epoch_time=mt, epoch_std=st, params=sum(p.numel() for p in m.parameters())))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/scaling.csv', index=False)

# ------------------------------------------------------------------ E6: compactness / fidelity with top-k antecedents per rule
def interp(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for zo in [False, True]:
            for lam in [0, 10]:
                m = ZadehNet(d, L=16, Xinit=init, seed=seed, zero_order=zo)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=batch, seed=seed)
                p_full = predict(m, Xte); yh = p_full >= .5
                full = m.rho.data.clone(); order = torch.argsort(full, dim=1, descending=True)
                for k in [1, 2, 3, 5, 10]:
                    keep = torch.zeros_like(full, dtype=torch.bool); keep.scatter_(1, order[:, :min(k, d)], True)
                    m.rho.data = torch.where(keep, full, torch.full_like(full, -30.)); pk = predict(m, Xte); m.rho.data = full
                    rows.append(dict(dataset=name, seed=seed, model='ZadehNet-0' if zo else 'ZadehNet', lam=lam, k=k, d=d,
                                     fidelity=float(((pk >= .5) == yh).mean()), acc_full=float((yh == yte).mean()),
                                     acc_compact=float(((pk >= .5) == yte).mean()), auc_full=float(roc_auc_score(yte, p_full)),
                                     auc_compact=float(roc_auc_score(yte, pk))))
        print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/interp_{name}_' + '-'.join(map(str, seeds)) + '.csv', index=False)

# ------------------------------------------------------------------ E3: rule-level audit (illustration)
def audit(name='Adult', seed=0, lam=0):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    sp, cols = split_and_encode(X, y, s, mu, groups, seed)
    (Xtr, ytr, _, mutr, itr), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, ite) = sp['tr'], sp['va'], sp['te']
    init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
    out = []
    for zo in [True, False]:
        m = ZadehNet(Xtr.shape[1], L=16, Xinit=init, seed=seed, zero_order=zo)
        train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=1024, seed=seed)
        with torch.no_grad():
            P, W, G = m(torch.as_tensor(Xte), return_parts=True)
            ro, wi, mG, mN, rG, rN = rule_terms(torch.as_tensor(mute, dtype=torch.float32), W, G)
        # linguistic description: original-scale value of each retained antecedent centre, mapped to a term
        raw = X.iloc[itr]; gam = m.gamma().detach().numpy(); C = m.c.detach().numpy()
        Xd_tr = Xtr
        for l in range(m.L):
            ants = []
            for j in np.argsort(-gam[l]):
                if gam[l, j] <= 0.5: break
                col = cols[j]; q = np.mean(Xd_tr[:, j] <= C[l, j])
                if col not in raw.columns:      # one-hot indicator "var_level"
                    lo, hi = np.unique(Xd_tr[:, j])[[0, -1]]
                    if C[l, j] > (lo + hi) / 2: ants.append((float(gam[l, j]), f"{col.replace('_', ' = ', 1) if False else col}"))
                    continue
                v = raw[col].astype(float); val = C[l, j] * v.std() + v.mean()
                where = (f"centre below observed min {v.min():.3g}" if val < v.min() else
                         f"centre above observed max {v.max():.3g}" if val > v.max() else f"centre ≈ {val:.3g}")
                term = f"{col} is {'Low' if q < 1/3 else ('Medium' if q < 2/3 else 'High')} ({where})"
                ants.append((float(gam[l, j]), term))
            out.append(dict(model='zero-order' if zo else 'first-order', lam=lam, rule=l + 1,
                            antecedents=' AND '.join(t for _, t in ants[:6]) + (f' (+{len(ants)-6})' if len(ants) > 6 else ''),
                            n_ant=len(ants), consequent=float(torch.sigmoid(m.b[l]).detach()) if zo else float('nan'),
                            cov_G=float(mG[l]), cov_N=float(mN[l]), rate_G=float(rG[l]), rate_N=float(rN[l]),
                            routing=float(ro[l]), within=float(wi[l]), contribution=float(ro[l] + wi[l]),
                            total_gap=float(ro.sum() + wi.sum()), direct_gap=float((mute * P.numpy()).sum() / mute.sum() - ((1 - mute) * P.numpy()).sum() / (1 - mute).sum()),
                            test_acc=float(((P.numpy() >= .5) == yte).mean())))
    pd.DataFrame(out).to_csv(f'{OUT}/audit_{name}_lam{lam}.csv', index=False)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd in ('threshold', 'ablation', 'interp') and os.path.exists(f"{OUT}/{cmd}_{sys.argv[2]}_" + '-'.join(sys.argv[3:]) + '.csv'): sys.exit(0)
    if cmd == 'audit' and os.path.exists(f"{OUT}/audit_{sys.argv[2]}_lam{float(sys.argv[4])}.csv"): sys.exit(0)
    if cmd == 'scaling' and os.path.exists(f"{OUT}/scaling.csv"): sys.exit(0)
    if cmd == 'threshold': threshold(sys.argv[2], [int(a) for a in sys.argv[3:]])
    elif cmd == 'ablation': ablation(sys.argv[2], [int(a) for a in sys.argv[3:]])
    elif cmd == 'scaling': scaling()
    elif cmd == 'interp': interp(sys.argv[2], [int(a) for a in sys.argv[3:]])
    elif cmd == 'audit': audit(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
