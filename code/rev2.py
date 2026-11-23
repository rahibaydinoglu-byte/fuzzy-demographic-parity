"""Revision experiments, batch 2.
  interp2  DATASET SEEDS : R3.3 + R3.6. Compact rule bases by top-k antecedents per rule, ranked either by the gate
                           gamma (as in the original paper; not identifiable, see R3.3) or by the identifiable effective
                           precision P_lj = gamma_lj / (2 s_lj^2). Reports prediction fidelity AND attribution fidelity
                           (routing/within totals and per-rule contributions of Theorem 1 before vs after pruning).
  noproxy  SEEDS         : R1.3. Adult without marital status and relationship.
  fdpage   DATASET SEEDS : R1.1. DNN+FDP re-run storing the residual linear age dependence of test decisions.
"""
import sys, os, json, numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from data import LOADERS, split_and_encode
from models import ZadehNet, MLP, train_torch, predict, rule_terms
from run import metrics
torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results')
TAG = os.environ.get('INTERP_TAG', 'interp2')

def prec(m):
    return (m.gamma() * 0.5 * torch.exp(-2 * m.log_s)).detach()

def parts(m, X, mu_t):
    with torch.no_grad():
        P, W, G = m(torch.as_tensor(X), return_parts=True)
        ro, wi, *_ = rule_terms(mu_t, W, G)
    return P.numpy(), ro.numpy(), wi.numpy()

def topk_overlap(a, b, k=4):
    return len(set(np.argsort(-np.abs(a))[:k]) & set(np.argsort(-np.abs(b))[:k])) / k

def interp2(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    fn = f'{OUT}/{TAG}_{name}_' + '-'.join(map(str, seeds)) + '.csv'
    if os.path.exists(fn): return
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=60000 if name == 'CensusKDD' else None)
        (Xtr, ytr, _, mutr, _), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        mt = torch.as_tensor(mute, dtype=torch.float32)
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for zo in [False, True]:
            for lam in [0, 10]:
                m = ZadehNet(d, L=16, Xinit=init, seed=seed, zero_order=zo)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=batch, seed=seed)
                p0, ro0, wi0 = parts(m, Xte, mt); c0 = ro0 + wi0; yh0 = p0 >= .5
                gam = m.gamma().detach(); P = prec(m)
                Ps = P * torch.as_tensor(Xtr.var(0), dtype=torch.float32)[None]   # variance-scaled precision: invariant to feature scale
                rho_sp = float(np.nanmean([spearmanr(gam[l].numpy(), P[l].numpy())[0] for l in range(m.L)]))
                full = m.rho.data.clone()
                for crit, score in ([('gate', gam), ('precision', P)] if TAG == 'interp2' else [('gate', gam), ('precision', P), ('scaled', Ps)]):
                    order = torch.argsort(score, dim=1, descending=True)
                    for k in [1, 2, 3, 5, 10]:
                        keep = torch.zeros_like(full, dtype=torch.bool); keep.scatter_(1, order[:, :min(k, d)], True)
                        m.rho.data = torch.where(keep, full, torch.full_like(full, -30.))
                        pk, rok, wik = parts(m, Xte, mt); m.rho.data = full; ck = rok + wik
                        rows.append(dict(dataset=name, seed=seed, model='ZadehNet-0' if zo else 'ZadehNet', lam=lam, crit=crit, k=k,
                                         fidelity=float(((pk >= .5) == yh0).mean()), acc_full=float((yh0 == yte).mean()),
                                         acc_compact=float(((pk >= .5) == yte).mean()),
                                         gap_full=float(c0.sum()), gap_compact=float(ck.sum()),
                                         R_full=float(ro0.sum()), R_compact=float(rok.sum()), W_full=float(wi0.sum()), W_compact=float(wik.sum()),
                                         contrib_corr=float(np.corrcoef(c0, ck)[0, 1]) if c0.std() > 0 and ck.std() > 0 else np.nan,
                                         contrib_l1=float(np.abs(c0 - ck).sum()), top4_overlap=topk_overlap(c0, ck),
                                         spearman_gate_prec=rho_sp))
        print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(fn, index=False)

def noproxy(seeds):
    X, y, s, mu, z, groups, meta = LOADERS['Adult']()
    rows = []
    for variant, Xv in [('with proxies', X), ('without marital/relationship', X.drop(columns=['marital', 'relationship']))]:
        for seed in seeds:
            fn = f'{OUT}/noproxy_{seed}.csv'
            sp, cols = split_and_encode(Xv, y, s, mu, groups, seed)
            (Xtr, ytr, _, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
            mt = torch.as_tensor(mute, dtype=torch.float32); d = Xtr.shape[1]
            init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
            for fair, lam in [('total', 0), ('total', 10), ('total', 100), ('rule', 100)]:
                if os.environ.get('NOPROXY_TAG') and fair == 'total' and lam == 100: continue
                if variant == 'with proxies' and not (fair == 'rule' and lam == 100) and lam != 0: continue   # main_all has these
                if os.environ.get('NOPROXY_TAG') and lam not in (0, 100): continue
                m = ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=1024, seed=seed)
                pte, ro, wi = parts(m, Xte, mt); pva = predict(m, Xva)
                r = dict(variant=variant, seed=seed, fair=fair, lam=lam, routing=float(ro.sum()), within=float(wi.sum()), d=d)
                r.update(metrics(pva, yva, sva, muva, 'va_')); r.update(metrics(pte, yte, ste, mute, 'te_'))
                # which features carry the largest-contribution rules: top-3 features by effective precision of the top-4 rules
                P = prec(m).numpy(); c = ro + wi; Ps = P * Xtr.var(0)[None]
                feats = [cols[j].split('_')[0] for l in np.argsort(-np.abs(c))[:4] for j in np.argsort(-P[l])[:3]]
                r['top_rule_features'] = json.dumps(feats)
                r['top_rule_features_scaled'] = json.dumps([cols[j] for l in np.argsort(-np.abs(c))[:4] for j in np.argsort(-Ps[l])[:3]])
                r['top_rules'] = json.dumps([dict(rule=int(l), contrib=float(c[l]), routing=float(ro[l]), within=float(wi[l]),
                                                  features=[cols[j] for j in np.argsort(-Ps[l])[:5]]) for l in np.argsort(-np.abs(c))[:4]])
                rows.append(r)
            print(variant, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/{os.environ.get("NOPROXY_TAG", "noproxy")}_' + '-'.join(map(str, seeds)) + '.csv', index=False)

def fdpage(name, seeds):
    X, y, s, mu, age, groups, meta = LOADERS[name]()
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, _, mutr, itr), (Xva, yva, sva, muva, iva), (Xte, yte, ste, mute, ite) = sp['tr'], sp['va'], sp['te']
        zte = age[ite]; d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        for lam in [0, 1, 3, 10, 30, 100]:
            m = MLP(d, seed=seed); train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lr=1e-3, batch=batch, seed=seed)
            pva, pte = predict(m, Xva), predict(m, Xte)
            r = dict(dataset=name, seed=seed, method='DNN+FDP' if lam else 'DNN', cfg=json.dumps({'lam': lam}))
            r.update(metrics(pva, yva, sva, muva, 'va_')); r.update(metrics(pte, yte, ste, mute, 'te_'))
            yh = (pte >= .5).astype(float); r['te_cov_age'] = float(np.corrcoef(zte, yh)[0, 1]) if yh.std() > 0 else 0.0
            rows.append(r)
        print(name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/fdpage_{name}.csv', index=False)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'interp2': interp2(sys.argv[2], [int(a) for a in sys.argv[3:]])
    elif cmd == 'noproxy': noproxy([int(a) for a in sys.argv[2:]])
    elif cmd == 'fdpage': fdpage(sys.argv[2], [int(a) for a in sys.argv[3:]])

def ramp(name, seeds):
    """R3.4: sensitivity to the membership specification. Train DNN+FDP and ZadehNet (lam=10) against ramps
    young_(25,b) with b in {30, 35, 45} (and the crisp core age<=25), then evaluate every model's decisions against
    every ramp. Diagonal = gap on the group the model was trained for; off-diagonal = gap under a different spec."""
    from data import young
    X, y, s, mu, age, groups, meta = LOADERS[name]()
    specs = {'crisp<=25': lambda a: (a <= 25).astype(float), 'ramp25-30': lambda a: young(a, 25, 30),
             'ramp25-35': lambda a: young(a, 25, 35), 'ramp25-45': lambda a: young(a, 25, 45)}
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, _, _, itr), (Xva, yva, _, _, iva), (Xte, yte, _, _, ite) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else 1024
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for model in ['DNN', 'ZadehNet']:
            for train_spec in ['none'] + list(specs):
                lam = 0 if train_spec == 'none' else 10
                f = specs.get(train_spec, specs['ramp25-35'])
                m = MLP(d, seed=seed) if model == 'DNN' else ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, f(age[itr]).astype(np.float32), Xva, yva, f(age[iva]).astype(np.float32), lam_f=lam,
                            lr=1e-3 if model == 'DNN' else 1e-2, lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
                yh = (predict(m, Xte) >= .5).astype(float)
                r = dict(dataset=name, seed=seed, model=model, train_spec=train_spec, acc=float((yh == yte).mean()))
                for es, g in specs.items():
                    mu_e = g(age[ite]); pi = mu_e.mean()
                    r[f'gap_{es}'] = float(abs(np.mean((mu_e - pi) * (yh - yh.mean())) / (pi * (1 - pi))))
                rows.append(r)
        print('ramp', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/ramp_{name}.csv', index=False)

if __name__ == '__main__' and sys.argv[1] == 'ramp':
    ramp(sys.argv[2], [int(a) for a in sys.argv[3:]])

def audit2(name='Adult', seed=0, lams=(0, 10)):
    """R3.3-consistent rule audit: antecedents ranked by variance-scaled effective precision P_lj*Var(x_j)
    (identifiable), not by the gate. Zero-order model as in the paper (pure routing, Corollary 3)."""
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    sp, cols = split_and_encode(X, y, s, mu, groups, seed)
    (Xtr, ytr, _, mutr, itr), (Xva, yva, _, muva, _), (Xte, yte, ste, mute, ite) = sp['tr'], sp['va'], sp['te']
    init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
    raw = X.iloc[itr]; var = Xtr.var(0); out = []
    for lam in lams:
        m = ZadehNet(Xtr.shape[1], L=16, Xinit=init, seed=seed, zero_order=True)
        train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lam_g=0.01, lam_a=1e-2, lr=1e-2, batch=1024, seed=seed)
        with torch.no_grad():
            P, W, G = m(torch.as_tensor(Xte), return_parts=True)
            ro, wi, mG, mN, rG, rN = rule_terms(torch.as_tensor(mute, dtype=torch.float32), W, G)
        Ps = (prec(m).numpy() * var[None]); C = m.c.detach().numpy(); p = P.numpy()
        cbar = float(p.mean())                                        # centring constant: pooled mean score (round 3)
        ro = (mG - mN) * ((rG + rN) / 2 - cbar)
        direct = float((mute * p).sum() / mute.sum() - ((1 - mute) * p).sum() / (1 - mute).sum())
        for l in range(m.L):
            ants = []
            for j in np.argsort(-Ps[l])[:6]:
                col = cols[j]
                if col in raw.columns:
                    v = raw[col].astype(float); val = C[l, j] * v.std() + v.mean()
                    q = np.mean(Xtr[:, j] <= C[l, j]); lvl = 'Low' if q < 1 / 3 else ('Medium' if q < 2 / 3 else 'High')
                    loc = (f'<{v.min():.3g}' if val < v.min() else f'>{v.max():.3g}' if val > v.max() else f'≈{val:.3g}')
                    ants.append(f'{col} {lvl} ({loc})')
                else:
                    lo, hi = np.unique(Xtr[:, j])[[0, -1]]
                    ants.append(f"{col.replace('_', ' = ', 1)}" + ('' if C[l, j] > (lo + hi) / 2 else ' [absent]'))
            out.append(dict(lam=lam, rule=l + 1, antecedents=' ∧ '.join(ants), relevance=json.dumps([round(float(x), 3) for x in np.sort(Ps[l])[::-1][:6]]),
                            consequent=float(torch.sigmoid(m.b[l]).detach()), cov_G=float(mG[l]), cov_N=float(mN[l]),
                            share=float(ro[l] + wi[l]), total=float(ro.sum() + wi.sum()), direct=direct,
                            test_acc=float(((p >= .5) == yte).mean())))
    pd.DataFrame(out).to_csv(f'{OUT}/audit3_{name}_seed{seed}.csv', index=False)

if __name__ == '__main__' and sys.argv[1] == 'audit2':
    audit2(sys.argv[2], int(sys.argv[3]))

def zmf(a, lo, hi):
    """Z-shaped (smooth spline) membership: 1 below lo, 0 above hi, C1-smooth in between."""
    t = np.clip((np.asarray(a, float) - lo) / (hi - lo), 0, 1)
    return np.where(t < 0.5, 1 - 2 * t ** 2, 2 * (1 - t) ** 2)

def ramp2(name, seeds):
    """Round 4 (Major 5): membership-function SHAPE. Same crossover (mu = 0.5 at 30 years) for linear 25-35, Z-spline 25-35
    and sigmoid (midpoint 30, scale 2 years); a plateau-free linear 18-35; and the crisp core <= 25.
    Train DNN and ZadehNet against each (lam = 10) or without penalty, evaluate decisions against all."""
    from data import young
    X, y, s, mu, age, groups, meta = LOADERS[name]()
    specs = {'crisp<=25': lambda a: (a <= 25).astype(float), 'linear25-35': lambda a: young(a, 25, 35),
             'zspline25-35': lambda a: zmf(a, 25, 35), 'sigmoid30': lambda a: 1 / (1 + np.exp((np.asarray(a, float) - 30) / 2)),
             'linear18-35': lambda a: young(a, 18, 35)}
    rows = []
    for seed in seeds:
        sp, cols = split_and_encode(X, y, s, mu, groups, seed)
        (Xtr, ytr, _, _, itr), (Xva, yva, _, _, iva), (Xte, yte, _, _, ite) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; batch = 64 if len(ytr) < 5000 else 1024
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        for model in ['DNN', 'ZadehNet']:
            for train_spec in ['none'] + list(specs):
                lam = 0 if train_spec == 'none' else 10
                f = specs.get(train_spec, specs['linear25-35'])
                m = MLP(d, seed=seed) if model == 'DNN' else ZadehNet(d, L=16, Xinit=init, seed=seed)
                train_torch(m, Xtr, ytr, f(age[itr]).astype(np.float32), Xva, yva, f(age[iva]).astype(np.float32), lam_f=lam,
                            lr=1e-3 if model == 'DNN' else 1e-2, lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
                yh = (predict(m, Xte) >= .5).astype(float)
                r = dict(dataset=name, seed=seed, model=model, train_spec=train_spec, acc=float((yh == yte).mean()))
                for es, g in specs.items():
                    mu_e = g(age[ite]); pi = mu_e.mean()
                    r[f'gap_{es}'] = float(abs(np.mean((mu_e - pi) * (yh - yh.mean())) / (pi * (1 - pi))))
                rows.append(r)
        print('ramp2', name, seed, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/ramp2_{name}.csv', index=False)

if __name__ == '__main__' and sys.argv[1] == 'ramp2':
    ramp2(sys.argv[2], [int(a) for a in sys.argv[3:]])
