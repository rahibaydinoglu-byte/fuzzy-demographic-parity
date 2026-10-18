"""Main comparison. Usage: python run.py DATASET SEED [SEED ...]  -> results/main_DATASET_SEED.csv
Every method x configuration is evaluated on validation and test; model selection happens later in analyze.py."""
import sys, os, json, time, warnings, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score
import lightgbm as lgb
from data import LOADERS, split_and_encode
from models import ZadehNet, MLP, AdvMLP, train_torch, predict, rule_terms
warnings.filterwarnings('ignore'); torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), '..', 'results'); os.makedirs(OUT, exist_ok=True)
LAMS = [0, 1, 3, 10, 30, 100]

def fdp(mu, yhat):
    pi = mu.mean(); return float(np.mean((mu - pi) * (yhat - yhat.mean())) / (pi * (1 - pi)))

def metrics(p, y, s, mu, prefix):
    yh = (p >= 0.5).astype(float)
    eo = max(abs(yh[(s == 1) & (y == 1)].mean() - yh[(s == 0) & (y == 1)].mean()),
             abs(yh[(s == 1) & (y == 0)].mean() - yh[(s == 0) & (y == 0)].mean()))
    return {f'{prefix}acc': accuracy_score(y, yh), f'{prefix}bacc': balanced_accuracy_score(y, yh),
            f'{prefix}auc': roc_auc_score(y, p), f'{prefix}f1': f1_score(y, yh),
            f'{prefix}fdp': abs(fdp(mu, yh)),          # parity gap w.r.t. the paper's (fuzzy or crisp) group
            f'{prefix}dp_crisp': abs(yh[s == 1].mean() - yh[s == 0].mean()),
            f'{prefix}eo': eo}

def main(name, seeds):
    X, y, s, mu, z, groups, meta = LOADERS[name]()
    max_train = 60000 if name == 'CensusKDD' else None
    for seed in seeds:
        fn = f'{OUT}/main_{name}_{seed}.csv'
        if os.path.exists(fn): continue
        sp, cols = split_and_encode(X, y, s, mu, groups, seed, max_train=max_train)
        (Xtr, ytr, str_, mutr, _), (Xva, yva, sva, muva, _), (Xte, yte, ste, mute, _) = sp['tr'], sp['va'], sp['te']
        d = Xtr.shape[1]; rows = []
        def rec(method, cfg, pva, pte, extra=None):
            r = dict(dataset=name, seed=seed, method=method, cfg=json.dumps(cfg))
            r.update(metrics(pva, yva, sva, muva, 'va_')); r.update(metrics(pte, yte, ste, mute, 'te_'))
            if extra: r.update(extra)
            rows.append(r)
        batch = 64 if len(ytr) < 5000 else (256 if len(ytr) < 20000 else 1024)
        # --- unconstrained references
        t = time.time(); lr_ = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
        rec('LR', {}, lr_.predict_proba(Xva)[:, 1], lr_.predict_proba(Xte)[:, 1], dict(train_time=time.time() - t))
        t = time.time(); gb = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed, n_jobs=1).fit(Xtr, ytr)
        rec('LightGBM', {}, gb.predict_proba(Xva)[:, 1], gb.predict_proba(Xte)[:, 1], dict(train_time=time.time() - t))
        # --- pre-processing: reweighing (Kamiran & Calders 2012) on LightGBM
        w = np.ones(len(ytr))
        for a in (0, 1):
            for c in (0, 1):
                m = (str_ == a) & (ytr == c)
                if m.any(): w[m] = (str_ == a).mean() * (ytr == c).mean() / m.mean()
        t = time.time(); gbw = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed, n_jobs=1).fit(Xtr, ytr, sample_weight=w)
        rec('Reweighing', {}, gbw.predict_proba(Xva)[:, 1], gbw.predict_proba(Xte)[:, 1], dict(train_time=time.time() - t))
        # --- in-processing: exponentiated gradient (Agarwal et al. 2018), LR base
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity
        for eps in [0.01, 0.02, 0.05, 0.1]:
            t = time.time()
            eg = ExponentiatedGradient(LogisticRegression(max_iter=2000), DemographicParity(difference_bound=eps), eps=0.01, max_iter=50)
            eg.fit(Xtr, ytr, sensitive_features=str_)
            rec('ExpGrad', {'eps': eps}, eg._pmf_predict(Xva)[:, 1], eg._pmf_predict(Xte)[:, 1], dict(train_time=time.time() - t))
        # --- post-processing: threshold optimizer (Hardt et al. 2016), LightGBM base; uses s at test time
        from fairlearn.postprocessing import ThresholdOptimizer
        t = time.time()
        to = ThresholdOptimizer(estimator=gb, constraints='demographic_parity', prefit=True, predict_method='predict_proba')
        to.fit(Xva, yva, sensitive_features=sva)   # fitted on validation split
        pva = to._pmf_predict(Xva, sensitive_features=sva)[:, 1]; pte = to._pmf_predict(Xte, sensitive_features=ste)[:, 1]
        rec('ThreshOpt', {}, pva, pte, dict(train_time=time.time() - t))
        # --- neural: DNN (+ same fuzzy-parity penalty) and ZadehNet variants
        for lam in LAMS:
            m = MLP(d, seed=seed); info = train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, lr=1e-3, batch=batch, seed=seed)
            rec('DNN+FDP' if lam else 'DNN', {'lam': lam}, predict(m, Xva), predict(m, Xte), info)
        for alpha in [0.3, 1, 3]:
            m = AdvMLP(d, alpha=alpha, seed=seed); info = train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lr=1e-3, batch=batch, seed=seed)
            rec('AdvDebias', {'alpha': alpha}, predict(m, Xva), predict(m, Xte), info)
        init = Xtr[np.random.default_rng(seed).choice(len(Xtr), min(5000, len(Xtr)), replace=False)]
        configs = [('total', False, lam) for lam in LAMS] + [('rule', False, lam) for lam in (3, 30, 100)] + \
                  [('total', True, lam) for lam in LAMS]
        mute_t = torch.as_tensor(mute, dtype=torch.float32)
        for fair, zo, lam in configs:
            m = ZadehNet(d, L=16, Xinit=init, seed=seed, zero_order=zo)
            info = train_torch(m, Xtr, ytr, mutr, Xva, yva, muva, lam_f=lam, fair=fair, lr=1e-2, lam_g=0.01, lam_a=1e-2, batch=batch, seed=seed)
            # Theorem-1 audit on the test split + interpretability statistics
            with torch.no_grad():
                P, W, G = m(torch.as_tensor(Xte), return_parts=True)
                ro, wi, *_ = rule_terms(mute_t, W, G)
                gam = m.gamma(); full = m.rho.data.clone()
                m.rho.data = torch.where(gam > 0.5, full, torch.full_like(full, -30.)); pc = predict(m, Xte); m.rho.data = full
            pte = P.numpy()
            info.update(routing=float(ro.sum()), within=float(wi.sum()), antecedents=float((gam > 0.5).sum(1).float().mean()),
                        active_rules=int((W.mean(0) > 1 / (4 * m.L)).sum()), fidelity=float(((pte >= .5) == (pc >= .5)).mean()),
                        consequent_nnz=float((m.A.abs() > 0.01).sum(1).float().mean()) if m.A is not None else 0.0)
            base = 'ZadehNet0' if zo else 'ZadehNet'
            meth = (base + '-noFair') if lam == 0 else (base + ('-rule' if fair == 'rule' else ''))
            rec(meth, {'lam': lam}, predict(m, Xva), pte, info)
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(name, seed, 'done', flush=True)

if __name__ == '__main__':
    main(sys.argv[1], [int(a) for a in sys.argv[2:]])
