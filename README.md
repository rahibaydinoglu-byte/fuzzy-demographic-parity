# Fuzzy demographic parity and rule-level disparity decomposition (ZadehNet)

Code and complete per-split results for the article

> R. Imamguluyev, *Fuzzy demographic parity and exact rule-level disparity decomposition for auditable fairness-aware TSK classifiers* (submitted).

The repository contains:

- **ZadehNet**, a fairness-aware first-order (and zero-order) Takagi–Sugeno–Kang classifier;
- the **fuzzy demographic parity** (FDP) penalty;
- the exact **routing / within-rule decomposition** of the parity gap (Theorem 1);
- every experiment, table and figure reported in the paper.

## Layout

```
code/
  data.py          dataset loaders, fuzzy group "young", 60/20/20 splits, encoding
  models.py        ZadehNet, DNN, adversarial debiasing, FDP penalty (Prop. 2), Theorem-1 terms
  run.py           main comparison: 13 method families x configurations, one CSV per (dataset, split)
  extra.py         threshold stability, ablation, rule-base compactness, rule audit, scaling
  analyze.py       operating-point selection, hypervolume, Friedman/Nemenyi, Wilcoxon+Holm, Nadeau-Bengio
  tables_md.py     tables of the paper (Markdown) -> results/tables_md/
  figs.py          figures of the paper -> figures/
  check_theory.py  numerical check of Prop. 2-3, Theorem 1 and Corollary 1
results/           outputs used in the paper: main_all, threshold_all, ablation_all, interp_all (one row per
                   method/configuration and split), audit_*, scaling
results/tables/    aggregated tables and statistical tests (summary.json)
figures/           figures as in the paper
data/              download instructions (raw data are not redistributed)
```

## Reproducing the results

```bash
pip install -r requirements.txt
# 1. put the raw data in data/raw/ as described in data/README.md
# 2. main comparison (CPU; about 5 h on 2 cores for everything)
for D in German Student COMPAS Default Bank Adult; do for s in 0 1 2 3 4 5 6 7 8 9; do python code/run.py $D $s; done; done
for D in Diabetes CensusKDD; do for s in 0 1 2 3 4; do python code/run.py $D $s; done; done
# 3. secondary experiments
for D in Adult Bank German; do for s in 0 1 2 3 4; do python code/extra.py threshold $D $s; done; done
for D in Adult COMPAS Bank; do for s in 0 1 2; do python code/extra.py ablation $D $s; done; done
for D in German Student COMPAS Default Bank Adult Diabetes CensusKDD; do for s in 0 1 2; do python code/extra.py interp $D $s; done; done
python code/extra.py audit Adult 0 0; python code/extra.py audit Adult 0 10; python code/extra.py audit COMPAS 0 0
python code/extra.py scaling
# 4. statistics, tables, figures
python code/analyze.py && python code/tables_md.py && python code/figs.py
python code/check_theory.py
```

`run.py` and `extra.py` write one file per (dataset, split), e.g. `results/main_Adult_3.csv`, and skip runs whose file already exists. The shipped `*_all.csv` files concatenate those per-split files; the analysis scripts read both forms. To recompute from scratch, move the `*_all.csv` files out of `results/`. Steps 4 and `check_theory.py` regenerate every table and figure from the CSV files shipped here, without retraining. They still need `data/raw/` for the dataset statistics in Table 2 and the hypervolume reference points.

Seeds 0–9 fix splits, initialisation and mini-batch order. Runs used one CPU thread each (`torch.set_num_threads(1)`). Small floating-point differences across platforms and library versions are possible.

## Using ZadehNet and the decomposition on your own data

```python
import torch
from models import ZadehNet, train_torch, predict, rule_terms
m = ZadehNet(d=X_train.shape[1], L=16, Xinit=X_train[:5000])
train_torch(m, X_train, y_train, mu_train, X_val, y_val, mu_val, lam_f=10, lam_g=0.01, lam_a=1e-2, lr=1e-2)
with torch.no_grad():
    p, W, G = m(torch.as_tensor(X_test), return_parts=True)
    routing, within, *_ = rule_terms(torch.as_tensor(mu_test, dtype=torch.float32), W, G)
print('gap =', float(routing.sum() + within.sum()))   # equals the FDP gap of the scores (Theorem 1)
```

Here `mu_*` is the membership of each individual in the protected group: 0/1 for a crisp group, or a membership function such as `data.young(age)`.

## Note on AI assistance

The code was written with the assistance of an AI coding tool (Claude, Anthropic) under the author's direction and review, as stated in the article's declaration. All results were produced by running this code.

## Licence

Code: MIT (see `LICENSE`). The datasets keep their original licences and are not part of this repository.
