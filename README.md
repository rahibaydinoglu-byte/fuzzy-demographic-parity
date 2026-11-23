# Fuzzy demographic parity, rule-level disparity decomposition and rule-local fairness constraints (FairTSK)

Code and complete per-split results for the article

> R. Imamguluyev, *Fuzzy demographic parity, rule-level disparity decomposition and the failure of rule-local fairness constraints in Takagi–Sugeno–Kang classifiers* (submitted to *Knowledge and Information Systems*).

The model called **FairTSK** in the article is the class `ZadehNet` in the code. Version 1.1 adds the experiments of the revised article (scripts listed below).

The repository contains:

- **ZadehNet** (FairTSK in the article), a fairness-aware first-order (and zero-order) Takagi–Sugeno–Kang classifier;
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

## Experiments of the revised article (v1.1)

All tables of the article and of Online Resource 1 are generated from result files by `code/tables_v3.py` and then `code/tables_v4.py` (no number is typed by hand); figures by `code/figs_v4.py`. The manuscript is assembled from `paper_v4/q*.md` and `paper_v4/s_supp.md` by `paper_v4/assemble_v4.py` and typeset with the Springer Nature LaTeX template by `paper_v4/build_latex.py TEMPLATE_DIR`.

```
code/synth.py            synthetic study with known Bayes score (Theorem 2; Section 6.4)
code/synth_gap.py        additional synthetic configurations: frozen antecedents with L up to 128, centres frozen, trainable L = 3
code/align.py            alignment of Theorem 2 / Corollary 3 with training (plain and regularised protocols)
code/cov_baselines.py    covariance and HGR penalties on age versus FDP (German, Bank)
code/interp_baselines.py fair decision tree and FairCORELS baselines
code/rev2.py, rev3.py    compact rule bases, attribution stability, rule audit, proxy removal
code/rev4.py             calibration and bootstrap variability, mixture of experts with a softmax gate
code/round5.py           score/decision correlation, HV reference sensitivity, age-axis analysis
code/round6.py           log-loss and consequent norms of first-order, rule-local and zero-order models; Oaxaca-Blinder reference weights
code/egb.py              Exponentiated Gradient with a LightGBM base learner
code/round7.py           Adult with the fuzzy group young (dataset AdultAge in data.py; run first: python code/run.py AdultAge 0 ... 9)
code/stats_revision.py, stats_round3.py, analysis_round4.py, analyze_revision.py   statistics of the revision
code/check_theory2.py    numerical check of Lemma 1, Proposition 3 and Corollary 2
```

Example: `python code/round6.py COMPAS 0 1 2 3 4 5 6 7 8 9`, `python code/egb.py Adult 0 1 2 3 4 5 6 7 8 9`, `python code/synth_gap.py nonlinear 0 1 2 3 4`.

## Using ZadehNet (FairTSK) and the decomposition on your own data

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
