| **Inputs** | **Model** | **Accuracy** | **FDP** | $\mathcal R$ / $\mathcal W$ | **Sex AUC of firing: top-4 / other rules** | **Sex AUC of features: top-4 rules / all** |
|---|---|---|---|---|---|---|
| all features | unconstrained | 0.852 | 0.189 | −0.101 / −0.086 | 0.628 / 0.605 | 0.590 / 0.522 |
| all features | rule-local, λ=100 | 0.853 | 0.187 | −0.188 / 0.000 | 0.693 / 0.688 | 0.601 / 0.522 |
| without marital, relationship | unconstrained | 0.825 | 0.108 | −0.056 / −0.052 | 0.585 / 0.554 | 0.552 / 0.512 |
| without marital, relationship | rule-local, λ=100 | 0.826 | 0.102 | −0.108 / 0.000 | 0.612 / 0.575 | 0.557 / 0.512 |
| without marital, relationship | total, λ=10 | 0.818 | 0.058 | −0.027 / −0.003 | — | — |
| without marital, relationship | total, λ=100 | 0.811 | 0.045 | −0.011 / +0.005 | — | — |
