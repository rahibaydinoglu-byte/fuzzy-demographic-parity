Section, equation, proposition and table numbers without the prefix S refer to the main article. All numbers below were computed by the released code \cite{software}.

# Hyperparameters and protocol details

| **Hyperparameter** | **Value** | **Origin** |
|---|---|---|
| Number of rules $L$ | 16 | A priori; ablation over 4–64 in Section 6.8 |
| Firing temperature $\tau$ | 1 | A priori; ablation with 0.5 and 2 in Section 6.8 |
| Load-balancing weight $\lambda_b$ | 0.1 | A priori; ablation without it in Section 6.8 |
| Consequent sparsity $\lambda_a$ | $10^{-2}$ | Pilot, from $\{10^{-3},10^{-2},3\cdot10^{-2}\}$; no effect on accuracy |
| Gate penalty $\lambda_\gamma$ | $10^{-2}$ | Pilot, from $\{10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\}$ |
| Initialisation | $k$-means++ seeding of centres on ≤5,000 training points; $s_{lj} = 1$; $\gamma_{lj} = 0.73$ | A priori |
| Learning rate (Adam) | $10^{-2}$ for TSK models and the penalised logistic regressions, $10^{-3}$ for networks | A priori |
| Mini-batch size | 64 / 256 / 1024 for training sets <5,000 / <20,000 / larger | Pilot (the zero-order model did not leave its initial constant prediction on German with batches of 256) |
| Epochs, early-stopping patience | 100, 10 | A priori |
| DNN architecture | 2 hidden layers × 64 ReLU units | A priori |
| LightGBM | 300 trees, learning rate 0.05, 31 leaves | A priori |

: Hyperparameters and their origin. "A priori": fixed before any experiment. "Pilot": chosen in pilot runs on split 0 of Adult and German for rule sparsity and training stability. "Validation": selected per split on validation data. Grids: Exponentiated Gradient $\epsilon \in \{0.01,0.02,0.05,0.1\}$; adversarial debiasing $\alpha \in \{0.3,1,3\}$; DNN+FDP, FairTSK and FairTSK-0 $\lambda_f \in \{0,1,3,10,30,100\}$; FairTSK-rule $\lambda \in \{3,30,100\}$; covariance and HGR penalties $\lambda \in \{1,3,10,30,100,300\}$ ($\{0.3,\dots,100\}$ for HGR); fair tree $\theta \in \{0,0.1,0.2,0.3,0.5,0.7,0.9\}$, depth 4 or 6; FairCORELS maximum training unfairness $\{0.01,0.02,0.05,0.1\}$ and unconstrained, rules of one condition, minimum support 0.01, $c = 10^{-3}$, search budget $2 \times 10^5$ nodes, numeric inputs binarised at training quintiles.

**Pilot choices.** On Adult, gate penalties $\lambda_{\gamma} \in \{10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\}$ gave accuracies of 0.8458–0.8464; $10^{-2}$ was chosen because larger values made the zero-order model on the small datasets nearly constant. $\lambda_{a} = 10^{-2}$ was chosen among $\{10^{-3},10^{-2},3\cdot10^{-2}\}$, which did not affect accuracy. The mini-batch size was made dependent on the training-set size after the zero-order model failed to leave its initial constant prediction on German with batches of 256. Split 0 is also an evaluation split; repeating all dataset-level tests without it leaves the conclusions unchanged (FairTSK minus DNN+FDP HV difference +0.0034, 90% CI [−0.0016, +0.0085]; Friedman $p = 0.018$).

# Specification of the protected group

**Proposition S1 (Stability with respect to the membership function).** Let $\mu,\mu'$ satisfy $\|\mu - \mu'\|_{\infty} \leq \varepsilon$, with means in $[\pi_{0},1 - \pi_{0}]$, and let $v_{0} := \pi_{0}(1 - \pi_{0})$. Then for any $h$ with values in $[0,1]$, $|\Delta_{G}(h) - \Delta_{G'}(h)| \leq \frac{\varepsilon}{2v_{0}}(1 + v_{0}^{-1/2})$.

*Proof.* Let $v = \pi(1 - \pi)$ and $C = \mathrm{Cov}(\mu,h)$, and $v',C'$ analogously. (1) $|C - C'| = |\mathbb{E}[(\mu - \mu')(h - \mathbb{E}h)]| \leq \varepsilon\sqrt{\mathrm{Var}\,h} \leq \varepsilon/2$. (2) $\mathrm{Var}\,\mu \leq v$, so $|C| \leq \sqrt{v}/2$. (3) $|v - v'| \leq \varepsilon$. Hence $|C/v - C'/v'| \leq |C - C'|/v' + |C|\,|v - v'|/(v v') \leq \varepsilon/(2v_0) + \varepsilon/(2v_0^{3/2})$. $\square$

For memberships $\mu_t(z) = S((t - z)/s)$ with a 1-Lipschitz $S$, $t \mapsto \Delta_{G_t}(h)$ is therefore Lipschitz with constant $(1 + v_{0}^{-1/2})/(2v_{0}s)$. On German ($v_0 \approx 0.0475$, $s = 10$ years) the constant is about 5.9 per year, larger than any gap, so the bound is qualitative. For crisp groups and integer ages, $t \mapsto \Delta_{\{z \leq t\}}$ is a step function with a jump at every atom.

**Measurement experiment.** On German, Bank and Adult (for Adult with age as the sensitive variable for this experiment only), DNN and FairTSK were trained without a penalty, with the penalty on the crisp group age ≤ 25 and with the penalty on *young* ($\lambda_f = 10$, 5 splits). For the test decisions of each of the 90 models, the crisp gap for $\{\text{age} \leq t\}$ and the fuzzy gap for a ramp of width 10 centred at $t$ were computed for $t$ from 20 to 40 in steps of 0.5 (both groups ≥ 5% of the data). A linear ramp is the average of crisp indicators over its window, so the fuzzy curve is a smoothed crisp curve by construction; the experiment measures the size of the effect. For all 90 models the crisp curve had the larger total variation (on average 0.158 versus 0.063) and the larger largest step (0.029 versus 0.004; Fig. S1). On German, moving the crisp threshold past a single integer age changes the measured gap of the same decisions by up to 0.16. Training against the fuzzy group did not make models more robust to the crisp threshold (Table S2): across the 30 paired runs, it gave a slightly larger maximum crisp gap (+0.014, Wilcoxon $p = 0.028$) and higher accuracy (+0.008, $p < 0.001$) than training against the crisp group.

![Gap of the same test decisions (unconstrained FairTSK, 5 splits) measured against crisp groups {age ≤ t} and against fuzzy groups with a ramp of width 10 centred at t](figures/fig_threshold.pdf){width=100%}

| **Dataset (valid** $t$**)** | **Model** | **Penalty group** | **Accuracy** | **Gap at** $t$**=25** | **Max gap over** $t$ | **Mean gap over** $t$ | **FDP (*young*)** |
|---|---|---|---|---|---|---|---|
| German (22--40)             | DNN       | none (λ=0)        | 0.748        | 0.105                 | 0.147 ± 0.037        | 0.073                 | 0.079             |
| German (22--40)             | DNN       | crisp, age ≤ 25   | 0.748        | 0.044                 | 0.107 ± 0.047        | 0.044                 | 0.022             |
| German (22--40)             | DNN       | fuzzy *young*     | 0.756        | 0.058                 | 0.102 ± 0.040        | 0.046                 | 0.032             |
| German (22--40)             | FairTSK  | none (λ=0)        | 0.754        | 0.114                 | 0.145 ± 0.060        | 0.064                 | 0.067             |
| German (22--40)             | FairTSK  | crisp, age ≤ 25   | 0.734        | 0.043                 | 0.104 ± 0.021        | 0.042                 | 0.026             |
| German (22--40)             | FairTSK  | fuzzy *young*     | 0.749        | 0.059                 | 0.106 ± 0.024        | 0.047                 | 0.030             |
| Bank (27--40)               | DNN       | none (λ=0)        | 0.893        | 0.071                 | 0.049 ± 0.020        | 0.021                 | 0.025             |
| Bank (27--40)               | DNN       | crisp, age ≤ 25   | 0.892        | 0.023                 | 0.020 ± 0.005        | 0.013                 | 0.011             |
| Bank (27--40)               | DNN       | fuzzy *young*     | 0.893        | 0.011                 | 0.017 ± 0.005        | 0.009                 | 0.006             |
| Bank (27--40)               | FairTSK  | none (λ=0)        | 0.894        | 0.062                 | 0.043 ± 0.016        | 0.018                 | 0.021             |
| Bank (27--40)               | FairTSK  | crisp, age ≤ 25   | 0.892        | 0.019                 | 0.015 ± 0.005        | 0.010                 | 0.009             |
| Bank (27--40)               | FairTSK  | fuzzy *young*     | 0.893        | 0.007                 | 0.014 ± 0.004        | 0.007                 | 0.004             |
| Adult (20--40)              | DNN       | none (λ=0)        | 0.849        | 0.224                 | 0.228 ± 0.006        | 0.191                 | 0.191             |
| Adult (20--40)              | DNN       | crisp, age ≤ 25   | 0.827        | 0.083                 | 0.104 ± 0.016        | 0.081                 | 0.093             |
| Adult (20--40)              | DNN       | fuzzy *young*     | 0.839        | 0.147                 | 0.149 ± 0.017        | 0.125                 | 0.128             |
| Adult (20--40)              | FairTSK  | none (λ=0)        | 0.849        | 0.225                 | 0.229 ± 0.005        | 0.193                 | 0.192             |
| Adult (20--40)              | FairTSK  | crisp, age ≤ 25   | 0.826        | 0.069                 | 0.099 ± 0.008        | 0.073                 | 0.084             |
| Adult (20--40)              | FairTSK  | fuzzy *young*     | 0.839        | 0.144                 | 0.146 ± 0.017        | 0.124                 | 0.128             |

: Models trained without penalty, with the crisp-group penalty and with the fuzzy-group penalty ($\lambda_{f} = 10$, 5 splits). Gaps are measured on test decisions; "max" and "mean" over the valid thresholds in parentheses.

**Location of the group.** Table S3 trains models against one specification of *young* (crisp ≤ 25, or ramps ending at 30, 35 or 45) and evaluates them against all four. The measured gap of the same unconstrained Bank DNN is 0.071 for the crisp group and 0.046, 0.025 and 0.008 for ramps ending at 30, 35 and 45. A model trained against the 25–35 ramp transfers well (Bank DNN: 0.006–0.011 under all four); one trained against the wide 25–45 ramp leaves a crisp gap of 0.042.

| **Dataset** | **Model** | **Trained against** | **Accuracy** | **gap: crisp ≤25** | **gap: 25–30** | **gap: 25–35** | **gap: 25–45** |
|---|---|---|---|---|---|---|---|
| Bank (5 splits) | DNN | none (λ=0) | 0.893 | 0.071 | 0.046 | 0.025 | 0.008 |
| Bank (5 splits) | DNN | crisp ≤25 | 0.892 | 0.023 | 0.013 | 0.011 | 0.013 |
| Bank (5 splits) | DNN | 25–30 | 0.892 | 0.015 | 0.011 | 0.010 | 0.013 |
| Bank (5 splits) | DNN | 25–35 | 0.893 | 0.011 | 0.006 | 0.006 | 0.009 |
| Bank (5 splits) | DNN | 25–45 | 0.893 | 0.042 | 0.028 | 0.013 | 0.003 |
| Bank (5 splits) | FairTSK | none (λ=0) | 0.894 | 0.062 | 0.041 | 0.021 | 0.007 |
| Bank (5 splits) | FairTSK | crisp ≤25 | 0.892 | 0.019 | 0.011 | 0.009 | 0.010 |
| Bank (5 splits) | FairTSK | 25–30 | 0.892 | 0.010 | 0.007 | 0.007 | 0.009 |
| Bank (5 splits) | FairTSK | 25–35 | 0.893 | 0.007 | 0.005 | 0.004 | 0.007 |
| Bank (5 splits) | FairTSK | 25–45 | 0.893 | 0.034 | 0.023 | 0.010 | 0.004 |
| German (10 splits) | DNN | none (λ=0) | 0.748 | 0.094 | 0.066 | 0.069 | 0.044 |
| German (10 splits) | DNN | crisp ≤25 | 0.742 | 0.041 | 0.033 | 0.030 | 0.032 |
| German (10 splits) | DNN | 25–30 | 0.742 | 0.043 | 0.028 | 0.032 | 0.029 |
| German (10 splits) | DNN | 25–35 | 0.750 | 0.055 | 0.030 | 0.028 | 0.025 |
| German (10 splits) | DNN | 25–45 | 0.750 | 0.069 | 0.044 | 0.039 | 0.032 |
| German (10 splits) | FairTSK | none (λ=0) | 0.751 | 0.099 | 0.064 | 0.060 | 0.033 |
| German (10 splits) | FairTSK | crisp ≤25 | 0.737 | 0.036 | 0.036 | 0.027 | 0.031 |
| German (10 splits) | FairTSK | 25–30 | 0.743 | 0.048 | 0.034 | 0.030 | 0.037 |
| German (10 splits) | FairTSK | 25–35 | 0.748 | 0.053 | 0.033 | 0.032 | 0.031 |
| German (10 splits) | FairTSK | 25–45 | 0.753 | 0.046 | 0.029 | 0.027 | 0.025 |

: Cross-evaluation of group specifications ($\lambda = 10$; decision-level test gaps, means over splits).

**Shape of the transition.** Table S4 varies the shape at a fixed crossover ($\mu = 0.5$ at 30 years): linear 25–35, a C¹ Z-spline on the same support, a sigmoid with scale 2 years, and a plateau-free ramp 18–35. On Bank, models trained against any fuzzy specification have gaps of 0.003–0.006 under every fuzzy specification, and the three shapes with the same crossover measure the gap of the unconstrained DNN as 0.023–0.029. The normative content lies in the location of the group, not in the form of the transition.

| **Dataset** | **Model** | **Trained against** | **Accuracy** | **gap: crisp ≤25** | **gap: linear 25–35** | **gap: Z-spline 25–35** | **gap: sigmoid (30, 2)** | **gap: linear 18–35** |
|---|---|---|---|---|---|---|---|---|
| Bank (5 splits) | DNN | none (λ=0) | 0.893 | 0.071 | 0.025 | 0.029 | 0.023 | 0.026 |
| Bank (5 splits) | DNN | crisp ≤25 | 0.892 | 0.023 | 0.011 | 0.011 | 0.011 | 0.011 |
| Bank (5 splits) | DNN | linear 25–35 | 0.893 | 0.011 | 0.006 | 0.006 | 0.005 | 0.005 |
| Bank (5 splits) | DNN | Z-spline 25–35 | 0.892 | 0.008 | 0.006 | 0.006 | 0.006 | 0.005 |
| Bank (5 splits) | DNN | sigmoid (30, 2) | 0.892 | 0.014 | 0.005 | 0.006 | 0.005 | 0.004 |
| Bank (5 splits) | DNN | linear 18–35 | 0.892 | 0.007 | 0.005 | 0.006 | 0.005 | 0.005 |
| Bank (5 splits) | FairTSK | none (λ=0) | 0.894 | 0.062 | 0.021 | 0.024 | 0.020 | 0.022 |
| Bank (5 splits) | FairTSK | crisp ≤25 | 0.892 | 0.019 | 0.009 | 0.009 | 0.009 | 0.008 |
| Bank (5 splits) | FairTSK | linear 25–35 | 0.893 | 0.007 | 0.004 | 0.004 | 0.004 | 0.003 |
| Bank (5 splits) | FairTSK | Z-spline 25–35 | 0.893 | 0.008 | 0.004 | 0.004 | 0.004 | 0.004 |
| Bank (5 splits) | FairTSK | sigmoid (30, 2) | 0.893 | 0.008 | 0.004 | 0.004 | 0.004 | 0.003 |
| Bank (5 splits) | FairTSK | linear 18–35 | 0.893 | 0.006 | 0.004 | 0.004 | 0.004 | 0.003 |
| German (10 splits) | DNN | none (λ=0) | 0.748 | 0.094 | 0.069 | 0.072 | 0.065 | 0.056 |
| German (10 splits) | DNN | crisp ≤25 | 0.742 | 0.041 | 0.030 | 0.032 | 0.030 | 0.026 |
| German (10 splits) | DNN | linear 25–35 | 0.750 | 0.055 | 0.028 | 0.029 | 0.026 | 0.027 |
| German (10 splits) | DNN | Z-spline 25–35 | 0.746 | 0.053 | 0.024 | 0.025 | 0.023 | 0.023 |
| German (10 splits) | DNN | sigmoid (30, 2) | 0.752 | 0.051 | 0.026 | 0.029 | 0.025 | 0.023 |
| German (10 splits) | DNN | linear 18–35 | 0.751 | 0.057 | 0.037 | 0.041 | 0.036 | 0.032 |
| German (10 splits) | FairTSK | none (λ=0) | 0.751 | 0.099 | 0.060 | 0.061 | 0.056 | 0.051 |
| German (10 splits) | FairTSK | crisp ≤25 | 0.737 | 0.036 | 0.027 | 0.030 | 0.027 | 0.024 |
| German (10 splits) | FairTSK | linear 25–35 | 0.748 | 0.053 | 0.032 | 0.035 | 0.031 | 0.027 |
| German (10 splits) | FairTSK | Z-spline 25–35 | 0.743 | 0.057 | 0.030 | 0.033 | 0.028 | 0.024 |
| German (10 splits) | FairTSK | sigmoid (30, 2) | 0.750 | 0.044 | 0.027 | 0.030 | 0.027 | 0.025 |
| German (10 splits) | FairTSK | linear 18–35 | 0.743 | 0.048 | 0.027 | 0.029 | 0.026 | 0.024 |

: Shape of the membership function ($\lambda = 10$; decision-level test gaps, means over splits).

# Additional results of the main comparison

| **Method** | **German** | **Student** | **COMPAS** | **Default** | **Bank** | **Adult** | **Diabetes** | **Census-KDD** |
|---|---|---|---|---|---|---|---|---|
| LR | 0.744 / 0.071 | 0.855 / 0.034 | 0.678 / 0.222 | 0.810 / 0.015 | 0.892 / 0.021 | 0.848 / 0.183 | 0.889 / 0.001 | 0.952 / 0.049 |
| LightGBM | 0.746 / 0.058 | 0.853 / 0.045 | 0.667 / 0.203 | 0.819 / 0.023 | 0.894 / 0.022 | 0.868 / 0.184 | 0.890 / 0.001 | 0.955 / 0.052 |
| Reweighing | 0.749 / 0.061 | 0.852 / 0.037 | 0.663 / 0.195 | 0.819 / 0.021 | 0.894 / 0.017 | 0.865 / 0.114 | 0.890 / 0.001 | 0.953 / 0.036 |
| Exp. Gradient | 0.739 / 0.041 | 0.850 / 0.043 | 0.567 / 0.073 | 0.810 / 0.014 | 0.892 / 0.009 | 0.833 / 0.031 | 0.889 / 0.001 | 0.952 / 0.042 |
| Threshold Opt. | 0.737 / 0.040 | 0.842 / 0.035 | 0.649 / 0.066 | 0.817 / 0.010 | 0.894 / 0.007 | 0.845 / 0.010 | 0.890 / 0.002 | 0.947 / 0.003 |
| Adv. debiasing | 0.747 / 0.063 | 0.855 / 0.027 | 0.671 / 0.210 | 0.818 / 0.024 | 0.893 / 0.017 | 0.845 / 0.106 | 0.890 / 0.000 | 0.952 / 0.045 |
| DNN+FDP | 0.744 / 0.044 | 0.852 / 0.043 | 0.565 / 0.024 | 0.818 / 0.022 | 0.893 / 0.014 | 0.838 / 0.043 | 0.890 / 0.001 | 0.952 / 0.045 |
| FairTSK | 0.742 / 0.032 | 0.852 / 0.026 | 0.565 / 0.020 | 0.819 / 0.018 | 0.893 / 0.014 | 0.840 / 0.049 | 0.889 / 0.000 | 0.953 / 0.044 |
| FairTSK-0 | 0.740 / 0.042 | 0.848 / 0.011 | 0.557 / 0.006 | 0.820 / 0.021 | 0.891 / 0.009 | 0.838 / 0.039 | 0.890 / 0.000 | 0.952 / 0.041 |

: Test accuracy / FDP at the operating point selected on validation data for $\epsilon = 0.05$ (mean over 10 splits; 5 for Diabetes and Census-KDD).

| **Dataset** | **Method** | **Accuracy** | **AUC** | **F1** | **FDP** | **EO diff.** | **HV** |
|---|---|---|---|---|---|---|---|
| German | LR | 0.744 ± 0.022 | 0.774 ± 0.034 | 0.826 ± 0.013 | 0.071 ± 0.032 | 0.143 ± 0.070 | — |
| German | LightGBM | 0.746 ± 0.017 | 0.755 ± 0.032 | 0.827 ± 0.012 | 0.058 ± 0.046 | 0.099 ± 0.076 | — |
| German | Reweighing | 0.749 ± 0.023 | 0.755 ± 0.026 | 0.829 ± 0.017 | 0.061 ± 0.050 | 0.136 ± 0.059 | 0.121 ± 0.075 |
| German | Exp. Gradient | 0.739 ± 0.022 | 0.663 ± 0.033 | 0.823 ± 0.015 | 0.041 ± 0.029 | 0.177 ± 0.060 | 0.113 ± 0.040 |
| German | Threshold Opt. | 0.737 ± 0.031 | 0.672 ± 0.025 | 0.823 ± 0.034 | 0.040 ± 0.041 | 0.193 ± 0.121 | 0.112 ± 0.067 |
| German | Adv. debiasing | 0.747 ± 0.013 | 0.774 ± 0.032 | 0.828 ± 0.008 | 0.063 ± 0.038 | 0.205 ± 0.068 | 0.130 ± 0.041 |
| German | DNN+FDP | 0.744 ± 0.027 | 0.774 ± 0.033 | 0.833 ± 0.012 | 0.044 ± 0.037 | 0.137 ± 0.082 | 0.174 ± 0.032 |
| German | FairTSK | 0.742 ± 0.014 | 0.775 ± 0.025 | 0.834 ± 0.010 | 0.032 ± 0.034 | 0.107 ± 0.071 | 0.172 ± 0.037 |
| German | FairTSK-0 | 0.740 ± 0.032 | 0.767 ± 0.030 | 0.831 ± 0.020 | 0.042 ± 0.029 | 0.120 ± 0.073 | 0.176 ± 0.058 |
| Student | LR | 0.855 ± 0.013 | 0.792 ± 0.050 | 0.918 ± 0.007 | 0.034 ± 0.037 | 0.210 ± 0.137 | — |
| Student | LightGBM | 0.853 ± 0.009 | 0.793 ± 0.046 | 0.916 ± 0.005 | 0.045 ± 0.034 | 0.191 ± 0.134 | — |
| Student | Reweighing | 0.852 ± 0.014 | 0.799 ± 0.035 | 0.916 ± 0.008 | 0.037 ± 0.024 | 0.178 ± 0.148 | 0.051 ± 0.060 |
| Student | Exp. Gradient | 0.850 ± 0.015 | 0.632 ± 0.034 | 0.915 ± 0.009 | 0.043 ± 0.030 | 0.245 ± 0.149 | 0.071 ± 0.079 |
| Student | Threshold Opt. | 0.842 ± 0.005 | 0.650 ± 0.061 | 0.913 ± 0.004 | 0.035 ± 0.031 | 0.125 ± 0.165 | 0.001 ± 0.001 |
| Student | Adv. debiasing | 0.855 ± 0.015 | 0.808 ± 0.070 | 0.920 ± 0.008 | 0.027 ± 0.049 | 0.115 ± 0.156 | 0.096 ± 0.107 |
| Student | DNN+FDP | 0.852 ± 0.019 | 0.822 ± 0.055 | 0.918 ± 0.010 | 0.043 ± 0.050 | 0.130 ± 0.147 | 0.081 ± 0.078 |
| Student | FairTSK | 0.852 ± 0.007 | 0.819 ± 0.050 | 0.918 ± 0.004 | 0.026 ± 0.045 | 0.082 ± 0.154 | 0.080 ± 0.037 |
| Student | FairTSK-0 | 0.848 ± 0.010 | 0.816 ± 0.057 | 0.917 ± 0.005 | 0.011 ± 0.011 | 0.091 ± 0.128 | 0.034 ± 0.054 |
| COMPAS | LR | 0.678 ± 0.007 | 0.731 ± 0.010 | 0.730 ± 0.007 | 0.222 ± 0.024 | 0.247 ± 0.029 | — |
| COMPAS | LightGBM | 0.667 ± 0.010 | 0.709 ± 0.012 | 0.711 ± 0.010 | 0.203 ± 0.024 | 0.193 ± 0.020 | — |
| COMPAS | Reweighing | 0.663 ± 0.011 | 0.704 ± 0.011 | 0.709 ± 0.011 | 0.195 ± 0.019 | 0.183 ± 0.020 | 0.012 ± 0.018 |
| COMPAS | Exp. Gradient | 0.567 ± 0.046 | 0.534 ± 0.030 | 0.687 ± 0.041 | 0.073 ± 0.018 | 0.107 ± 0.029 | 0.067 ± 0.024 |
| COMPAS | Threshold Opt. | 0.649 ± 0.012 | 0.656 ± 0.015 | 0.689 ± 0.015 | 0.066 ± 0.031 | 0.101 ± 0.051 | 0.153 ± 0.039 |
| COMPAS | Adv. debiasing | 0.671 ± 0.017 | 0.721 ± 0.018 | 0.723 ± 0.015 | 0.210 ± 0.037 | 0.235 ± 0.020 | 0.012 ± 0.023 |
| COMPAS | DNN+FDP | 0.565 ± 0.010 | 0.644 ± 0.023 | 0.711 ± 0.004 | 0.024 ± 0.014 | 0.037 ± 0.023 | 0.106 ± 0.020 |
| COMPAS | FairTSK | 0.565 ± 0.012 | 0.661 ± 0.024 | 0.712 ± 0.004 | 0.020 ± 0.013 | 0.030 ± 0.017 | 0.104 ± 0.022 |
| COMPAS | FairTSK-0 | 0.557 ± 0.006 | 0.666 ± 0.016 | 0.709 ± 0.002 | 0.006 ± 0.007 | 0.013 ± 0.010 | 0.090 ± 0.020 |
| Default | LR | 0.810 ± 0.004 | 0.722 ± 0.007 | 0.888 ± 0.003 | 0.015 ± 0.005 | 0.015 ± 0.010 | — |
| Default | LightGBM | 0.819 ± 0.004 | 0.773 ± 0.007 | 0.891 ± 0.003 | 0.023 ± 0.010 | 0.023 ± 0.012 | — |
| Default | Reweighing | 0.819 ± 0.004 | 0.772 ± 0.007 | 0.891 ± 0.003 | 0.021 ± 0.010 | 0.025 ± 0.015 | 0.160 ± 0.017 |
| Default | Exp. Gradient | 0.810 ± 0.004 | 0.607 ± 0.004 | 0.888 ± 0.003 | 0.014 ± 0.005 | 0.014 ± 0.010 | 0.131 ± 0.020 |
| Default | Threshold Opt. | 0.817 ± 0.003 | 0.665 ± 0.009 | 0.890 ± 0.002 | 0.010 ± 0.010 | 0.049 ± 0.027 | 0.165 ± 0.019 |
| Default | Adv. debiasing | 0.818 ± 0.005 | 0.772 ± 0.007 | 0.890 ± 0.003 | 0.024 ± 0.010 | 0.027 ± 0.014 | 0.161 ± 0.022 |
| Default | DNN+FDP | 0.818 ± 0.005 | 0.772 ± 0.008 | 0.890 ± 0.003 | 0.022 ± 0.010 | 0.027 ± 0.013 | 0.173 ± 0.022 |
| Default | FairTSK | 0.819 ± 0.004 | 0.772 ± 0.008 | 0.891 ± 0.002 | 0.018 ± 0.007 | 0.024 ± 0.013 | 0.179 ± 0.022 |
| Default | FairTSK-0 | 0.820 ± 0.004 | 0.774 ± 0.004 | 0.892 ± 0.002 | 0.021 ± 0.008 | 0.022 ± 0.014 | 0.182 ± 0.018 |
| Bank | LR | 0.892 ± 0.001 | 0.767 ± 0.008 | 0.286 ± 0.013 | 0.021 ± 0.005 | 0.082 ± 0.044 | — |
| Bank | LightGBM | 0.894 ± 0.003 | 0.797 ± 0.008 | 0.364 ± 0.020 | 0.022 ± 0.009 | 0.068 ± 0.053 | — |
| Bank | Reweighing | 0.894 ± 0.002 | 0.796 ± 0.007 | 0.359 ± 0.016 | 0.017 ± 0.006 | 0.057 ± 0.034 | 0.083 ± 0.014 |
| Bank | Exp. Gradient | 0.892 ± 0.001 | 0.584 ± 0.007 | 0.274 ± 0.014 | 0.009 ± 0.005 | 0.038 ± 0.027 | 0.078 ± 0.012 |
| Bank | Threshold Opt. | 0.894 ± 0.002 | 0.619 ± 0.022 | 0.325 ± 0.050 | 0.007 ± 0.005 | 0.117 ± 0.051 | 0.088 ± 0.019 |
| Bank | Adv. debiasing | 0.893 ± 0.001 | 0.785 ± 0.008 | 0.354 ± 0.024 | 0.017 ± 0.005 | 0.039 ± 0.022 | 0.088 ± 0.012 |
| Bank | DNN+FDP | 0.893 ± 0.002 | 0.787 ± 0.009 | 0.347 ± 0.037 | 0.014 ± 0.006 | 0.064 ± 0.046 | 0.095 ± 0.009 |
| Bank | FairTSK | 0.893 ± 0.002 | 0.787 ± 0.008 | 0.315 ± 0.013 | 0.014 ± 0.011 | 0.067 ± 0.051 | 0.095 ± 0.011 |
| Bank | FairTSK-0 | 0.891 ± 0.001 | 0.778 ± 0.011 | 0.303 ± 0.020 | 0.009 ± 0.007 | 0.070 ± 0.035 | 0.082 ± 0.009 |
| Adult | LR | 0.848 ± 0.003 | 0.903 ± 0.003 | 0.663 ± 0.008 | 0.183 ± 0.008 | 0.095 ± 0.021 | — |
| Adult | LightGBM | 0.868 ± 0.004 | 0.925 ± 0.003 | 0.712 ± 0.008 | 0.184 ± 0.006 | 0.082 ± 0.014 | — |
| Adult | Reweighing | 0.865 ± 0.004 | 0.922 ± 0.003 | 0.696 ± 0.010 | 0.114 ± 0.007 | 0.101 ± 0.027 | 0.195 ± 0.017 |
| Adult | Exp. Gradient | 0.833 ± 0.003 | 0.728 ± 0.011 | 0.597 ± 0.010 | 0.031 ± 0.012 | 0.249 ± 0.032 | 0.303 ± 0.025 |
| Adult | Threshold Opt. | 0.845 ± 0.004 | 0.759 ± 0.007 | 0.639 ± 0.018 | 0.010 ± 0.006 | 0.319 ± 0.023 | 0.357 ± 0.022 |
| Adult | Adv. debiasing | 0.845 ± 0.003 | 0.898 ± 0.006 | 0.647 ± 0.013 | 0.106 ± 0.031 | 0.113 ± 0.061 | 0.182 ± 0.061 |
| Adult | DNN+FDP | 0.838 ± 0.004 | 0.891 ± 0.008 | 0.614 ± 0.015 | 0.043 ± 0.022 | 0.239 ± 0.054 | 0.354 ± 0.011 |
| Adult | FairTSK | 0.840 ± 0.004 | 0.893 ± 0.007 | 0.618 ± 0.014 | 0.049 ± 0.018 | 0.222 ± 0.037 | 0.354 ± 0.009 |
| Adult | FairTSK-0 | 0.838 ± 0.003 | 0.893 ± 0.005 | 0.610 ± 0.008 | 0.039 ± 0.015 | 0.247 ± 0.037 | 0.351 ± 0.014 |
| Diabetes | LR | 0.889 ± 0.002 | 0.672 ± 0.008 | 0.941 ± 0.001 | 0.001 ± 0.000 | 0.004 ± 0.002 | — |
| Diabetes | LightGBM | 0.890 ± 0.003 | 0.672 ± 0.008 | 0.942 ± 0.001 | 0.001 ± 0.001 | 0.008 ± 0.005 | — |
| Diabetes | Reweighing | 0.890 ± 0.003 | 0.672 ± 0.007 | 0.942 ± 0.001 | 0.001 ± 0.001 | 0.006 ± 0.003 | 0.013 ± 0.022 |
| Diabetes | Exp. Gradient | 0.889 ± 0.002 | 0.505 ± 0.002 | 0.941 ± 0.001 | 0.001 ± 0.000 | 0.003 ± 0.001 | 0.010 ± 0.021 |
| Diabetes | Threshold Opt. | 0.890 ± 0.003 | 0.510 ± 0.004 | 0.942 ± 0.002 | 0.002 ± 0.001 | 0.011 ± 0.007 | 0.013 ± 0.024 |
| Diabetes | Adv. debiasing | 0.890 ± 0.003 | 0.675 ± 0.009 | 0.942 ± 0.001 | 0.000 ± 0.001 | 0.004 ± 0.004 | 0.012 ± 0.022 |
| Diabetes | DNN+FDP | 0.890 ± 0.003 | 0.676 ± 0.008 | 0.942 ± 0.001 | 0.001 ± 0.001 | 0.002 ± 0.003 | 0.012 ± 0.022 |
| Diabetes | FairTSK | 0.889 ± 0.003 | 0.677 ± 0.008 | 0.941 ± 0.001 | 0.000 ± 0.000 | 0.002 ± 0.002 | 0.012 ± 0.022 |
| Diabetes | FairTSK-0 | 0.890 ± 0.003 | 0.671 ± 0.009 | 0.942 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.011 ± 0.022 |
| Census-KDD | LR | 0.952 ± 0.002 | 0.940 ± 0.003 | 0.500 ± 0.014 | 0.049 ± 0.002 | 0.176 ± 0.010 | — |
| Census-KDD | LightGBM | 0.955 ± 0.001 | 0.946 ± 0.002 | 0.546 ± 0.010 | 0.052 ± 0.003 | 0.154 ± 0.019 | — |
| Census-KDD | Reweighing | 0.953 ± 0.002 | 0.942 ± 0.004 | 0.510 ± 0.012 | 0.036 ± 0.004 | 0.055 ± 0.036 | 0.202 ± 0.024 |
| Census-KDD | Exp. Gradient | 0.952 ± 0.002 | 0.685 ± 0.006 | 0.481 ± 0.026 | 0.042 ± 0.007 | 0.139 ± 0.029 | 0.199 ± 0.028 |
| Census-KDD | Threshold Opt. | 0.947 ± 0.002 | 0.647 ± 0.016 | 0.383 ± 0.021 | 0.003 ± 0.001 | 0.206 ± 0.031 | 0.145 ± 0.026 |
| Census-KDD | Adv. debiasing | 0.952 ± 0.002 | 0.940 ± 0.003 | 0.504 ± 0.011 | 0.045 ± 0.003 | 0.136 ± 0.036 | 0.190 ± 0.027 |
| Census-KDD | DNN+FDP | 0.952 ± 0.002 | 0.940 ± 0.003 | 0.506 ± 0.029 | 0.045 ± 0.007 | 0.146 ± 0.027 | 0.216 ± 0.030 |
| Census-KDD | FairTSK | 0.953 ± 0.002 | 0.942 ± 0.003 | 0.492 ± 0.020 | 0.044 ± 0.005 | 0.156 ± 0.015 | 0.219 ± 0.028 |
| Census-KDD | FairTSK-0 | 0.952 ± 0.002 | 0.940 ± 0.004 | 0.488 ± 0.018 | 0.041 ± 0.004 | 0.139 ± 0.028 | 0.214 ± 0.027 |

: Full results at the operating point for $\epsilon = 0.05$ (mean ± s.d. over splits). EO diff.: equalised-odds difference (crisp group). HV of the test front for families with a trade-off.

| **Method** | **German** | **Student** | **COMPAS** | **Default** | **Bank** | **Adult** | **Diabetes** | **Census-KDD** |
|---|---|---|---|---|---|---|---|---|
| LR | 0.143 | 0.210 | 0.247 | 0.015 | 0.082 | 0.095 | 0.004 | 0.176 |
| LightGBM | 0.099 | 0.191 | 0.193 | 0.023 | 0.068 | 0.082 | 0.008 | 0.154 |
| Reweighing | 0.136 | 0.178 | 0.183 | 0.025 | 0.057 | 0.101 | 0.006 | 0.055 |
| Exp. Gradient | 0.177 | 0.245 | 0.107 | 0.014 | 0.038 | 0.249 | 0.003 | 0.139 |
| Threshold Opt. | 0.193 | 0.125 | 0.101 | 0.049 | 0.117 | 0.319 | 0.011 | 0.206 |
| Adv. debiasing | 0.205 | 0.115 | 0.235 | 0.027 | 0.039 | 0.113 | 0.004 | 0.136 |
| DNN+FDP | 0.137 | 0.130 | 0.037 | 0.027 | 0.064 | 0.239 | 0.002 | 0.146 |
| FairTSK | 0.107 | 0.082 | 0.030 | 0.024 | 0.067 | 0.222 | 0.002 | 0.156 |
| FairTSK-0 | 0.120 | 0.091 | 0.013 | 0.022 | 0.070 | 0.247 | 0.000 | 0.139 |

: Equalised-odds difference at the operating point ($\epsilon = 0.05$; mean over splits).

| **Reference FDP** | **Datasets** | **Friedman** $p$ | **vs DNN+FDP** | **vs FairTSK-0** | **vs Threshold Opt.** | **vs Exp. Gradient** | **vs Adv. debiasing** | **vs Reweighing** |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 8 | 0.011 | 0.0000 (0.641) | +0.0077 (0.234) | +0.0176 (0.500) | +0.0334 (0.117) | +0.0553 (0.219) | +0.0666 (0.094) |
| 0.1 | 7 (without Diabetes) | 0.004 | +0.0001 (0.813) | +0.0088 (0.234) | +0.0202 (0.438) | +0.0379 (0.156) | +0.0632 (0.188) | +0.0763 (0.094) |
| 0.2 | 8 | 0.017 | +0.0005 (0.844) | +0.0093 (0.328) | +0.0226 (0.391) | +0.0303 (0.047) | +0.0429 (0.219) | +0.0472 (0.078) |
| 0.2 | 7 (without Diabetes) | 0.009 | +0.0006 (1.000) | +0.0106 (0.328) | +0.0260 (0.328) | +0.0344 (0.094) | +0.0491 (0.188) | +0.0541 (0.094) |
| 0.3 | 8 | 0.009 | +0.0008 (0.547) | +0.0089 (0.313) | +0.0298 (0.313) | +0.0295 (0.047) | +0.0318 (0.313) | +0.0341 (0.078) |
| 0.3 | 7 (without Diabetes) | 0.003 | +0.0010 (0.375) | +0.0102 (0.313) | +0.0341 (0.313) | +0.0334 (0.094) | +0.0364 (0.313) | +0.0392 (0.094) |

: Sensitivity of the dataset-level comparison to the HV reference point and to excluding the Diabetes control. Cells: mean HV difference FairTSK minus competitor (Holm-adjusted Wilcoxon $p$).

| **FairTSK vs.** | **Δ HV, test front [90% CI]** | **wins/8** | $p_{Holm}$ | **Δ acc@0.05 [90% CI]** | $p_{Holm}$ |
|---|---|---|---|---|---|
| DNN+FDP | +0.0034 [−0.0016, +0.0085] | 4 | 0.586 | +0.0004 [−0.0010, +0.0017] | 1.000 |
| FairTSK-0 | +0.0088 [−0.0025, +0.0202] | 6 | 0.586 | +0.0019 [+0.0000, +0.0038] | 0.438 |
| Threshold Opt. | +0.0241 [−0.0066, +0.0547] | 5 | 0.586 | −0.0083 [−0.0288, +0.0122] | 1.000 |
| Exp. Gradient | +0.0322 [+0.0178, +0.0467] | 8 | 0.047 | +0.0036 [+0.0014, +0.0057] | 0.047 |
| Adv. debiasing | +0.0461 [+0.0074, +0.0849] | 7 | 0.078 | −0.0143 [−0.0389, +0.0103] | 0.586 |
| Reweighing | +0.0502 [+0.0135, +0.0869] | 7 | 0.078 | −0.0160 [−0.0386, +0.0067] | 0.234 |

: Dataset-level comparison with split 0, which was also used in the pilot runs, removed (Wilcoxon over datasets, Holm-adjusted; confidence intervals unadjusted).

**Evaluation against the crisp core group.** On German and Bank the fuzzy-group methods are trained against *young*, whereas the crisp baselines are trained against its core (age ≤ 25). To check that this does not favour the fuzzy-group methods, Table S10 evaluates every family against the crisp core: the HV of the front of accuracy against the crisp gap, and the operating point selected on the validation crisp gap ($\epsilon = 0.05$). The fuzzy-penalty families keep the largest HV on both datasets (German: FairTSK-0 0.174, FairTSK 0.161, DNN+FDP 0.158 against at most 0.110 for the crisp baselines; Bank: DNN+FDP and FairTSK 0.093 against at most 0.086).

| **Method** | **German: HV (crisp)** | **German: acc / crisp gap** | **Bank: HV (crisp)** | **Bank: acc / crisp gap** |
|---|---|---|---|---|
| Reweighing | 0.100 | 0.749 / 0.085 | 0.072 | 0.894 / 0.043 |
| Exp. Gradient | 0.098 | 0.741 / 0.054 | 0.075 | 0.892 / 0.017 |
| Threshold Opt. | 0.089 | 0.737 / 0.076 | 0.086 | 0.894 / 0.012 |
| Adv. debiasing | 0.110 | 0.748 / 0.095 | 0.080 | 0.894 / 0.039 |
| DNN+FDP | 0.158 | 0.736 / 0.043 | 0.093 | 0.893 / 0.019 |
| FairTSK | 0.161 | 0.745 / 0.060 | 0.093 | 0.894 / 0.024 |
| FairTSK-0 | 0.174 | 0.736 / 0.029 | 0.080 | 0.892 / 0.022 |

: German and Bank evaluated against the crisp group age ≤ 25 (means over 10 splits).

| **Dataset** | **Exp. Gradient (LR): HV** | **Exp. Gradient (LightGBM): HV** | **acc / FDP at** $\epsilon = 0.05$ | **FairTSK: HV** |
|---|---|---|---|---|
| German | 0.113 | 0.104 | 0.742 / 0.050 | 0.172 |
| Student | 0.071 | 0.058 | 0.854 / 0.036 | 0.080 |
| COMPAS | 0.067 | 0.093 | 0.616 / 0.108 | 0.104 |
| Default | 0.131 | 0.168 | 0.819 / 0.021 | 0.179 |
| Bank | 0.078 | 0.095 | 0.893 / 0.011 | 0.095 |
| Adult | 0.303 | 0.359 | 0.853 / 0.040 | 0.354 |
| Diabetes | 0.010 | 0.012 | 0.890 / 0.001 | 0.012 |
| Census-KDD | 0.199 | 0.238 | 0.955 / 0.048 | 0.219 |

: Exponentiated Gradient with a LightGBM base learner (100 trees, learning rate 0.1, 31 leaves; same grid $\epsilon \in \{0.01,0.02,0.05,0.1\}$) compared with the LR base learner of the main comparison and with FairTSK: test-front HV and operating point at $\epsilon = 0.05$ (means over splits). On COMPAS no configuration met $\epsilon = 0.05$ on validation data in any split, and the least unfair one is reported. Dataset-level difference FairTSK minus Exponentiated Gradient (LightGBM): +0.011, Wilcoxon $p = 0.38$.

| **Method** | **HV (FDP, young)** | **acc / FDP at** $\epsilon = 0.05$ | **crisp gap (age ≤ 25)** | **HV (crisp gap)** | **Δ HV vs FairTSK** ($p_{Holm}$) |
|---|---|---|---|---|---|
| LR | — | 0.847 / 0.191 | 0.222 | — | — |
| LightGBM | — | 0.864 / 0.197 | 0.225 | — | — |
| Reweighing | 0.036 ± 0.010 | 0.862 / 0.183 | 0.210 | 0.000 | +0.152 (< 0.001) |
| Exp. Gradient | 0.081 ± 0.005 | 0.845 / 0.157 | 0.179 | 0.039 | +0.107 (0.003) |
| Threshold Opt. | 0.226 ± 0.018 | 0.830 / 0.054 | 0.027 | 0.272 | −0.038 (0.274) |
| Adv. debiasing | 0.060 ± 0.028 | 0.845 / 0.169 | 0.196 | 0.018 | +0.129 (0.003) |
| DNN+FDP | 0.150 ± 0.017 | 0.835 / 0.118 | 0.134 | 0.122 | +0.038 (0.274) |
| FairTSK | 0.188 ± 0.031 | 0.820 / 0.087 | 0.098 | 0.165 | — |
| FairTSK-0 | 0.146 ± 0.021 | 0.835 / 0.121 | 0.138 | 0.110 | +0.042 (0.274) |

: Adult with the fuzzy group *young* as protected group (age and sex are not inputs; 10 splits; means, HV with s.d.). Operating point: no in-processing family reached validation FDP $\leq 0.05$ with the penalty grid of the main comparison, so the least unfair configuration is reported; Threshold Optimizer met it in 4 of 10 splits. Crisp gap: decision-rate difference for the core age $\leq 25$. Δ HV: FairTSK minus competitor, Nadeau–Bengio test, Holm-adjusted over six comparisons.

![Mean ranks of the fairness-aware methods by dataset-level hypervolume (test fronts, lower is better), with the Nemenyi critical difference at α = 0.05](figures/fig_cd.pdf){width=100%}

| **Dataset** | **FairTSK: acc / FDP** | **HV test / val.** | **Fair tree: acc / FDP** | **HV test / val.** | **Leaves** | **FairCORELS: acc / FDP** | **HV test / val.** | **Rules** |
|---|---|---|---|---|---|---|---|---|
| German | 0.742 / 0.032 | 0.172/0.149 | 0.716 / 0.029 | 0.089/0.049 | 15.8 | 0.703 / 0.047 | 0.045/0.035 | 5.0 |
| Student | 0.852 / 0.026 | 0.080/0.047 | 0.842 / 0.036 | 0.091/0.071 | 18.4 | 0.851 / 0.040 | 0.080/0.076 | 5.1 |
| COMPAS | 0.565 / 0.020 | 0.104/0.104 | 0.570 / 0.020 | 0.070/0.066 | 17.8 | 0.572 / 0.026 | 0.106/0.106 | 4.7 |
| Default | 0.819 / 0.018 | 0.179/0.174 | 0.819 / 0.022 | 0.178/0.176 | 20.8 | 0.819 / 0.016 | 0.173/0.173 | 2.2 |
| Bank | 0.893 / 0.014 | 0.095/0.091 | 0.893 / 0.007 | 0.091/0.087 | 53.6 | 0.893 / 0.008 | 0.084/0.084 | 2.2 |
| Adult | 0.840 / 0.049 | 0.354/0.345 | 0.804 / 0.047 | 0.225/0.224 | 33.2 | 0.791 / 0.035 | 0.180/0.178 | 5.0 |
| Diabetes | 0.889 / 0.000 | 0.012/0.011 | 0.890 / 0.001 | 0.013/0.011 | 24.2 | 0.890 / 0.000 | 0.011/0.011 | 1.0 |
| Census-KDD | 0.953 / 0.044 | 0.219/0.218 | 0.943 / 0.014 | 0.081/0.072 | 60.6 | 0.940 / 0.006 | 0.033/0.033 | 2.8 |

: FairTSK versus interpretable fairness-aware models: test accuracy / FDP at the operating point ($\epsilon = 0.05$), HV of the test front and of the validation-selected front, and model size at the operating point (leaves; rules of the list including the default rule). Means over splits. Dataset-level differences FairTSK minus fair tree / FairCORELS: test-front HV +0.047 (90% CI [+0.006, +0.088]) / +0.063 ([+0.006, +0.119]), Holm-adjusted $p = 0.11$ for both; validation-selected fronts $p = 0.30$.

**Calibration.** Without a penalty, FairTSK and the DNN are equally well calibrated (ECE 0.010–0.012 on Adult, 0.035–0.038 on COMPAS). The parity penalty worsens calibration within groups for both, as expected when parity is enforced against unequal base rates: on Adult the within-group ECE rises from 0.016–0.019 to 0.066–0.106 at $\lambda_f = 10$, while the overall ECE stays below 0.02 (Table S14).

| **Dataset** | **Model** | **Acc.** | **FDP** | **ECE** | **ECE group / rest** | **Brier** | **Bootstrap s.d.: gap / $\mathcal R$ / $\mathcal W$** |
|---|---|---|---|---|---|---|---|
| German (10 splits) | DNN | 0.748 | 0.069 | 0.084 | 0.204 / 0.088 | 0.168 | — |
| German (10 splits) | DNN+FDP, λ=10 | 0.750 | 0.028 | 0.073 | 0.210 / 0.079 | 0.170 | — |
| German (10 splits) | FairTSK, λ=0 | 0.751 | 0.060 | 0.077 | 0.206 / 0.090 | 0.168 | 0.028 / 0.002 / 0.026 |
| German (10 splits) | FairTSK, λ=10 | 0.748 | 0.032 | 0.081 | 0.233 / 0.077 | 0.170 | 0.024 / 0.001 / 0.024 |
| German (10 splits) | FairTSK-rule, λ=100 | 0.734 | 0.036 | 0.086 | 0.207 / 0.088 | 0.176 | 0.019 / 0.014 / 0.006 |
| COMPAS (10 splits) | DNN | 0.683 | 0.235 | 0.038 | 0.048 / 0.053 | 0.206 | — |
| COMPAS (10 splits) | DNN+FDP, λ=10 | 0.658 | 0.171 | 0.086 | 0.097 / 0.082 | 0.223 | — |
| COMPAS (10 splits) | FairTSK, λ=0 | 0.681 | 0.227 | 0.035 | 0.052 / 0.051 | 0.206 | 0.011 / 0.006 / 0.006 |
| COMPAS (10 splits) | FairTSK, λ=10 | 0.660 | 0.177 | 0.089 | 0.097 / 0.090 | 0.224 | 0.005 / 0.002 / 0.004 |
| COMPAS (10 splits) | FairTSK-rule, λ=100 | 0.683 | 0.237 | 0.033 | 0.049 / 0.054 | 0.206 | 0.011 / 0.011 / 0.000 |
| Bank (10 splits) | DNN | 0.893 | 0.024 | 0.010 | 0.074 / 0.010 | 0.083 | — |
| Bank (10 splits) | DNN+FDP, λ=10 | 0.893 | 0.006 | 0.010 | 0.102 / 0.010 | 0.083 | — |
| Bank (10 splits) | FairTSK, λ=0 | 0.894 | 0.022 | 0.012 | 0.073 / 0.012 | 0.083 | 0.004 / 0.001 / 0.003 |
| Bank (10 splits) | FairTSK, λ=10 | 0.893 | 0.004 | 0.011 | 0.101 / 0.012 | 0.084 | 0.003 / 0.001 / 0.003 |
| Bank (10 splits) | FairTSK-rule, λ=100 | 0.893 | 0.024 | 0.010 | 0.080 / 0.010 | 0.083 | 0.004 / 0.004 / 0.001 |
| Adult (10 splits) | DNN | 0.851 | 0.184 | 0.012 | 0.016 / 0.019 | 0.102 | — |
| Adult (10 splits) | DNN+FDP, λ=10 | 0.833 | 0.015 | 0.017 | 0.106 / 0.066 | 0.116 | — |
| Adult (10 splits) | FairTSK, λ=0 | 0.852 | 0.189 | 0.010 | 0.017 / 0.016 | 0.102 | 0.006 / 0.003 / 0.004 |
| Adult (10 splits) | FairTSK, λ=10 | 0.833 | 0.015 | 0.019 | 0.106 / 0.066 | 0.116 | 0.006 / 0.002 / 0.004 |
| Adult (10 splits) | FairTSK-rule, λ=100 | 0.853 | 0.187 | 0.012 | 0.017 / 0.017 | 0.101 | 0.006 / 0.005 / 0.000 |

: Calibration and bootstrap variability of the decomposition (test data, means over 10 splits). ECE: expected calibration error (15 bins), overall and within the crisp group and its complement; bootstrap s.d.: over 1,000 test-set resamples for the fixed trained model.

# Where FDP moves disparity along the age axis

| **Dataset** | **Model** | λ | **Accuracy** | **abs. corr(age)** | **≤25** | **26–30** | **31–35** | **36–45** | **46–60** | **>60** |
|---|---|---|---|---|---|---|---|---|---|---|
| Bank | DNN | 0 | 0.893 | 0.055 | +0.067 | +0.020 | −0.012 | −0.018 | −0.012 | +0.268 |
| Bank | DNN | 10 | 0.893 | 0.122 | +0.001 | −0.001 | −0.013 | −0.014 | −0.005 | +0.323 |
| Bank | FairTSK | 0 | 0.894 | 0.043 | +0.062 | +0.016 | −0.009 | −0.015 | −0.011 | +0.217 |
| Bank | FairTSK | 10 | 0.893 | 0.106 | +0.002 | +0.000 | −0.009 | −0.012 | −0.005 | +0.259 |
| German | DNN | 0 | 0.748 | 0.051 | −0.065 | −0.030 | +0.031 | +0.081 | −0.035 | +0.060 |
| German | DNN | 10 | 0.751 | 0.057 | −0.012 | +0.001 | +0.016 | +0.042 | −0.068 | +0.010 |
| German | FairTSK | 0 | 0.752 | 0.046 | −0.076 | −0.006 | +0.026 | +0.077 | −0.046 | +0.036 |
| German | FairTSK | 10 | 0.749 | 0.068 | −0.023 | +0.013 | +0.021 | +0.033 | −0.067 | +0.006 |
| Bank | test instances per band | | | | 267 | 1123 | 1974 | 2777 | 2657 | 245 |
| German | test instances per band | | | | 38 | 48 | 35 | 42 | 29 | 8 |

: Decision rate in each age band minus the overall decision rate (test data, means over 10 splits), without penalty ($\lambda = 0$) and with the FDP penalty for *young* ($\lambda = 10$). The last rows give the mean number of test instances per band; the German bands above 45 are small.

# Decomposition: further results

**Corollary S1 (Bound).** With $D := \max_{l}\bar{r}_{l} - \min_{l}\bar{r}_{l}$ and $TV(m^{G},m^{\neg G}) = \frac{1}{2}\sum_{l}|m_{l}^{G} - m_{l}^{\neg G}|$, $\mathrm{FDP}_{G}(\hat{y}) \leq TV\cdot D + \sum_{l}\bar{m}_{l}\,|r_{l}^{G} - r_{l}^{\neg G}|$. *Proof.* Choose $c$ as the midpoint of the $\bar r_l$ in $\mathcal R = \sum_l (m_l^G - m_l^{\neg G})(\bar r_l - c)$; the second term is the triangle inequality. $\square$ Hence $\mathrm{FDP}_G(\hat y) = 0$ if $r_l^G = r_l^{\neg G}$ for all $l$ and either $m^G = m^{\neg G}$ or all $\bar r_l$ are equal.

| **Dataset** | **Models** | **Pearson** $r$ |
|---|---|---|
| Adult | 150 | 0.99 |
| Bank | 150 | 0.70 |
| COMPAS | 150 | 0.86 |
| Census-KDD | 75 | 0.98 |
| Default | 150 | 0.87 |
| Diabetes | 75 | −0.04 |
| German | 150 | 0.79 |
| Student | 150 | 0.71 |
| All | 1050 | 0.82 |

: Pearson correlation between the absolute score-level gap and the decision-level FDP over the trained TSK models of the main comparison (FairTSK, FairTSK-0, FairTSK-rule; all splits and penalty weights). On Diabetes both gaps are essentially zero.

| **Dataset** | **Penalty** | **Accuracy** | **FDP** | $\mathcal R$ | $\mathcal W$ |
|---|---|---|---|---|---|
| COMPAS (10 splits) | none | 0.682 ± 0.010 | 0.228 ± 0.027 | −0.081 | −0.037 |
| COMPAS (10 splits) | rule-local, λ=100 | 0.682 ± 0.011 | 0.231 ± 0.028 | −0.118 | 0.000 |
| COMPAS (10 splits) | total, λ=10 | 0.659 ± 0.011 | 0.168 ± 0.028 | −0.022 | −0.017 |
| COMPAS (10 splits) | total, λ=100 | 0.568 ± 0.011 | 0.027 ± 0.015 | −0.001 | −0.005 |
| Adult (10 splits) | none | 0.852 ± 0.003 | 0.185 ± 0.012 | −0.152 | −0.033 |
| Adult (10 splits) | rule-local, λ=100 | 0.851 ± 0.003 | 0.185 ± 0.012 | −0.187 | 0.000 |
| Adult (10 splits) | total, λ=10 | 0.835 ± 0.004 | 0.027 ± 0.012 | −0.040 | +0.012 |
| Adult (10 splits) | total, λ=100 | 0.826 ± 0.005 | 0.011 ± 0.006 | −0.009 | +0.003 |

: Mixture of experts with a linear softmax gate instead of fuzzy antecedents (16 experts, same consequents, load balancing, sparsity and training protocol; test data, mean ± s.d. over splits; $\mathcal R$ and $\mathcal W$ at score level).

| **Data** | **Model** | **Log-loss** | **Acc.** | **Cons. norm** | $\mathcal R$ / $\mathcal W$ **midpoint** | $\mathcal R$ / $\mathcal W$ **ref. ¬G** | $\mathcal R$ / $\mathcal W$ **ref. G** |
|---|---|---|---|---|---|---|---|
| Adult | first-order | 0.318 | 0.852 | 1.77 ± 0.07 | −0.101 / −0.086 | −0.112 / −0.076 | −0.090 / −0.097 |
| Adult | rule-local | 0.316 | 0.853 | 1.60 ± 0.11 | −0.188 / 0.000 | −0.188 / 0.000 | −0.188 / 0.000 |
| Adult | zero-order | 0.318 | 0.851 | — | −0.186 / 0.000 | −0.186 / 0.000 | −0.186 / 0.000 |
| COMPAS | first-order | 0.601 | 0.681 | 1.15 ± 0.12 | −0.054 / −0.063 | −0.051 / −0.066 | −0.056 / −0.061 |
| COMPAS | rule-local | 0.600 | 0.683 | 0.28 ± 0.06 | −0.118 / 0.000 | −0.118 / 0.000 | −0.118 / 0.000 |
| COMPAS | zero-order | 0.600 | 0.684 | — | −0.117 / 0.000 | −0.117 / 0.000 | −0.117 / 0.000 |

: Fit and decomposition of first-order (FairTSK, $\lambda_f = 0$), rule-local (FairTSK-rule, $\lambda = 100$) and zero-order (FairTSK-0, $\lambda_f = 0$) rule bases (test data, means over 10 splits; s.d. of the log-loss at most 0.007). Cons. norm (mean ± s.d.): usage-weighted Euclidean norm of the consequent slopes per standard deviation of each input. $\mathcal R$ / $\mathcal W$ at score level with the midpoint weights of (4); reference ¬G: rates of ¬G in $\mathcal R$ and coverage of G in $\mathcal W$; reference G: the converse. All three sum to the gap.

| **Model** | $\mathcal R$ | $\mathcal W$ | **score-level gap** | **FDP (decisions)** | **Accuracy** |
|---|---|---|---|---|---|
| unconstrained | −0.116 | −0.085 | −0.200 | 0.191 | 0.848 |
| rule-local, λ=30 | −0.200 | −0.001 | −0.201 | 0.191 | 0.850 |
| rule-local, λ=100 | −0.201 | 0.000 | −0.201 | 0.189 | 0.849 |
| total, λ=10 | −0.042 | 0.014 | −0.028 | 0.127 | 0.838 |

: Theorem 1 decomposition for the fuzzy group *young* on Adult (FairTSK, test data, means over 10 splits; $\mathcal R$, $\mathcal W$ and the gap at score level).

**Sanity checks.** In both audited models of Table 7 of the main article, the rule shares sum to the directly computed score-level gap to four decimals, and the bound of Proposition 3 held in all 700 synthetic runs (largest ratio of deviation to bound 0.67). Lemma 1, Proposition 3 and Corollary 2 were also checked numerically on 2,000 random instances without violation. These checks verify the code, not the results.

# Alignment of Theorem 2 with training

| **Scenario** | $L$ | **Protocol** | **Retained score gap** | **Retained decision gap** | $\Psi(f')$ | $\delta$ | **Certificate** | **Deviation** |
|---|---|---|---|---|---|---|---|---|
| linear | 2 | plain | 0.99 ± 0.01 | 0.99 | 0.0000 | −0.0001 | 0.042 | 0.0057 |
| linear | 4 | plain | 0.99 ± 0.01 | 0.99 | 0.0000 | +0.0001 | 0.049 | 0.0057 |
| linear | 8 | plain | 0.99 ± 0.01 | 0.99 | 0.0000 | +0.0004 | 0.063 | 0.0047 |
| linear | 16 | plain | 0.99 ± 0.01 | 0.99 | 0.0000 | +0.0002 | 0.071 | 0.0046 |
| linear | 2 | regularised | 1.01 ± 0.01 | 1.00 | 0.0039 | 0.0000 | 0.105 | 0.0017 |
| linear | 4 | regularised | 1.01 ± 0.01 | 1.01 | 0.0066 | −0.0013 | 0.126 | 0.0031 |
| linear | 8 | regularised | 1.00 ± 0.01 | 0.99 | 0.0060 | −0.0008 | 0.126 | 0.0022 |
| linear | 16 | regularised | 1.01 ± 0.02 | 1.01 | 0.0040 | −0.0002 | 0.113 | 0.0024 |
| nonlinear | 2 | plain | 0.66 ± 0.19 | 0.86 | 0.0000 | −0.0345 | 0.606 | 0.0694 |
| nonlinear | 4 | plain | 0.99 ± 0.01 | 1.00 | 0.0000 | −0.0075 | 0.202 | 0.0034 |
| nonlinear | 8 | plain | 0.99 ± 0.02 | 1.00 | 0.0000 | +0.0024 | 0.155 | 0.0037 |
| nonlinear | 16 | plain | 1.00 ± 0.02 | 1.02 | 0.0000 | +0.0006 | 0.117 | 0.0030 |
| nonlinear | 2 | regularised | 0.59 ± 0.24 | 0.82 | 0.0055 | −0.0152 | 0.642 | 0.0833 |
| nonlinear | 4 | regularised | 0.98 ± 0.02 | 1.02 | 0.0127 | +0.0015 | 0.323 | 0.0087 |
| nonlinear | 8 | regularised | 1.01 ± 0.02 | 1.00 | 0.0072 | −0.0015 | 0.181 | 0.0033 |
| nonlinear | 16 | regularised | 1.01 ± 0.01 | 1.00 | 0.0060 | +0.0003 | 0.176 | 0.0023 |

: Synthetic data ($\rho = 0.6$, trainable antecedents, rule-local penalty $\lambda = 100$, mean over 5 seeds). "Plain": log-loss and rule-local penalty only, 100 epochs, last iterate; "regularised": objective (3) with early stopping. $\Psi(f')$ and $\delta$: regulariser value of the zero-order reference and relative optimisation gap of Corollary 3 (nats), with $\Omega(f') = 0$ as Lemma 1(b) requires; certificate: the bound of Corollary 3 on $|\Delta_G(\hat f) - \Delta_G(\eta)|$. Both bounds of Corollary 3 held in all 80 runs; the largest ratio of deviation to certificate was 0.21.

| **Scenario** | **Antecedents** | $L$ | $\hat\varepsilon_0$ (nats) | **Retained gap** | $\mathcal W$ **after penalty** | **Accuracy: none / rule-local** |
|---|---|---|---|---|---|---|
| Linear | trainable | 3 | 0.001 ± 0.000 | 1.01 ± 0.01 | 0.000 | 0.787 / 0.787 |
| Linear | centres frozen | 4 | 0.029 ± 0.026 | 0.93 ± 0.12 | 0.000 | 0.787 / 0.777 |
| Linear | centres frozen | 16 | 0.005 ± 0.002 | 1.00 ± 0.01 | 0.000 | 0.786 / 0.784 |
| Linear | frozen | 32 | 0.129 ± 0.013 | 0.34 ± 0.03 | −0.003 | 0.787 / 0.753 |
| Linear | frozen | 64 | 0.132 ± 0.008 | 0.32 ± 0.02 | −0.004 | 0.787 / 0.753 |
| Linear | frozen | 128 | 0.131 ± 0.006 | 0.32 ± 0.02 | −0.004 | 0.787 / 0.751 |
| Nonlinear | trainable | 3 | 0.061 ± 0.001 | 0.88 ± 0.14 | −0.002 | 0.805 / 0.776 |
| Nonlinear | centres frozen | 4 | 0.125 ± 0.030 | 0.88 ± 0.13 | −0.001 | 0.800 / 0.750 |
| Nonlinear | centres frozen | 16 | 0.031 ± 0.008 | 0.97 ± 0.03 | 0.000 | 0.810 / 0.798 |
| Nonlinear | frozen | 32 | 0.220 ± 0.005 | 0.54 ± 0.04 | −0.002 | 0.783 / 0.699 |
| Nonlinear | frozen | 64 | 0.221 ± 0.003 | 0.52 ± 0.02 | −0.002 | 0.783 / 0.695 |
| Nonlinear | frozen | 128 | 0.221 ± 0.001 | 0.51 ± 0.01 | −0.002 | 0.782 / 0.696 |

: Synthetic configurations with zero-order excess risk between the design points of Section 6.4 ($\rho = 0.6$, mean ± s.d. over 5 seeds, rule-local penalty $\lambda = 100$). Centres frozen: Gaussian centres fixed at their k-means++ initialisation, widths, feature gates and consequents trained; frozen: centres, widths and gates fixed. $\hat\varepsilon_0$: excess risk of a zero-order rule base with the same $L$ and antecedent mode; retained gap = $\Delta_\lambda/\Delta_0$ at score level.

# Compaction and stability of the attribution

| **Dataset** | $d$ | **FairTSK: active rules / non-zero consequent weights** | **FairTSK, k=3: fidelity / abs. Δgap / contribution corr. / top-4 overlap** | **FairTSK-0, k=3: fidelity / abs. Δgap / contribution corr. / top-4 overlap** |
|---|---|---|---|---|
| German | 60 | 16.0 / 57.5 | 0.993 / 0.000 / 0.90 / 0.79 | 0.803 / 0.021 / 0.52 / 0.33 |
| Student | 54 | 16.0 / 51.9 | 0.992 / 0.004 / 0.89 / 0.92 | 0.991 / 0.022 / 0.49 / 0.38 |
| COMPAS | 9 | 16.0 / 7.5 | 0.987 / 0.002 / 0.77 / 0.75 | 0.947 / 0.011 / 0.97 / 0.92 |
| Default | 31 | 16.0 / 27.8 | 0.972 / 0.002 / 0.97 / 0.88 | 0.921 / 0.013 / 0.88 / 0.62 |
| Bank | 49 | 16.0 / 43.6 | 0.975 / 0.003 / 0.74 / 0.62 | 0.619 / 0.020 / −0.01 / 0.46 |
| Adult | 86 | 16.0 / 56.3 | 0.944 / 0.018 / 0.71 / 0.75 | 0.821 / 0.058 / 0.12 / 0.38 |
| Diabetes | 221 | 16.0 / 169.3 | 1.000 / 0.000 / 0.98 / 0.96 | 1.000 / 0.002 / 0.62 / 0.50 |
| Census-KDD | 469 | 16.0 / 385.5 | 0.999 / 0.001 / 0.95 / 0.92 | 0.939 / 0.026 / 0.65 / 0.54 |
| **Mean** | | | 0.983 / 0.004 / 0.86 / 0.82 | 0.880 / 0.022 / 0.53 / 0.52 |

: Compact rule bases with $k = 3$ antecedents per rule, ranked by $\varrho_{lj}$ (unconstrained and $\lambda_f = 10$ models, 3 splits). Fidelity: agreement of test decisions with the full model; abs. Δgap: absolute change of the score-level gap; contribution corr.: correlation of the 16 centred rule shares before and after compaction; top-4 overlap: share of the four largest-share rules that remain among the four largest. Active rules and consequent weights: selected first-order models.

Compact rule bases keep the decisions of first-order models (mean fidelity 0.983) and the total gap (mean absolute change 0.004), but not the rule-level attribution: the shares correlate at 0.86 (first-order) and 0.53 (zero-order), the four largest-share rules are preserved in 82% and 52% of cases, and for zero-order models on Bank and Adult the correlation is −0.01 and 0.12. Ranking antecedents by $\varrho_{lj}$ instead of the non-identifiable gate leaves first-order fidelity unchanged (0.983 versus 0.984) and raises zero-order fidelity from 0.859 to 0.880. Compacting the consequents instead (the five terms with the largest $|a_{lj}|\,\mathrm{sd}(x_j)$ per rule) is less faithful (fidelity 0.84, share correlation 0.69, top-4 overlap 0.61) than compacting the antecedents (0.98, 0.87, 0.78), and compacting both is worse still (0.83, 0.58, 0.53). Audits should therefore be computed on the full model.

| **Dataset** | $d$ | **λ** | **FairTSK: across splits (Spearman / top-5)** | **FairTSK: across L (Spearman / top-5)** | **Fair tree: across splits (Spearman / top-5)** | **Chance top-5** | **Most important features (FairTSK, L = 16)** |
|---|---|---|---|---|---|---|---|
| German | 60 | 0 | 0.83 / 0.76 | 0.89 / 0.80 | 0.50 / 0.54 | 0.08 | credit amount, duration, instalment rate, existing credits, dependants |
| German | 60 | 10 | 0.75 / 0.82 | 0.91 / 0.87 | 0.46 / 0.48 | 0.08 | credit amount, instalment rate, duration, dependants, existing credits |
| Student | 54 | 0 | 0.66 / 0.50 | 0.88 / 0.77 | 0.16 / 0.18 | 0.09 | family relations, free time, going out, age, travel time |
| Student | 54 | 10 | 0.72 / 0.32 | 0.91 / 0.76 | 0.12 / 0.12 | 0.09 | family relations, going out, weekend alcohol, mother education, free time |
| COMPAS | 9 | 0 | 0.85 / 0.92 | 0.98 / 0.96 | 0.73 / 0.76 | 0.56 | priors count, age, juv other count, juv misd count, juv fel count |
| COMPAS | 9 | 10 | 0.73 / 0.74 | 0.90 / 0.89 | 0.81 / 0.76 | 0.56 | priors count, age, juv misd count, juv other count, juv fel count |
| Default | 31 | 0 | 0.81 / 0.74 | 0.84 / 0.68 | 0.69 / 0.48 | 0.16 | PAY 0, PAY 2, AGE, PAY 5, PAY 3 |
| Default | 31 | 10 | 0.65 / 0.66 | 0.76 / 0.73 | 0.54 / 0.48 | 0.16 | PAY 0, PAY 2, AGE, PAY 4, PAY 5 |
| Bank | 49 | 0 | 0.84 / 0.66 | 0.90 / 0.81 | 0.71 / 0.62 | 0.10 | contact unknown, balance, pdays, contact cellular, housing yes |
| Bank | 49 | 10 | 0.70 / 0.50 | 0.80 / 0.59 | 0.66 / 0.58 | 0.10 | contact unknown, pdays, balance, contact cellular, month feb |
| Adult | 86 | 0 | 0.88 / 1.00 | 0.92 / 0.97 | 0.92 / 1.00 | 0.06 | age, hours-per-week, education-num, capital-gain, marital-status = married |
| Adult | 86 | 10 | 0.71 / 0.58 | 0.77 / 0.63 | 0.82 / 0.80 | 0.06 | age, hours-per-week, relationship = husband, occupation = craft-repair, occupation = other-service |
| Diabetes | 221 | 0 | 0.90 / 0.54 | 0.93 / 0.76 | 0.70 / 0.70 | 0.02 | number diagnoses, discharge disposition id 11, number emergency, number inpatient, age [50-60) |
| Diabetes | 221 | 10 | 0.90 / 0.46 | 0.94 / 0.68 | 0.67 / 0.74 | 0.02 | discharge disposition id 11, number diagnoses, number emergency, number inpatient, age [50-60) |
| Census-KDD | 469 | 0 | 0.98 / 0.84 | 0.99 / 0.83 | 0.69 / 0.82 | 0.01 | age, wage per hour, cap gain, dividends, weeks worked |
| Census-KDD | 469 | 10 | 0.99 / 0.76 | 0.99 / 0.81 | 0.79 / 0.72 | 0.01 | age, wage per hour, cap gain, dividends, household summary Spouse of householder |

: Stability of the feature-level attribution. Each rule's centred share is distributed over its features in proportion to the normalised $\varrho_{lj}$, giving a profile $\phi_j$ that sums to the gap and does not depend on the numbering of the rules; for the fair tree, each leaf's share is distributed equally over the features on its path. Spearman: mean pairwise rank correlation of $|\phi_j|$; top-5: mean pairwise overlap of the five features with the largest $|\phi_j|$, across 5 splits ($L = 16$) and across $L \in \{8,16,32\}$; chance top-5 $= 5/d$. $\lambda = 10$ for FairTSK corresponds to $\theta = 0.3$ for the tree (depth 4).

The five features that carry most of the gap agree across splits in 50–100% of cases without a penalty (32–82% with $\lambda_f = 10$), against 1–56% by chance, and across rule-base sizes in 68–97% (59–89%). Without a penalty the fair tree's attributions are less stable on six of eight datasets, equally stable on Adult and more stable on Diabetes; with the penalty the comparison is mixed.

# Ablation and computational cost

Each epoch of FairTSK costs $O(nLd)$ time and $O(BL + Ld)$ memory per mini-batch of size $B$, with $L(4d + 1)$ parameters: (2) evaluates $L \times d$ standardised distances per sample, the consequents are $L$ inner products of length $d$, and the softmax and the regularisers of (3) add $O(L)$ and $O(Ld)$.

| **Variant**         | **COMPAS: acc (λ=0) / FDP (λ=10) / usage entropy** | **Bank: acc (λ=0) / FDP (λ=10) / usage entropy** | **Adult: acc (λ=0) / FDP (λ=10) / usage entropy** |
|---|---|---|---|
| full (L=16)         | 0.687 / 0.181 / 1.00                               | 0.894 / 0.004 / 1.00                             | 0.850 / 0.016 / 1.00                              |
| no load-balancing   | 0.687 / 0.178 / 0.96                               | 0.893 / 0.006 / 0.75                             | 0.850 / 0.023 / 0.56                              |
| no antecedent gates | 0.688 / 0.182 / 1.00                               | 0.893 / 0.006 / 1.00                             | 0.851 / 0.023 / 1.00                              |
| L=4                 | 0.687 / 0.182 / 1.00                               | 0.893 / 0.005 / 1.00                             | 0.850 / 0.024 / 1.00                              |
| L=8                 | 0.688 / 0.186 / 1.00                               | 0.894 / 0.005 / 1.00                             | 0.850 / 0.025 / 1.00                              |
| L=32                | 0.689 / 0.179 / 1.00                               | 0.893 / 0.005 / 1.00                             | 0.851 / 0.018 / 1.00                              |
| L=64                | 0.690 / 0.182 / 1.00                               | 0.893 / 0.004 / 1.00                             | 0.850 / 0.016 / 1.00                              |
| tau=0.5             | 0.687 / 0.181 / 1.00                               | 0.893 / 0.005 / 1.00                             | 0.850 / 0.021 / 1.00                              |
| tau=2               | 0.691 / 0.185 / 1.00                               | 0.894 / 0.003 / 1.00                             | 0.849 / 0.018 / 1.00                              |

: Ablation (mean over 3 splits): accuracy at $\lambda_{f} = 0$, FDP at $\lambda_{f} = 10$, and normalised entropy of mean rule usage (1 = uniform) at $\lambda_{f} = 0$.

Accuracy without a penalty changes by at most 0.4 points when the number of rules varies from 4 to 64 or the temperature or the gates change; FDP under $\lambda_f = 10$ varies by up to 0.009 on Adult. Without load balancing the normalised entropy of rule usage falls from 1.00 to 0.56 on Adult and 0.75 on Bank, i.e. a few rules absorb most samples, which makes an audit less informative.

| $n$ | **DNN** | **FairTSK L=16** | **FairTSK L=64** |
|---|---|---|---|
| 10,000 | 0.033 | 0.043 | 0.063 |
| 30,000 | 0.123 | 0.117 | 0.215 |
| 100,000 | 0.366 | 0.379 | 0.634 |
| 179,571 | 0.696 | 0.726 | 1.207 |

: Seconds per epoch (mean of 3 epochs, one CPU thread, batch size 1024) on Census-KDD.

On Census-KDD ($d = 469$), the epoch time with 16 rules grows from 0.043 s at $n = 10{,}000$ to 0.726 s at $n = 179{,}571$, a factor of 16.9 for a 17.96-fold increase in $n$; at full size it is close to that of the DNN (0.696 s). Quadrupling the rules to 64 multiplies the epoch time by 1.66. On synthetic data with $n = 20{,}000$, increasing $d$ 32-fold from 25 to 800 multiplies it by 2.9 (Fig. S3). A grid-partitioned ANFIS with three terms per input would need $3^{86} \approx 10^{41}$ rules on Adult.

![Training time per epoch versus the number of instances (Census-KDD, d = 469) and versus input dimension (synthetic, n = 20,000), log–log axes](figures/fig_scaling.pdf){width=100%}
