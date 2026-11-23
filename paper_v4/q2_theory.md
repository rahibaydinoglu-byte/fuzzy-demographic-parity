# 3. Fairness with respect to fuzzy protected groups

Let $(X,Z,Y)$ be drawn from a distribution $P$, where $X \in \mathbb{R}^{d}$ are the model inputs, $Z$ is a sensitive variable (numeric, such as age, or categorical, such as sex) that is not an input, and $Y \in \{0,1\}$ is the label, with $Y = 1$ the favourable outcome. The definitions apply to any function $h$ of $X$ with values in $[0,1]$: to scores $\hat y$ and to decisions $\mathbf{1}[\hat{y} \geq 0.5]$ alike. All statements hold verbatim for the empirical distribution of a sample.

**Definition 1 (Fuzzy protected group and fuzzy demographic parity).** A fuzzy protected group $G$ is a fuzzy set on the range of $Z$ with membership function $\mu_G$; write $\mu := \mu_G(Z)$, $\pi := \mathbb{E}[\mu] \in (0,1)$. The group expectations and the parity gap are
$$\mathbb{E}_{G}[h] := \frac{\mathbb{E}[\mu\, h(X)]}{\pi},\qquad \mathbb{E}_{\neg G}[h] := \frac{\mathbb{E}[(1 - \mu)\, h(X)]}{1 - \pi},\qquad \Delta_{G}(h) := \mathbb{E}_{G}[h] - \mathbb{E}_{\neg G}[h], \tag{1}$$
and $\mathrm{FDP}_{G}(h) := |\Delta_{G}(h)|$. We call $\Delta_G(\hat y)$ the *score-level* and $\Delta_G(\mathbf{1}[\hat{y} \geq 0.5])$ the *decision-level* gap; all tables report decision-level FDP.

$\mathbb{E}_G$ is the expectation under $P$ weighted by $\mu$, i.e. Zadeh's probability of a fuzzy event [@zadeh68] normalised to a conditional expectation: a person with membership 0.4 contributes with weight 0.4 to the group mean and 0.6 to the complement mean. Intersectional groups are formed with a t-norm, $\mu_{G_1 \cap G_2} = T(\mu_{G_1},\mu_{G_2})$, which recovers the usual subgroup for crisp memberships.

**Proposition 1 (Basic properties).** For $h$ with values in $[0,1]$: (i) if $\mu \in \{0,1\}$ almost surely, $\mathrm{FDP}_{G}(h)$ is the classical demographic-parity difference of $h$; (ii) $0 \leq \mathrm{FDP}_{G}(h) \leq 1$; (iii) $\Delta_{\neg G} = -\Delta_{G}$, so FDP does not depend on which of the two groups is called protected.

*Proof.* (i) For crisp $\mu$, $\mathbb{E}[\mu h]/\pi = \mathbb{E}[h \mid \mu = 1]$, and likewise for the complement. (ii) Both terms of (1) are averages of values in $[0,1]$. (iii) Exchange $\mu$ and $1 - \mu$. $\square$

**Proposition 2 (Covariance form).** For integrable $h$, $\Delta_{G}(h) = \mathrm{Cov}(\mu,h)/(\pi(1 - \pi))$; hence $\mathrm{FDP}_{G}(h) = 0$ if and only if $\mu$ and $h$ are uncorrelated.

*Proof.* $\mathbb{E}[\mu h]/\pi - (\mathbb{E}[h] - \mathbb{E}[\mu h])/(1 - \pi) = (\mathbb{E}[\mu h] - \pi\,\mathbb{E}[h])/(\pi(1 - \pi))$. $\square$

*Remark 1 (What is and is not new).* Three readings relate FDP to existing notions. (a) A penalty on $\Delta_G^2$ is a covariance penalty, as in Zafar et al. [@zafar]; what differs from a penalty on $\mathrm{Cov}(Z,\hat y)$ is the variable whose dependence is removed: the membership of a named group, not the raw attribute. (b) If $\tilde G$ is a crisp label drawn with probability $\mu(Z)$ independently of $(X,Y)$ given $Z$, then $\Delta_G(h)$ is the classical parity gap of $\tilde G$. (c) The plug-in estimate of (1) is the probability-weighted estimator proposed for disparities when the protected class is unobserved [@chen2019; @kallus2022]. The estimands differ: there the weights are probabilities of an unknown crisp class, here the fuzzy group is the object of interest by definition. The contribution of FDP is therefore not a new estimator but the explicit, inspectable specification of the group and its use in training and in the decomposition of Section 4.

**The group *young*.** For age we use $\mu_{\text{young}}(a) = \min\{1,\max\{0,(35 - a)/10\}\}$, which equals 1 up to 25 and 0 from 35. The core reproduces the crisp convention age ≤ 25 [@lequy]; the support up to 35 is a normative choice of the analyst, to be agreed like a crisp threshold, by reference to a legal definition, a policy target or expert elicitation, with the support stating explicitly how far protection extends. Section S2 of Online Resource 1 proves that $\Delta_G$ is Lipschitz in $\mu$ (Proposition S1), but the constant is conservative (about 5.9 per year of threshold shift on German), so the practical evidence is empirical: the measured gap depends strongly on the location of the group and little on the shape of the transition (@stbl:threshold; @stbl:ramp; @stbl:shape).

# 4. A fairness-aware TSK classifier and its disparity decomposition

## 4.1 Model

A TSK rule base [@tsk] has $L$ rules "IF $x_1$ is $A_{l1}$ AND … AND $x_d$ is $A_{ld}$ THEN $y = g_l(x)$". The firing strengths are normalised to a soft partition $\bar w_l(x)$ of the input space and the output is $\sum_l \bar w_l(x)\, g_l(x)$. FairTSK uses $L$ rules on the standardised inputs with Gaussian memberships of centres $c_{lj}$ and widths $s_{lj}$, raised to powers $\gamma_{lj} \in (0,1)$:
$$\begin{gathered}\log w_{l}(x) = \frac{1}{d}\sum_{j = 1}^{d}\gamma_{lj}\left(-\frac{(x_{j} - c_{lj})^{2}}{2s_{lj}^{2}}\right),\qquad \bar{w}_{l}(x) = \frac{w_l(x)^{1/\tau}}{\sum_{k = 1}^{L}w_k(x)^{1/\tau}},\\ \hat{y}(x) = \sum_{l = 1}^{L}\bar{w}_{l}(x)\,\mathrm{sigmoid}(a_{l}^{\top}x + b_{l}).\end{gathered} \tag{2}$$
The factor $1/d$ is the HTSK device [@htsk] that prevents softmax saturation in high dimension; the temperature is $\tau = 1$. The zero-order variant FairTSK-0 fixes $g_l \equiv \mathrm{sigmoid}(b_l)$, so that each rule reads "IF (antecedents) THEN $P(Y = 1) = g_l$". Feature $j$ enters rule $l$ only through the effective precision $P_{lj} = \gamma_{lj}/(2s_{lj}^2)$, so $\gamma_{lj}$ alone is not identifiable; we measure antecedent relevance by the scale-invariant $\varrho_{lj} := P_{lj}\mathrm{Var}(x_j)$. The cost of a training epoch is $O(nLd)$, linear in the numbers of samples, rules and features (Section S8 of Online Resource 1). The architecture combines known components; we do not claim it as a contribution.

FairTSK minimises
$$\mathcal{L} = \mathrm{BCE}(y,\hat{y}) + \lambda_{f}\,\Omega_{\text{fair}} + \lambda_{b}\,\mathrm{KL}(\bar{u}\,\|\,\mathcal{U}_{L}) + \frac{\lambda_{a}}{Ld}\sum_{l,j}|a_{lj}| + \frac{\lambda_{\gamma}}{Ld}\sum_{l,j}\gamma_{lj}, \tag{3}$$
where $\bar{u}$ is the mean rule usage in a mini-batch; the load-balancing term prevents rule collapse, as in mixtures of experts [@shazeer], and the last two terms regularise the consequents and the gates. The total fairness penalty is $\Omega_{\text{fair}} = \hat{\Delta}_{G}(\hat y)^{2}$, the squared mini-batch estimate of Proposition 2.

## 4.2 Rule-level decomposition

For rule $l$ and $H \in \{G,\neg G\}$ let $m_{l}^{H} := \mathbb{E}_{H}[\bar{w}_{l}]$ (coverage) and $r_{l}^{H} := \mathbb{E}_{H}[\bar{w}_{l}\, g_{l}]/m_{l}^{H}$ (within-rule rate), with midpoints $\bar{m}_{l}$ and $\bar{r}_{l}$.

**Theorem 1 (Rule-level decomposition).** If $\hat y = \sum_l \bar w_l g_l$ with $\bar w_l > 0$, $\sum_l \bar w_l = 1$ and $g_l \in [0,1]$, then for any fuzzy group $G$
$$\Delta_{G}(\hat{y}) = \mathcal{R} + \mathcal{W},\qquad \mathcal{R} := \sum_{l = 1}^{L}(m_{l}^{G} - m_{l}^{\neg G})\,\bar{r}_{l},\qquad \mathcal{W} := \sum_{l = 1}^{L}\bar{m}_{l}\,(r_{l}^{G} - r_{l}^{\neg G}). \tag{4}$$

*Proof.* By linearity and $\sum_l \bar w_l = 1$, $\mathbb{E}_{H}[\hat{y}] = \sum_{l}m_{l}^{H}r_{l}^{H}$; apply $ab - a'b' = (a - a')(b + b')/2 + (a + a')(b - b')/2$ termwise. $\square$

$\mathcal R$, the *routing* disparity, is non-zero when the groups are covered by different rules that carry different outcomes; $\mathcal W$, the *within-rule* disparity, when individuals who fire the same rule receive different scores by group. **Corollary 1.** If every $g_l$ is constant (FairTSK-0), then $\mathcal W = 0$: every disparity is routing.

Because $\sum_l (m_l^G - m_l^{\neg G}) = 0$, $\mathcal R = \sum_l (m_l^G - m_l^{\neg G})(\bar r_l - c)$ for any constant $c$. The midpoint weights in (4) avoid the index-number choice of Oaxaca–Blinder analysis, in which the coverage or the rates of one group serve as reference; Section 6.3 reports the split under both alternatives. We define the *share* of rule $l$ as $(m_l^G - m_l^{\neg G})(\bar r_l - c) + \bar m_l (r_l^G - r_l^{\neg G})$ with $c$ the mean score; the shares sum to the gap and do not change when a constant is added to all rule outputs. A rule contributes to the gap if one group is over-represented in it and its outcome differs from the average.

Theorem 1 is an Oaxaca–Blinder-type identity [@oaxaca; @blinder] and holds for any convex mixture, including mixtures of experts [@jacobs; @shazeer] and the leaves of a decision tree. Two qualifications matter for its use. First, the split between $\mathcal R$ and $\mathcal W$ depends on how the predictor is parametrised: with one rule all disparity is within-rule, with many narrow rules most of it is routing. It describes the computation of the fitted model, not the causes of the disparity in the data. Second, it decomposes the score-level gap, the difference of mean predicted probabilities; thresholding is not linear, and no exact identity holds for decisions. The two gaps are different quantities rather than one being a part of the other: on COMPAS without a penalty the score-level gap is −0.117 and the decision-level gap 0.227, because thresholding at 0.5 amplifies differences of scores near the threshold. Across the 1,050 trained TSK models of the main comparison, the absolute score-level gap and the decision-level FDP correlate at 0.82 (per dataset 0.70–0.99, except Diabetes, where both are essentially zero; @stbl:scoredec).

## 4.3 When must a rule-local penalty fail?

Conditional statistical parity requires parity within strata of legitimate factors [@kamiran2013; @corbett]. In a rule base, the rules are learned soft strata, and requiring that "individuals who match the same rule be treated alike" gives the *rule-local* penalty
$$\Omega(\hat y) := \sum_{l}\bar{m}_{l}\,(r_{l}^{G} - r_{l}^{\neg G})^{2}, \tag{5}$$
which penalises $\mathcal W$ (Lemma 1) and leaves routing free. We call the resulting model FairTSK-rule. The question is whether such a constraint reduces the group gap when the strata are chosen by the optimiser. Let $\eta(x) = P(Y = 1 \mid X = x)$, $\mathcal{L}_{\log}(f)$ the population log-loss and $\varepsilon(f) := \mathcal{L}_{\log}(f) - \mathcal{L}_{\log}(\eta) = \mathbb{E}\,\mathrm{KL}(\mathrm{Ber}(\eta)\,\|\,\mathrm{Ber}(f))$ the excess risk.

**Lemma 1.** For any convex mixture: (a) $|\mathcal{W}| \leq \sqrt{\Omega}$; (b) if every $g_l$ is constant, $\Omega = 0$; (c) if $L = 1$, then $\mathcal{R} = 0$ and $\Omega = \mathcal W^2 = \Delta_{G}^{2}$.

*Proof.* (a) Cauchy–Schwarz with weights $\bar m_l$ summing to one. (b) Corollary 1. (c) With one rule $\bar w_1 \equiv 1$, so $m_1^G = m_1^{\neg G} = 1$. $\square$

**Proposition 3 (Excess risk controls the gap).** For every score $f$,
$$|\Delta_{G}(f) - \Delta_{G}(\eta)| \leq \kappa_{\mu}\sqrt{\frac{\varepsilon(f)}{2\pi(1 - \pi)}},\qquad \kappa_{\mu} := \sqrt{\frac{\mathrm{Var}(\mu)}{\pi(1 - \pi)}} \in [0,1], \tag{6}$$
with $\kappa_{\mu} = 1$ if and only if $\mu$ is crisp.

*Proof.* By Proposition 2, $\Delta_{G}(f) - \Delta_{G}(\eta) = \mathrm{Cov}(\mu,f - \eta)/(\pi(1 - \pi))$. Cauchy–Schwarz gives $|\mathrm{Cov}(\mu,f - \eta)| \leq \sqrt{\mathrm{Var}(\mu)\,\mathbb{E}[(f - \eta)^{2}]}$, and Pinsker's inequality [@cover] gives $\mathbb{E}[(f - \eta)^{2}] \leq \varepsilon(f)/2$. Finally $\mathrm{Var}(\mu) \leq \pi - \pi^{2}$ for $\mu \in [0,1]$, with equality if and only if $\mu^2 = \mu$. $\square$

**Theorem 2 (Disparity displacement).** Let $\mathcal{M}$ be a class of $L$-component convex mixtures, $f_{\lambda}$ a minimiser of $\mathcal{L}_{\log}(f) + \lambda\,\Omega(f)$ over $\mathcal{M}$ (assumed to exist; otherwise Corollary 3 applies to any $f$ whose objective is within $\delta > 0$ of the infimum), and $\varepsilon_{\Omega} := \inf\{\varepsilon(f): f \in \mathcal{M},\ \Omega(f) = 0\}$. Then for every $\lambda \geq 0$: (i) $\varepsilon(f_{\lambda}) \leq \varepsilon_{\Omega}$; (ii) $|\Delta_{G}(f_{\lambda}) - \Delta_{G}(\eta)| \leq \kappa_{\mu}\sqrt{\varepsilon_{\Omega}/(2\pi(1 - \pi))}$; (iii) for $\lambda > 0$, $|\mathcal{W}(f_{\lambda})| \leq \sqrt{\varepsilon_{\Omega}/\lambda}$. If $\mathcal{M}$ contains its zero-order members, $\varepsilon_{\Omega}$ is at most the excess risk $\varepsilon_{0}$ of the best zero-order rule base.

*Proof.* Let $f' \in \mathcal{M}$ with $\Omega(f') = 0$. (i) $\mathcal{L}_{\log}(f_{\lambda}) \leq \mathcal{L}_{\log}(f_{\lambda}) + \lambda\Omega(f_\lambda) \leq \mathcal{L}_{\log}(f')$; subtract $\mathcal{L}_{\log}(\eta)$ and take the infimum. (ii) Proposition 3 and (i). (iii) $\lambda\Omega(f_{\lambda}) \leq \mathcal{L}_{\log}(f') - \mathcal{L}_{\log}(f_\lambda) \leq \varepsilon(f')$; take the infimum and apply Lemma 1(a). The last claim follows from Lemma 1(b). $\square$

**Corollary 2 (One rule).** If $L = 1$, the rule-local penalty equals $\lambda\Delta_G^2$, the group penalty (Lemma 1(c)); if $\mathcal M$ contains a constant score $f_c$, then $\lambda\Delta_{G}(f_\lambda)^2 \le \mathcal{L}_{\log}(f_c)$, so $\Delta_{G}(f_\lambda) \to 0$ as $\lambda \to \infty$.

**Corollary 3 (Regularised objectives and approximate minimisers).** Let $\Psi \geq 0$ be any additional regulariser (for FairTSK, the last three terms of (3)), $\tilde J_\lambda = \mathcal{L}_{\log} + \lambda\Omega + \Psi$, and let $\hat f$ satisfy $\tilde J_\lambda(\hat f) \leq \tilde J_\lambda(f') + \delta$ for some $f'$ with $\Omega(f') = 0$. Then (i)–(iii) of Theorem 2 hold for $\hat f$ with $\varepsilon_\Omega$ replaced by $\tilde\varepsilon := \varepsilon(f') + \Psi(f') + \delta$.

*Proof.* $\mathcal{L}_{\log}(\hat f) \leq \tilde J_\lambda(\hat f) \leq \mathcal{L}_{\log}(f') + \Psi(f') + \delta$; subtract $\mathcal{L}_{\log}(\eta)$ and apply Proposition 3. Similarly $\lambda\Omega(\hat f) \leq \tilde J_\lambda(\hat f) - \mathcal{L}_{\log}(\hat f) \leq \tilde\varepsilon$ because $\mathcal{L}_{\log}(\hat f) \geq \mathcal{L}_{\log}(\eta)$. $\square$

**Interpretation.** A rule-local penalty cannot remove a group gap when (i) the inputs carry information about the group that the Bayes score uses, so that $\Delta_{G}(\eta) \neq 0$, and (ii) the antecedents can partition the input space finely enough that some rule base without within-rule disparity is nearly Bayes-optimal. The penalty is then free: $\mathcal W$ vanishes at rate $\lambda^{-1/2}$ while the gap stays near the Bayes gap and is carried by routing. With a single rule or fixed antecedents, routing capacity is small and the penalty acts on the gap. Part (ii) holds for every $\lambda$, including $\lambda = 0$; its content is that the penalty does not pull the solution away from the Bayes score. The constant from Cauchy–Schwarz and Pinsker is not tight, and $\varepsilon_\Omega$ requires $\eta$, so the theorem can be verified on synthetic data (Section 6.4) but not on real data. There, a zero-order rule base that is nearly as accurate as the first-order one is a necessary sign that $\varepsilon_0$ is small, not a sufficient one.
