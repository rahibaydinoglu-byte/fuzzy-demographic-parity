| **Dataset** | $d$ | **λ** | **ZadehNet: across splits (Spearman / top-5)** | **ZadehNet: across L (Spearman / top-5)** | **Fair tree: across splits (Spearman / top-5)** | **Chance top-5** | **Most important features (ZadehNet, L = 16)** |
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
