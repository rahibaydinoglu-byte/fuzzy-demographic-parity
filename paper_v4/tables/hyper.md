
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
