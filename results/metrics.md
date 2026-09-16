# OptAB Results & Metrics

## Prediction Accuracy

### SOFA Score Prediction
- MSE (normalized by variance): < 0.2 across all forecast horizons up to 45h
- Stable accuracy even with sparse observations

### Side-Effect Biomarker Prediction
- **Creatinine**: Accurate prediction of nephrotoxicity trajectory
- **Bilirubin**: Handles sparse measurements (typically daily)
- **ALT (Alanine Aminotransferase)**: Hepatotoxicity monitoring

All predictions use 5-point moving average MSE for sparsely-measured variables.

## Counterfactual Validation

| Metric | Counterfactual Matching | Factual Matching (Baseline) |
|---|---|---|
| MAE (SOFA points) | **1.5 - 2.0** | 1.8 - 3.1 |
| Trend over time | Stable | Increasing (patient divergence) |
| Interpretation | Model prediction bias | Natural patient variability |

**Key finding**: Counterfactual MAE is consistently *below* the factual matching baseline, indicating the model's systematic bias is lower than irreducible patient-to-patient variability.

## Side-Effect Prevention

### Vancomycin (Nephrotoxic)
- 125 patients received Vancomycin in test set
- 39.2% had renal contraindications
- OptAB recommended alternative treatment in all high-risk cases
- 30% of Vancomycin patients develop AKI in practice

### Ceftriaxone (Hepatotoxic)
- 298 patients received Ceftriaxone
- 19.1% had hepatic contraindications or side effects
- OptAB excluded 10.9% of patients from Ceftriaxone

## Treatment Optimization

- 6 feasible antibiotic combinations evaluated per patient per decision point
- Decision interval: every 24 hours
- OptAB-recommended regimens achieve **faster SOFA score reduction** vs empiric therapy
- De-escalation trigger: SOFA drop >= 2 points in 48h

## Model Configuration

| Hyperparameter | Encoder | Decoder |
|---|---|---|
| hidden_channels | 17 | 17 |
| hidden_states | 33 | 33 |
| num_depth | 15 | 15 |
| activation | tanh | tanh |
| learning_rate | 0.00507 | 0.00507 |
| batch_size | 500 | 1000 |
| pred_states | 128 | 128 |
| interpolation | linear (rectilinear) | linear (rectilinear) |

## Figures

Key result figures from the original paper:
- `framework_overview.png`: OptAB iterative decision loop diagram
- `prediction_results.png`: MSE heatmaps + counterfactual side-effect density plots

*(Figures are from the OptAB paper by Wendland et al., 2024, reproduced during this project.)*
