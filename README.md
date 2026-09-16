# OptAB: AI-Driven Optimal Antibiotic Selection for Sepsis Patients

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Neural CDE](https://img.shields.io/badge/Model-Neural%20CDE-9cf)](https://github.com/patrick-kidger/torchcde)

> **First data-driven, online-updateable optimal antibiotic selection model for sepsis, accounting for antibiotic-induced nephrotoxicity and hepatotoxicity.**

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Technical Highlights](#technical-highlights)
- [Evaluation Methodology](#evaluation-methodology)
- [References](#references)

---

## Problem Statement

**Sepsis** is a life-threatening organ dysfunction caused by a dysregulated host response to infection. It affects **1.7 million Americans annually** and is the #1 cause of in-hospital death.

### The Antibiotic Dilemma

| Challenge | Impact |
|---|---|
| **Pathogen detection delay** | 30–70% of sepsis patients have no identified pathogen; culture results take 2–3 days |
| **Empiric broad-spectrum therapy** | Leads to antibiotic overuse, resistance, and avoidable side effects |
| **Nephrotoxicity & hepatotoxicity** | Vancomycin causes AKI in ~30% of patients; Ceftriaxone causes liver injury in ~19% |
| **One-size-fits-all guidelines** | Ignore patient-specific factors (weight, comorbidities, lab trajectories) |

> Existing ML research in sepsis focuses on **early detection** — data-driven **antibiotic selection** has been largely overlooked.

## Solution Overview

**OptAB** (Optimal Antibiotic selection) is an AI framework that:

1. **Continuously assimilates** patient data (vitals, labs, treatments) as it arrives
2. **Predicts disease progression** (SOFA score + side-effect biomarkers) under each possible antibiotic combination
3. **Recommends the optimal regimen** that minimizes SOFA score while respecting contraindication thresholds
4. **Updates every 24 hours** as new measurements become available

### Iterative Decision Loop

```
┌─────────────┐     Measuring      ┌──────────────┐
│  Sepsis     │ ──────────────────►│  SOFA / Labs │
│  Patient    │                    │  / Vitals    │
└──────┬──────┘                    └──────┬───────┘
       │                                  │
       │  Final treatment decision        ▼
       │                          ┌──────────────┐
       │                          │    OptAB     │
       │                          │  (Encoder +  │
       │                          │   Decoder)   │
       │                          └──────┬───────┘
       │                                 │
       │         ┌───────────────────────┼───────────────────────┐
       │         ▼                       ▼                       ▼
       │  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
       └──┤  Proposing  │        │ Predicting  │        │  Optimizing │
          │  treatments │        │  progression│        │  (min SOFA  │
          └─────────────┘        └─────────────┘        │  + constraints)│
                                                        └─────────────┘
                              Next iteration (24h) ──────►
```

## Architecture

OptAB is built on **Neural Controlled Differential Equations (Neural CDE)** — a continuous-time model that naturally handles the irregular sampling, massive missingness, and time-dependent confounding inherent in ICU data.

### Encoder-Decoder Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER (Neural CDE)                      │
│  Input:  Irregular time series [vitals, labs, treatments]       │
│          + missing-value masks + rectilinear time channel        │
│  Method: dz(t) = f_θ(z(t)) dX(t)   (continuous latent state)    │
│  Output: Latent patient state z_T at decision time               │
│  Loss:   MSE(SOFA) + MSE(toxicity) + μ·treatment_classification │
└──────────────────────────────┬──────────────────────────────────┘
                               │ z_T + static + treatment
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER (Neural CDE)                      │
│  Input:  Time-only control path + treatment-conditioned z0       │
│  Method: Counterfactual rollout under each antibiotic combo      │
│  Output: Predicted SOFA, Creatinine, Bilirubin, ALT trajectories │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TREATMENT OPTIMIZER                          │
│  Enumerate: powerset({Vanco, Pip/Taz, Ceftriaxone}) → 6 combos   │
│  Objective: minimize terminal SOFA at 24h/48h                    │
│  Constraints: creatinine ≤ threshold, bilirubin ≤ threshold,     │
│               ALT ≤ threshold (contraindication avoidance)       │
└─────────────────────────────────────────────────────────────────┘
```

### Why Neural CDE?

| Property | Standard RNN/LSTM | Neural CDE |
|---|---|---|
| Irregular time steps | Requires imputation/bucketing | **Native support** via continuous control path |
| Missing values | Treated as zeros or imputed | **Missing masks** as additional channels |
| Online updates | Retrain or stateful hack | **Natural assimilation** of new observations |
| Time-dependent confounding | Hard to handle | **Treatment as control signal** in the ODE |

## Key Results

### Prediction Accuracy (MSE heatmaps)

The model accurately predicts SOFA score and side-effect biomarkers across all forecast horizons. MSE is normalized by variable variance — **darker = more accurate**.

- **SOFA score**: MSE stays below 0.2 (normalized) even at 45h horizons
- **Creatinine / Bilirubin / ALT**: Sparser measurements handled via rectilinear interpolation

### Counterfactual Validation

Since we cannot run RCTs on model recommendations, we use **counterfactual matching**:

| Metric | Counterfactual Matching | Factual Matching (baseline) |
|---|---|---|
| **MAE in SOFA points** | **1.5 – 2.0** | 1.8 – 3.1 |
| Interpretation | Model prediction bias | Patient-to-patient variability |

> The counterfactual MAE being **lower and stable** indicates OptAB's predictions have low systematic bias — the remaining error is inherent patient variability, not model error.

### Side-Effect Prevention

| Antibiotic | Patients with contraindication | OptAB excludes |
|---|---|---|
| **Vancomycin** (nephrotoxic) | 39.2% of 125 patients | Recommends alternative |
| **Ceftriaxone** (hepatotoxic) | 19.1% of 298 patients | Excludes 10.9% |

### Treatment Optimization

OptAB-recommended regimens achieve **faster SOFA score reduction** compared to actual empiric therapy, while simultaneously reducing exposure to high-risk antibiotics in vulnerable patients.

## Project Structure

```
OptAB-Project/
├── README.md                          ← You are here
├── requirements.txt
├── configs/
│   └── default_config.yaml            ← Hyperparameters (Bayesian-optimized)
├── src/
│   ├── model/
│   │   └── neural_cde.py              ← Core Neural CDE (encoder + decoder)
│   ├── training/
│   │   ├── train_encoder.py           ← Encoder training loop
│   │   └── early_stopping.py          ← Early stopping utility
│   ├── optimization/
│   │   └── treatment_optimizer.py     ← Antibiotic combo optimization
│   ├── evaluation/
│   │   └── counterfactual_matching.py ← Counterfactual evaluation
│   └── preprocessing/
│       └── mimiciv_preprocessing.py   ← MIMIC-IV data pipeline
├── docs/
│   ├── architecture.md                ← Deep technical architecture
│   ├── methodology.md                 ← Methodology & mathematical details
│   └── interview_guide.md             ← Interview talking points & Q&A
├── results/
│   └── figures/                       ← Result visualizations
└── notebooks/
    └── exploratory_analysis.ipynb     ← Data exploration
```

## Getting Started

### Prerequisites
- Python 3.10+
- MIMIC-IV dataset access (credentialed via [PhysioNet](https://physionet.org/content/mimiciv/2.2/))
- PostgreSQL (for MIMIC-IV database construction)

### Installation

```bash
git clone https://github.com/houdan-hub/OptAB-Sepsis-Antibiotic-Optimization.git
cd OptAB-Sepsis-Antibiotic-Optimization
pip install -r requirements.txt
```

### Data Preprocessing

```bash
# 1. Build MIMIC-IV PostgreSQL database (per MIT-LCP mimic-code)
# 2. Run the OpenSEP pipeline for SOFA score + Sepsis-3 classification
# 3. Process into Neural CDE format:
python -m src.preprocessing.mimiciv_preprocessing
```

### Training

```bash
# Hyperparameter optimization
python scripts/hypopt_enc.py
python scripts/hypopt_dec.py

# Train encoder (patient state assimilation)
python scripts/train_encoder.py

# Train decoder (counterfactual progression prediction)
python scripts/train_decoder.py
```

### Inference & Optimization

```python
from src.model.neural_cde import NeuralCDE
from src.optimization.treatment_optimizer import optimize_treatment

# Load trained models
encoder = NeuralCDE(...)
encoder.load_state_dict(torch.load('Trained_Encoder.pth'))
decoder = NeuralCDE(...)
decoder.load_state_dict(torch.load('Trained_Decoder.pth'))

# Get optimal antibiotic recommendation
best_combo, sofa_trajectory, all_results = optimize_treatment(
    encoder, decoder, patient_data, static_features,
    treatment_history, decision_time=24, horizon=48,
    creatinine_threshold=2.0,  # mg/dL
)
```

## Technical Highlights

### 1. Treatment-Effect Controlled Differential Equations (TE-CDE)
The encoder learns a **treatment-invariant latent representation** via an auxiliary treatment-classification loss (adversarial-style), ensuring the latent state captures patient physiology rather than treatment patterns. The decoder then performs **counterfactual rollouts** by conditioning on hypothetical treatments.

### 2. Rectilinear Interpolation
Instead of standard linear interpolation, OptAB uses **rectilinear interpolation** — the path moves horizontally in time (holding values) then vertically in value (jumping at observation times). This is mathematically equivalent to carrying forward the last observation, the clinical standard for irregular ICU data.

### 3. Shifted Softplus for Non-Negativity
SOFA scores, creatinine, and bilirubin are physiologically non-negative. A **shifted softplus** activation enforces this constraint while preserving gradient flow:
```
output = softplus(x - threshold) + threshold
```
where `threshold` is the normalized value of zero.

### 4. Stratified Offset Sampling for Decoder
The decoder is trained at randomly sampled starting offsets (stratified by observation density) to robustly handle predictions from arbitrary decision points, not just sepsis onset.

### 5. Knowledge Graph Extension (Explored)
Complementary exploration of **GRAPHCARE**-style personalized knowledge graphs using Neo4j to model antibiotic-pathogen relationships, drug-drug interactions, and contraindication networks — enabling explainable, rule-augmented treatment recommendations.

## Evaluation Methodology

### Counterfactual Matching Protocol

1. For each test patient, OptAB recommends an optimal treatment at onset.
2. Find the **most similar patient** (by Euclidean distance on 11 core covariates) who actually received that treatment.
3. Compare OptAB's predicted SOFA trajectory to the matched patient's actual SOFA trajectory.
4. **Control**: repeat matching using each patient's *actual* treatment (factual matching) to establish a variability baseline.

### Why This Works
- If counterfactual MAE ≈ factual MAE → model is as accurate as real patient variability allows
- If counterfactual MAE < factual MAE → model has low bias (observed: 1.5–2.0 vs 1.8–3.1)

## Datasets

| Dataset | Patients | Source |
|---|---|---|
| **MIMIC-IV** v2.2 | ~30k sepsis ICU stays | MIT/Beth Israel Deaconess Medical Center |
| **AmsterdamUMCdb** | ~5k sepsis ICU stays | Amsterdam UMC (external validation) |

## References

1. Wendland, P., Schenkel-Häger, C., & Kschischo, M. "OptAB — an optimal antibiotic selection framework for Sepsis." (2024)
2. Seedat, N. et al. "Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations." **ICML 2022**.
3. Kidger, P. et al. "Neural Controlled Differential Equations for Irregular Time Series." **NeurIPS 2020**.
4. Johnson, A.E.W. et al. "MIMIC-IV, a freely accessible electronic health record dataset." **Scientific Data 2023**.
5. Hofford, M.R. et al. "OpenSep: a generalizable open source pipeline for SOFA score calculation and Sepsis-3 classification." **JAMIA Open 2022**.
6. Jiang, P. et al. "GRAPHCARE: Enhancing Healthcare Predictions with Personalized Knowledge Graphs." **ICLR 2024**.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*This project is a research reproduction and extension. MIMIC-IV and AmsterdamUMCdb data are not included and require credentialed access.*