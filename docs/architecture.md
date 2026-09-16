# OptAB Technical Architecture Deep Dive

## 1. Mathematical Foundation

### Neural Controlled Differential Equations

A Neural CDE models a latent state z(t) that evolves continuously:

    z(t) = z0 + integral_0^t f_theta(z(s)) dX(s)

where:
- X(t) is the **control path** - an interpolation of the observed time series
- f_theta(z) is a neural network outputting a (H x D) matrix
- dX(s) is the derivative of the control path

This is equivalent to an ODE dz/dt = f_theta(z(t)) dX/dt, solved numerically via `torchcde.cdeint` (Adaptive Heun method).

### Why This Beats RNNs for ICU Data

ICU data has three pathological properties:
1. **Irregular sampling**: Heart rate every 5 min, labs every 6-24h
2. **Massive missingness**: Some variables measured only once per stay
3. **Time-dependent confounding**: Treatment decisions depend on past outcomes, which depend on past treatments

Neural CDE handles all three natively:
- The control path X(t) is defined for all t, so no bucketing needed
- Missing-value mask channels tell the model "this value was not observed"
- Treatment is part of the control path, so the latent state's evolution is explicitly driven by treatment changes

## 2. Encoder Architecture

### Input Tensor

Shape: `(N_patients, T_hours, D_variables)` where:
- `D = 1 (time) + ~70 (vitals+labs) + 1 (SOFA) + 3 (treatments) + ~74 (missing masks)`
- Each variable has a corresponding binary mask channel
- Time channel at index 0, normalized to [0, 1]

### Forward Pass

1. **Rectilinear interpolation**: `torchcde.linear_interpolation_coeffs(data, rectilinear=0)`
   - The path holds values constant between observations, then jumps vertically
   - Mathematically equivalent to last-observation-carried-forward (LOCF)

2. **Initial state**: z0 = W_init [X(0); static]
   - Static features: age, gender, ethnicity

3. **CDE solve**: Integrate from t=0 to t=T with adaptive step size

4. **Readout heads** at each time step:
   - `pred_dec`: deep MLP -> 4 outputs (SOFA, creatinine, bilirubin, ALT)
   - `treatment`: linear -> 3 treatment logits (softmax)

### Loss Function

    L = MSE(SOFA_hat, SOFA) + sum_k MSE(y_k_hat, y_k) + mu * (-mean softsign(CE(a_hat, a)))

The treatment loss uses **negative cross-entropy with softsign** - this is an adversarial-style objective that encourages the latent state to be **uninformative about treatment**, so the decoder can cleanly attribute outcome changes to treatment choices rather than confounding.

### Hyperparameters (Bayesian-optimized)

| Parameter | Value | Rationale |
|---|---|---|
| hidden_channels | 17 | Compression ratio ~4:1 from input |
| hidden_states | 33 | Max width of CDE vector field MLP |
| num_depth | 15 | Progressive-width layers (15 encoder + 15 decoder) |
| activation | tanh | Stable for ODE integration |
| lr | 0.00507 | Found via Optuna-style search |
| batch_size | 500 | Memory-constrained by 72h x ~150 channels |

## 3. Decoder Architecture

### Purpose
The decoder performs **counterfactual prediction**: given the patient's encoded state at decision time t0, predict the future trajectory under each hypothetical antibiotic combination.

### Initial State
    z0_dec = W_init_dec [zT_enc; static; a_hypothetical]
The hypothetical treatment is concatenated to the initial state, conditioning the entire rollout.

### Control Path
The decoder's control path is **time-only** (plus treatment signal), because future covariates are unknown. The model learns to extrapolate physiology from the latent state alone.

### Stratified Offset Training
To make the decoder robust at any decision point (not just sepsis onset), training samples starting offsets:
- Group time indices into bins of 10
- Sample 2 offsets per bin for early bins (dense data), 1 per bin for later bins
- This ensures the decoder sees predictions starting from various patient states

## 4. Treatment Optimization

### Enumeration
All non-empty subsets of 3 antibiotics = 6 combinations:
1. Vancomycin alone
2. Pip/Taz alone
3. Ceftriaxone alone
4. Vanco + Pip/Taz
5. Vanco + Ceftriaxone
6. Pip/Taz + Ceftriaxone

### Objective
For each combination c:
    score(c) = SOFA_hat^(c)(t0 + H)
where H in {24, 48} hours. Select c* = argmin_c score(c).

### Constraints (Contraindications)
A combination is **infeasible** if:
- max_t creatinine_hat^(c)(t) > threshold_renal
- max_t bilirubin_hat^(c)(t) > threshold_hepatic
- max_t ALT_hat^(c)(t) > threshold_hepatic

If no combination is feasible, fall back to unconstrained minimum.

### Iterative Updates
Every 24 hours:
1. Assimilate new measurements into the encoder
2. Re-encode patient state
3. Re-run optimization over all 6 combinations
4. Recommend updated regimen (or de-escalation if SOFA dropped in 48h)

## 5. Data Pipeline

### MIMIC-IV Processing
1. **PostgreSQL build**: Load MIMIC-IV CSV files per MIT-LCP `mimic-code`
2. **OpenSEP pipeline**: Compute SOFA scores (6 sub-scores: respiration, coagulation, liver, cardiovascular, CNS, renal) and identify Sepsis-3 patients (SOFA >= 2 + suspected infection)
3. **Hourly aggregation**: Resample all events to 1-hour bins
4. **Variable selection**: 70+ variables including vitals, labs, blood gases, medications
5. **Missing mask construction**: Binary indicator per variable per hour
6. **Standardization**: Z-score normalization using training set mean/std
7. **Train/test split**: 80/20 patient-level split (no leakage)

### Key Variables
- **Vitals** (14): HR, SBP, DBP, MBP, RR, Temp, SpO2, Glucose, FiO2, GCS (x4), Weight
- **Labs** (32): Creatinine, Bilirubin, ALT, AST, Lactate, WBC, Platelets, etc.
- **Blood gases** (7): pH, PaO2, PaCO2, A-a gradient, PaO2/FiO2 ratio, etc.
- **Treatments** (3): Vancomycin, Pip/Taz, Ceftriaxone (binary per hour)
- **Outcome** (1): SOFA score
- **Static** (3): Age, gender, ethnicity

## 6. Knowledge Graph Extension (Neo4j)

As a complementary exploration, we investigated integrating a **medical knowledge graph** built in Neo4j:

### Graph Schema
- **Nodes**: Antibiotics, Pathogens, Infection Sites, Side Effects, Lab Tests
- **Edges**: `SUSCEPTIBLE_TO` (pathogen->antibiotic), `CAUSES` (antibiotic->side_effect), `CONTRAINDICATED_WITH` (antibiotic<->condition), `MEASURES` (lab_test->biomarker)

### Use Cases
1. **Explainability**: When OptAB recommends Vancomycin, the KG can show "covers MRSA, risk of nephrotoxicity, monitor creatinine"
2. **Rule augmentation**: Hard contraindications from the KG override model recommendations
3. **Pathogen-aware selection**: When culture results arrive, KG edges inform antibiotic narrowing

This extension was inspired by **GRAPHCARE** (ICLR 2024), which uses LLM-extracted personalized knowledge graphs to enhance healthcare predictions.
