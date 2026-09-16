# OptAB Methodology

## 1. Research Question

> Can a data-driven AI model recommend optimal antibiotic combinations for sepsis patients that minimize organ failure (SOFA score) while avoiding antibiotic-induced nephrotoxicity and hepatotoxicity?

## 2. Causal Framework

### The Fundamental Problem of Causal Inference
For each patient, we observe only **one** treatment outcome (the factual). We cannot observe what would have happened under a different treatment (the counterfactual). OptAB addresses this by:

1. Learning a **treatment-invariant patient representation** (encoder)
2. Using it to **simulate counterfactual outcomes** (decoder)
3. **Optimizing** over the simulated outcomes

### Time-Dependent Confounding
In observational data:

    Disease severity -> Treatment choice -> Future outcome
         |                                    ^
         +------------ confounds -------------+

Sicker patients get stronger antibiotics AND have worse outcomes. A naive model would attribute bad outcomes to the antibiotic. OptAB's treatment-invariance loss breaks this confound.

## 3. Model Specification

### Encoder: Treatment-Effect CDE
    z(t) = z0 + integral_0^t f_theta(z(s)) dX(s),  X(s) = [vitals, labs, treatments, masks]
    y_hat(t) = g_phi(z(t)),  a_hat(t) = softmax(h_psi(z(t)))

Training objective:
    min_{theta,phi} MSE(y_hat, y) + mu * max_psi CE(a_hat, a)

The inner maximization (implemented as negative loss) makes z treatment-invariant.

### Decoder: Counterfactual Rollout
    z_cf(t) = z0_cf + integral_{t0}^t f_theta_dec(z_cf(s)) dt,  z0_cf = W[z(t0); static; a_hyp]

For each antibiotic combination c in P({1,2,3}) \ {empty}:
    y_hat^(c)(t0 + H) = g_phi_dec(z_cf^(c)(t0 + H))

### Optimizer
    c* = argmin_{c in Feasible} SOFA_hat^(c)(t0 + H)
    Feasible(c) = intersection_{k in {crea,bili,ALT}} {max_t y_k_hat^(c)(t) <= tau_k}

## 4. Evaluation Protocol

### Counterfactual Matching
For patient i with recommended treatment c_i*:
1. M_i = {j : a_j(0) = c_i*, t_j_onset < 3h}
2. j* = argmin_{j in M_i} ||x_j(0) - x_i(0)||_2  (11 core covariates)
3. MAE_i = (1/H) sum_{t=1}^H |SOFA_hat^(c_i*)(t) - SOFA_j*(t)|

### Factual Matching (Baseline)
Same procedure but matching on **actual** treatment a_i(0), comparing two real patients' trajectories. Measures irreducible patient variability.

### Interpretation
- CF MAE < Factual MAE -> model bias is below natural variability (check)
- CF MAE stable over time -> no error accumulation (check)

## 5. Clinical Deployment Considerations

- **Update frequency**: Every 24h (matches clinical reassessment cadence)
- **De-escalation**: If SOFA drops >=2 points in 48h, recommend narrowing antibiotics
- **Human-in-the-loop**: Model provides recommendation + predicted trajectories; clinician makes final decision
- **Contraindications**: Hard constraints from clinical guidelines override model output
