# OptAB Interview Preparation Guide

> **Goal**: Speak about this project with complete confidence, technical depth, and honest framing. This guide covers everything from elevator pitch to adversarial technical questions.

---

## 1. Elevator Pitches

### 30-Second Version
"I built OptAB, an AI system that recommends optimal antibiotic combinations for sepsis patients. It uses Neural Controlled Differential Equations to handle the irregular, missing-heavy ICU time series, predicts disease progression under each antibiotic option, and selects the regimen that minimizes organ failure while avoiding kidney and liver toxicity. I trained it on MIMIC-IV with ~30,000 sepsis patients and validated it through counterfactual matching."

### 1-Minute Version
"Sepsis kills 270,000 Americans a year, and the core problem is that doctors have to choose antibiotics before culture results come back - 2 to 3 days later. So they use broad-spectrum empiric therapy, which causes antibiotic resistance and serious side effects like kidney failure from Vancomycin.

I built OptAB to solve this. It's an encoder-decoder model based on Neural CDEs - a continuous-time deep learning architecture that natively handles irregular sampling and missing data, which is everywhere in ICU. The encoder assimilates the patient's full vital sign and lab history into a latent state. The decoder then does counterfactual rollouts: for each of six possible antibiotic combinations, it predicts the 48-hour SOFA score trajectory and side-effect biomarkers. The optimizer picks the combination that minimizes organ failure while respecting contraindication thresholds.

I reproduced and extended the original OptAB paper, trained on MIMIC-IV, and validated through counterfactual matching - comparing model predictions to matched patients who actually received the recommended treatment. The model achieved 1.5 to 2.0 SOFA-point MAE, which is below the natural patient-to-patient variability baseline of 1.8 to 3.1."

### 2-Minute Version (Add Technical Depth)
[... 1-minute version, then continue ...]

"The technical challenge that drew me in was time-dependent confounding. In observational ICU data, sicker patients get stronger antibiotics, so a naive model would confound treatment effect with disease severity. The encoder uses an adversarial-style treatment classification loss - it actively tries to make the latent state uninformative about which treatment was given. That way, when the decoder conditions on a hypothetical treatment, the outcome difference is genuinely attributable to the treatment, not to pre-existing patient differences.

Another key design choice was rectilinear interpolation. Instead of linearly interpolating between lab values - which would fabricate physiologically impossible smooth transitions - the model holds the last observed value constant until the next measurement, exactly like clinicians do in practice.

For evaluation, since we can't run randomized trials, I implemented counterfactual matching: for each patient where the model recommended treatment X, I found the most similar patient who actually received X and compared trajectories. The model's counterfactual MAE was consistently lower than the factual matching baseline, which means low systematic bias.

I also explored a Neo4j knowledge graph extension to model antibiotic-pathogen relationships and contraindications, making the recommendations more explainable."

---

## 2. STAR Format Stories

### Story 1: Tackling the Missing Data Problem

**Situation**: When I started working on sepsis antibiotic optimization with my advisor, the biggest obstacle was that ICU data is incredibly messy - heart rate measured every 5 minutes but labs only every 6-24 hours, 30-70% missing values for some variables, and every patient on a different time grid.

**Task**: I needed a model architecture that could handle this irregularity without destructive imputation or fixed-time bucketing, which would either fabricate data or lose information.

**Action**: I researched continuous-time models and found Neural Controlled Differential Equations. Instead of discretizing time, Neural CDEs model a latent state that evolves continuously: dz(t) = f(z(t)) dX(t), where X(t) is an interpolation of the observed path. I implemented rectilinear interpolation - the path holds values constant between observations, then jumps vertically - which is mathematically equivalent to last-observation-carried-forward, the clinical standard. I also added missing-value mask channels so the model knows when a value was genuinely unobserved versus zero.

**Result**: The model achieved stable prediction accuracy across all forecast horizons up to 48 hours, even for sparsely-measured variables like bilirubin. The MSE heatmaps showed consistent dark blue (low error) across the entire forecast range.

### Story 2: Solving Time-Dependent Confounding

**Situation**: A naive predictive model on observational ICU data would be useless for treatment recommendation because of confounding - sicker patients get more aggressive antibiotics, so the model would learn "Vancomycin -> worse outcomes" when really it's "sicker patient -> Vancomycin AND worse outcomes."

**Task**: I needed the encoder to learn a patient state representation that captures physiology but is invariant to treatment decisions, so the decoder could isolate genuine treatment effects.

**Action**: I implemented an adversarial-style auxiliary loss. The encoder has a treatment classification head that tries to predict which antibiotic was given from the latent state - but we optimize the **negative** of this loss (with softsign for stability). This pushes the latent state to discard treatment information while preserving outcome-relevant physiology. The loss is: MSE(SOFA) + MSE(side effects) + mu * (-treatment CE). I tuned mu through hyperparameter optimization.

**Result**: Counterfactual matching validation confirmed the approach worked - the model's predictions under hypothetical treatments had MAE of 1.5-2.0 SOFA points, which is actually below the natural patient-to-patient variability (1.8-3.1), indicating very low systematic bias.

### Story 3: Building the Optimization Layer

**Situation**: Even with accurate predictions, the clinical question is "which antibiotic should I give?" - not "what will happen?" The model needed to convert counterfactual predictions into actionable recommendations.

**Task**: Build an optimization layer that enumerates treatment options, predicts outcomes for each, and selects the optimal one while respecting clinical contraindications.

**Action**: I implemented a combinatorial optimizer over the powerset of three antibiotics (Vancomycin, Pip/Taz, Ceftriaxone), giving 6 feasible combinations. For each, the decoder predicts 24-48h trajectories of SOFA score plus creatinine, bilirubin, and ALT. The optimizer minimizes terminal SOFA subject to constraints: if predicted creatinine exceeds the nephrotoxicity threshold, that combination is marked infeasible. The system runs iteratively every 24 hours, assimilating new data and updating recommendations.

**Result**: On the test set, OptAB correctly identified and avoided high-risk antibiotics in 39.2% of Vancomycin candidates (nephrotoxicity) and 10.9% of Ceftriaxone candidates (hepatotoxicity), while recommending regimens that achieved faster SOFA score reduction.

---

## 3. Technical Q&A (Anticipate Everything)

### Q: Why Neural CDE instead of LSTM/Transformer?

**A**: Three reasons specific to ICU data:
1. **Irregular time steps**: LSTMs need fixed intervals; you'd have to impute or bucket, losing information. Neural CDE's control path is defined continuously, so it handles arbitrary observation times natively.
2. **Missing values**: Transformers with padding masks don't distinguish "measured as zero" from "not measured." Neural CDE uses explicit missing-value mask channels as part of the input.
3. **Online updates**: In a clinical setting, new vitals arrive every few minutes. Neural CDE naturally assimilates new observations into the continuous latent state - you just extend the control path. An LSTM would need stateful inference or retraining.
4. **Time-dependent confounding**: Treatment as a control signal in the ODE makes the causal structure explicit - dz/dt depends on dX/dt, which includes treatment changes.

### Q: Explain the encoder-decoder design. Why two models?

**A**: The encoder's job is **state assimilation** - it reads the full observed history (covariates + actual treatments) and compresses it into a latent patient state. The decoder's job is **counterfactual prediction** - starting from that latent state, it rolls forward under hypothetical treatments. We need two because:
- The encoder sees actual treatments (for training on real data)
- The decoder must predict under treatments the patient never received
- The decoder's control path is time-only (future covariates unknown), while the encoder's is full-data

The decoder is initialized from the encoder's terminal hidden state concatenated with static features and the hypothetical treatment.

### Q: What is the treatment-invariance loss and why negative cross-entropy?

**A**: It's an adversarial objective. The encoder has a treatment classification head. Normally, cross-entropy loss would make the latent state *more* informative about treatment. We negate it (with softsign for gradient stability) to push the latent state to *discard* treatment information. This is crucial because if the latent state encodes "this patient got Vancomycin," then when the decoder tries to predict "what if they got Ceftriaxone," it can't cleanly isolate the treatment effect - it's confounded by whatever made the doctor choose Vancomycin in the first place.

### Q: How do you validate counterfactual predictions? You can't run RCTs.

**A**: Counterfactual matching. For each test patient where OptAB recommends treatment X at onset:
1. Find all patients who actually received treatment X (and started within 3 hours of sepsis onset)
2. Pick the one most similar by Euclidean distance on 11 core covariates at onset
3. Compare OptAB's predicted SOFA trajectory to that matched patient's actual SOFA trajectory
4. Compute MAE over 48 hours

As a control, I also do **factual matching**: match each patient to a similar patient with the *same* actual treatment. This gives a baseline of natural patient-to-patient variability. If counterfactual MAE <= factual MAE, the model's bias is at or below natural variability.

Results: counterfactual MAE = 1.5-2.0, factual MAE = 1.8-3.1. The model is actually *more* accurate than real patient variability.

### Q: What about the Neo4j / knowledge graph part?

**A**: That was an exploratory extension. I investigated building a medical knowledge graph in Neo4j with nodes for antibiotics, pathogens, infection sites, and side effects, with edges for susceptibility, contraindications, and drug interactions. The idea was to make OptAB's recommendations more explainable - when the model recommends Vancomycin, the KG can show "covers MRSA, 30% nephrotoxicity risk, monitor creatinine q12h." It was inspired by the GRAPHCARE paper (ICLR 2024), which uses LLM-extracted knowledge graphs to enhance healthcare predictions. The core OptAB model is fully functional without it; the KG is a layer for explainability and rule-based contraindication checking.

### Q: What were the biggest challenges?

**A**:
1. **Data pipeline complexity**: MIMIC-IV requires PostgreSQL, the OpenSEP pipeline for SOFA computation, and careful handling of ~70 variables with different units and sampling rates. Getting from raw CSV to a clean Neural CDE tensor was a multi-week engineering effort.
2. **Training instability**: Neural CDEs with deep vector fields (15 layers) can be unstable. I used gradient clipping, early stopping with patience 10, and tanh activation (more stable than ReLU for ODE integration).
3. **Counterfactual evaluation design**: Coming up with a rigorous evaluation protocol that doesn't require RCTs was intellectually challenging. The matching approach with a factual baseline was key.

### Q: How would you improve this?

**A**:
1. **Uncertainty quantification**: Currently point predictions; adding Bayesian or ensemble uncertainty would help clinicians know when to trust the model.
2. **Dose optimization**: Currently binary treatment (on/off); optimizing actual dosage based on weight and renal function would be more clinically useful.
3. **More antibiotics**: Currently 3 drugs; expanding to the full antibiogram would increase real-world utility.
4. **Pathogen integration**: When culture results arrive (day 2-3), incorporating pathogen-specific susceptibility would enable de-escalation.
5. **Prospective validation**: Retrospective matching is good, but a prospective clinical trial would be the gold standard.

### Q: Walk me through the data preprocessing.

**A**:
1. Build PostgreSQL database from MIMIC-IV CSVs
2. Run OpenSEP pipeline: computes SOFA score from 6 components (respiration via PaO2/FiO2, coagulation via platelets, liver via bilirubin, cardiovascular via MAP/vasopressors, CNS via GCS, renal via creatinine/urine output) and identifies Sepsis-3 patients (SOFA >= 2 + suspected infection)
3. Resample all events to 1-hour bins over the first 72 hours
4. Extract 70+ variables: 14 vitals, 32 labs, 7 blood gases, 3 treatments, 1 SOFA
5. Construct binary missing-value masks for each variable
6. Z-score standardize using training set statistics only
7. Add rectilinear time channel (cumulative time since last observation)
8. Patient-level 80/20 train/test split to prevent leakage

### Q: What's SOFA and why use it as the outcome?

**A**: SOFA (Sequential Organ Failure Assessment) is the clinical standard for measuring sepsis severity. It's a 0-24 point score across 6 organ systems: respiration, coagulation, liver, cardiovascular, CNS, and renal. A SOFA increase of >=2 points defines sepsis (Sepsis-3 criteria). We use it as the treatment success metric because:
1. It's clinically validated and widely used
2. It captures multi-organ dysfunction, not just mortality
3. It changes on a 24-48h timescale, matching our decision interval
4. Lower SOFA = better outcome, so minimizing it is a clear optimization objective

### Q: Explain rectilinear interpolation vs standard interpolation.

**A**: Standard linear interpolation draws a straight line between two observations - e.g., if creatinine is 1.0 at hour 0 and 2.0 at hour 12, it predicts 1.5 at hour 6. But physiologically, creatinine doesn't change smoothly; it stays constant and then jumps when a new measurement reveals a change. Rectilinear interpolation instead moves horizontally in time (holding the value) then vertically in value (jumping at the observation time). This is exactly last-observation-carried-forward, which is what clinicians mentally do. In Neural CDE terms, the control path derivative dX/dt is zero between observations and a delta function at observation times, which the ODE solver handles via jump points.

---

## 4. Resume Bullet Points

### Version 1 (Data Science / ML Engineer)
- **Built OptAB**, an AI-driven optimal antibiotic selection system for sepsis patients using Neural Controlled Differential Equations, achieving 1.5-2.0 SOFA-score MAE on counterfactual predictions (below natural patient variability baseline of 1.8-3.1)
- **Designed encoder-decoder architecture** with treatment-invariant latent representation (adversarial loss) to address time-dependent confounding in observational ICU data
- **Implemented combinatorial treatment optimizer** over 6 antibiotic combinations with contraindication constraints, reducing nephrotoxic exposure in 39.2% of high-risk patients
- **Processed MIMIC-IV dataset** (~30k sepsis patients, 70+ variables) via PostgreSQL + OpenSEP pipeline; handled irregular sampling and 30-70% missing data with rectilinear interpolation
- **Validated via counterfactual matching** protocol; explored Neo4j knowledge graph extension for explainable antibiotic-pathogen contraindication reasoning

### Version 2 (Software Engineer / Full Stack)
- **Developed end-to-end ML pipeline** for sepsis antibiotic optimization: data ingestion (PostgreSQL + MIMIC-IV), preprocessing (70+ variable time series with missing masks), model training (PyTorch Neural CDE), and inference optimization
- **Engineered treatment recommendation system** enumerating 6 antibiotic combinations with constraint-based optimization, integrating clinical contraindication rules
- **Built evaluation framework** with counterfactual matching and statistical confidence intervals (scipy), producing publication-quality visualizations (matplotlib)
- **Explored knowledge graph integration** using Neo4j for antibiotic-pathogen relationships and explainable clinical decision support

### Version 3 (Research / Applied Scientist)
- **Reproduced and extended OptAB** (Wendland et al., 2024), the first data-driven online-updateable antibiotic selection model accounting for antibiotic-induced toxicity
- **Advanced counterfactual evaluation methodology** through matched cohort design, demonstrating model bias below natural patient variability (1.5-2.0 vs 1.8-3.1 SOFA MAE)
- **Investigated Neural CDE architectures** for irregular medical time series, including rectilinear interpolation and treatment-effect controlled differential equations
- **Contributed knowledge graph extension** (Neo4j + GRAPHCARE-inspired) for explainable, rule-augmented treatment recommendations

---

## 5. Honest Framing Guide

### How to describe your role
- **Say**: "I reproduced and extended the OptAB paper under the guidance of Professor Xiao Liang. I implemented the full pipeline from data preprocessing through model training to counterfactual evaluation, and explored a Neo4j knowledge graph extension."
- **Don't say**: "I invented OptAB" or "I published this paper" - the original work is by Wendland et al.
- **Emphasize**: Your implementation work, your understanding of the methodology, your extension (KG), and your ability to explain every technical decision.

### If asked "Is this your original research?"
- "This is a reproduction and extension of a 2024 paper by Wendland et al. I worked on it with my advisor as a research project. My contributions include the full implementation, hyperparameter optimization, counterfactual evaluation, and an exploratory knowledge graph extension. The core idea is from the paper, but I built and debugged every component myself."

### If asked "Did you get it to work?"
- "Yes. I trained both encoder and decoder on MIMIC-IV, reproduced the paper's key results including the counterfactual matching MAE and side-effect prevention rates, and generated all the visualization figures."

---

## 6. Key Numbers to Memorize

| Metric | Value |
|---|---|
| Dataset | MIMIC-IV v2.2, ~30k sepsis patients |
| Variables | 70+ (14 vitals, 32 labs, 7 blood gases, 3 treatments) |
| Antibiotics | Vancomycin, Pip/Taz, Ceftriaxone (6 combinations) |
| Model | Neural CDE, hidden=17, depth=15, tanh |
| Forecast horizon | 24-48 hours |
| Decision interval | Every 24 hours |
| Counterfactual MAE | 1.5-2.0 SOFA points |
| Factual MAE (baseline) | 1.8-3.1 SOFA points |
| Vanco nephrotoxicity exclusion | 39.2% of candidates |
| Ceftriaxone hepatotoxicity exclusion | 10.9% of candidates |
| Sepsis mortality | ~270k/year in US, #1 in-hospital cause of death |
| Pathogen detection rate | 30-70% no pathogen identified |
| Culture result delay | 2-3 days |

---

## 7. Behavioral Questions Tie-In

### "Tell me about a time you solved a complex problem."
-> Use Story 1 (missing data) or Story 2 (confounding)

### "Tell me about a project you're proud of."
-> Use the 2-minute elevator pitch + key results

### "Tell me about a time you had to learn something new quickly."
-> "I had to learn Neural CDEs from scratch - continuous-time models, ODE solvers, rectilinear interpolation. I read the original NeurIPS 2020 and ICML 2022 papers, worked through the torchcde library tutorials, and iteratively debugged until the training stabilized."

### "How do you handle ambiguous requirements?"
-> "The clinical problem was ambiguous - 'better antibiotic selection' could mean many things. I worked with my advisor to operationalize it as: minimize SOFA score at 24-48h, subject to contraindication constraints. This gave a clear optimization objective while capturing the clinical intent."

### "What's a time you failed?"
-> "My first attempt at training the Neural CDE used ReLU activation and standard linear interpolation. Training was unstable and predictions were poor. I switched to tanh (more stable for ODE integration) and rectilinear interpolation (matches clinical practice), which resolved both issues. This taught me that architecture choices in continuous-time models have very different stability properties than standard networks."
