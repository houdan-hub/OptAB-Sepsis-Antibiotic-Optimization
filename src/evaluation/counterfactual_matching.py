"""
Counterfactual matching evaluation for OptAB.

Since we cannot run randomized controlled trials on the model's
recommendations, we evaluate counterfactual predictions by matching each
patient (for whom OptAB recommends treatment X) to the most similar patient
who actually received treatment X, then comparing SOFA trajectories.

Two matching strategies:
- Counterfactual matching: patient i's OptAB-recommended treatment -> match
  to patients who actually received that treatment. Measures model bias.
- Factual matching: patient i's actual treatment -> match to similar patients
  with same treatment. Measures baseline patient-to-patient variability.
"""

import numpy as np
import torch
from scipy.stats import t, sem


def counterfactual_matching(opt_treatments, actual_treatments, covariables,
                            predicted_sofa, actual_sofa, opt_times,
                            n_matches=1, covariate_slice=slice(1, 12)):
    """
    Match patients by OptAB-recommended treatment and compute MAE.

    Args:
        opt_treatments: list of treatment tuples recommended by OptAB per patient.
        actual_treatments: (N, T, 3) actual treatment one-hot.
        covariables: (N, T, D) patient covariates for similarity matching.
        predicted_sofa: (N, horizon) model-predicted SOFA under recommended treatment.
        actual_sofa: (N, T) observed SOFA trajectories.
        opt_times: list of decision time indices per patient.
        covariate_slice: which covariate dimensions to use for distance.

    Returns:
        mae_counterfactual: (horizon,) mean absolute error with 95% CI.
        mae_factual: (horizon,) baseline MAE.
    """
    N = len(opt_treatments)
    horizon = predicted_sofa.shape[1] if hasattr(predicted_sofa, 'shape') else len(predicted_sofa[0])

    mae_cf = np.full((N, horizon), np.nan)
    mae_fac = np.full((N, horizon), np.nan)

    for i in range(N):
        # --- Counterfactual matching ---
        recommended = set(opt_treatments[i])
        # Find patients whose actual treatment at onset matches the recommendation
        candidates_cf = []
        for j in range(N):
            if j == i:
                continue
            actual_j = set(np.where(actual_treatments[j, opt_times[j][0]] > 0.5)[0])
            if actual_j == recommended and opt_times[j][0] < 3:
                candidates_cf.append(j)

        if candidates_cf:
            dists = [np.nanmean((covariables[j, 0, covariate_slice] -
                                 covariables[i, 0, covariate_slice]) ** 2)
                     for j in candidates_cf]
            match_j = candidates_cf[int(np.argmin(dists))]
            t0 = opt_times[match_j][0]
            actual_traj = actual_sofa[match_j, t0:t0 + horizon]
            mae_cf[i, :len(actual_traj)] = np.abs(predicted_sofa[i, :len(actual_traj)] - actual_traj)

        # --- Factual matching ---
        actual_i = set(np.where(actual_treatments[i, opt_times[i][0]] > 0.5)[0])
        candidates_fac = []
        for j in range(N):
            if j == i:
                continue
            actual_j = set(np.where(actual_treatments[j, opt_times[j][0]] > 0.5)[0])
            if actual_j == actual_i and opt_times[j][0] < 3:
                candidates_fac.append(j)

        if candidates_fac:
            dists = [np.nanmean((covariables[j, 0, covariate_slice] -
                                 covariables[i, 0, covariate_slice]) ** 2)
                     for j in candidates_fac]
            match_j = candidates_fac[int(np.argmin(dists))]
            t0_i = opt_times[i][0]
            t0_j = opt_times[match_j][0]
            traj_i = actual_sofa[i, t0_i:t0_i + horizon]
            traj_j = actual_sofa[match_j, t0_j:t0_j + horizon]
            min_len = min(len(traj_i), len(traj_j))
            mae_fac[i, :min_len] = np.abs(traj_i[:min_len] - traj_j[:min_len])

    def _ci(x):
        mean = np.nanmean(x, axis=0)
        se = sem(x, axis=0, nan_policy='omit')
        n = np.sum(~np.isnan(x), axis=0)
        ci = t.ppf(0.975, np.maximum(n - 1, 1)) * se / np.sqrt(np.maximum(n, 1))
        return mean, ci

    return _ci(mae_cf), _ci(mae_fac)
