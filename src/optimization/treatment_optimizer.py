"""
Treatment optimization for OptAB.

For each patient at each decision point, enumerate all feasible antibiotic
combinations (from the powerset of {Vancomycin, Pip/Taz, Ceftriaxone}),
predict the 24-48h SOFA trajectory under each combination using the decoder,
and select the combination that minimizes SOFA while respecting
contraindication thresholds (nephrotoxicity via creatinine, hepatotoxicity
via bilirubin / ALT).
"""

import itertools
import torch
import numpy as np


ANTIBIOTICS = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ceftriaxone']


def powerset(iterable):
    """All non-empty proper subsets (excluding empty set and full set)."""
    s = list(iterable)
    return list(itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)))


def optimize_treatment(encoder, decoder, patient_covariables, patient_static,
                       treatment_history, decision_time, horizon=48,
                       creatinine_threshold=None, bilirubin_threshold=None,
                       alt_threshold=None, device='cpu'):
    """
    Recommend the optimal antibiotic combination for a single patient.

    Args:
        encoder: trained NeuralCDE encoder.
        decoder: trained NeuralCDE decoder.
        patient_covariables: (1, T, D) observed covariates up to decision_time.
        patient_static: (1, S) static features.
        treatment_history: (1, T, 3) observed treatments up to decision_time.
        decision_time: int, index of the current decision point.
        horizon: prediction horizon in hours.
        creatinine_threshold: max allowed creatinine (nephrotoxicity contraindication).
        bilirubin_threshold: max allowed bilirubin (hepatotoxicity).
        alt_threshold: max allowed ALT (hepatotoxicity).

    Returns:
        best_treatment: tuple of antibiotic indices.
        best_sofa_trajectory: predicted SOFA over horizon.
        all_results: dict mapping treatment combo -> (sofa_traj, toxic_traj, feasible).
    """
    encoder.eval()
    decoder.eval()

    combos = list(powerset(range(len(ANTIBIOTICS))))
    results = {}

    with torch.no_grad():
        # Encode patient state up to decision time
        _, _, z_T = encoder(patient_covariables[:, :decision_time + 1],
                            max_T=decision_time + 1, static=patient_static)
        z0 = z_T[:, -1, :]  # terminal latent state

        for combo in combos:
            # Build hypothetical treatment signal for the horizon
            trt_signal = torch.zeros(1, horizon, len(ANTIBIOTICS), device=device)
            for idx in combo:
                trt_signal[:, :, idx] = 1.0

            # Decoder predicts future SOFA + side effects
            pred_y, _, _ = decoder(
                torch.cat([torch.zeros(1, horizon, 1, device=device), trt_signal], dim=-1),
                max_T=horizon, z0=torch.cat([z0, patient_static, trt_signal[:, 0]], dim=-1))

            sofa_traj = pred_y[0, :, 0].cpu().numpy()
            creatinine_traj = pred_y[0, :, 1].cpu().numpy()
            bilirubin_traj = pred_y[0, :, 2].cpu().numpy()
            alt_traj = pred_y[0, :, 3].cpu().numpy()

            # Check contraindications
            feasible = True
            if creatinine_threshold and creatinine_traj.max() > creatinine_threshold:
                feasible = False
            if bilirubin_threshold and bilirubin_traj.max() > bilirubin_threshold:
                feasible = False
            if alt_threshold and alt_traj.max() > alt_threshold:
                feasible = False

            results[combo] = (sofa_traj, creatinine_traj, bilirubin_traj, alt_traj, feasible)

    # Select feasible treatment with minimum terminal SOFA
    feasible_results = {k: v for k, v in results.items() if v[4]}
    if not feasible_results:
        # Fallback: lowest SOFA even if constraints violated
        best = min(results.items(), key=lambda x: x[1][0][-1])
    else:
        best = min(feasible_results.items(), key=lambda x: x[1][0][-1])

    return best[0], best[1][0], results
