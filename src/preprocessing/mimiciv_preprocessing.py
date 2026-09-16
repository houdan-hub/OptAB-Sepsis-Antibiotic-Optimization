"""
MIMIC-IV preprocessing pipeline for OptAB.

Steps:
1. Build PostgreSQL database from MIMIC-IV CSV files (per MIT-LCP mimic-code).
2. Extract Sepsis-3 patients and compute SOFA scores via the OpenSEP pipeline.
3. Construct hourly time series with 70+ variables (vitals, labs, treatments).
4. Add missing-value masks and rectilinear time channel for Neural CDE.
5. Standardize continuous variables; split into train/test.

Data access: MIMIC-IV requires credentialed access via PhysioNet.
"""

import os
import pickle
import numpy as np
import pandas as pd

# Variables extracted from MIMIC-IV for OptAB
VITALS = [
    'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature',
    'spo2', 'glucose', 'fio2', 'gcs', 'gcs_eyes', 'gcs_motor', 'gcs_verbal',
    'weight',
]

LABS = [
    'albumin', 'alp', 'alt', 'aniongap', 'ast', 'bicarbonate', 'bilirubin_total',
    'bun', 'calcium', 'chloride', 'creatinine', 'hematocrit', 'hemoglobin',
    'inr', 'lactate', 'lymphocytes', 'magnesium', 'mcv', 'neutrophils',
    'platelet', 'potassium', 'pt', 'ptt', 'rbc', 'rdw', 'sodium', 'wbc',
    'basophils', 'eosinophils', 'monocytes', 'aado2_calc', 'pao2fio2ratio',
    'pco2', 'ph', 'po2', 'totalco2', 'ionized calcium', 'ld_ldh', 'mch', 'mchc',
]

SOFA_COMPONENTS = ['SOFA']

TREATMENTS = ['Vancomycin', 'Piperacillin-Tazobactam', 'Ceftriaxon']

STATIC_VARS = ['age', 'gender', 'ethnicity']

ALL_VARIABLES = ['Time'] + VITALS + LABS + SOFA_COMPONENTS + TREATMENTS


def build_patient_tensor(sepsis_cohort, chartevents, labevents, inputevents,
                         stay_id, max_hours=72):
    """
    Construct an hourly time-series tensor for a single ICU stay.

    Returns:
        data: (T, D) array with Time, vitals, labs, SOFA, treatments, masks.
    """
    # This is a simplified skeleton - the full pipeline uses the OpenSEP
    # Jupyter notebook (Mimic_preprocessing.ipynb) for SOFA computation.
    hours = np.arange(max_hours)
    n_vars = len(ALL_VARIABLES)
    data = np.full((max_hours, n_vars), np.nan)
    data[:, 0] = hours  # Time channel

    # Forward-fill vitals and labs within each stay
    # (Implementation follows OpenSEP pipeline - see notebooks/)
    return data


def add_missing_masks(data):
    """Append binary missing-value masks and normalize time channel.

    Neural CDE with rectilinear interpolation requires:
    - A time channel at index 0.
    - Missing-value indicator channels after each variable.
    """
    N, T, D = data.shape
    masks = (~np.isnan(data)).astype(np.float32)
    # Concatenate masks; time channel mask is always 1
    augmented = np.concatenate([data, masks], axis=-1)
    augmented[:, :, 0] = augmented[:, :, 0] / T  # normalize time
    augmented[:, :, D:] = augmented[:, :, D:] / T  # normalize mask cumulative time
    return augmented


def save_processed_data(data, key_times, variables, static_tensor,
                        static_variables, variables_mean, variables_std,
                        indices_train, indices_test, output_dir):
    """Save all processed artifacts as pickle files."""
    os.makedirs(output_dir, exist_ok=True)
    artifacts = {
        'sepsis_all_1_lab.pkl': data,
        'sepsis_all_1_keys.pkl': key_times,
        'sepsis_all_1_variables_complete.pkl': variables,
        'sepsis_all_1_static.pkl': static_tensor,
        'sepsis_all_1_static_variables.pkl': static_variables,
        'sepsis_all_1_variables_mean.pkl': variables_mean,
        'sepsis_all_1_variables_std.pkl': variables_std,
        'sepsis_all_1_indices_train.pkl': indices_train,
        'sepsis_all_1_indices_test.pkl': indices_test,
    }
    for name, obj in artifacts.items():
        with open(os.path.join(output_dir, name), 'wb') as f:
            pickle.dump(obj, f)
    print(f"Saved {len(artifacts)} artifacts to {output_dir}")
