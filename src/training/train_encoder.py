"""
Encoder training for OptAB.

The encoder is trained to predict the observed SOFA score and side-effect
biomarkers (creatinine, bilirubin, ALT) from the full patient trajectory,
while also learning a treatment-invariant latent representation via an
auxiliary treatment-classification loss.

Loss = MSE(SOFA) + MSE(side effects) + mu * treatment_classification_loss
"""

import torch
import torch.nn as nn
import numpy as np
import torchcde

from ..model.neural_cde import NeuralCDE
from .early_stopping import EarlyStopping


def train_encoder(model, train_output, train_toxic, train_treatments,
                  covariables, active_entries, static=None,
                  val_output=None, val_toxic=None, val_treatments=None,
                  val_covariables=None, val_active=None, val_static=None,
                  lr=0.005, batch_size=500, patience=10, delta=0.0001,
                  max_epochs=1000, weight_loss=True, rectilinear_index=0,
                  checkpoint_path='Trained_Encoder.pth'):
    """Train the OptAB encoder.

    Args:
        model: NeuralCDE instance in encoder mode.
        train_output: (N, T, 1) observed SOFA scores.
        train_toxic: (N, T, 3) creatinine, bilirubin, ALT.
        train_treatments: (N, T, 3) one-hot antibiotic indicators.
        covariables: (N, T, D) full covariate tensor incl. time channel & masks.
        active_entries: (N, T, 1) boolean mask - patient still in ICU.
        static: (N, S) optional static patient features.

    Returns:
        Final validation loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.train()
    early_stopping = EarlyStopping(patience=patience, delta=delta, path=checkpoint_path)

    # Interpolate the control path (rectilinear interpolation handles missing data)
    train_coeffs = torchcde.linear_interpolation_coeffs(
        covariables, rectilinear=rectilinear_index)
    if val_covariables is not None:
        val_coeffs = torchcde.linear_interpolation_coeffs(
            val_covariables, rectilinear=rectilinear_index)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    softsign = nn.Softsign()

    # mu balances outcome prediction vs treatment-invariance
    mu = model.output_channels / model.treatment_options if weight_loss else 1.0

    datasets = [train_coeffs, train_output, active_entries, train_toxic, train_treatments]
    if static is not None:
        datasets.append(static)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*datasets), batch_size=batch_size, shuffle=True)

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            if static is not None:
                b_coeffs, b_out, b_active, b_toxic, b_trt, b_static = batch
            else:
                b_coeffs, b_out, b_active, b_toxic, b_trt = batch
                b_static = None

            pred_out, pred_a, _ = model(b_coeffs, max_T=covariables.shape[1], static=b_static)

            # Treatment classification loss (encourages treatment-invariant latent state)
            loss_a_comps = []
            for i in range(pred_a.shape[1]):
                for j in range(pred_a.shape[2]):
                    mask = b_active[:, i + 1, 0].bool()
                    if mask.sum() > 0:
                        loss_a_comps.append(softsign(ce(pred_a[:, i, j][mask], b_trt[:, i + 1, j][mask])))
            loss_a = -torch.mean(torch.stack(loss_a_comps)) if loss_a_comps else torch.tensor(0.0, device=model.device)

            # Outcome (SOFA) MSE - only on observed, active entries
            out_mask = b_active[:, 1:, :].bool() & ~b_out.isnan()
            loss_out = mse(pred_out[:, :, 0:1][out_mask], b_out[out_mask])

            # Side-effect MSE
            loss_toxic = 0.0
            for j in range(b_toxic.shape[2]):
                tox_mask = b_active[:, 1:, :].bool() & ~b_toxic[:, :, j:j+1].isnan()
                if tox_mask.sum() > 0:
                    loss_toxic += mse(pred_out[:, :, j+1:j+2][tox_mask], b_toxic[:, :, j:j+1][tox_mask])

            loss = loss_out + loss_toxic + mu * loss_a
            if torch.isnan(loss):
                return loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        if val_output is not None:
            model.eval()
            with torch.no_grad():
                val_pred, _, _ = model(val_coeffs, max_T=val_covariables.shape[1], static=val_static)
                val_mask = val_active[:, 1:, :].bool() & ~val_output.isnan()
                val_loss = mse(val_pred[:, :, 0:1][val_mask], val_output[val_mask])
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break

    return early_stopping.val_loss_min
