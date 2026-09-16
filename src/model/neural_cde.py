"""
Neural Controlled Differential Equation (Neural CDE) core implementation.

This module implements the encoder-decoder architecture used in OptAB:
- Encoder: processes irregularly-sampled patient time series (vitals, labs, treatments)
  into a continuous latent state via Neural CDE.
- Decoder: initialized from the encoder's terminal hidden state, predicts future
  disease progression (SOFA score + side-effect biomarkers) under hypothetical treatments.

Reference:
    Seedat et al., "Continuous-Time Modeling of Counterfactual Outcomes Using
    Neural Controlled Differential Equations", ICML 2022.
    Kidger et al., "Neural Controlled Differential Equations for Irregular Time
    Series", NeurIPS 2020.
"""

import torch
import torchcde
import numpy as np
import torch.nn as nn


class CDEFunc(nn.Module):
    """
    The vector field f_theta(z) inside the Neural CDE ODE: dz(t) = f_theta(z(t)) dX(t).

    The output is reshaped to (batch, hidden_channels, input_channels) so that
    it acts as a matrix multiplying the control path derivative dX(t).
    """

    def __init__(self, input_channels, hidden_channels, hidden_states=128,
                 activation='tanh', num_depth=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        activations = {
            'leakyrelu': nn.LeakyReLU,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'identity': nn.Identity,
        }
        act = activations[activation]()

        # Progressive-width hidden layers (encoder path)
        diff = hidden_states - hidden_channels
        layers = []
        for i in range(num_depth):
            in_f = hidden_channels + int(np.round(diff * i / num_depth))
            out_f = hidden_channels + int(np.round(diff * (i + 1) / num_depth))
            layers.append(nn.Linear(in_f, out_f))
            layers.append(act)

        # Progressive-width output layers (decoder path -> matrix)
        diff2 = hidden_states - input_channels * hidden_channels
        for i in range(num_depth):
            in_f = hidden_channels * input_channels + int(np.round(diff2 * (num_depth - i) / num_depth))
            out_f = hidden_channels * input_channels + int(np.round(diff2 * (num_depth - i - 1) / num_depth))
            layers.append(nn.Linear(in_f, out_f))
            layers.append(act)

        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        # z: (batch, hidden_channels) -> (batch, hidden_channels * input_channels)
        z = self.net(z)
        return z.view(z.size(0), self.hidden_channels, self.input_channels)


class NeuralCDE(nn.Module):
    """
    Neural CDE model that serves as both Encoder and Decoder in OptAB.

    Encoder mode (z0_dimension_dec=None):
        Input: full observed patient trajectory (covariates + treatments).
        Output: predicted SOFA + side effects, treatment logits, latent states.

    Decoder mode (z0_dimension_dec set):
        Input: time-only control path; initial state z0 comes from encoder.
        Output: counterfactual future trajectory under a chosen treatment.
    """

    def __init__(self, input_channels, hidden_channels, hidden_states=128,
                 output_channels=4, treatment_options=3, activation='tanh',
                 num_depth=1, z0_dimension_dec=None, interpolation='linear',
                 pos=True, thresh=None, pred_comp=True, pred_act='tanh',
                 pred_states=128, pred_depth=1, static_dim=0):
        super().__init__()

        self.func = CDEFunc(input_channels, hidden_channels, hidden_states,
                            activation, num_depth)

        # Initial hidden state network
        if z0_dimension_dec is not None:
            self.initial = nn.Linear(z0_dimension_dec, hidden_channels)
        else:
            self.initial = nn.Linear(input_channels + static_dim, hidden_channels)

        # Readout head (deep MLP if pred_comp)
        self.pred_comp = pred_comp
        if pred_comp:
            act = {'leakyrelu': nn.LeakyReLU, 'tanh': nn.Tanh, 'relu': nn.ReLU,
                   'sigmoid': nn.Sigmoid, 'identity': nn.Identity}[pred_act]()
            diff = pred_states - hidden_channels
            diff2 = pred_states - output_channels
            layers = []
            for i in range(pred_depth):
                in_f = hidden_channels + int(np.round(diff * i / pred_depth))
                out_f = hidden_channels + int(np.round(diff * (i + 1) / pred_depth))
                layers.append(nn.Linear(in_f, out_f))
                layers.append(act)
            for i in range(pred_depth):
                in_f = output_channels + int(np.round(diff2 * (pred_depth - i) / pred_depth))
                out_f = output_channels + int(np.round(diff2 * (pred_depth - i - 1) / pred_depth))
                layers.append(nn.Linear(in_f, out_f))
                if i < pred_depth - 1:
                    layers.append(act)
            self.pred_dec = nn.Sequential(*layers)
        else:
            self.readout = nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation
        self.softplus = nn.Softplus()
        self.treatment_act = nn.Softmax(dim=1)
        self.output_channels = output_channels
        self.treatment = nn.Linear(hidden_channels, treatment_options)
        self.treatment_options = treatment_options
        self.pos = pos
        self.thresh = thresh

        # Auto-detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, coeffs, max_T=None, z0=None, static=None):
        """
        Args:
            coeffs: interpolation coefficients for the control path X(t).
            max_T: number of future time steps to predict (linear mode only).
            z0: optional initial hidden state (decoder mode).
            static: optional static patient features concatenated to X0.

        Returns:
            pred_y: (batch, time, output_channels) - SOFA + side-effect predictions.
            pred_a: (batch, time, treatment_options) - treatment probability logits.
            z_T: (batch, time, hidden_channels) - latent state trajectory.
        """
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
            t_eval = X.interval
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs.float())
            if max_T is not None:
                # Start at t=0.5 so the first step produces a prediction;
                # prepend 0 for the initial state.
                t = np.array([i * 2 for i in range(max_T)], dtype=float)
                t[0] = 0.5
                t = torch.from_numpy(np.insert(t, 0, 0, axis=0)).float().to(self.device)
            else:
                t_eval = X.interval
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation are supported.")

        # Initial hidden state from first observation (+ static features)
        X0 = X.evaluate(X.interval[0]).float()
        if static is not None:
            X0 = torch.cat([X0, static[:, 0, :] if len(static.shape) == 3 else static], dim=1)

        z0 = self.initial(z0 if z0 is not None else X0).float()

        # Solve the CDE: dz(t) = f(z(t)) dX(t)
        if self.interpolation == 'cubic':
            z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=X.interval)
        else:
            z_T = torchcde.cdeint(X=X, z0=z0, func=self.func, t=t,
                                  options=dict(jump_t=X.grid_points))

        z_T = z_T[:, 1:, :]  # drop initial state

        # Readout at each time step
        pred_y = torch.empty(z0.size(0), z_T.size(1), self.output_channels, device=X0.device)
        pred_a = torch.empty(z0.size(0), z_T.size(1), self.treatment_options, device=X0.device)
        for i in range(z_T.size(1)):
            pred_y[:, i, :] = self.pred_dec(z_T[:, i, :]) if self.pred_comp else self.readout(z_T[:, i, :])
            pred_a[:, i] = self.treatment_act(self.treatment(z_T[:, i, :]))

        # Enforce non-negative outputs (SOFA, creatinine, etc.) via shifted softplus
        if self.pos:
            for i in range(self.output_channels):
                pred_y[:, :, i] = self.softplus(pred_y[:, :, i] - self.thresh[i]) + self.thresh[i]

        return pred_y, pred_a, z_T
