"""Early stopping utility for training OptAB models."""

import torch
import numpy as np


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        delta: Minimum change to qualify as an improvement.
        path: File path to save the best model checkpoint.
    """

    def __init__(self, patience=10, delta=0.0001, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, aux_model=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model, aux_model)
        elif score < self.best_score + self.delta or torch.isnan(score).any():
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model, aux_model)
            self.counter = 0

    def _save(self, val_loss, model, aux_model=None):
        torch.save(model.state_dict(), self.path)
        if aux_model is not None:
            torch.save(aux_model.state_dict(), self.path + '.decoder')
        self.val_loss_min = val_loss
