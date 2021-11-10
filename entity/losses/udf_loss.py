"""
-*- coding: utf-8 -*-

Copyright (c) 2021, Citic-Lab. All rights reserved.
Authors: citic-lab
"""
import numpy as np
from scipy import special

from entity.losses.base_loss import BaseLoss
from entity.losses.base_loss import LossResult
from utils.constant_values import ConstantValues


# regression loss
class MSELoss(BaseLoss):
    """
    mse loss class, user can get loss, grad and hess by loss_fn method,
    and this object can be used in regression
    """
    def __repr__(self):
        pass

    def __init__(self, **params):
        assert "name" in params, "Key: name must be in params."
        assert "label_name" in params, "Key: label_name must be in params"

        super(MSELoss, self).__init__(name=params["name"],
                                      task_name=ConstantValues.regression)
        self._label_name = params.get("label_name")

    def loss_fn(self,
                score: np.ndarray,
                label: np.ndarray) -> LossResult:

        assert len(label) and len(label) == len(score)
        loss, grad, hess = self._mse_loss(score=score, label=label)

        loss_dict = {"loss": loss,
                     "grad": grad,
                     "hess": hess}
        return LossResult(name="loss_result", result=loss_dict)

    @classmethod
    def _mse_loss(cls, score, label):
        loss = (score - label) * (score - label)
        grad = 2 * (score - label)
        hess = 2
        return loss, grad, hess

# binary classification loss
class BinaryLogLoss(BaseLoss):
    """
    mse loss class, user can get loss, grad and hess by loss_fn method,
    and this object can be used in regression
    """
    def __repr__(self):
        pass

    def __init__(self, **params):
        assert "name" in params, "Key: name must be in params."
        super(BinaryLogLoss, self).__init__(name=params["name"],
                                            task_name=ConstantValues.binary_classification)

        self._label_name = params.get("label_name")
        self._task_name = ConstantValues.binary_classification

    def loss_fn(self,
                score: np.ndarray,
                label: np.ndarray) -> LossResult:
        assert score.ndim == 1, "Numpy array: score must be the shape: (n, )."
        assert label.ndim == 1, "Numpy array: label must be the shape: (n, )."

        assert len(label) and len(label) == len(score)
        loss, grad, hess = self._log_loss(score=score, label=label)

        loss_dict = {"loss": loss,
                     "grad": grad,
                     "hess": hess}
        return LossResult(name="loss_result", result=loss_dict)

    @classmethod
    def _log_loss(cls, score, label):
        prob = special.expit(score)
        loss = np.sum(- label * np.log(prob) - (1 - label) * np.log(1 - prob))
        grad = prob - label
        hess = prob * (1 - prob)
        return loss, grad, hess

    @classmethod
    def _loss_on_point(cls, label: float, prob: float):
        epsilon = 1e-15

        if label <= 0:
            if 1.0 - prob > epsilon:
                return -np.log(1.0 - prob)
        else:
            if prob > epsilon:
                return -np.log(prob)

        return -np.log(epsilon)
