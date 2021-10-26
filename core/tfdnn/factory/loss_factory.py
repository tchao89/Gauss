# -*- coding: utf-8 -*-

from core.tfdnn.losses.classification_losses import CrossEntropyLoss
from core.tfdnn.losses.regression_losses import (
    MeanAbsoluteErrorLoss, 
    MeanSquareErrorLoss,
    HuberLoss
    )

class LossFunctionFactory():

    @staticmethod
    def get_loss_function(func_name):
        # TODO: handle Multiclass and Regression tasks. 
        if func_name == "BinaryCrossEntropy":
            return CrossEntropyLoss
        elif func_name == "MeanSquareError":
            return MeanSquareErrorLoss
        elif func_name == "MeanAbsoluteError":
            return MeanAbsoluteErrorLoss
        elif func_name == "Huber":
            return HuberLoss
        else:
            return None
