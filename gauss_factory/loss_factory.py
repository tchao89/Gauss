# -*- coding: utf-8 -*-

from core.tfdnn.losses.classification_loss import CrossEntropyLoss
from core.tfdnn.losses.regression_loss import MSELoss

class LossFunctionFactory():

    @staticmethod
    def get_loss_function(func_name):
        # TODO: handle Multiclass and Regression tasks. 
        if func_name == "BinaryCrossEntropy":
            return CrossEntropyLoss
        elif func_name == "mse":
            return MSELoss
        else:
            raise "Loss function `{name}` not supported.".format(
                name=func_name
            )
