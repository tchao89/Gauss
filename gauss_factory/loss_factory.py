# -*- coding: utf-8 -*-

from core.tfdnn.losses.cross_entropy_loss import CrossEntropyLoss

class LossFunctionFactory():

    @staticmethod
    def get_loss_function(func_name):
        # TODO: handle Multiclass and Regression tasks. 
        if func_name == "BinaryCrossEntropy":
            return CrossEntropyLoss
        else:
            return None
