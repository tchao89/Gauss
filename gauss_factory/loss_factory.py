# -*- coding: utf-8 -*-

from core.tfdnn.losses.cross_entropy_loss import CrossEntropyLoss

class LossFunctionFactory():

    @staticmethod
    def get_loss_function(task_type):
        # TODO: handle Multiclass and Regression tasks. 
        if task_type == "classification":
            return CrossEntropyLoss
        elif task_type == "regression":
            return None
